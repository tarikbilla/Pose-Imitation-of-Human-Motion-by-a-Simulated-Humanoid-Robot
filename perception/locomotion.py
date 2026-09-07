import os

import numpy as np

from .skeleton import A3_INDEX

FOCAL_RATIO = float(os.environ.get("A3_FOCAL_RATIO", "0.72"))
FRAME_LONG_PX = float(os.environ.get("A3_FRAME_LONG_PX", "1920"))
BODY_M = float(os.environ.get("A3_BODY_M", "1.75"))
NOSE_ANKLE_RATIO = 0.897

CALIBRATION_S = 2.0
SIZE_TAU = float(os.environ.get("A3_SIZE_TAU", "0.35"))
MIN_CONFIDENCE = 0.30
MIN_SIZE_PX = 20.0
MAX_RATIO = 4.0
PLANE_WEIGHT = float(os.environ.get("A3_PLANE_WEIGHT", "0.5"))
PLANE_MIN_SAMPLES = 90
PLANE_MIN_SPREAD = 0.02
PLANE_MIN_FIT = 0.4
GROUND_KEYS = ("left_ankle", "right_ankle")

MEASURES = (
    ("nose_ankle", ("nose",), ("left_ankle", "right_ankle"), 1),
    ("nose_hip", ("nose",), ("left_hip", "right_hip"), 1),
    ("shoulder_hip", ("left_shoulder", "right_shoulder"),
     ("left_hip", "right_hip"), 1),
    ("hip_ankle", ("left_hip", "right_hip"), ("left_ankle", "right_ankle"), 1),
    ("shoulder_width", ("left_shoulder",), ("right_shoulder",), 0),
)


def _mean(keypoints, names, axis):
    return float(np.mean([keypoints[A3_INDEX[name]][axis] for name in names]))


def _confidence(scores, names):
    return min(float(scores[A3_INDEX[name]]) for name in names)


def body_measures(keypoints, scores):
    sizes = {}
    for name, first, second, axis in MEASURES:
        if _confidence(scores, first + second) < MIN_CONFIDENCE:
            continue
        value = abs(_mean(keypoints, first, axis)
                    - _mean(keypoints, second, axis))
        if value >= MIN_SIZE_PX:
            sizes[name] = value
    return sizes


class LocomotionTracker:
    def __init__(self, focal=None, body=BODY_M, frame_long=FRAME_LONG_PX):
        self.focal = float(focal if focal else FOCAL_RATIO * frame_long)
        self.body = float(body)
        self.reset()

    def reset(self):
        self.samples = {}
        self.plane = None
        self.plane_fit = 0.0
        self.sums = np.zeros(6)
        self.reference = None
        self.reference_hip = None
        self.reference_distance = None
        self.ratio = None
        self.distance = None
        self.forward = 0.0
        self.lateral = 0.0
        self.velocity = 0.0
        self.spread = 0.0
        self.start = None
        self.time = None

    def _calibrate(self, sizes, hip, timestamp):
        for name, value in sizes.items():
            self.samples.setdefault(name, []).append(value)
        self.samples.setdefault("_hip", []).append(hip)
        if timestamp - self.start < CALIBRATION_S:
            return False
        self.reference = {name: float(np.median(values))
                          for name, values in self.samples.items()
                          if not name.startswith("_") and len(values) >= 3}
        if not self.reference:
            return False
        self.reference_hip = float(np.median(self.samples["_hip"]))
        anchor = self.reference.get("nose_ankle")
        self.reference_distance = (
            self.focal * self.body * NOSE_ANKLE_RATIO / anchor if anchor
            else self.focal * self.body / max(
                self.reference.get("nose_hip", 1.0) * 2.0, MIN_SIZE_PX))
        self.samples = {}
        return True

    def _ground_row(self, keypoints, scores):
        if _confidence(scores, GROUND_KEYS) < MIN_CONFIDENCE:
            return None
        return max(float(keypoints[A3_INDEX[name]][1]) for name in GROUND_KEYS)

    def _fit_plane(self, inverse, row):
        self.sums += np.array([1.0, inverse, row, inverse * inverse,
                               inverse * row, row * row])
        count, sx, sy, sxx, sxy, syy = self.sums
        if count < PLANE_MIN_SAMPLES:
            return
        variance = sxx - sx * sx / count
        if variance <= PLANE_MIN_SPREAD * PLANE_MIN_SPREAD * count:
            return
        covariance = sxy - sx * sy / count
        slope = covariance / variance
        intercept = (sy - slope * sx) / count
        spread = syy - sy * sy / count
        self.plane_fit = float(np.clip(
            covariance * covariance / max(variance * spread, 1e-9), 0.0, 1.0))
        if slope > 1.0 and self.plane_fit >= PLANE_MIN_FIT:
            self.plane = (float(slope), float(intercept))

    def update(self, keypoints, scores, timestamp):
        step = 0.0 if self.time is None else max(0.0, timestamp - self.time)
        self.time = timestamp
        if self.start is None:
            self.start = timestamp

        sizes = body_measures(keypoints, scores)
        if not sizes:
            return self.state()
        hip = _mean(keypoints, ("left_hip", "right_hip"), 0)

        if self.reference is None:
            if not self._calibrate(sizes, hip, timestamp):
                return self.state()

        ratios = [self.reference[name] / value
                  for name, value in sizes.items() if name in self.reference]
        if not ratios:
            return self.state()
        raw = float(np.clip(np.median(ratios), 1.0 / MAX_RATIO, MAX_RATIO))
        self.spread = float(np.percentile(ratios, 75)
                            - np.percentile(ratios, 25)) if len(ratios) > 2 else 0.0

        if self.ratio is None:
            self.ratio = raw
        else:
            self.ratio += min(1.0, step / max(SIZE_TAU, 1e-6)) * (raw - self.ratio)

        distance = self.reference_distance * self.ratio
        row = self._ground_row(keypoints, scores)
        if row is not None:
            self._fit_plane(1.0 / max(distance, 0.2), row)
            if self.plane is not None:
                slope, intercept = self.plane
                plane = slope / max(row - intercept, 1.0)
                if 0.2 < plane < MAX_RATIO * self.reference_distance:
                    distance += PLANE_WEIGHT * (plane - distance)
        previous = self.forward
        self.distance = distance
        self.forward = self.reference_distance - distance
        if step > 1e-6:
            self.velocity = (self.forward - previous) / step
        self.lateral = -(hip - self.reference_hip) * distance / self.focal
        return self.state()

    def state(self):
        return {"forward": float(self.forward),
                "lateral": float(self.lateral),
                "velocity": float(self.velocity),
                "moving": bool(abs(self.velocity) > 0.05),
                "ready": self.reference is not None,
                "spread": float(self.spread),
                "plane_fit": float(self.plane_fit),
                "size_px": float(self.ratio or 0.0),
                "distance_m": float(self.distance or 0.0)}
