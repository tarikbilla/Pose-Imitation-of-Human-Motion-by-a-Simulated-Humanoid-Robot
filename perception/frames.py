import math

import numpy as np

from .skeleton import A3_INDEX

DECAY = 0.9995
MIN_SAMPLES = 15
MIN_RATIO = 0.15


class SegmentReference:
    def __init__(self, decay=DECAY):
        self.decay = decay
        self._reference = {}
        self._samples = {}

    def update(self, name, length):
        if length <= 0.0:
            return self._reference.get(name, 0.0)
        current = self._reference.get(name, 0.0) * self.decay
        self._reference[name] = max(current, length)
        self._samples[name] = self._samples.get(name, 0) + 1
        return self._reference[name]

    def ready(self, name):
        return self._samples.get(name, 0) >= MIN_SAMPLES

    def get(self, name):
        return self._reference.get(name, 0.0)

    def reset(self):
        self._reference.clear()
        self._samples.clear()


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.zeros(3)
    return vector / norm


def depth_component(measured, reference):
    if reference <= 0.0 or measured <= 0.0:
        return 0.0
    ratio = min(1.0, measured / reference)
    return reference * math.sqrt(max(0.0, 1.0 - ratio * ratio))


def image_delta(keypoints, start, end):
    a = keypoints[A3_INDEX[start]]
    b = keypoints[A3_INDEX[end]]
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64)


def lift_segment(keypoints, start, end, reference, name, scale=1.0, depth_sign=1.0):
    delta = image_delta(keypoints, start, end) / scale
    length = float(np.linalg.norm(delta))
    ref = reference.update(name, length)
    if not reference.ready(name) or ref <= 0.0:
        return None, 0.0
    depth = depth_component(length, ref) * depth_sign
    vector = np.array([depth, delta[0], -delta[1]], dtype=np.float64)
    return unit(vector), length / ref


SCALE_TAU = 2.0


class BodyFrame:
    def __init__(self):
        self.reference = SegmentReference()
        self._yaw_sign = 1.0
        self._scale = None
        self._last_time = None

    def reset(self):
        self.reference.reset()
        self._yaw_sign = 1.0
        self._scale = None
        self._last_time = None

    def update_scale(self, keypoints, timestamp=None):
        nose = keypoints[A3_INDEX["nose"]]
        left = keypoints[A3_INDEX["left_ankle"]]
        right = keypoints[A3_INDEX["right_ankle"]]
        ankle_y = (left[1] + right[1]) / 2.0
        stature = abs(ankle_y - nose[1])

        shoulders = float(np.linalg.norm(
            image_delta(keypoints, "right_shoulder", "left_shoulder")))
        trunk = float(np.linalg.norm(image_delta(keypoints, "hip_center", "neck")))
        raw = max(stature, 3.0 * trunk, 4.0 * shoulders)

        if self._scale is None:
            self._scale = raw
            self._last_time = timestamp
            return self._scale

        dt = 1.0 / 30.0
        if timestamp is not None and self._last_time is not None:
            delta = timestamp - self._last_time
            if 0.0 < delta < 1.0:
                dt = delta
        self._last_time = timestamp

        alpha = dt / (SCALE_TAU + dt)
        self._scale += alpha * (raw - self._scale)
        return self._scale

    @property
    def scale(self):
        return self._scale if self._scale and self._scale > 1e-6 else 1.0

    def torso_yaw(self, keypoints, scores):
        shoulders = float(np.linalg.norm(
            image_delta(keypoints, "right_shoulder", "left_shoulder")))
        hips = float(np.linalg.norm(image_delta(keypoints, "right_hip", "left_hip")))
        if hips <= 1e-6:
            return 0.0, 0.0

        twist = shoulders / hips
        ref = self.reference.update("shoulder_over_hip", twist)
        if not self.reference.ready("shoulder_over_hip") or ref <= 0.0:
            return 0.0, 0.0

        ratio = min(1.0, twist / ref)
        magnitude = math.acos(max(-1.0, min(1.0, ratio)))

        left_ear = scores[A3_INDEX["left_ear"]]
        right_ear = scores[A3_INDEX["right_ear"]]
        if abs(left_ear - right_ear) > 0.15:
            self._yaw_sign = 1.0 if left_ear > right_ear else -1.0

        return magnitude * self._yaw_sign, ratio

    def torso_tilt(self, keypoints):
        neck = keypoints[A3_INDEX["neck"]]
        hip = keypoints[A3_INDEX["hip_center"]]
        delta = np.array([neck[0] - hip[0], neck[1] - hip[1]],
                         dtype=np.float64) / self.scale
        length = float(np.linalg.norm(delta))
        ref = self.reference.update("spine", length)

        roll = math.atan2(delta[0], -delta[1]) if length > 1e-6 else 0.0
        pitch = 0.0
        if self.reference.ready("spine") and ref > 0.0:
            ratio = min(1.0, length / ref)
            pitch = math.acos(max(-1.0, min(1.0, ratio)))
        return pitch, roll

    def head_angles(self, keypoints):
        nose = keypoints[A3_INDEX["nose"]]
        neck = keypoints[A3_INDEX["neck"]]
        left_ear = keypoints[A3_INDEX["left_ear"]]
        right_ear = keypoints[A3_INDEX["right_ear"]]

        ear_delta = np.array([left_ear[0] - right_ear[0],
                              left_ear[1] - right_ear[1]], dtype=np.float64)
        ear_width = float(np.linalg.norm(ear_delta)) / self.scale
        ref = self.reference.update("ears", ear_width)

        yaw = 0.0
        if self.reference.ready("ears") and ref > 0.0:
            ratio = min(1.0, ear_width / ref)
            centre_x = (left_ear[0] + right_ear[0]) / 2.0
            direction = 1.0 if nose[0] > centre_x else -1.0
            yaw = math.acos(max(-1.0, min(1.0, ratio))) * direction

        neck_to_nose = np.array([nose[0] - neck[0], nose[1] - neck[1]],
                                dtype=np.float64) / self.scale
        length = float(np.linalg.norm(neck_to_nose))
        head_ref = self.reference.update("head", length)
        pitch = 0.0
        if self.reference.ready("head") and head_ref > 0.0:
            vertical = -neck_to_nose[1] / max(length, 1e-6)
            pitch = math.asin(max(-1.0, min(1.0, 1.0 - vertical))) if length > 1e-6 else 0.0
        return yaw, pitch

    def arm_directions(self, keypoints, side):
        prefix = "left" if side == "L" else "right"
        upper, upper_ratio = lift_segment(
            keypoints, f"{prefix}_shoulder", f"{prefix}_elbow",
            self.reference, f"{prefix}_upper", scale=self.scale,
        )
        fore, fore_ratio = lift_segment(
            keypoints, f"{prefix}_elbow", f"{prefix}_wrist",
            self.reference, f"{prefix}_fore", scale=self.scale,
        )
        return upper, fore, min(upper_ratio, fore_ratio)

    def hand_normal(self, keypoints, side):
        prefix = "left" if side == "L" else "right"
        wrist = keypoints[A3_INDEX[f"{prefix}_wrist"]]
        index = keypoints[A3_INDEX[f"{prefix}_index_mcp"]]
        pinky = keypoints[A3_INDEX[f"{prefix}_pinky_mcp"]]

        a = np.array([index[0] - wrist[0], -(index[1] - wrist[1]), 0.0])
        b = np.array([pinky[0] - wrist[0], -(pinky[1] - wrist[1]), 0.0])
        span = float(np.linalg.norm(index - pinky)) / self.scale
        ref = self.reference.update(f"{prefix}_palm", span)
        if not self.reference.ready(f"{prefix}_palm") or ref <= 0.0:
            return None

        depth = depth_component(span, ref)
        a3 = np.array([depth * 0.5, a[0], a[1]])
        b3 = np.array([-depth * 0.5, b[0], b[1]])
        normal = np.cross(a3, b3)
        if side == "R":
            normal = -normal
        return unit(normal)
