import math
import os

import numpy as np

from .frames import BodyFrame, lift_segment
from .lift3d import H36M
from .skeleton import A3_INDEX

THIGH = 0.37733
SHANK = 0.422
HIP_OFFSET = np.array([0.05, 0.089, -0.05])
SOLE_OFFSET = np.array([0.048, 0.0, -0.076119])

TORSO_LIMITS = {"BackLbz": (-0.610865, 0.610865),
                "BackMby": (-1.2, 1.28),
                "BackUbx": (-0.790809, 0.790809)}
NECK_LIMITS = (-0.610865238, 1.13446401)
TORSO_GAIN = float(os.environ.get("A3_TORSO_GAIN", "1.0"))

MIN_SCORE = 0.30
DEPTH_BIAS_TAU = 8.0
REACH_MARGIN = 0.97
ELBOW_AUTHORITY = 0.35
RATE_LIMIT = {"Arm": 8.0, "Back": 4.0, "Neck": 4.0, "Leg": 10.0}
DEFAULT_RATE = 8.0
LATERAL_LIMIT = 0.26
STRIDE_GAIN = float(os.environ.get("A3_STRIDE_GAIN", "1.0"))
HEAD_GAIN = float(os.environ.get("A3_HEAD_GAIN", "1.0"))
HEAD_SCALE = math.radians(float(os.environ.get("A3_HEAD_SCALE", "60.0")))
HEAD_NAMES = ("nose", "left_ear", "right_ear")
LEG_NAMES = ("LegUhz", "LegMhx", "LegLhy", "LegKny", "LegUay", "LegLax")


def unit(vector):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / norm if norm > 1e-9 else None


def to_body(vector):
    return unit(np.array([-vector[2], vector[0], -vector[1]],
                         dtype=np.float64))


def clamp(value, low, high):
    return max(low, min(high, value))


def foot_frame(direction, up=(0.0, 0.0, 1.0)):
    forward = unit(direction)
    if forward is None:
        return np.eye(3)
    reference = np.asarray(up, dtype=np.float64)
    side = unit(np.cross(reference, forward))
    if side is None:
        return np.eye(3)
    vertical = np.cross(forward, side)
    return np.column_stack([forward, side, vertical])


class FullBodyRetargeter:
    def __init__(self, leg_ik, arm_solver=None, lifter=None):
        self.lifter = lifter
        self.depth_bias = None
        self.bias_time = None
        self.pitch_bias = None
        self.torso_time = None
        self.head_bias = None
        self.head_time = None
        self.rate_time = None
        self.leg_ik = leg_ik
        self.arm_solver = arm_solver
        self.body = BodyFrame()
        self.seeds = {"L": None, "R": None}
        self.arm_seeds = {"L": None, "R": None}
        self.last = {}
        self.seq = 0

    def reset(self):
        self.body.reset()
        self.seeds = {"L": None, "R": None}
        self.arm_seeds = {"L": None, "R": None}
        self.last = {}
        self.seq = 0

    def _visible(self, scores, names):
        return all(scores[A3_INDEX[name]] >= MIN_SCORE for name in names)

    def leg_angles(self, keypoints, scores, side):
        prefix = "left" if side == "L" else "right"
        needed = (f"{prefix}_hip", f"{prefix}_knee", f"{prefix}_ankle")
        if not self._visible(scores, needed):
            return None, None

        thigh, thigh_ratio = lift_segment(
            keypoints, f"{prefix}_hip", f"{prefix}_knee",
            self.body.reference, f"{prefix}_thigh", scale=self.body.scale)
        shank, shank_ratio = lift_segment(
            keypoints, f"{prefix}_knee", f"{prefix}_ankle",
            self.body.reference, f"{prefix}_shank", scale=self.body.scale)
        if thigh is None or shank is None:
            return None, None

        rotation = np.eye(3)

        hip = HIP_OFFSET.copy()
        if side == "R":
            hip[1] = -hip[1]
        ankle = hip + np.asarray(thigh) * THIGH + np.asarray(shank) * SHANK
        target = ankle + rotation @ SOLE_OFFSET

        values, position_error, angle_error = self.leg_ik[side].solve(
            target, rotation, seed=self.seeds[side])
        self.seeds[side] = values
        angles = {side + name: float(value)
                  for name, value in zip(LEG_NAMES, values)}
        quality = {"position_mm": position_error * 1000.0,
                   "angle_deg": np.degrees(angle_error),
                   "ratio": min(thigh_ratio, shank_ratio)}
        return angles, quality

    def head_from_2d(self, keypoints, scores, timestamp):
        if not self._visible(scores, HEAD_NAMES):
            return None
        left = keypoints[A3_INDEX["left_ear"]]
        right = keypoints[A3_INDEX["right_ear"]]
        nose = keypoints[A3_INDEX["nose"]]
        width = float(np.hypot(left[0] - right[0], left[1] - right[1]))
        shoulders = float(np.hypot(
            keypoints[A3_INDEX["left_shoulder"]][0]
            - keypoints[A3_INDEX["right_shoulder"]][0],
            keypoints[A3_INDEX["left_shoulder"]][1]
            - keypoints[A3_INDEX["right_shoulder"]][1]))
        scale = max(width, 0.35 * shoulders, 1.0)
        raised = (0.5 * (left[1] + right[1]) - nose[1]) / scale
        step = (timestamp - self.head_time
                if self.head_time is not None else 0.0)
        self.head_time = timestamp
        if self.head_bias is None:
            self.head_bias = raised
        else:
            alpha = min(1.0, max(0.0, step / DEPTH_BIAS_TAU))
            self.head_bias += alpha * (raised - self.head_bias)
        return -HEAD_SCALE * (raised - self.head_bias)

    def head_from_3d(self, points, torso_pitch, timestamp):
        head = to_body(points[H36M["head"]] - points[H36M["neck"]])
        if head is None:
            return None
        pitch = math.atan2(head[0], max(head[2], 1e-6)) - torso_pitch
        step = (timestamp - self.head_time
                if self.head_time is not None else 0.0)
        self.head_time = timestamp
        if self.head_bias is None:
            self.head_bias = pitch
        else:
            alpha = min(1.0, max(0.0, step / DEPTH_BIAS_TAU))
            self.head_bias += alpha * (pitch - self.head_bias)
        return pitch - self.head_bias

    def torso_from_3d(self, points, timestamp):
        spine = to_body(points[H36M["neck"]] - points[H36M["hip"]])
        if spine is None:
            return None
        pitch = math.atan2(spine[0], max(spine[2], 1e-6))
        roll = math.atan2(-spine[1], max(spine[2], 1e-6))
        shoulders = to_body(points[H36M["left_shoulder"]]
                            - points[H36M["right_shoulder"]])
        yaw = (math.atan2(-shoulders[0], max(shoulders[1], 1e-6))
               if shoulders is not None else 0.0)

        step = (timestamp - self.torso_time
                if self.torso_time is not None else 0.0)
        self.torso_time = timestamp
        if self.pitch_bias is None:
            self.pitch_bias = pitch
        else:
            alpha = min(1.0, max(0.0, step / DEPTH_BIAS_TAU))
            self.pitch_bias += alpha * (pitch - self.pitch_bias)
        return pitch - self.pitch_bias, roll, yaw

    def leg_target_3d(self, points, side):
        prefix = "left" if side == "L" else "right"
        hip = points[H36M[prefix + "_hip"]]
        knee = points[H36M[prefix + "_knee"]]
        ankle = points[H36M[prefix + "_ankle"]]
        thigh = to_body(knee - hip)
        shank = to_body(ankle - knee)
        if thigh is None or shank is None:
            return None
        origin = HIP_OFFSET.copy()
        if side == "R":
            origin[1] = -origin[1]
        offset = thigh * THIGH + shank * SHANK
        offset[1] = float(np.clip(offset[1], -LATERAL_LIMIT, LATERAL_LIMIT))
        reach = float(np.linalg.norm(offset))
        limit = REACH_MARGIN * (THIGH + SHANK)
        if reach > limit:
            offset = offset * (limit / reach)
        return origin + offset

    def solve_leg(self, side, ankle):
        rotation = np.eye(3)
        target = ankle + rotation @ SOLE_OFFSET
        values, position_error, angle_error = self.leg_ik[side].solve(
            target, rotation, seed=self.seeds[side])
        self.seeds[side] = values
        return ({side + name: float(value)
                 for name, value in zip(LEG_NAMES, values)},
                {"position_mm": position_error * 1000.0,
                 "angle_deg": np.degrees(angle_error), "ratio": 1.0})

    def __call__(self, keypoints, scores, timestamp):
        self.seq += 1
        if keypoints is None:
            return None

        torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not self._visible(scores, torso_names):
            return None
        self.body.update_scale(keypoints, timestamp)

        angles = dict(self.last)
        points = self.lifter(keypoints, scores) if self.lifter else None
        if points is not None:
            points = np.asarray(points)
        torso = self.torso_from_3d(points, timestamp) if points is not None else None
        if torso is not None:
            pitch, roll, yaw = torso
        else:
            yaw, _ = self.body.torso_yaw(keypoints, scores)
            pitch, roll = self.body.torso_tilt(keypoints)
        for name, value in (("BackLbz", yaw), ("BackMby", pitch),
                            ("BackUbx", -roll)):
            angles[name] = clamp(TORSO_GAIN * value, *TORSO_LIMITS[name])

        head_pitch = self.head_from_2d(keypoints, scores, timestamp)
        if head_pitch is None and points is not None:
            head_pitch = self.head_from_3d(points, pitch, timestamp)
        if head_pitch is None and self._visible(scores, HEAD_NAMES):
            _, head_pitch = self.body.head_angles(keypoints)
        if head_pitch is not None:
            angles["NeckAy"] = clamp(HEAD_GAIN * head_pitch, *NECK_LIMITS)

        quality = {}
        if points is not None:
            targets = {side: self.leg_target_3d(points, side)
                       for side in ("L", "R")}
            if all(value is not None for value in targets.values()):
                mean_x = 0.5 * (targets["L"][0] + targets["R"][0])
                for side in ("L", "R"):
                    targets[side][0] = (mean_x + STRIDE_GAIN
                                        * (targets[side][0] - mean_x))
                step = (timestamp - self.bias_time
                        if self.bias_time is not None else 0.0)
                self.bias_time = timestamp
                if self.depth_bias is None:
                    self.depth_bias = mean_x
                else:
                    alpha = min(1.0, max(0.0, step / DEPTH_BIAS_TAU))
                    self.depth_bias += alpha * (mean_x - self.depth_bias)
                for side in ("L", "R"):
                    targets[side][0] -= self.depth_bias
                    leg, info = self.solve_leg(side, targets[side])
                    angles.update(leg)
                    quality[side] = info
        else:
            for side in ("L", "R"):
                leg, info = self.leg_angles(keypoints, scores, side)
                if leg is not None:
                    angles.update(leg)
                    quality[side] = info

        if self.arm_solver is not None:
            for side in ("L", "R"):
                upper, fore, _ = self.body.arm_directions(keypoints, side)
                if upper is None:
                    continue
                solved, _, _ = self.arm_solver(side, upper, fore,
                                               seed=self.arm_seeds[side])
                self.arm_seeds[side] = {
                    "upper": (solved["usy"], solved["shx"]),
                    "fore": (solved.get("ely", 0.0), solved.get("elx", 0.0)),
                }
                mapping = {"usy": "ArmUsy", "shx": "ArmShx",
                           "ely": "ArmEly", "elx": "ArmElx"}
                flexion = abs(float(solved.get("elx", 0.0)))
                authority = min(1.0, flexion / ELBOW_AUTHORITY)
                for key, value in solved.items():
                    if key not in mapping:
                        continue
                    name = side + mapping[key]
                    value = float(value)
                    if key == "ely":
                        previous = self.last.get(name)
                        if previous is not None:
                            value = previous + authority * (value - previous)
                    angles[name] = value

        step = (timestamp - self.rate_time
                if self.rate_time is not None else 0.0)
        self.rate_time = timestamp
        if step > 1e-6 and self.last:
            for name, value in list(angles.items()):
                previous = self.last.get(name)
                if previous is None:
                    continue
                for key, rate in RATE_LIMIT.items():
                    if key in name:
                        break
                else:
                    rate = DEFAULT_RATE
                span = rate * step
                angles[name] = previous + max(-span, min(span, value - previous))

        self.last = angles
        return {"seq": self.seq, "t": timestamp, "angles": angles,
                "quality": quality, "scale": self.body.scale}
