from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from src.types import JointCommand, Keypoint, PoseFrame


def _vector(a: Keypoint, b: Keypoint) -> Tuple[float, float, float]:
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-8


def _angle_between(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    cosine = max(-1.0, min(1.0, dot / (_norm(a) * _norm(b))))
    return math.acos(cosine)


@dataclass
class JointLimit:
    min_rad: float
    max_rad: float


@dataclass
class RetargetingMapper:
    joint_limits: Dict[str, JointLimit]

    def _clip(self, joint: str, angle: float) -> float:
        lim = self.joint_limits.get(joint)
        if lim is None:
            return angle
        return max(lim.min_rad, min(lim.max_rad, angle))

    def map_pose(self, pose: PoseFrame) -> JointCommand:
        kp = pose.keypoints
        if not kp:
            return JointCommand(timestamp_s=pose.timestamp_s, joint_angles_rad={}, frame_index=pose.frame_index)

        left_upper = _vector(kp["left_shoulder"], kp["left_elbow"])
        left_lower = _vector(kp["left_elbow"], kp["left_wrist"])
        right_upper = _vector(kp["right_shoulder"], kp["right_elbow"])
        right_lower = _vector(kp["right_elbow"], kp["right_wrist"])

        left_hip_to_shoulder = _vector(kp["left_hip"], kp["left_shoulder"])
        right_hip_to_shoulder = _vector(kp["right_hip"], kp["right_shoulder"])

        left_knee_vec = _vector(kp["left_hip"], kp["left_knee"])
        right_knee_vec = _vector(kp["right_hip"], kp["right_knee"])

        raw = {
            "LShoulderPitch": math.atan2(-left_upper[1], abs(left_upper[0]) + 1e-6),
            "RShoulderPitch": math.atan2(-right_upper[1], abs(right_upper[0]) + 1e-6),
            "LElbowRoll": math.pi - _angle_between(left_upper, left_lower),
            "RElbowRoll": -(math.pi - _angle_between(right_upper, right_lower)),
            "LHipPitch": math.atan2(left_knee_vec[1], abs(left_knee_vec[0]) + 1e-6),
            "RHipPitch": math.atan2(right_knee_vec[1], abs(right_knee_vec[0]) + 1e-6),
            "TorsoPitch": 0.5 * (
                math.atan2(-left_hip_to_shoulder[1], abs(left_hip_to_shoulder[0]) + 1e-6)
                + math.atan2(-right_hip_to_shoulder[1], abs(right_hip_to_shoulder[0]) + 1e-6)
            ),
        }

        clipped = {joint: self._clip(joint, angle) for joint, angle in raw.items()}
        return JointCommand(
            timestamp_s=pose.timestamp_s,
            joint_angles_rad=clipped,
            frame_index=pose.frame_index,
        )


def default_joint_limits() -> Dict[str, JointLimit]:
    deg = math.radians
    return {
        "LShoulderPitch": JointLimit(deg(-119), deg(119)),
        "RShoulderPitch": JointLimit(deg(-119), deg(119)),
        "LElbowRoll": JointLimit(deg(0), deg(135)),
        "RElbowRoll": JointLimit(deg(-135), deg(0)),
        "LHipPitch": JointLimit(deg(-88), deg(27)),
        "RHipPitch": JointLimit(deg(-88), deg(27)),
        "TorsoPitch": JointLimit(deg(-30), deg(30)),
    }
