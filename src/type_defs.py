from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseFrame:
    timestamp_s: float
    keypoints: Dict[str, Keypoint]
    frame_index: int


@dataclass(frozen=True)
class JointCommand:
    timestamp_s: float
    joint_angles_rad: Dict[str, float]
    frame_index: int
