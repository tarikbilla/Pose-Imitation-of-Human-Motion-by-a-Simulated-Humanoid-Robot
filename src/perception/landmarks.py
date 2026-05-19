"""MediaPipe Pose 33 landmarks definition.

Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
Diagram: see `docs/` (skeleton figure with IDs 0-32).
"""
from __future__ import annotations

from typing import List, Tuple

# Ordered list of all 33 MediaPipe Pose landmarks (index == landmark ID).
MEDIAPIPE_POSE_LANDMARKS: List[str] = [
    "nose",               # 0
    "left_eye_inner",     # 1
    "left_eye",           # 2
    "left_eye_outer",     # 3
    "right_eye_inner",    # 4
    "right_eye",          # 5
    "right_eye_outer",    # 6
    "left_ear",           # 7
    "right_ear",          # 8
    "mouth_left",         # 9
    "mouth_right",        # 10
    "left_shoulder",      # 11
    "right_shoulder",     # 12
    "left_elbow",         # 13
    "right_elbow",        # 14
    "left_wrist",         # 15
    "right_wrist",        # 16
    "left_pinky",         # 17
    "right_pinky",        # 18
    "left_index",         # 19
    "right_index",        # 20
    "left_thumb",         # 21
    "right_thumb",        # 22
    "left_hip",           # 23
    "right_hip",          # 24
    "left_knee",          # 25
    "right_knee",         # 26
    "left_ankle",         # 27
    "right_ankle",        # 28
    "left_heel",          # 29
    "right_heel",         # 30
    "left_foot_index",    # 31
    "right_foot_index",   # 32
]

NUM_LANDMARKS: int = len(MEDIAPIPE_POSE_LANDMARKS)


def landmark_id(name: str) -> int:
    """Return the MediaPipe landmark index for a given lowercase landmark name."""
    return MEDIAPIPE_POSE_LANDMARKS.index(name)


def enumerate_landmarks() -> List[Tuple[int, str]]:
    return list(enumerate(MEDIAPIPE_POSE_LANDMARKS))
