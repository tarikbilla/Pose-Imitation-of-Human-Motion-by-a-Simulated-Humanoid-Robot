from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from src.perception.landmarks import MEDIAPIPE_POSE_LANDMARKS
from src.types import Keypoint, PoseFrame


@dataclass
class PoseEstimator:
    use_mediapipe: bool = True

    def __post_init__(self) -> None:
        self._mp_pose = None
        self._pose = None
        if self.use_mediapipe:
            try:
                import mediapipe as mp  # type: ignore

                self._mp_pose = mp.solutions.pose
                self._pose = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._pose = None

    def estimate(self, image_bgr: np.ndarray, timestamp_s: float, frame_index: int) -> PoseFrame:
        if self._pose is None:
            return self._estimate_fallback(image_bgr, timestamp_s, frame_index)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return PoseFrame(timestamp_s=timestamp_s, keypoints={}, frame_index=frame_index)

        landmarks = result.pose_landmarks.landmark
        # MediaPipe always returns 33 landmarks in canonical order; map by index.
        keypoints: Dict[str, Keypoint] = {}
        for idx, name in enumerate(MEDIAPIPE_POSE_LANDMARKS):
            lm = landmarks[idx]
            keypoints[name] = Keypoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
        return PoseFrame(timestamp_s=timestamp_s, keypoints=keypoints, frame_index=frame_index)

    def _estimate_fallback(self, image_bgr: np.ndarray, timestamp_s: float, frame_index: int) -> PoseFrame:
        """Synthetic deterministic pose generator for environments without MediaPipe."""
        t = frame_index / 20.0
        cx, cy = 0.5, 0.45
        swing = 0.08 * math.sin(t)

        base: Dict[str, Keypoint] = {
            "nose":             Keypoint(cx,         cy - 0.18),
            "left_eye_inner":   Keypoint(cx - 0.01,  cy - 0.20),
            "left_eye":         Keypoint(cx - 0.02,  cy - 0.20),
            "left_eye_outer":   Keypoint(cx - 0.03,  cy - 0.20),
            "right_eye_inner":  Keypoint(cx + 0.01,  cy - 0.20),
            "right_eye":        Keypoint(cx + 0.02,  cy - 0.20),
            "right_eye_outer":  Keypoint(cx + 0.03,  cy - 0.20),
            "left_ear":         Keypoint(cx - 0.05,  cy - 0.18),
            "right_ear":        Keypoint(cx + 0.05,  cy - 0.18),
            "mouth_left":       Keypoint(cx - 0.015, cy - 0.14),
            "mouth_right":      Keypoint(cx + 0.015, cy - 0.14),
            "left_shoulder":    Keypoint(cx - 0.08,  cy),
            "right_shoulder":   Keypoint(cx + 0.08,  cy),
            "left_elbow":       Keypoint(cx - 0.16,  cy + swing),
            "right_elbow":      Keypoint(cx + 0.16,  cy - swing),
            "left_wrist":       Keypoint(cx - 0.24,  cy + swing * 1.3),
            "right_wrist":      Keypoint(cx + 0.24,  cy - swing * 1.3),
            "left_pinky":       Keypoint(cx - 0.26,  cy + swing * 1.4),
            "right_pinky":      Keypoint(cx + 0.26,  cy - swing * 1.4),
            "left_index":       Keypoint(cx - 0.27,  cy + swing * 1.4),
            "right_index":      Keypoint(cx + 0.27,  cy - swing * 1.4),
            "left_thumb":       Keypoint(cx - 0.255, cy + swing * 1.35),
            "right_thumb":      Keypoint(cx + 0.255, cy - swing * 1.35),
            "left_hip":         Keypoint(cx - 0.07,  cy + 0.20),
            "right_hip":        Keypoint(cx + 0.07,  cy + 0.20),
            "left_knee":        Keypoint(cx - 0.07,  cy + 0.38),
            "right_knee":       Keypoint(cx + 0.07,  cy + 0.38),
            "left_ankle":       Keypoint(cx - 0.07,  cy + 0.55),
            "right_ankle":      Keypoint(cx + 0.07,  cy + 0.55),
            "left_heel":        Keypoint(cx - 0.075, cy + 0.57),
            "right_heel":       Keypoint(cx + 0.075, cy + 0.57),
            "left_foot_index":  Keypoint(cx - 0.05,  cy + 0.58),
            "right_foot_index": Keypoint(cx + 0.05,  cy + 0.58),
        }
        return PoseFrame(timestamp_s=timestamp_s, keypoints=base, frame_index=frame_index)
