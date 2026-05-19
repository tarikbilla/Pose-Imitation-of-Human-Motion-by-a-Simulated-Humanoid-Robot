"""Skeleton overlay visualizer for live camera feed."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from src.perception.landmarks import MEDIAPIPE_POSE_LANDMARKS
from src.types import Keypoint, PoseFrame

logger = logging.getLogger(__name__)

# Edges follow MediaPipe's POSE_CONNECTIONS layout (33-landmark skeleton).
POSE_CONNECTIONS: Tuple[Tuple[str, str], ...] = (
    # Face
    ("left_ear", "left_eye_outer"),
    ("left_eye_outer", "left_eye"),
    ("left_eye", "left_eye_inner"),
    ("left_eye_inner", "nose"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("left_index", "left_pinky"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("right_index", "right_pinky"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
)

LANDMARK_COLOR = (0, 200, 255)   # cyan-orange landmarks
SKELETON_COLOR = (255, 255, 255) # white bones
HUD_COLOR = (50, 220, 50)        # green HUD
LOW_VIS_THRESHOLD = 0.4


class SkeletonOverlay:
    """Draws keypoints, bones, and HUD onto a BGR frame."""

    def __init__(self, window_name: str = "Pose Imitation - Camera Feed", show: bool = True) -> None:
        self.window_name = window_name
        self.show = show
        self._window_created = False

    def _ensure_window(self) -> None:
        if self.show and not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

    def draw(
        self,
        frame_bgr: np.ndarray,
        pose: PoseFrame,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        extra_hud: Iterable[str] = (),
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()
        keypoints = pose.keypoints

        # Draw bones
        for a_name, b_name in POSE_CONNECTIONS:
            a = keypoints.get(a_name)
            b = keypoints.get(b_name)
            if a is None or b is None:
                continue
            if a.visibility < LOW_VIS_THRESHOLD or b.visibility < LOW_VIS_THRESHOLD:
                continue
            pa = (int(a.x * w), int(a.y * h))
            pb = (int(b.x * w), int(b.y * h))
            cv2.line(canvas, pa, pb, SKELETON_COLOR, 2, cv2.LINE_AA)

        # Draw landmarks
        for name in MEDIAPIPE_POSE_LANDMARKS:
            kp = keypoints.get(name)
            if kp is None or kp.visibility < LOW_VIS_THRESHOLD:
                continue
            cx, cy = int(kp.x * w), int(kp.y * h)
            cv2.circle(canvas, (cx, cy), 4, LANDMARK_COLOR, -1, cv2.LINE_AA)

        # HUD
        self._draw_hud(canvas, pose, fps, latency_ms, extra_hud)
        return canvas

    def _draw_hud(
        self,
        canvas: np.ndarray,
        pose: PoseFrame,
        fps: float,
        latency_ms: float,
        extra_hud: Iterable[str],
    ) -> None:
        detected = sum(1 for kp in pose.keypoints.values() if kp.visibility >= LOW_VIS_THRESHOLD)
        total = len(MEDIAPIPE_POSE_LANDMARKS)
        status = "HUMAN DETECTED" if detected > total * 0.3 else "NO HUMAN"
        lines = [
            f"FPS: {fps:5.1f}   Latency: {latency_ms:5.1f} ms",
            f"Frame: {pose.frame_index}   Landmarks: {detected}/{total}",
            f"Status: {status}",
            *extra_hud,
        ]
        y = 28
        for line in lines:
            cv2.putText(
                canvas, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2, cv2.LINE_AA,
            )
            y += 24

    def show_frame(self, canvas: np.ndarray) -> bool:
        """Display the canvas. Returns False if the user requested exit."""
        if not self.show:
            return True
        self._ensure_window()
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))  # ESC or 'q' quits

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
