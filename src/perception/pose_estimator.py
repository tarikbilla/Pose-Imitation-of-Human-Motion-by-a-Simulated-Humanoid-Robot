"""MediaPipe-based human pose estimator.

Design goals:
- Loud, explicit failures (no silent fallback) when MediaPipe cannot load.
- Optional synthetic fallback only when explicitly enabled in config.
- Returns canonical 33-landmark `PoseFrame` per frame, with visibility scores.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from src.perception.landmarks import MEDIAPIPE_POSE_LANDMARKS
from src.types import Keypoint, PoseFrame

logger = logging.getLogger(__name__)


class PoseEstimatorError(RuntimeError):
    """Raised when MediaPipe cannot be initialised and fallback is disabled."""


@dataclass
class PoseEstimator:
    """Wraps MediaPipe Pose with cross-platform initialisation and robust logging."""

    use_mediapipe: bool = True
    model_complexity: int = 1            # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_landmarks: bool = True
    allow_synthetic_fallback: bool = False

    def __post_init__(self) -> None:
        self._mp_pose = None
        self._pose = None

        if not self.use_mediapipe:
            self._warn_about_fallback("pose.use_mediapipe is False")
            return

        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            msg = (
                "MediaPipe is not installed. Install with:\n"
                "    pip install 'mediapipe>=0.10.14'\n"
                "If you are on Python 3.13+, downgrade to Python 3.10–3.12 or use\n"
                "    pip install --pre mediapipe\n"
                f"Original error: {exc}"
            )
            if self.allow_synthetic_fallback:
                logger.error(msg)
                self._warn_about_fallback("MediaPipe import failed")
                return
            raise PoseEstimatorError(msg) from exc

        try:
            # Access the official MediaPipe solutions API. Some binary builds
            # or incompatible wheels (e.g. for Python 3.13) may import but not
            # expose `solutions` — provide a clearer error in that case.
            self._mp_pose = mp.solutions.pose
        except AttributeError as exc:
            msg = (
                "Imported `mediapipe` module does not expose `solutions` (mp.solutions).\n"
                "This commonly happens when MediaPipe is not compatible with the current Python\n"
                "version (MediaPipe supports Python 3.10-3.12). See docs/RUN_INSTRUCTIONS.md\n"
                "for guidance.\nOriginal error: {}".format(exc)
            )
            if self.allow_synthetic_fallback:
                logger.error(msg)
                self._warn_about_fallback("MediaPipe missing `solutions` attribute")
                self._pose = None
                return
            raise PoseEstimatorError(msg) from exc

        try:
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info(
                "MediaPipe Pose initialised (complexity=%d, det=%.2f, trk=%.2f).",
                self.model_complexity,
                self.min_detection_confidence,
                self.min_tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            if self.allow_synthetic_fallback:
                logger.exception("MediaPipe Pose() init failed; using synthetic fallback.")
                self._pose = None
                return
            raise PoseEstimatorError(f"MediaPipe Pose init failed: {exc}") from exc

    # ------------------------------------------------------------------ public

    @property
    def is_real(self) -> bool:
        """True if the real MediaPipe model is active (not the synthetic fallback)."""
        return self._pose is not None

    def estimate(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
        if self._pose is None:
            return self._estimate_fallback(image_bgr, timestamp_s, frame_index)

        # MediaPipe requires an RGB, contiguous, writeable=False array for speed.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return PoseFrame(timestamp_s=timestamp_s, keypoints={}, frame_index=frame_index)

        landmarks = result.pose_landmarks.landmark
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

    def close(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:  # noqa: BLE001
                pass
            self._pose = None

    # ----------------------------------------------------------------- helpers

    def _warn_about_fallback(self, reason: str) -> None:
        logger.warning(
            "Using SYNTHETIC pose fallback (%s). The skeleton will NOT follow the "
            "human; install MediaPipe to enable real pose tracking.",
            reason,
        )

    def _estimate_fallback(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
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
