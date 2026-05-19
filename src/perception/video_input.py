"""Cross-platform video capture wrapper.

Selects the best OpenCV backend per OS (V4L2 on Linux, AVFoundation on macOS,
DirectShow on Windows) and falls back to the default backend if needed.
"""
from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import Generator, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SourceType = Union[int, str]


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray


def _preferred_backend() -> int:
    system = platform.system().lower()
    if system == "linux":
        return cv2.CAP_V4L2
    if system == "darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class VideoCaptureError(RuntimeError):
    """Raised when the camera or video file cannot be opened."""


class VideoSource:
    """Robust OpenCV-backed video source with platform-aware backend selection."""

    def __init__(
        self,
        source: SourceType,
        width: int = 1280,
        height: int = 720,
        preferred_fps: float = 30.0,
    ) -> None:
        self.source = source
        backend = _preferred_backend()
        logger.info("Opening video source %r with backend=%d", source, backend)

        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            logger.warning("Preferred backend failed; retrying with CAP_ANY")
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise VideoCaptureError(
                f"Cannot open video source {source!r}. "
                "Check camera index, device permissions, or file path."
            )

        # Configure capture parameters; some webcams ignore these silently.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, preferred_fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency where supported
        except cv2.error:
            pass

        logger.info(
            "Capture opened: %dx%d @ %.1f FPS",
            self.width, self.height, self.cap.get(cv2.CAP_PROP_FPS),
        )

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_loop(self, target_period_s: float = 0.0) -> Generator[VideoFrame, None, None]:
        """Yield frames until the source ends or too many failures occur."""
        frame_index = 0
        consecutive_failures = 0
        max_failures = 30

        while True:
            start = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive read failures; stopping capture.")
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            yield VideoFrame(
                frame_index=frame_index,
                timestamp_s=time.time(),
                image_bgr=frame,
            )
            frame_index += 1

            if target_period_s > 0:
                elapsed = time.perf_counter() - start
                sleep_s = target_period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
