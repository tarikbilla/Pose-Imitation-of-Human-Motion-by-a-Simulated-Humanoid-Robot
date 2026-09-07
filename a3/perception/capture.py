import math
import os
from dataclasses import dataclass, field

import cv2
import yaml

ROTATION_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "cameras")
)


@dataclass
class Intrinsics:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    calibrated: bool = False

    def rotated(self, degrees, width, height):
        if not self.calibrated or degrees == 0:
            return Intrinsics(self.fx, self.fy, self.cx, self.cy, self.calibrated)
        if degrees == 180:
            return Intrinsics(self.fx, self.fy, width - self.cx, height - self.cy, True)
        if degrees == 90:
            return Intrinsics(self.fy, self.fx, height - self.cy, self.cx, True)
        return Intrinsics(self.fy, self.fx, self.cy, width - self.cx, True)


@dataclass
class CameraProfile:
    name: str = "brio100"
    source: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"
    rotation: int = 90
    autofocus: bool = False
    diagonal_fov_deg: float = 58.0
    focal_px: float = 0.0
    intrinsics: Intrinsics = field(default_factory=Intrinsics)

    def focal_for(self, width, height):
        if self.intrinsics.calibrated and self.intrinsics.fx > 1.0:
            return float(self.intrinsics.fx), "calibrated intrinsics"
        if self.focal_px > 1.0:
            return float(self.focal_px), "focal_px from profile"
        if self.diagonal_fov_deg > 1.0:
            diagonal = math.hypot(width, height)
            angle = math.radians(0.5 * self.diagonal_fov_deg)
            return 0.5 * diagonal / math.tan(angle),                 f"{self.diagonal_fov_deg:.0f} deg diagonal field of view"
        return 0.72 * max(width, height), "generic estimate, UNCALIBRATED"

    @property
    def frame_size(self):
        if self.rotation in (90, 270):
            return self.capture_height, self.capture_width
        return self.capture_width, self.capture_height

    @classmethod
    def load(cls, name):
        path = name
        if not os.path.isfile(path):
            path = os.path.join(CONFIG_DIR, f"{name}.yaml")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        intrinsics = Intrinsics(**(data.pop("intrinsics", {}) or {}))
        return cls(intrinsics=intrinsics, **data)

    def save(self, path=None):
        path = path or os.path.join(CONFIG_DIR, f"{self.name}.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "name": self.name,
            "source": self.source,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "fps": self.fps,
            "fourcc": self.fourcc,
            "rotation": self.rotation,
            "autofocus": self.autofocus,
            "diagonal_fov_deg": self.diagonal_fov_deg,
            "focal_px": self.focal_px,
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "calibrated": self.intrinsics.calibrated,
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        return path


def apply_rotation(frame, degrees):
    if degrees == 0:
        return frame
    code = ROTATION_CODES.get(degrees % 360)
    if code is None:
        raise ValueError(f"rotation must be one of 0/90/180/270, got {degrees}")
    return cv2.rotate(frame, code)


def mirror_for_display(frame, keypoints=None):
    mirrored = cv2.flip(frame, 1)
    if keypoints is None:
        return mirrored, None
    width = frame.shape[1]
    flipped = keypoints.copy()
    flipped[..., 0] = width - 1 - flipped[..., 0]
    return mirrored, flipped


class Camera:
    def __init__(self, profile, backend=cv2.CAP_DSHOW):
        self.profile = profile
        self.backend = backend
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.profile.source, self.backend)
        if not self.capture.isOpened():
            fallback = (
                cv2.CAP_MSMF if self.backend == cv2.CAP_DSHOW else cv2.CAP_DSHOW
            )
            self.capture = cv2.VideoCapture(self.profile.source, fallback)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera source {self.profile.source}")

        if self.profile.fourcc:
            self.capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.profile.fourcc)
            )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.profile.capture_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.profile.capture_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.profile.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.profile.autofocus:
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        return self

    def actual_settings(self):
        if self.capture is None:
            return {}
        return {
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.capture.get(cv2.CAP_PROP_FPS),
            "fourcc": int(self.capture.get(cv2.CAP_PROP_FOURCC)),
        }

    def read(self):
        if self.capture is None:
            raise RuntimeError("camera not opened")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return apply_rotation(frame, self.profile.rotation)

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
