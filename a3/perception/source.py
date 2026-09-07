import os
import time

import cv2

from .capture import Camera, CameraProfile, apply_rotation

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


class FrameSource:
    is_live = False

    def read(self):
        raise NotImplementedError

    def timestamp(self):
        raise NotImplementedError

    def release(self):
        pass

    def describe(self):
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class CameraSource(FrameSource):
    is_live = True

    def __init__(self, profile):
        self.profile = profile
        self.camera = Camera(profile)
        self._started = None

    def open(self):
        self.camera.open()
        self._started = time.perf_counter()
        return self

    def timestamp(self):
        if self._started is None:
            return 0.0
        return time.perf_counter() - self._started

    def read(self):
        return self.camera.read()

    def release(self):
        self.camera.release()

    def describe(self):
        settings = self.camera.actual_settings()
        return {
            "kind": "camera",
            "source": self.profile.source,
            "rotation": self.profile.rotation,
            "sensor": f"{settings.get('width')}x{settings.get('height')}",
            "fps": settings.get("fps"),
        }

    def __enter__(self):
        return self.open()


class VideoSource(FrameSource):
    is_live = False

    def __init__(self, path, rotation=0, loop=False, realtime=False, start_frame=0):
        self.path = path
        self.rotation = rotation
        self.loop = loop
        self.realtime = realtime
        self.start_frame = start_frame
        self.capture = None
        self.frame_index = 0
        self.frame_count = 0
        self.source_fps = 0.0
        self._next_deadline = None

    def open(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        if self.start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.frame_index = self.start_frame
        self._next_deadline = None
        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError("video not opened")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            if not self.loop:
                return None
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.frame_index = self.start_frame
            ok, frame = self.capture.read()
            if not ok or frame is None:
                return None

        self.frame_index += 1

        if self.realtime and self.source_fps > 0:
            period = 1.0 / self.source_fps
            now = time.perf_counter()
            if self._next_deadline is None:
                self._next_deadline = now + period
            else:
                remaining = self._next_deadline - now
                if remaining > 0:
                    time.sleep(remaining)
                self._next_deadline += period

        return apply_rotation(frame, self.rotation)

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def timestamp(self):
        if self.source_fps <= 0:
            return 0.0
        return (self.frame_index - 1) / self.source_fps

    @property
    def progress(self):
        if not self.frame_count:
            return 0.0
        return min(1.0, self.frame_index / self.frame_count)

    def describe(self):
        return {
            "kind": "video",
            "path": self.path,
            "rotation": self.rotation,
            "frames": self.frame_count,
            "fps": self.source_fps,
            "loop": self.loop,
            "realtime": self.realtime,
        }

    def __enter__(self):
        return self.open()


def looks_like_video(spec):
    if isinstance(spec, int):
        return False
    text = str(spec)
    if text.isdigit():
        return False
    return text.lower().endswith(VIDEO_SUFFIXES) or os.path.isfile(text)


FALLBACK_REASON = []


def open_source(spec, profile=None, loop=False, realtime=False, rotation=None,
                prefer_ffmpeg=True, backend=None, device_hint=None):
    if looks_like_video(spec):
        return VideoSource(
            str(spec),
            rotation=0 if rotation is None else rotation,
            loop=loop,
            realtime=realtime,
        ).open()

    profile = profile or CameraProfile()
    if spec is not None and str(spec).isdigit():
        profile.source = int(spec)
    if rotation is not None:
        profile.rotation = rotation

    FALLBACK_REASON.clear()
    if prefer_ffmpeg:
        try:
            from . import ffmpeg_source

            source = ffmpeg_source.from_profile(
                profile, device_hint=device_hint, backend=backend)
            source.open()
            probe = source.read()
            if probe is not None:
                source._pending = probe
                return source
            FALLBACK_REASON.append(
                "ffmpeg opened but delivered no frame: "
                + (source.stderr_text()[:300] or "no stderr output"))
            source.release()
        except Exception as error:
            FALLBACK_REASON.append(f"{type(error).__name__}: {error}")

    return CameraSource(profile).open()
