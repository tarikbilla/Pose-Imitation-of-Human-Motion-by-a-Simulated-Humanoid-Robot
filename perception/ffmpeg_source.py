import os
import re
import subprocess
import threading
import sys
import time

import numpy as np

from .capture import apply_rotation
from .source import FrameSource

DEVICE_PATTERN = re.compile(r'"([^"]+)"\s*\(video\)')
AVFOUNDATION_PATTERN = re.compile(r"\[(\d+)\]\s+(.+)")

BACKEND_BY_PLATFORM = {"win32": "dshow", "linux": "v4l2",
                       "darwin": "avfoundation"}


def default_backend():
    for key, backend in BACKEND_BY_PLATFORM.items():
        if sys.platform.startswith(key):
            return backend
    return "dshow"


def ffmpeg_executable():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def list_video_devices(backend=None):
    backend = backend or default_backend()
    if backend == "dshow":
        command = [ffmpeg_executable(), "-hide_banner", "-f", "dshow",
                   "-list_devices", "true", "-i", "dummy"]
        result = subprocess.run(command, capture_output=True, text=True)
        output = (result.stderr or "") + (result.stdout or "")
        return DEVICE_PATTERN.findall(output)
    if backend == "avfoundation":
        command = [ffmpeg_executable(), "-hide_banner", "-f", "avfoundation",
                   "-list_devices", "true", "-i", ""]
        result = subprocess.run(command, capture_output=True, text=True)
        output = (result.stderr or "") + (result.stdout or "")
        names = []
        for index, label in AVFOUNDATION_PATTERN.findall(output):
            if "AVFoundation audio" in output.split(label)[0][-200:]:
                continue
            names.append(index)
        return names
    return sorted(os.path.join("/dev", entry)
                  for entry in os.listdir("/dev")
                  if entry.startswith("video")) if os.path.isdir("/dev") else []


def find_device(hint, backend=None):
    backend = backend or default_backend()
    devices = list_video_devices(backend)
    if not devices:
        return None
    if hint:
        lowered = str(hint).lower()
        for name in devices:
            if lowered in str(name).lower():
                return name
    return devices[0]


class FFmpegCameraSource(FrameSource):
    is_live = True

    def __init__(
        self,
        device_name,
        width=1920,
        height=1080,
        fps=30,
        vcodec="mjpeg",
        rotation=0,
        low_latency=True,
        drain=True,
        backend=None,
    ):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.vcodec = vcodec
        self.rotation = rotation
        self.backend = backend or default_backend()
        self.low_latency = low_latency
        self.process = None
        self.frame_bytes = width * height * 3
        self.frames_read = 0
        self.frames_dropped = 0
        self.drain = drain
        self._pending = None
        self._started = None
        self._thread = None
        self._latest = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._ended = False

    def _command(self):
        command = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error"]
        if self.low_latency:
            command += ["-fflags", "nobuffer", "-flags", "low_delay"]
        size = f"{self.width}x{self.height}"
        if self.backend == "dshow":
            command += ["-f", "dshow", "-rtbufsize", "64M",
                        "-vcodec", self.vcodec, "-video_size", size,
                        "-framerate", str(self.fps),
                        "-i", f"video={self.device_name}"]
        elif self.backend == "v4l2":
            command += ["-f", "v4l2", "-input_format", self.vcodec,
                        "-video_size", size, "-framerate", str(self.fps),
                        "-i", str(self.device_name)]
        elif self.backend == "avfoundation":
            command += ["-f", "avfoundation", "-video_size", size,
                        "-framerate", str(self.fps),
                        "-i", f"{self.device_name}:none"]
        else:
            raise ValueError(f"unsupported capture backend '{self.backend}'")
        command += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return command

    def open(self):
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
            creationflags=creationflags,
        )
        self._started = time.perf_counter()
        if self.drain:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
        return self

    def _pump(self):
        stream = self.process.stdout
        while not self._stop.is_set():
            raw = stream.read(self.frame_bytes)
            if not raw or len(raw) < self.frame_bytes:
                break
            with self._lock:
                if self._latest is not None:
                    self.frames_dropped += 1
                self._latest = raw
            self._ready.set()
        self._ended = True
        self._ready.set()

    def timestamp(self):
        if self._started is None:
            return 0.0
        return time.perf_counter() - self._started

    def read(self):
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        if self.process is None:
            raise RuntimeError("source not opened")
        if not self.drain:
            raw = self.process.stdout.read(self.frame_bytes)
            if not raw or len(raw) < self.frame_bytes:
                return None
            return self._decode(raw)

        while True:
            self._ready.wait(timeout=5.0)
            with self._lock:
                raw, self._latest = self._latest, None
                if raw is None:
                    self._ready.clear()
            if raw is not None:
                return self._decode(raw)
            if self._ended:
                return None

    def _decode(self, raw):
        self.frames_read += 1
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 3)
        return apply_rotation(frame, self.rotation)

    def stderr_text(self):
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read().decode("utf-8", "replace")
        except Exception:
            return ""

    def release(self):
        self._stop.set()
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()
            self.process = None

    def describe(self):
        return {
            "kind": "ffmpeg-camera",
            "device": self.device_name,
            "sensor": f"{self.width}x{self.height}",
            "vcodec": self.vcodec,
            "backend": self.backend,
            "fps": self.fps,
            "rotation": self.rotation,
            "drain": self.drain,
        }

    def __enter__(self):
        return self.open()


def from_profile(profile, device_hint=None, backend=None):
    backend = backend or default_backend()
    name = find_device(device_hint or profile.name, backend)
    if name is None:
        raise RuntimeError(
            f"no video capture device found for backend '{backend}'")
    return FFmpegCameraSource(
        name,
        width=profile.capture_width,
        height=profile.capture_height,
        fps=profile.fps,
        vcodec="mjpeg",
        rotation=profile.rotation,
        backend=backend,
    )
