import os
import sys
from dataclasses import dataclass, field

import yaml

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "sites")
)
DEFAULT_SITE = os.environ.get("A3_SITE", "home")

BACKEND_BY_PLATFORM = {"win32": "dshow", "linux": "v4l2", "darwin": "avfoundation"}


def platform_backend():
    for key, backend in BACKEND_BY_PLATFORM.items():
        if sys.platform.startswith(key):
            return backend
    return "dshow"


@dataclass
class Site:
    name: str = "home"
    description: str = ""
    camera: str = "brio100"
    device: str = "dml"
    device_fallback: str = "cpu"
    capture_backend: str = "auto"
    capture_device_hint: str = ""
    pose_mode: str = "lightweight"
    body_height_m: float = 1.75
    environment: dict = field(default_factory=dict)

    @property
    def backend(self):
        if self.capture_backend and self.capture_backend != "auto":
            return self.capture_backend
        return platform_backend()

    @classmethod
    def load(cls, name=None):
        name = name or DEFAULT_SITE
        path = name
        if not os.path.isfile(path):
            path = os.path.join(CONFIG_DIR, f"{name}.yaml")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"site '{name}' not found; expected {path}. "
                f"Available: {', '.join(available()) or 'none'}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(**data)

    def apply_environment(self):
        applied = {}
        for key, value in (self.environment or {}).items():
            if key not in os.environ:
                os.environ[key] = str(value)
                applied[key] = str(value)
        return applied

    def describe(self):
        return {"site": self.name, "camera": self.camera, "device": self.device,
                "backend": self.backend, "pose_mode": self.pose_mode}


def available():
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(os.path.splitext(entry)[0]
                  for entry in os.listdir(CONFIG_DIR)
                  if entry.endswith(".yaml"))
