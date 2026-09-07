import collections
import time
import os

import numpy as np
import onnxruntime as ort

from . import runtime
from .skeleton import A3_INDEX

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
DEFAULT_MODEL = os.path.join(MODEL_DIR, "motionbert_lite.onnx")

WINDOW = 27
EVERY_NTH = 2

H36M_FROM_A3 = (
    ("hip_center",), ("right_hip",), ("right_knee",), ("right_ankle",),
    ("left_hip",), ("left_knee",), ("left_ankle",),
    ("neck", "hip_center"), ("neck",), ("nose",), ("left_ear", "right_ear"),
    ("left_shoulder",), ("left_elbow",), ("left_wrist",),
    ("right_shoulder",), ("right_elbow",), ("right_wrist",),
)

H36M = {
    "hip": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
    "left_hip": 4, "left_knee": 5, "left_ankle": 6,
    "spine": 7, "neck": 8, "nose": 9, "head": 10,
    "left_shoulder": 11, "left_elbow": 12, "left_wrist": 13,
    "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16,
}

# measured empirically (tools/check_lifter.py): x = image right,
# y = image down, z = depth away from the camera
AXIS_LATERAL, AXIS_VERTICAL, AXIS_DEPTH = 0, 1, 2

_A3_ROWS = tuple(tuple(A3_INDEX[name] for name in group) for group in H36M_FROM_A3)


def to_h36m(keypoints, scores):
    out = np.zeros((17, 3), dtype=np.float32)
    for row, indices in enumerate(_A3_ROWS):
        out[row, :2] = keypoints[list(indices)].mean(axis=0)
        out[row, 2] = float(np.mean(scores[list(indices)]))
    return out


def crop_scale(motion):
    result = motion.copy()
    valid = motion[motion[..., 2] > 0.0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(motion)
    xmin, xmax = float(valid[:, 0].min()), float(valid[:, 0].max())
    ymin, ymax = float(valid[:, 1].min()), float(valid[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale <= 0.0:
        return np.zeros_like(motion)
    xs = (xmin + xmax - scale) / 2.0
    ys = (ymin + ymax - scale) / 2.0
    result[..., :2] = (motion[..., :2] - np.array([xs, ys], dtype=np.float32)) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2.0
    return np.clip(result, -1.0, 1.0)


class Lifter3D:
    """MotionBERT 2-D to 3-D lifter, causal window, DirectML."""

    def __init__(self, model=DEFAULT_MODEL, window=WINDOW,
                 every_nth=EVERY_NTH, device=runtime.DEVICE_DML):
        self.available = os.path.isfile(model)
        self.window = window
        self.every_nth = max(1, every_nth)
        self.model_path = model
        self.session = None
        self.device = device
        self.warning = None
        self._buffer = collections.deque(maxlen=window)
        self._frame = 0
        self._last = None
        self._last_ms = 0.0

        if not self.available:
            self.warning = (f"lifter model missing: {model}\n"
                            "         build it once with tools/build_lifter.py")
            return

        resolved, warning = runtime.resolve_device(device)
        self.device = resolved
        if warning:
            self.warning = warning
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model, options, providers=runtime.session_providers(resolved)
        )
        self.input_name = self.session.get_inputs()[0].name

    @property
    def ready(self):
        return self.session is not None and len(self._buffer) == self.window

    def reset(self):
        self._buffer.clear()
        self._frame = 0
        self._last = None

    def __call__(self, keypoints, scores):
        if self.session is None or keypoints is None:
            return self._last

        self._buffer.append(to_h36m(keypoints, scores))
        self._frame += 1
        if len(self._buffer) < self.window:
            return self._last
        if (self._frame % self.every_nth) and self._last is not None:
            return self._last


        chunk = crop_scale(np.asarray(self._buffer, dtype=np.float32))[None, ...]
        start = time.perf_counter()
        result = self.session.run(None, {self.input_name: chunk})[0]
        self._last_ms = (time.perf_counter() - start) * 1000.0
        self._last = result[0, -1]
        return self._last

    @property
    def inference_ms(self):
        return self._last_ms

    def describe(self):
        return {
            "model": os.path.basename(self.model_path),
            "available": self.available,
            "device": self.device,
            "window": self.window,
            "every_nth": self.every_nth,
        }
