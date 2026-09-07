import time

import numpy as np

from . import runtime, skeleton
from .filters import KeypointFilter
from .skeleton import A3_INDEX, A3_JOINTS, SKELETON_LINKS

MODEL_BODY = "body"
MODEL_WHOLEBODY = "wholebody"

FILTER_MIN_CUTOFF = 3.0
FILTER_BETA = 0.02


class PoseResult:
    def __init__(self, keypoints, scores, inference_ms, raw_count=0):
        self.keypoints = keypoints
        self.scores = scores
        self.inference_ms = inference_ms
        self.raw_count = raw_count

    @property
    def found(self):
        return self.keypoints is not None

    def person(self):
        return self.keypoints, self.scores

    def point(self, name):
        if self.keypoints is None:
            return None, 0.0
        index = A3_INDEX[name]
        return self.keypoints[index], float(self.scores[index])

    def has_hands(self, threshold=0.3):
        return skeleton.has_hands(self.scores, threshold)


class Pose2D:
    def __init__(self, model=MODEL_WHOLEBODY, mode="lightweight",
                 device=runtime.DEVICE_DML, backend="onnxruntime",
                 smooth=True, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA):
        runtime.patch_rtmlib()
        resolved, warning = runtime.resolve_device(device)
        self.device = resolved
        self.warning = warning
        self.model_name = model
        self.mode = mode

        import rtmlib

        if model == MODEL_WHOLEBODY:
            self.model = rtmlib.Wholebody(mode=mode, backend=backend, device=resolved)
        elif model == MODEL_BODY:
            self.model = rtmlib.BodyWithFeet(mode=mode, backend=backend, device=resolved)
        else:
            raise ValueError(f"unknown model '{model}'")

        self.filter = (
            KeypointFilter(len(A3_JOINTS), min_cutoff=min_cutoff, beta=beta)
            if smooth
            else None
        )
        self._frame = 0

    @property
    def supports_hands(self):
        return self.model_name == MODEL_WHOLEBODY

    def reset(self):
        if self.filter is not None:
            self.filter.reset()
        self._frame = 0

    def __call__(self, frame, timestamp=None):
        start = time.perf_counter()
        keypoints, scores = self.model(frame)
        elapsed = (time.perf_counter() - start) * 1000.0

        if keypoints is None or len(keypoints) == 0:
            return PoseResult(None, None, elapsed)

        raw_count = len(keypoints[0])
        mapped, mapped_scores = skeleton.to_a3(keypoints[0], scores[0])

        if self.filter is not None:
            if timestamp is None:
                timestamp = self._frame / 30.0
            mapped, mapped_scores = self.filter(mapped, mapped_scores, timestamp)
            mapped = mapped.astype(np.float32)

        self._frame += 1
        return PoseResult(mapped, mapped_scores, elapsed, raw_count)

    def warmup(self, size=(640, 480), rounds=3):
        blank = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for _ in range(rounds):
            self.model(blank)
        self.reset()


HALPE26_NAMES = A3_JOINTS
HALPE26_INDEX = A3_INDEX
SKELETON = SKELETON_LINKS
