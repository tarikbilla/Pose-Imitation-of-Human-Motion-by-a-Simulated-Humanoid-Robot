import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._value = None
        self._derivative = None
        self._last_time = None

    def reset(self):
        self._value = None
        self._derivative = None
        self._last_time = None

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)

        if self._value is None:
            self._value = value.copy()
            self._derivative = np.zeros_like(value)
            self._last_time = timestamp
            return value.copy()

        dt = timestamp - self._last_time
        if dt <= 0.0 or dt > 1.0:
            dt = 1.0 / 30.0
        self._last_time = timestamp

        derivative = (value - self._value) / dt
        alpha_d = _alpha(self.d_cutoff, dt)
        self._derivative = alpha_d * derivative + (1.0 - alpha_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)

        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value.copy()


class KeypointFilter:
    def __init__(self, count, min_cutoff=1.0, beta=0.007, d_cutoff=1.0,
                 score_threshold=0.3, hold_frames=5):
        self.count = count
        self.score_threshold = score_threshold
        self.hold_frames = hold_frames
        self._filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._last_good = None
        self._missing = np.zeros(count, dtype=np.int32)

    def reset(self):
        self._filter.reset()
        self._last_good = None
        self._missing[:] = 0

    def __call__(self, keypoints, scores, timestamp):
        if keypoints is None:
            return None, None

        keypoints = np.asarray(keypoints, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        weak = scores < self.score_threshold
        if self._last_good is not None:
            self._missing[weak] += 1
            self._missing[~weak] = 0
            substitute = weak & (self._missing <= self.hold_frames)
            keypoints = keypoints.copy()
            keypoints[substitute] = self._last_good[substitute]

        smoothed = self._filter(keypoints, timestamp)

        good = ~weak
        if self._last_good is None:
            self._last_good = smoothed.copy()
        else:
            self._last_good[good] = smoothed[good]

        return smoothed, scores
