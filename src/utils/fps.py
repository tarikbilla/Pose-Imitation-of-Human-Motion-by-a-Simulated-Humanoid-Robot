from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AdaptiveFPSController:
    min_fps: float = 25.0
    max_fps: float = 100.0
    latency_budget_ms: float = 150.0
    step_fps: float = 5.0
    _current_fps: float = 25.0
    _latency_window_ms: deque[float] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def current_fps(self) -> float:
        return max(self.min_fps, min(self._current_fps, self.max_fps))

    @property
    def target_period_s(self) -> float:
        return 1.0 / self.current_fps

    def update(self, measured_latency_ms: float, dropped_frame: bool = False) -> float:
        self._latency_window_ms.append(measured_latency_ms)
        avg_latency = sum(self._latency_window_ms) / len(self._latency_window_ms)

        if dropped_frame or avg_latency > self.latency_budget_ms:
            self._current_fps -= self.step_fps
        elif avg_latency < self.latency_budget_ms * 0.65:
            self._current_fps += self.step_fps

        self._current_fps = max(self.min_fps, min(self._current_fps, self.max_fps))
        return self.current_fps
