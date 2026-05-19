from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ExponentialSmoother:
    alpha: float = 0.3
    _state: Dict[str, float] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in values.items():
            prev = self._state.get(key, value)
            smoothed = self.alpha * value + (1.0 - self.alpha) * prev
            self._state[key] = smoothed
            out[key] = smoothed
        return out
