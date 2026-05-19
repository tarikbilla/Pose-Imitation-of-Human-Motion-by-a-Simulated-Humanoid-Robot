from __future__ import annotations

from src.utils.filtering import ExponentialSmoother
from src.utils.fps import AdaptiveFPSController


def test_smoother_converges() -> None:
    smoother = ExponentialSmoother(alpha=0.5)
    out = smoother.update({"a": 1.0})
    assert out["a"] == 1.0
    out = smoother.update({"a": 3.0})
    assert out["a"] == 2.0


def test_fps_controller_backs_off_on_high_latency() -> None:
    ctl = AdaptiveFPSController(min_fps=25, max_fps=100, latency_budget_ms=100, step_fps=10, _current_fps=80)
    for _ in range(40):
        ctl.update(measured_latency_ms=300.0)
    assert ctl.current_fps == 25


def test_fps_controller_ramps_up_when_fast() -> None:
    ctl = AdaptiveFPSController(min_fps=25, max_fps=100, latency_budget_ms=200, step_fps=10, _current_fps=30)
    for _ in range(40):
        ctl.update(measured_latency_ms=20.0)
    assert ctl.current_fps == 100
