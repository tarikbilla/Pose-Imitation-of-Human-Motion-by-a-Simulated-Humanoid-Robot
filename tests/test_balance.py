"""Tests for the model-based CoM balance feedback (main/libraries/balance.py)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import (  # noqa: E402
    BalanceController,
    NaoCoMModel,
    fibonacci_spiral,
)


def test_total_mass_is_realistic() -> None:
    # NAO H25 is ~4.8-5.3 kg depending on version.
    assert 4.5 < NaoCoMModel().total_mass < 5.6


def test_standing_com_sits_over_the_feet() -> None:
    m = NaoCoMModel()
    frames = m.frames({})
    com = m.com({}, frames)
    support = 0.5 * (m.foot_sole_center("L", frames) + m.foot_sole_center("R", frames))
    # Standing, the CoM must be within a few cm (a foot half-length) of centre.
    assert abs(com[0] - support[0]) < 0.05
    assert abs(com[1] - support[1]) < 0.02  # near-symmetric left/right


def test_com_is_above_the_feet() -> None:
    m = NaoCoMModel()
    frames = m.frames({})
    com = m.com({}, frames)
    sole = m.foot_sole_center("L", frames)
    assert com[2] > sole[2] + 0.2  # CoM well above the soles


def test_crouch_keeps_com_centred() -> None:
    # Symmetric crouch should not move the CoM far horizontally.
    m = NaoCoMModel()
    u = 0.35
    crouch = {
        "LHipPitch": -u, "RHipPitch": -u,
        "LKneePitch": 2 * u, "RKneePitch": 2 * u,
        "LAnklePitch": -u, "RAnklePitch": -u,
    }
    frames = m.frames(crouch)
    com = m.com(crouch, frames)
    support = 0.5 * (m.foot_sole_center("L", frames) + m.foot_sole_center("R", frames))
    assert abs(com[0] - support[0]) < 0.06


def test_balanced_pose_needs_no_correction() -> None:
    bc = BalanceController(NaoCoMModel())
    corr = bc.compute_correction({}, (0.0, 0.0))
    assert all(abs(v) < 1e-6 for v in corr.values())


def test_search_reduces_predicted_imbalance() -> None:
    # A strong forward torso tilt is an imbalance the search must respond to and
    # reduce (in the model's own cost metric).
    bc = BalanceController(NaoCoMModel())
    tilt = (0.0, 0.25)  # 0.25 rad nose-down
    before = bc._imbalance({}, tilt)[0]
    corr = bc.compute_correction({}, tilt)
    assert any(abs(v) > 1e-3 for v in corr.values())  # it did something
    after = bc._imbalance(bc._apply_corr({}, bc._state), tilt)[0]
    assert after < before  # and the prediction improved


def test_corrections_are_clamped() -> None:
    bc = BalanceController(NaoCoMModel())
    # Drive many cycles of a large imbalance; corrections must stay bounded.
    for _ in range(50):
        corr = bc.compute_correction({}, (0.4, 0.4))
    assert all(abs(v) <= bc.params.max_ankle_corr + 1e-9 for v in corr.values())


def test_fibonacci_spiral_is_bounded_and_even() -> None:
    pts = fibonacci_spiral(50, 0.2)
    assert len(pts) == 50
    assert all(math.hypot(x, y) <= 0.2 + 1e-9 for x, y in pts)
    # Golden-angle spacing => points are spread, not collinear.
    angles = {round(math.atan2(y, x), 2) for x, y in pts}
    assert len(angles) > 25
