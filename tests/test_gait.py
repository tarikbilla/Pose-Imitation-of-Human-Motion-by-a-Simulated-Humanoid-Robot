"""Tests for the on-robot walk engine (main/libraries/gait.py).

Verifies the safety-critical invariants that can be checked off-simulation:
the degenerate (no-command) output is exactly the proven crouch; the Tier-A
march keeps the CoM between both feet (double support stays valid); the engine
stops by decaying amplitude (never freezes mid-step); the Tier-B single-support
lift refuses to fire without a positive stance margin; and every output respects
the NAO joint limits.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import HIP_OFFSET_Y, NaoCoMModel  # noqa: E402
from gait import GaitEngine, GaitParams, LEG_JOINTS  # noqa: E402
from pose_control_utils import get_default_motor_configs  # noqa: E402

CONFIGS = get_default_motor_configs()
TWO_PI = 2.0 * math.pi


def _march_cmd(cadence=1.0, phase=0.0, swing=1, intensity=1.0, conf=1.0):
    return {
        "state": "march", "cadence_hz": cadence, "phase": phase,
        "swing_side": swing, "intensity": intensity, "turn": 0.0, "conf": conf,
    }


def _full_pose(leg_targets):
    """A realistic whole-body angle dict: rest posture with leg targets overlaid."""
    pose = {name: cfg.rest_angle for name, cfg in CONFIGS.items()}
    pose.update(leg_targets)
    return pose


def test_idle_output_is_the_symmetric_crouch() -> None:
    eng = GaitEngine()
    eng.set_command(None)
    # Two steps so a dt exists; amp_gain stays 0 with no command.
    eng.step(0.0, tier="march")
    targets, meta = eng.step(0.05, tier="march")
    u = eng.params.base_crouch_u
    assert meta["single_support"] is False
    assert abs(targets["LHipPitch"] - (-u)) < 1e-6
    assert abs(targets["LKneePitch"] - (2 * u)) < 1e-6
    assert abs(targets["LAnklePitch"] - (-u)) < 1e-6
    assert abs(targets["LHipRoll"]) < 1e-6 and abs(targets["LAnkleRoll"]) < 1e-6
    # Symmetric left/right.
    assert abs(targets["LHipPitch"] - targets["RHipPitch"]) < 1e-6


def test_stand_tier_is_pure_crouch_regardless_of_command() -> None:
    eng = GaitEngine()
    eng.set_command(_march_cmd(cadence=1.0))
    eng.step(0.0, tier="stand")
    targets, meta = eng.step(0.1, tier="stand")
    u = eng.params.base_crouch_u
    assert abs(targets["LKneePitch"] - 2 * u) < 1e-6
    assert meta["single_support"] is False


def test_amp_gain_rises_then_phase_and_cadence_track_human() -> None:
    eng = GaitEngine()
    eng.set_command(_march_cmd(cadence=0.8, phase=0.0))
    t = 0.0
    meta = {}
    for _ in range(200):  # ~4 s at 50 Hz
        # Feed a moving human phase so phase-lock has something to track.
        human_phase = (TWO_PI * 0.8 * t) % TWO_PI
        eng.set_command(_march_cmd(cadence=0.8, phase=human_phase))
        _, meta = eng.step(t, tier="march")
        t += 0.02
    assert meta["amp_gain"] > 0.9              # ramped up
    assert abs(meta["cadence"] - 0.8) < 0.1    # cadence locked to human
    assert 0.0 <= meta["phase"] < TWO_PI


def test_march_keeps_com_between_both_feet() -> None:
    # Sample the whole gait cycle at full amplitude; the model CoM must stay
    # laterally between the feet (double support remains valid -> no topple).
    eng = GaitEngine(GaitParams())
    model = NaoCoMModel()
    eng.state.amp_gain = 1.0  # force full amplitude
    eng._cmd_intensity = 1.0
    worst = 0.0
    for k in range(48):
        eng.state.phase = TWO_PI * k / 48
        targets, _ = eng._tier_a_march(eff=1.0)
        pose = _full_pose(targets)
        frames = model.frames(pose)
        com = model.com(pose, frames)
        support = 0.5 * (model.foot_sole_center("L", frames)
                         + model.foot_sole_center("R", frames))
        worst = max(worst, abs(com[1] - support[1]))
    # CoM lateral excursion stays well inside half the hip spacing -> both feet
    # remain loaded (it never shifts out over a single foot).
    assert worst < HIP_OFFSET_Y, f"CoM left the double-support region: {worst:.4f} m"


def test_march_outputs_within_joint_limits() -> None:
    eng = GaitEngine()
    eng.state.amp_gain = 1.0
    eng._cmd_intensity = 1.0
    for k in range(48):
        eng.state.phase = TWO_PI * k / 48
        targets, _ = eng._tier_a_march(eff=1.0)
        for name, val in targets.items():
            cfg = CONFIGS.get(name)
            if cfg is not None:
                assert cfg.min_angle - 1e-9 <= val <= cfg.max_angle + 1e-9


def test_knees_are_commanded_every_step() -> None:
    eng = GaitEngine()
    eng.set_command(_march_cmd())
    targets, _ = eng.step(0.0, tier="march")
    assert "LKneePitch" in targets and "RKneePitch" in targets


def test_stop_decays_amplitude_back_to_crouch() -> None:
    eng = GaitEngine()
    # Walk for a while.
    t = 0.0
    for _ in range(150):
        eng.set_command(_march_cmd(cadence=0.8))
        eng.step(t, tier="march")
        t += 0.02
    assert eng.state.amp_gain > 0.5
    # Human stops -> idle command -> amplitude must decay to ~0 (return to crouch).
    for _ in range(150):
        eng.set_command(None)
        targets, meta = eng.step(t, tier="march")
        t += 0.02
    assert meta["amp_gain"] < 0.02
    u = eng.params.base_crouch_u
    assert abs(targets["LKneePitch"] - 2 * u) < 1e-3


def test_high_tilt_aborts_even_with_active_command() -> None:
    eng = GaitEngine()
    t = 0.0
    for _ in range(120):
        eng.set_command(_march_cmd(cadence=0.8))
        # Torso tilted far past the abort threshold every step.
        _, meta = eng.step(t, tier="march", torso_rp=(0.0, 0.6))
        t += 0.02
    assert meta["amp_gain"] < 0.05  # never built up amplitude while tilted


def test_tier_b_refuses_to_lift_without_stance_proof() -> None:
    # No measured pose / no FSR -> the gate cannot prove safety -> never lifts.
    eng = GaitEngine()
    t = 0.0
    lifted = False
    for _ in range(200):
        eng.set_command(_march_cmd(cadence=0.6))
        _, meta = eng.step(t, tier="step", measured=None, fsr=None)
        lifted = lifted or meta["single_support"]
        t += 0.02
    assert lifted is False


def test_tier_b_lifts_when_com_is_over_stance_and_fsr_confirms() -> None:
    eng = GaitEngine()
    eng.state.amp_gain = 1.0
    eng._cmd_intensity = 1.0
    model = NaoCoMModel()
    # Force a stance/swing assignment: phase in [0,pi) -> stance LEFT.
    eng.state.phase = math.pi / 2
    # A pose strongly shifted onto the left foot so its stance margin is positive.
    shifted = _full_pose({"LHipRoll": 0.18, "RHipRoll": 0.18,
                          "LAnkleRoll": 0.10, "RAnkleRoll": 0.10})
    assert model.stance_margin(shifted, "L") > 0.0  # CoM really is over left foot
    targets, meta = eng._tier_b_step(eff=1.0, measured=shifted,
                                     fsr={"L": 0.85, "R": 0.15})
    assert meta["stance"] == "L"
    assert meta["gate_ok"] is True
    assert meta["single_support"] is True


def test_tier_b_gate_blocks_when_fsr_shows_weight_not_transferred() -> None:
    eng = GaitEngine()
    eng.state.amp_gain = 1.0
    eng.state.phase = math.pi / 2
    shifted = _full_pose({"LHipRoll": 0.18, "RHipRoll": 0.18})
    # FSR says weight is still mostly on the swing (right) foot -> no lift.
    _, meta = eng._tier_b_step(eff=1.0, measured=shifted, fsr={"L": 0.3, "R": 0.7})
    assert meta["gate_ok"] is False
    assert meta["single_support"] is False


def test_all_leg_joints_are_known_nao_motors() -> None:
    for name in LEG_JOINTS:
        assert name in CONFIGS
