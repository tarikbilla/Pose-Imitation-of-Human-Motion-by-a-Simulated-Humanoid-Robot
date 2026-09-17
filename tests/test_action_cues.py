"""The lower-body action classifier: does it decide safely?

A wrong verdict here is not one bad frame -- it is a clip, and a clip is several
seconds of the robot doing something open loop. So these tests are about the
FAILURE MODES this project actually recorded, not about coverage:

    false walking     18.8% of one session spent marching at nobody
    late stopping     1218 ms mean, because the stop was judged on a windowed
                      amplitude that could not fall until the window emptied
    flicker           one walk chopped into fragments, each paying a clip
                      restart, a ramp and a handover

Every number asserted below was measured, and the module docstring says where.
"""
from __future__ import annotations

import math

import pytest

from src.perception.action_cues import (
    MIN_TRAVEL_M,
    WALK_ENTER_MPS,
    ActionCue,
    _Debounce,
)
from src.type_defs import Keypoint, PoseFrame

FPS = 13.0
DT = 1.0 / FPS


def frame(t, *, forward_m=0.0, lateral_m=0.0, yaw=0.0, crouch_m=0.0,
          lift_m=0.0, lift_side=0, visible=True):
    """One synthetic pose. ``forward_m`` is where the pelvis has got to along
    the body's own forward axis, so a caller integrates it to make a walk."""
    vis = 1.0 if visible else 0.0
    # Body facing +z (away from camera); forward axis is +z, left is -x.
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    fx, fz = -sin_y, cos_y
    lx, lz = -fz, fx
    px = 1000.0 * (forward_m * fx + lateral_m * lx)
    pz = 4000.0 * 1.0 + 1000.0 * (forward_m * fz + lateral_m * lz)
    half = 190.0
    # Shoulder line perpendicular to forward, so _measure recovers this yaw.
    sx, sz = half * lx, half * lz
    hip_y = -450.0 + 1000.0 * crouch_m
    kps = {
        "left_shoulder": Keypoint(px + sx, -700.0, pz + sz, vis),
        "right_shoulder": Keypoint(px - sx, -700.0, pz - sz, vis),
        "left_hip": Keypoint(px + 100.0 * lx, hip_y, pz + 100.0 * lz, vis),
        "right_hip": Keypoint(px - 100.0 * lx, hip_y, pz - 100.0 * lz, vis),
        "left_ankle": Keypoint(px + 100.0 * lx,
                               0.0 - (1000.0 * lift_m if lift_side > 0 else 0.0),
                               pz + 100.0 * lz, vis),
        "right_ankle": Keypoint(px - 100.0 * lx,
                                0.0 - (1000.0 * lift_m if lift_side < 0 else 0.0),
                                pz - 100.0 * lz, vis),
    }
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=int(t * FPS))


def run(cue, frames):
    out = []
    for f in frames:
        out.append(cue.update(f))
    return out


def walking(seconds, speed_mps, **kw):
    """Frames of someone walking at a steady speed."""
    n = int(seconds / DT)
    return [frame(i * DT, forward_m=speed_mps * i * DT, **kw) for i in range(n)]


def standing(seconds, t0=0.0, **kw):
    n = int(seconds / DT)
    return [frame(t0 + i * DT, **kw) for i in range(n)]


# ---------------------------------------------------------------------------
# The three states that are not the same
# ---------------------------------------------------------------------------
def test_nobody_in_frame_is_unknown_not_idle() -> None:
    """The distinction the whole design rests on. 'Not seen' and 'seen, still'
    demand different responses from the robot -- hold versus stand down -- and
    collapsing them is how a robot keeps walking at a departed human."""
    cue = ActionCue()
    out = run(cue, [frame(i * DT, visible=False) for i in range(20)])
    assert out[-1].action == "unknown"
    assert out[-1].observed is False


def test_a_still_human_is_idle_with_full_confidence() -> None:
    cue = ActionCue()
    out = run(cue, standing(3.0))
    assert out[-1].action == "idle"
    assert out[-1].observed is True
    assert out[-1].confidence == 1.0


def test_losing_the_human_mid_walk_goes_to_unknown_not_on_walking() -> None:
    cue = ActionCue()
    run(cue, walking(3.0, 0.4))
    out = run(cue, [frame(3.0 + i * DT, visible=False) for i in range(30)])
    assert out[-1].action == "unknown", "kept an opinion about a human it cannot see"


# ---------------------------------------------------------------------------
# Walking: start, stop, direction
# ---------------------------------------------------------------------------
def test_a_steady_walk_is_detected() -> None:
    cue = ActionCue()
    out = run(cue, walking(4.0, 0.45))
    assert out[-1].action == "walk_forward", out[-1].reason


def test_walking_backward_is_told_apart_from_forward() -> None:
    """These are different clips. Before the direction window was separated from
    the speed window they alternated within a single stride."""
    cue = ActionCue()
    out = run(cue, walking(4.0, -0.45))
    assert out[-1].action == "walk_backward", out[-1].reason


def test_stopping_is_noticed_quickly() -> None:
    """The recorded failure was 1218 ms mean, because the evidence was a count
    over a window and a count cannot fall until the window empties. Travel has
    no such memory."""
    cue = ActionCue()
    walk = walking(4.0, 0.45)
    run(cue, walk)
    end = walk[-1].timestamp_s
    held = walk[-1].keypoints["left_hip"].z
    stop = [frame(end + i * DT, forward_m=(held - 4000.0) / 1000.0)
            for i in range(1, 30)]
    latency = None
    for cmd, f in zip(run(cue, stop), stop, strict=True):
        if cmd.action != "walk_forward":
            latency = f.timestamp_s - end
            break
    assert latency is not None, "never stopped"
    assert latency < 0.8, f"took {latency*1000:.0f} ms to notice the stop"


def test_a_stationary_human_never_triggers_a_walk() -> None:
    """The safety property, and the one with a measured false-positive rate:
    0.0% over 622 provably-parked frames."""
    cue = ActionCue()
    for cmd in run(cue, standing(20.0)):
        assert cmd.action in ("idle", "unknown"), cmd.reason


def test_a_velocity_spike_without_travel_does_not_trigger_a_walk() -> None:
    """The second witness. One mis-placed landmark makes a velocity spike that
    clears any speed threshold; it cannot make DISPLACEMENT. Measured with speed
    as the only witness, 40 of 74 locomotion episodes involved under 60 mm of
    real travel."""
    cue = ActionCue()
    frames = standing(2.0)
    t = frames[-1].timestamp_s
    # One frame a long way off, then straight back: a spike, not a step.
    frames.append(frame(t + DT, forward_m=0.25))
    frames += [frame(t + (2 + i) * DT) for i in range(30)]
    for cmd in run(cue, frames):
        assert not cmd.action.startswith(("walk", "step")), (
            f"a spike produced {cmd.action}: {cmd.reason}")


def test_creeping_below_the_threshold_is_not_a_walk() -> None:
    cue = ActionCue()
    out = run(cue, walking(6.0, WALK_ENTER_MPS * 0.4))
    assert out[-1].action == "idle", out[-1].reason


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------
def test_a_squat_is_detected_when_standing_still() -> None:
    cue = ActionCue()
    run(cue, standing(3.0))                       # learn the standing height
    out = run(cue, standing(3.0, t0=3.0, crouch_m=0.09))
    assert out[-1].action == "squat", out[-1].reason


def test_a_crouch_while_walking_is_a_stride_not_a_squat() -> None:
    """Asking the robot to squat while the human walks is two clips fighting."""
    cue = ActionCue()
    run(cue, standing(3.0))
    frames = [frame(3.0 + i * DT, forward_m=0.45 * i * DT, crouch_m=0.09)
              for i in range(int(4.0 / DT))]
    out = run(cue, frames)
    assert out[-1].action == "walk_forward", out[-1].reason


@pytest.mark.parametrize(("side", "expected"),
                         [(1, "raise_left"), (-1, "raise_right")])
def test_a_raised_foot_is_detected_on_the_right_side(side, expected) -> None:
    cue = ActionCue()
    run(cue, standing(2.0))
    out = run(cue, standing(3.0, t0=2.0, lift_m=0.12, lift_side=side))
    assert out[-1].action == expected, out[-1].reason


# ---------------------------------------------------------------------------
# Not flickering
# ---------------------------------------------------------------------------
def test_one_dropped_frame_does_not_wipe_the_evidence() -> None:
    """The exact shape of the bug that produced 18.8% false march was the
    opposite -- evidence wiped on a single low-confidence frame. Both directions
    are wrong; this pins the one that costs responsiveness."""
    cue = ActionCue()
    frames = walking(3.0, 0.45)
    t = frames[-1].timestamp_s
    frames.append(frame(t + DT, visible=False))          # one dropout
    frames += [frame(t + (2 + i) * DT, forward_m=0.45 * (t + (2 + i) * DT))
               for i in range(10)]
    out = run(cue, frames)
    assert out[-1].action == "walk_forward", out[-1].reason


def test_a_walk_is_not_chopped_into_fragments() -> None:
    """Fragmentation is the expensive failure: each fragment is a clip restart,
    a ramp and a handover. A steady walk must be ONE episode."""
    cue = ActionCue()
    out = run(cue, walking(8.0, 0.45))
    actions = [c.action for c in out]
    # Count transitions into walk_forward after it has first been entered.
    first = actions.index("walk_forward")
    entries = sum(1 for a, b in zip(actions[first:], actions[first + 1:], strict=False)
                  if a != "walk_forward" and b == "walk_forward")
    assert entries == 0, f"the walk restarted {entries} times"


def test_debounce_is_slower_to_enter_than_to_leave() -> None:
    gate = _Debounce(enter_s=0.30, exit_s=0.10)
    assert gate.update(0.0, True) is False
    assert gate.update(0.20, True) is False, "entered before its hold elapsed"
    assert gate.update(0.31, True) is True
    assert gate.update(0.35, False) is True, "left before its hold elapsed"
    assert gate.update(0.42, False) is True, "0.07 s is still inside the 0.10 s hold"
    assert gate.update(0.46, False) is False


def test_the_travel_threshold_is_the_one_that_was_measured() -> None:
    """Guard against a well-meaning round number replacing a measured one."""
    assert WALK_ENTER_MPS == pytest.approx(0.12)
    assert MIN_TRAVEL_M == pytest.approx(0.08)
