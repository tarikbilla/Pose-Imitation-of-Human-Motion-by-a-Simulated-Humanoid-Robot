"""Tests for monocular gait-cue extraction (src/perception/gait_cues.py).

The extractor must turn a stream of 2D keypoints into a stable gait command:
detect marching cadence under non-uniform frame timing, reject arm-swing
aliasing, report idle when the legs are still or out of frame, and stay
scale-invariant as the subject moves toward/away from the camera.
"""
from __future__ import annotations

import math

from src.perception.gait_cues import GaitCueExtractor
from src.type_defs import Keypoint, PoseFrame

TWO_PI = 2.0 * math.pi


def _march_frame(
    t: float,
    idx: int,
    *,
    freq_hz: float = 1.0,
    scale: float = 1.0,
    knee_amp: float = 0.10,
    cx: float = 0.5,
    leg_vis: float = 1.0,
) -> PoseFrame:
    """A synthetic frontal marcher: shoulders span ``scale``; knees oscillate
    anti-phase by ``knee_amp`` (image y is DOWN, raised knee -> smaller y)."""
    phase = TWO_PI * freq_hz * t
    hip_y = 0.5
    knee_base_y = 0.7
    left_knee_y = knee_base_y - knee_amp * math.sin(phase)   # up when sin>0
    right_knee_y = knee_base_y + knee_amp * math.sin(phase)  # anti-phase
    kps = {
        "left_shoulder": Keypoint(cx - scale / 2, 0.2, 0.0, 1.0),
        "right_shoulder": Keypoint(cx + scale / 2, 0.2, 0.0, 1.0),
        "left_hip": Keypoint(cx - 0.2 * scale, hip_y, 0.0, leg_vis),
        "right_hip": Keypoint(cx + 0.2 * scale, hip_y, 0.0, leg_vis),
        "left_knee": Keypoint(cx - 0.2 * scale, left_knee_y, 0.0, leg_vis),
        "right_knee": Keypoint(cx + 0.2 * scale, right_knee_y, 0.0, leg_vis),
        "left_ankle": Keypoint(cx - 0.2 * scale, 0.9, 0.0, leg_vis),
        "right_ankle": Keypoint(cx + 0.2 * scale, 0.9, 0.0, leg_vis),
    }
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=idx)


def _still_frame(t: float, idx: int, *, arm_swing: bool = False) -> PoseFrame:
    """A standing person; optionally swinging arms (legs perfectly still)."""
    wrist_y = 0.5 + (0.1 * math.sin(TWO_PI * 1.0 * t) if arm_swing else 0.0)
    kps = {
        "left_shoulder": Keypoint(0.4, 0.2, 0.0, 1.0),
        "right_shoulder": Keypoint(0.6, 0.2, 0.0, 1.0),
        "left_hip": Keypoint(0.45, 0.5, 0.0, 1.0),
        "right_hip": Keypoint(0.55, 0.5, 0.0, 1.0),
        "left_knee": Keypoint(0.45, 0.7, 0.0, 1.0),
        "right_knee": Keypoint(0.55, 0.7, 0.0, 1.0),
        "left_wrist": Keypoint(0.4, wrist_y, 0.0, 1.0),
        "right_wrist": Keypoint(0.6, wrist_y, 0.0, 1.0),
    }
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=idx)


def _feed(ex: GaitCueExtractor, frame_fn, *, duration_s: float, start_t: float = 0.0):
    """Feed frames with deliberately NON-uniform dt (mimics AdaptiveFPSController)."""
    t = start_t
    idx = 0
    last = None
    while t < start_t + duration_s:
        last = ex.update(frame_fn(t, idx))
        dt = 0.03 if idx % 2 == 0 else 0.05  # ~25-33 fps, jittered
        t += dt
        idx += 1
    return last


def test_detects_marching_cadence_under_jittered_timing() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    assert cmd.state == "march"
    assert 0.7 <= cmd.cadence_hz <= 1.3  # ~1 Hz despite non-uniform dt
    assert cmd.conf >= 0.9
    assert 0.0 <= cmd.phase < TWO_PI
    assert cmd.intensity > 0.0


def test_faster_march_reads_higher_cadence() -> None:
    slow = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=0.8), duration_s=5.0)
    fast = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.6), duration_s=5.0)
    assert fast.cadence_hz > slow.cadence_hz


def test_arm_swing_while_legs_still_is_rejected() -> None:
    # Vigorous arm swing, legs perfectly still -> must NOT read as marching.
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _still_frame(t, i, arm_swing=True), duration_s=4.0)
    assert cmd.state == "idle"
    assert cmd.cadence_hz == 0.0


def test_standing_still_is_idle() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _still_frame(t, i, arm_swing=False), duration_s=3.0)
    assert cmd.state == "idle"


def test_low_leg_visibility_forces_idle() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _march_frame(t, i, leg_vis=0.1), duration_s=3.0)
    assert cmd.state == "idle"
    assert cmd.conf < 0.6


def test_cadence_is_scale_invariant() -> None:
    # Same march, subject near (large) vs far (small) from the camera.
    near = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0, scale=1.6),
                 duration_s=4.0)
    far = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0, scale=0.5),
                duration_s=4.0)
    assert near.state == "march" and far.state == "march"
    assert abs(near.cadence_hz - far.cadence_hz) < 0.25


def test_swing_side_flips_within_a_cycle() -> None:
    ex = GaitCueExtractor()
    # Warm up to "march", then sample swing_side across half a cycle.
    _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=3.0)
    sides = set()
    t, idx = 3.0, 1000
    for _ in range(40):
        cmd = ex.update(_march_frame(t, idx, freq_hz=1.0))
        sides.add(cmd.swing_side)
        t += 0.03
        idx += 1
    assert 1 in sides and -1 in sides  # both knees lead at some point


def test_stops_promptly_when_marching_ceases() -> None:
    ex = GaitCueExtractor()
    _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    # Now stand still; the command must fall back to idle quickly.
    cmd = _feed(ex, lambda t, i: _still_frame(t, i), duration_s=2.0, start_t=4.0)
    assert cmd.state == "idle"


def test_as_dict_is_json_friendly() -> None:
    import json
    cmd = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    d = cmd.as_dict()
    json.dumps(d)  # must not raise
    assert set(d) == {"state", "cadence_hz", "phase", "swing_side", "intensity", "turn", "conf"}
