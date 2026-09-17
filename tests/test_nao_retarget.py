"""Tests for full-body retargeting (main/libraries/nao_retarget.py).

The leg solve is checked by *round trip*: a synthetic camera-frame 3D pose
(mm, camera coordinates -- see nao_retarget.py's module docstring) is built
from known NAO joint angles via the same swing-twist forward-kinematics
relationship the retargeter inverts, and the retargeter has to recover those
angles from the projected landmarks. That is a much stronger check than
asserting on hand-picked numbers -- it verifies the actual geometry (NAO's
HipRoll -> HipPitch -> KneePitch chain, the ankle levelling, the mirrored
roll signs) rather than just that the code runs.

Unlike the MediaPipe-era version of this test file, landmarks here are
absolute 3D (mm) rather than a 2D image-plane projection, so there is no
approximation/foreshortening error to tolerate in the round trip -- the
recovered angles should match the input angles almost exactly.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from nao_retarget import (  # noqa: E402
    HEAD_PITCH_BASELINE,
    HeadGeometry,
    LowerBodyRetargeter,
    PeakHold,
    _side_sign,
    crouch_posture,
    retarget_full_body,
    retarget_upper_body,
)

# Synthetic subject geometry, in millimeters.
TORSO = 250.0
THIGH = 180.0
SHANK = 180.0
HIP_Y = 550.0
HALF_HIP = 40.0
HALF_SHOULDER = 60.0
CENTER_X = 500.0

# The synthetic shoulders/hips below are placed so that ``_torso_frame``
# computes exactly this identity basis: right=(1,0,0), up=(0,-1,0),
# forward=(0,0,-1). Note this is a construction choice for the test (it does
# not assert anything about which way a real subject faces the camera -- see
# gait_cues.py for that empirical, separately-flagged convention) -- it only
# needs to be internally self-consistent with the forward-kinematics helpers
# below, which is what the round trip actually checks.
def _leg_dir(side, roll, pitch):
    s = _side_sign(side)
    return (
        s * math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
        -math.sin(pitch),
    )


def _leg_landmarks(side, roll_mag, hip_pitch, knee_pitch):
    """Landmarks for one leg posed at the given NAO angles."""
    hip = (CENTER_X + _side_sign(side) * HALF_HIP, HIP_Y, 0.0)
    d1 = _leg_dir(side, roll_mag, hip_pitch)
    knee = tuple(hip[i] + THIGH * d1[i] for i in range(3))
    d2 = _leg_dir(side, roll_mag, hip_pitch + knee_pitch)
    ankle = tuple(knee[i] + SHANK * d2[i] for i in range(3))
    pre = "left_" if side == "L" else "right_"
    return {
        pre + "hip": [hip[0], hip[1], hip[2], 1.0],
        pre + "knee": [knee[0], knee[1], knee[2], 1.0],
        pre + "ankle": [ankle[0], ankle[1], ankle[2], 1.0],
    }


def figure(left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0)):
    """A whole synthetic subject; each leg is ``(roll_mag, hip_pitch, knee)``."""
    sh_y = HIP_Y - TORSO
    kps = {
        "left_shoulder": [CENTER_X - HALF_SHOULDER, sh_y, 0.0, 1.0],
        "right_shoulder": [CENTER_X + HALF_SHOULDER, sh_y, 0.0, 1.0],
        # Arms hanging down, elbows and wrists included so the upper-body
        # retarget has something to solve (not round-trip tested here).
        "left_elbow": [CENTER_X - HALF_SHOULDER - 10.0, sh_y + 110.0, 0.0, 1.0],
        "right_elbow": [CENTER_X + HALF_SHOULDER + 10.0, sh_y + 110.0, 0.0, 1.0],
        "left_wrist": [CENTER_X - HALF_SHOULDER - 20.0, sh_y + 220.0, 0.0, 1.0],
        "right_wrist": [CENTER_X + HALF_SHOULDER + 20.0, sh_y + 220.0, 0.0, 1.0],
        "nose": [CENTER_X, sh_y - 120.0, 0.0, 1.0],
    }
    kps.update(_leg_landmarks("L", *left))
    kps.update(_leg_landmarks("R", *right))
    return kps


def warm(retargeter, frames=80):
    """Let the geometry EMA settle on the standing reference lengths."""
    obs = None
    for _ in range(frames):
        obs = retargeter.observe(figure())
    return obs


# ---------------------------------------------------------------- calibration
def test_peak_hold_rises_fast_and_decays_slowly() -> None:
    ph = PeakHold(rise=0.5, decay=0.01)
    ph.update(1.0)
    assert ph.update(2.0) > 1.4          # rises quickly toward a new peak
    before = ph.value
    ph.update(0.1)                        # a low sample
    assert ph.value > 0.9 * before        # barely moves the reference
    assert ph.update(float("nan")) == ph.value


def test_calibration_recovers_segment_lengths() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    # Real mm measurements, EMA-smoothed but not foreshortened -- tight
    # tolerance compared to the old MediaPipe-era scale-recovery test.
    assert abs(r.geom.torso - TORSO) < 1e-2
    assert abs(r.geom.thigh - THIGH) < 1e-2
    assert abs(r.geom.shank - SHANK) < 1e-2
    assert r.geom.calibrated


def test_uncalibrated_observation_is_invalid() -> None:
    # One frame is not enough to trust the proportions... but it must never throw.
    obs = LowerBodyRetargeter().observe({})
    assert obs.valid is False
    assert obs.leg("L") is None


# --------------------------------------------------------------- the leg solve
def test_standing_leg_solves_to_zero() -> None:
    obs = warm(LowerBodyRetargeter())
    for side in ("L", "R"):
        leg = obs.leg(side)
        assert abs(leg.hip_pitch) < 0.02
        assert abs(leg.hip_roll) < 0.02
        assert abs(leg.knee_pitch) < 0.02
        assert leg.lift < 0.02
    assert obs.crouch_u == 0.0
    assert obs.stance_side == ""


def test_round_trip_recovers_hip_and_knee_angles() -> None:
    cases = [
        (0.0, -0.60, 1.20),     # knee lifted forward, shank folded under
        (0.0, -1.20, 1.40),     # high march step
        (0.35, -0.30, 0.50),    # abducted and flexed
        (0.60, 0.0, 0.0),       # pure abduction
        (0.0, -0.40, 0.80),     # shallow crouch on one leg
    ]
    for roll, hip, knee in cases:
        r = LowerBodyRetargeter()
        warm(r)
        obs = r.observe(figure(left=(roll, hip, knee)))
        leg = obs.left
        assert abs(leg.hip_roll - roll) < 0.01, (roll, hip, knee, leg.hip_roll)
        assert abs(leg.hip_pitch - hip) < 0.01, (roll, hip, knee, leg.hip_pitch)
        assert abs(leg.knee_pitch - knee) < 0.01, (roll, hip, knee, leg.knee_pitch)


def test_roll_signs_are_mirrored_between_legs() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.5, 0.0, 0.0), right=(0.5, 0.0, 0.0)))
    # Both legs abducted outward: NAO wants LHipRoll positive, RHipRoll negative.
    assert obs.left.hip_roll > 0.4
    assert obs.right.hip_roll < -0.4
    # The ankle cancels the hip so each sole stays level.
    assert abs(obs.left.ankle_roll + obs.left.hip_roll) < 1e-9
    assert abs(obs.right.ankle_roll + obs.right.hip_roll) < 1e-9


def test_ankle_keeps_the_sole_flat() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.0, -0.5, 1.0), right=(0.0, -0.5, 1.0)))
    for leg in (obs.left, obs.right):
        # Hip + knee + ankle == 0 => torso vertical and sole flat.
        assert abs(leg.hip_pitch + leg.knee_pitch + leg.ankle_pitch) < 1e-9


def test_knee_never_hyperextends() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    for hip in (-1.2, -0.6, 0.0):
        obs = r.observe(figure(left=(0.0, hip, 0.0)))
        assert obs.left.knee_pitch >= 0.0


# ------------------------------------------------------------------- lift/stance
def test_single_leg_lift_is_detected_on_the_right_side() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    # A high march step on the left: hip flexed, knee folded -> the foot rises.
    obs = r.observe(figure(left=(0.0, -1.1, 1.5)))
    assert obs.left.lift > 0.5
    assert obs.right.lift < 0.05
    # Stance is the OTHER foot -- this is what tells the controller which way to
    # transfer weight before the step.
    assert obs.stance_side == "R"


def test_both_feet_planted_is_not_a_step() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.0, -0.4, 0.8), right=(0.0, -0.4, 0.8)))
    assert obs.stance_side == ""
    assert obs.left.lift < 0.05 and obs.right.lift < 0.05


def test_symmetric_squat_reports_a_crouch() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    shallow = r.observe(figure(left=(0.0, -0.3, 0.6), right=(0.0, -0.3, 0.6)))
    deep = r.observe(figure(left=(0.0, -0.7, 1.4), right=(0.0, -0.7, 1.4)))
    assert 0.0 < shallow.crouch_u < deep.crouch_u
    from nao_retarget import MAX_CROUCH
    assert deep.crouch_u <= MAX_CROUCH


def test_the_crouch_keeps_the_ankle_under_the_hip_at_any_depth() -> None:
    """This is why the crouch cap is a joint-range limit and not a safety one:
    NAO's thigh and shank are within 3 mm of the same length, so the hip stays
    over the ankle however deep the squat goes."""
    from balance import THIGH_LENGTH, TIBIA_LENGTH
    for u in (0.0, 0.35, 0.70, 1.0):
        p = crouch_posture(u)
        # Forward offset of the ankle from the hip, from the two segment pitches.
        offset = (THIGH_LENGTH * math.sin(u)
                  + TIBIA_LENGTH * math.sin(-(u)))
        assert abs(offset) < 0.004
        for side in ("L", "R"):
            total = (p[f"{side}HipPitch"] + p[f"{side}KneePitch"]
                     + p[f"{side}AnklePitch"])
            assert abs(total) < 1e-12          # torso vertical, sole flat


def test_lift_is_invariant_to_camera_distance() -> None:
    """Real 3D coordinates are already metric, so moving the whole subject
    farther from the camera (a uniform z shift) must not read as a lift --
    unlike MediaPipe's foreshortened 2D projection, there is no scale
    ambiguity left to introduce spurious lift, but a uniform depth offset is
    still worth checking since the leg solve uses z for the fwd/backward
    sign."""
    r = LowerBodyRetargeter()
    warm(r)
    far = {}
    for name, v in figure().items():
        far[name] = [v[0], v[1], v[2] + 1500.0, v[3]]
    obs = r.observe(far)
    assert obs.left.lift < 0.1 and obs.right.lift < 0.1


# ------------------------------------------------------------------ robustness
def test_invisible_leg_is_omitted_not_guessed() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    kps = figure()
    for name in ("left_hip", "left_knee", "left_ankle"):
        kps[name] = kps[name][:3] + [0.1]      # visibility below threshold
    obs = r.observe(kps)
    assert obs.left is None
    assert obs.right is not None
    # A one-legged read is explicitly less trusted.
    assert obs.confidence <= 0.5


def test_garbage_landmarks_do_not_raise() -> None:
    r = LowerBodyRetargeter()
    for payload in ({}, {"left_hip": []}, {"left_hip": ["x", "y"]},
                    {"left_hip": [float("nan"), 0.0, 0.0, 1.0]}):
        assert r.observe(payload).valid is False


# ------------------------------------------------------------------ upper body
def test_upper_body_returns_no_leg_joints() -> None:
    targets = retarget_upper_body(figure())
    assert targets
    assert not any("Hip" in n or "Knee" in n or "Ankle" in n for n in targets)


def test_upper_body_needs_no_hips_visible() -> None:
    """A desk-framed webcam (waist up only) should still drive arms/head --
    see nao_retarget._torso_frame's camera-vertical fallback."""
    kps = figure()
    for name in ("left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle"):
        kps.pop(name, None)
    targets = retarget_upper_body(kps)
    assert "LShoulderPitch" in targets and "RShoulderPitch" in targets


def test_full_body_includes_legs_only_when_asked() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    without = retarget_full_body(figure(), drive_legs=False)
    with_legs = retarget_full_body(figure(left=(0.0, -0.5, 1.0)),
                                  drive_legs=True, retargeter=r)
    assert "LKneePitch" not in without
    assert "LKneePitch" in with_legs


def test_swap_sides_mirrors_every_joint() -> None:
    normal = retarget_upper_body(figure())
    mirrored = retarget_upper_body(figure(), swap_sides=True)
    assert abs(normal["LShoulderPitch"] - mirrored["RShoulderPitch"]) < 1e-9


def test_crouch_posture_is_statically_balanced() -> None:
    for u in (0.0, 0.1, 0.35):
        p = crouch_posture(u)
        for side in ("L", "R"):
            total = p[f"{side}HipPitch"] + p[f"{side}KneePitch"] + p[f"{side}AnklePitch"]
            assert abs(total) < 1e-12   # torso vertical, sole flat
        assert p["LHipRoll"] == p["RHipRoll"] == 0.0


# ------------------------------------------------- head-pitch self-calibration
# ``figure()``'s nose sits 120 mm above a 120 mm shoulder span, i.e. exactly
# 1.0 shoulder-widths -- a subject whose neck is longer than the population
# average encoded in HEAD_PITCH_BASELINE.
def _with_neck(nose_height: float):
    """``figure()`` with the nose at ``nose_height`` shoulder-widths up."""
    kps = figure()
    shoulder_w = 2.0 * HALF_SHOULDER
    kps["nose"] = [CENTER_X, (HIP_Y - TORSO) - nose_height * shoulder_w, 0.0, 1.0]
    return kps


def test_an_off_average_neck_biases_the_head_without_calibration() -> None:
    """The regression this class exists for: a subject looking straight ahead
    gets a standing head tilt purely because their neck is not average."""
    pitch = retarget_upper_body(_with_neck(1.30))["HeadPitch"]
    assert abs(pitch) > 0.5   # ~ -37 deg of permanent nod


def test_calibration_removes_the_bias_for_a_neutral_head() -> None:
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        targets = retarget_upper_body(_with_neck(1.30), head_geom=geom)
    assert geom.calibrated
    assert abs(targets["HeadPitch"]) < 1e-6
    assert abs(geom.baseline - 1.30) < 1e-6


def test_calibration_does_not_flatten_real_head_motion() -> None:
    """Calibrating away the OFFSET must not calibrate away the SIGNAL."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        retarget_upper_body(_with_neck(1.30), head_geom=geom)
    # Nose drops toward the shoulders -> looking down -> positive pitch.
    assert retarget_upper_body(_with_neck(1.10), head_geom=geom)["HeadPitch"] > 0.2
    # ... and the opposite way for looking up.
    geom_up = HeadGeometry()
    for _ in range(geom_up.warmup):
        retarget_upper_body(_with_neck(1.30), head_geom=geom_up)
    assert retarget_upper_body(_with_neck(1.50), head_geom=geom_up)["HeadPitch"] < -0.2


def test_the_first_frame_replaces_the_population_default() -> None:
    geom = HeadGeometry()
    assert geom.baseline == HEAD_PITCH_BASELINE
    assert geom.update(1.40) == 1.40      # not averaged with the default


def test_non_finite_samples_cannot_poison_the_neutral() -> None:
    geom = HeadGeometry()
    geom.update(1.20)
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert geom.update(bad) == 1.20


def test_a_brief_glance_does_not_move_a_settled_neutral() -> None:
    """After warm-up the neutral drifts slowly, so looking away for a second
    must not drag the robot's idea of 'straight ahead' with it."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        geom.update(1.00)
    for _ in range(50):                    # ~1.7 s of looking down
        geom.update(0.60)
    assert abs(geom.baseline - 1.00) < 0.05


def test_a_new_subject_eventually_recalibrates() -> None:
    """The slow decay is what stops the second person in front of the camera
    from inheriting the first one's neck."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        geom.update(1.00)
    for _ in range(2000):
        geom.update(1.40)
    assert abs(geom.baseline - 1.40) < 0.05


# ------------------------------------------------- arm fore/aft axis (chirality)
# These use a subject built to face the camera (left shoulder at POSITIVE x --
# your left is on my right), because that is what a real estimator reports and
# it is the orientation under which the fore/aft sign actually matters.
#
# The suite could not catch a front-to-back inversion before: figure() hangs the
# arms straight down with z = 0, where the fore/aft component of the arm bone is
# zero and its sign therefore cannot change the answer.
def _facing_subject(left_arm_forward_rad: float = 0.0):
    """A camera-facing figure whose LEFT arm is swung forward by the given angle.

    Camera frame: x right, y DOWN, z away from the camera -- so "forward" for the
    subject (toward the camera) is NEGATIVE z.
    """
    sh_y, half_sh, upper, fore = -500.0, 190.0, 300.0, 260.0
    dy, dz = math.cos(left_arm_forward_rad), -math.sin(left_arm_forward_rad)
    ls = (half_sh, sh_y, 0.0)          # left on POSITIVE x: subject faces us
    rs = (-half_sh, sh_y, 0.0)
    return {
        "left_shoulder": [ls[0], ls[1], ls[2], 1.0],
        "right_shoulder": [rs[0], rs[1], rs[2], 1.0],
        "left_hip": [100.0, 0.0, 0.0, 1.0],
        "right_hip": [-100.0, 0.0, 0.0, 1.0],
        "left_elbow": [ls[0], ls[1] + upper * dy, ls[2] + upper * dz, 1.0],
        "left_wrist": [ls[0], ls[1] + (upper + fore) * dy, ls[2] + (upper + fore) * dz, 1.0],
        "right_elbow": [rs[0], rs[1] + upper, rs[2], 1.0],
        "right_wrist": [rs[0], rs[1] + upper + fore, rs[2], 1.0],
        "nose": [0.0, sh_y - 240.0, 0.0, 1.0],
    }


def test_a_hanging_arm_is_shoulder_pitch_ninety() -> None:
    """NAO's ShoulderPitch is +90 deg for an arm at the side."""
    t = retarget_upper_body(_facing_subject(0.0))
    assert math.degrees(t["LShoulderPitch"]) == pytest.approx(90.0, abs=1.0)


def test_an_arm_reaching_at_the_camera_is_shoulder_pitch_zero() -> None:
    """The regression this pair exists for. TorsoFrame.forward is right x up,
    which points out of the subject's BACK; NAO's ShoulderPitch is 0 for an arm
    held FORWARD, so the arm solve must read the negated axis. Without that, an
    arm pointing at the camera solved to atan2(0, -1) = 180 deg and the joint
    limit clamped it to 119.5 -- the arm hit its mechanical stop instead of
    reaching forward, on every frame of every forward reach."""
    t = retarget_upper_body(_facing_subject(math.radians(90.0)))
    assert math.degrees(t["LShoulderPitch"]) == pytest.approx(0.0, abs=1.0)


def test_the_forward_reach_is_monotonic_and_never_saturates() -> None:
    """A smooth human motion must produce a smooth robot one. The broken version
    was not merely offset: it saturated at +119.5 for most of the range and then
    flipped sign to -119.5 at the end."""
    limit = math.radians(119.5)
    pitches = [
        retarget_upper_body(_facing_subject(math.radians(d)))["LShoulderPitch"]
        for d in range(0, 91, 10)
    ]
    assert all(abs(p) < limit - 1e-3 for p in pitches), "a joint hit its limit"
    for earlier, later in zip(pitches, pitches[1:], strict=False):
        assert later < earlier + 1e-9, "shoulder pitch must fall as the arm rises"
    assert math.degrees(pitches[0] - pitches[-1]) == pytest.approx(90.0, abs=2.0)


def test_the_swing_is_defined_when_the_bone_lies_on_the_second_axis() -> None:
    """An arm straight out to the side puts the bone along the roll axis, so the
    swing angle is undefined and only atan2's SIGNED ZEROS would decide it --
    which is a coin flip between 0 and 180 deg, and 180 saturates the joint."""
    from nao_retarget import _swing_twist

    first, second = _swing_twist(0.0, 0.0, 1.0)
    assert first == 0.0
    assert second == pytest.approx(math.pi / 2.0)
    first_neg, second_neg = _swing_twist(-0.0, -0.0, -1.0)
    assert first_neg == 0.0
    assert second_neg == pytest.approx(-math.pi / 2.0)


def test_the_leg_solve_still_reads_the_unnegated_axis() -> None:
    """The arm fix must NOT be applied to the legs: NAO's HipPitch is NEGATIVE
    for a thigh swung forward, so the leg solve wants the back-pointing axis that
    TorsoFrame.forward already provides."""
    r = LowerBodyRetargeter()
    for _ in range(90):
        r.observe(figure())
    obs = r.observe(figure(left=(0.0, -0.6, 1.0)))
    leg = obs.leg("L")
    assert leg is not None
    assert leg.as_targets("L")["LHipPitch"] < 0.0


# ---------------------------------------------------------------------------
# Elbow range compression (2026-09-10). Reported as "upper body sometimes works
# wrong"; measured as RElbowRoll sitting on its hardware stop 26.9% of the run.
# ---------------------------------------------------------------------------
def test_ordinary_elbow_bends_are_untouched() -> None:
    """Below the knee the mapping must be exactly 1:1.

    76% of the frames in the live session were under 60 deg of bend. A plain
    rescale would have shrunk all of them -- trading a visible fault (a frozen
    arm) for a dull one (a timid arm).
    """
    from nao_retarget import ELBOW_LINEAR_RAD, _elbow_bend_to_nao
    for deg in (0.0, 5.0, 23.6, 45.0, 59.0):
        bend = math.radians(deg)
        assert _elbow_bend_to_nao(bend) == pytest.approx(bend), (
            f"{deg} deg of bend was rescaled; ordinary gestures must pass through")
    assert _elbow_bend_to_nao(ELBOW_LINEAR_RAD) == pytest.approx(ELBOW_LINEAR_RAD)


def test_a_deeply_bent_human_elbow_never_pins_the_joint() -> None:
    """Past 88.5 deg the old 1:1 map clamped, and a clamped joint has stopped
    imitating -- it holds still through the part of the motion with the most
    travel in it."""
    from nao_retarget import ELBOW_NAO_MAX_RAD, _elbow_bend_to_nao
    for deg in (90.0, 110.0, 131.7, 150.0, 175.0, 180.0):
        out = _elbow_bend_to_nao(math.radians(deg))
        assert out <= ELBOW_NAO_MAX_RAD + 1e-9, f"{deg} deg mapped past the stop"
    # A fully folded human arm reaches the stop but nothing beyond it saturates
    # EARLIER than that, so the response is still live at 110 and 130 deg.
    assert _elbow_bend_to_nao(math.radians(110.0)) < ELBOW_NAO_MAX_RAD - 1e-6
    assert _elbow_bend_to_nao(math.radians(131.7)) < ELBOW_NAO_MAX_RAD - 1e-6


def test_the_elbow_map_is_monotonic() -> None:
    """More human bend must always mean at least as much robot bend: a
    non-monotonic map would make the elbow travel BACKWARDS mid-gesture."""
    from nao_retarget import _elbow_bend_to_nao
    prev = -1.0
    for i in range(0, 361):
        out = _elbow_bend_to_nao(math.radians(i * 0.5))
        assert out >= prev - 1e-12, f"map decreased at {i * 0.5} deg"
        prev = out


def test_the_elbow_map_is_robust_to_junk() -> None:
    from nao_retarget import ELBOW_NAO_MAX_RAD, _elbow_bend_to_nao
    assert _elbow_bend_to_nao(-1.0) == 0.0
    assert _elbow_bend_to_nao(0.0) == 0.0
    assert _elbow_bend_to_nao(50.0) <= ELBOW_NAO_MAX_RAD + 1e-9


# ---------------------------------------------------------------------------
# Hand / roll joints (2026-09-16). ElbowYaw and WristYaw sat at 0.0 rad for the
# project's whole life: a roll joint turns about the axis its own bone lies
# along, so shoulder/elbow/wrist positions -- however accurate -- cannot see it.
# The cure is more landmarks, so these tests are built on the hand markers.
#
# Verified by ROUND TRIP through NAO's own arm chain, Ry.Rz.Rx.Rz.Rx, which is
# the chain balance.py builds from Nao.urdf. Landmarks are generated from known
# joint angles and the retargeter has to recover them.
# ---------------------------------------------------------------------------
UPPER_ARM_MM, FOREARM_MM, HAND_MM, THUMB_MM = 300.0, 260.0, 190.0, 110.0


def _nao_to_camera(v):
    """NAO torso axes -> the camera frame _facing_subject() sets up.

    That figure makes TorsoFrame right=(-1,0,0), up=(0,-1,0), forward=(0,0,1),
    so NAO's +X (chest) is -forward, +Y (left) is -right and +Z (up) is up.
    """
    a, b, c = v
    return (b, -c, -a)


def _chain(sp, sr, ey, er, wy):
    """Upper-arm, forearm and thumb directions in NAO torso axes."""
    from nao_retarget import _rot_x, _rot_y, _rot_z

    def shoulder(v):
        return _rot_y(_rot_z(v, sr), sp)

    upper = shoulder((1.0, 0.0, 0.0))
    fore = shoulder(_rot_x(_rot_z((1.0, 0.0, 0.0), er), ey))
    # The thumb is a hand-fixed axis: +Y of the hand frame, which is what
    # HAND_YAW_SIGN names. WristYaw is the only joint that can move it.
    thumb = shoulder(_rot_x(_rot_z(_rot_x((0.0, 1.0, 0.0), wy), er), ey))
    return upper, fore, thumb


def _hand_subject(side="L", sp=0.0, sr=0.0, ey=0.0, er=-0.8, wy=0.0, grip=0.0):
    """A camera-facing figure whose ``side`` arm is at the given NAO angles."""
    sh_y, half_sh = -500.0, 190.0
    ls, rs = (half_sh, sh_y, 0.0), (-half_sh, sh_y, 0.0)
    shoulder = ls if side == "L" else rs
    upper, fore, thumb = _chain(sp, sr, ey, er, wy)
    u, f, t = (_nao_to_camera(v) for v in (upper, fore, thumb))

    def step(origin, direction, length):
        return tuple(origin[i] + direction[i] * length for i in range(3))

    elbow = step(shoulder, u, UPPER_ARM_MM)
    wrist = step(elbow, f, FOREARM_MM)
    # A closing hand shortens the visible hand and swings the thumb across the
    # palm toward the fingers -- i.e. ALONG the forearm, the one direction
    # WristYaw cannot see. That is deliberate: it is what makes the
    # grip-invariance test below mean something.
    reach = HAND_MM * (1.0 - 0.6 * grip)
    lean = 0.5 * grip
    finger = step(wrist, f, reach)
    thumb_pt = tuple(
        wrist[i] + THUMB_MM * (t[i] * (1.0 - lean) + f[i] * lean) for i in range(3)
    )
    pre = "left_" if side == "L" else "right_"
    other = "right_" if side == "L" else "left_"
    other_sh = rs if side == "L" else ls
    return {
        "left_shoulder": [*ls, 1.0],
        "right_shoulder": [*rs, 1.0],
        "left_hip": [100.0, 0.0, 0.0, 1.0],
        "right_hip": [-100.0, 0.0, 0.0, 1.0],
        "nose": [0.0, sh_y - 240.0, 0.0, 1.0],
        pre + "elbow": [*elbow, 1.0],
        pre + "wrist": [*wrist, 1.0],
        pre + "hand_root": [*wrist, 1.0],
        pre + "finger": [*finger, 1.0],
        pre + "thumb": [*thumb_pt, 1.0],
        other + "elbow": [other_sh[0], other_sh[1] + UPPER_ARM_MM, other_sh[2], 1.0],
        other + "wrist": [other_sh[0], other_sh[1] + UPPER_ARM_MM + FOREARM_MM,
                          other_sh[2], 1.0],
    }


ROLL_POSES = [
    # (shoulder pitch, shoulder roll, elbow yaw, elbow roll, wrist yaw)
    (0.0, 0.0, 0.0, -0.80, 0.0),
    (0.0, 0.0, 0.70, -0.80, 0.0),
    (0.0, 0.0, -0.70, -0.80, 0.0),
    (0.5, 0.3, 0.40, -1.00, 0.6),
    (1.2, 0.2, -1.00, -0.60, -0.9),
    (-0.6, 0.4, 0.90, -1.20, 1.2),
]


@pytest.mark.parametrize(("sp", "sr", "ey", "er", "wy"), ROLL_POSES)
def test_the_roll_joints_round_trip_through_naos_own_arm_chain(sp, sr, ey, er, wy):
    """Generate landmarks from known angles; the solve must recover them."""
    kps = _hand_subject("L", sp=sp, sr=sr, ey=ey, er=er, wy=wy)
    t = retarget_upper_body(kps)
    assert t["LElbowYaw"] == pytest.approx(ey, abs=2e-3)
    assert t["LWristYaw"] == pytest.approx(wy, abs=2e-3)


def test_the_roll_joints_round_trip_on_the_right_arm_too():
    """The chain is the same on both sides even though ShoulderRoll's SIGN is
    not -- the solve has to convert back out of the outward-positive convention
    the shoulder reports in, and forgetting to would only show up here."""
    for sp, sr, ey, er, wy in ROLL_POSES:
        kps = _hand_subject("R", sp=sp, sr=-sr, ey=ey, er=-er, wy=wy)
        t = retarget_upper_body(kps)
        assert t["RElbowYaw"] == pytest.approx(ey, abs=2e-3), (sp, sr, ey, er, wy)
        assert t["RWristYaw"] == pytest.approx(wy, abs=2e-3), (sp, sr, ey, er, wy)


def test_a_straight_arm_omits_the_elbow_yaw_rather_than_guessing():
    """With the elbow straight the forearm lies ON the ElbowYaw axis: every
    value of the joint puts it in the same place, so there is nothing to read.
    Omitting it makes the driver hold the last value; returning 0 would snap the
    forearm flat every time the subject let their arm hang."""
    t = retarget_upper_body(_hand_subject("L", er=-0.05, ey=0.9))
    assert "LElbowYaw" not in t
    assert "LWristYaw" not in t, "wrist yaw is solved in the forearm frame, which "\
        "is not known when the elbow yaw is not"


def test_a_bent_arm_does_solve_the_elbow_yaw():
    t = retarget_upper_body(_hand_subject("L", er=-0.8, ey=0.9))
    assert t["LElbowYaw"] == pytest.approx(0.9, abs=2e-3)


def test_wrist_yaw_is_unchanged_by_closing_the_hand():
    """Grip moves the thumb ALONG the forearm, which is the one axis WristYaw
    cannot see -- so the solve has to project that component out. Without it,
    making a fist would twist the robot's wrist."""
    angles = [
        retarget_upper_body(_hand_subject("L", er=-0.9, ey=0.3, wy=0.7, grip=g))
        .get("LWristYaw")
        for g in (0.0, 0.3, 0.6, 0.9)
    ]
    assert all(a is not None for a in angles)
    assert max(angles) - min(angles) < 1e-6, angles


def test_grip_tracks_the_thumb_to_finger_distance():
    grips = [retarget_upper_body(_hand_subject("L", grip=g)).get("LHand")
             for g in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(g is not None for g in grips)
    for earlier, later in zip(grips, grips[1:], strict=False):
        assert later > earlier - 1e-9, f"grip must be monotonic: {grips}"
    assert grips[0] < 0.25 and grips[-1] > 0.75, grips


def test_grip_is_invariant_to_how_far_away_the_subject_stands():
    """Measured as a ratio of the hand's own length, so it cannot drift with
    camera distance the way an absolute millimetre threshold would."""
    near = retarget_upper_body(_hand_subject("L", grip=0.5))["LHand"]
    far = _hand_subject("L", grip=0.5)
    scaled = {k: [v[0] * 0.4, v[1] * 0.4, v[2] * 0.4, v[3]] for k, v in far.items()}
    assert retarget_upper_body(scaled)["LHand"] == pytest.approx(near, abs=1e-6)


def test_elbow_yaw_needs_no_hand_landmarks_at_all():
    """The happy surprise, and worth pinning because it is easy to assume
    otherwise: ElbowYaw is fixed by the FOREARM's direction given a known
    shoulder and elbow bend, so it is solvable from plain coco_19 -- no hand
    markers, no skeleton change. Only WristYaw and the grip need the hand.

    (It also survives the elbow's range compression: the solve reads only the
    DIRECTION of the off-axis part, and _elbow_bend_to_nao scales its magnitude
    without touching its sign.)
    """
    kps = _hand_subject("L", sp=0.4, er=-0.9, ey=0.55)
    for name in ("left_hand_root", "left_thumb", "left_finger"):
        kps.pop(name)
    t = retarget_upper_body(kps)
    assert t["LElbowYaw"] == pytest.approx(0.55, abs=2e-3)
    assert "LWristYaw" not in t and "LHand" not in t
    assert "LShoulderPitch" in t and "LElbowRoll" in t


def test_an_invisible_thumb_still_allows_the_elbow_yaw():
    """The two joints have different evidence: ElbowYaw needs only the forearm,
    WristYaw needs the thumb as well. Losing the thumb must not cost both."""
    kps = _hand_subject("L", er=-0.9, ey=0.5)
    kps["left_thumb"][3] = 0.0
    t = retarget_upper_body(kps)
    assert t["LElbowYaw"] == pytest.approx(0.5, abs=2e-3)
    assert "LWristYaw" not in t
