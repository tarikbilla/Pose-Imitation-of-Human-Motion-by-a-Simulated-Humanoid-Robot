"""The locomotion clip library: is every clip one the robot can actually survive?

A clip is played OPEN LOOP -- for its whole duration the legs follow keyframes
and the balance controller has no say (measured: 71.8% of one tracked window
inside a clip, mean commitment 6.84 s). So these tests are the only thing between
a bad keyframe and a fall, and they are deliberately about the ROBOT rather than
about the code: every assertion is a statement about NAO's mass, its sole
geometry or its motors.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "main", "libraries"))

from clip_forge import (  # noqa: E402
    LEG_JOINTS,
    crouch,
    lateral_shift,
    make_leg_raise,
    make_squat,
    mirror_pose,
    peak_velocity_fraction,
    refine_clip,
    retime_for_velocity,
    solve_weight_shift,
    standing_pose,
)
from clip_safety import certify  # noqa: E402
from pose_control_utils import get_default_motor_configs  # noqa: E402

pytest.importorskip("numpy", reason="the clip certifier needs the CoM model")

MOTIONS = os.path.join(REPO, "main", "controllers", "pose_imitation_controller",
                       "motions")


def _model():
    from balance import NaoCoMModel
    return NaoCoMModel()


# ---------------------------------------------------------------------------
# The certifier itself. A ruler has to be checked before its readings mean
# anything -- these are poses whose balance is known independently.
# ---------------------------------------------------------------------------
def test_the_model_calls_standing_balanced() -> None:
    configs = get_default_motor_configs()
    stand = {j: configs[j].rest_angle for j in LEG_JOINTS}
    fore_aft, lateral = _model().support_margins(stand)
    assert fore_aft > 0.04, f"standing reads {fore_aft*1000:.1f} mm of fore/aft margin"
    assert lateral > 0.06, f"standing reads {lateral*1000:.1f} mm of lateral margin"


def test_the_crouch_relation_is_balanced_at_every_depth() -> None:
    """``nao_retarget`` claims Hip=-u / Knee=+2u / Ankle=-u keeps the mass over
    the feet at ANY depth, because NAO's thigh and shank are within 3 mm of the
    same length. The whole squat clip rests on that, so it is checked here
    against the mass model rather than taken on trust."""
    model = _model()
    for u in (0.0, 0.2, 0.4, 0.6, 0.8):
        fore_aft, lateral = model.support_margins(crouch(u))
        assert fore_aft > 0.04, f"u={u}: fore/aft {fore_aft*1000:.1f} mm"
        assert lateral > 0.06, f"u={u}: lateral {lateral*1000:.1f} mm"


# ---------------------------------------------------------------------------
# Mirroring. This found a real bug: HipYawPitch's axis is the 45-degree
# (0, .707, -.707*sign), already opposite between the legs, so mirroring must
# NOT negate it -- and negating it produced a clip whose CoM sat -58.6 mm
# outside the polygon against the original's -18.0 mm, while the keyframes that
# carry no hip yaw mirrored perfectly and hid it.
# ---------------------------------------------------------------------------
MIRROR_POSES = [
    crouch(0.4),
    lateral_shift(crouch(0.25), -0.3),
    lateral_shift(crouch(0.25), +0.3),
    {**crouch(0.3), "LHipYawPitch": -0.4, "RHipYawPitch": -0.4},
    {**crouch(0.3), "LHipYawPitch": 0.35, "RHipYawPitch": 0.35,
     "LHipRoll": 0.2, "RAnkleRoll": -0.15},
]


@pytest.mark.parametrize("pose", MIRROR_POSES)
def test_mirroring_a_pose_preserves_its_balance(pose) -> None:
    """NAO is symmetric about the sagittal plane, so a mirrored pose must be
    exactly as balanced as its original. Any difference is a sign error."""
    model = _model()
    before = model.support_margins(pose)
    after = model.support_margins(mirror_pose(pose))
    assert before[0] == pytest.approx(after[0], abs=1e-9), "fore/aft margin moved"
    assert before[1] == pytest.approx(after[1], abs=1e-9), "lateral margin moved"


def test_mirroring_twice_is_the_identity() -> None:
    for pose in MIRROR_POSES:
        twice = mirror_pose(mirror_pose(pose))
        for joint, angle in pose.items():
            assert twice[joint] == pytest.approx(angle, abs=1e-12), joint


def test_mirroring_actually_swaps_the_sides() -> None:
    """Guard against a 'mirror' that preserves balance by doing nothing."""
    pose = {**crouch(0.3), "LHipRoll": 0.30, "RHipRoll": 0.05}
    out = mirror_pose(pose)
    assert out["RHipRoll"] == pytest.approx(-0.30)
    assert out["LHipRoll"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# The generated clips
# ---------------------------------------------------------------------------
def test_the_squat_is_quasi_static_and_actually_squats() -> None:
    model = _model()
    clip = make_squat()
    cert = certify("Squat", clip, kind="static")
    assert cert.passed, [str(v) for v in cert.violations[:3]]

    def height(pose):
        frames = model.frames(pose)
        return float(model.com_height(frames, model.com(pose, frames)))

    drop = height(clip[0][1]) - min(height(p) for _t, p in clip)
    # NAO's leg is only 203 mm, so hip height goes as 0.203*cos(u) -- a shallow
    # crouch is invisible. Anything under ~30 mm is not a squat, it is a wobble.
    assert drop > 0.030, f"the squat only lowers the mass by {drop*1000:.1f} mm"


@pytest.mark.parametrize("side", ["L", "R"])
def test_a_leg_raise_lifts_that_leg_and_stays_balanced(side) -> None:
    model = _model()
    clip = make_leg_raise(side)
    cert = certify(f"RaiseLeg{side}", clip, kind="static")
    assert cert.passed, [str(v) for v in cert.violations[:3]]

    def clearance(pose):
        frames = model.frames(pose)
        lows = {s: float(model.foot_corners(s, frames)[:, 2].min()) for s in ("L", "R")}
        return lows[side] - min(lows.values())

    assert max(clearance(p) for _t, p in clip) > 0.020, "the foot never leaves the floor"


@pytest.mark.parametrize("side", ["L", "R"])
def test_the_weight_moves_before_the_foot_does(side) -> None:
    """The ordering that matters. Lifting first and shifting after is how a robot
    falls sideways, and it is what the live per-leg imitation was doing."""
    model = _model()
    stance = "R" if side == "L" else "L"
    clip = make_leg_raise(side)

    def clearance(pose):
        frames = model.frames(pose)
        lows = {s: float(model.foot_corners(s, frames)[:, 2].min()) for s in ("L", "R")}
        return lows[side] - min(lows.values())

    lift_starts = next(i for i, (_t, p) in enumerate(clip) if clearance(p) > 0.005)
    margin_at_lift = model.stance_margin(clip[lift_starts][1], stance)
    assert margin_at_lift > 0.0, (
        f"the {side} foot leaves the floor while the mass is still "
        f"{margin_at_lift*1000:.1f} mm outside the {stance} foot")


def test_the_weight_shift_is_solved_toward_the_right_foot() -> None:
    """Solved against the model, not derived -- and the derivation gives the
    opposite sign, because in the torso frame it is the feet that appear to move."""
    model = _model()
    for stance in ("L", "R"):
        phi = solve_weight_shift(crouch(0.25), stance, model)
        shifted = lateral_shift(crouch(0.25), phi)
        assert model.stance_margin(shifted, stance) > 0.0, (
            f"solved lean {phi:+.3f} does not put the mass over the {stance} foot")


# ---------------------------------------------------------------------------
# Refining a Cyberbotics clip
# ---------------------------------------------------------------------------
def _cyberbotics(name):
    from walk_motion import default_motion_search_dirs, motion_poses
    for directory in default_motion_search_dirs():
        if os.path.abspath(directory) == os.path.abspath(MOTIONS):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return motion_poses(path)
    pytest.skip(f"{name} is not on this install")


def test_refining_frees_the_upper_body() -> None:
    """Backwards drives the arms AND the head, and release_to_motion suspends
    whatever a clip owns -- so playing it froze the upper-body imitation for its
    whole duration."""
    raw = _cyberbotics("Backwards.motion")
    assert any(j.startswith(("LShoulder", "RShoulder", "Head"))
               for _t, p in raw for j in p), "fixture no longer has upper-body joints"
    for _t, pose in refine_clip(raw):
        assert set(pose) <= set(LEG_JOINTS)


def test_refining_removes_the_posture_gap() -> None:
    """Every shipped clip opens and closes ~1.05 rad from the pose the controller
    stands in; that gap is what the prepare ramp and the handover absorb, and the
    handover is where 13 of 13 recorded transitions pushed the margin negative."""
    raw = _cyberbotics("Forwards.motion")
    stand = standing_pose()
    before = max(abs(raw[0][1].get(j, stand[j]) - stand[j]) for j in LEG_JOINTS)
    assert before > 0.5, "fixture no longer has a posture gap to close"
    refined = refine_clip(raw)
    for end in (refined[0][1], refined[-1][1]):
        for joint in LEG_JOINTS:
            assert end[joint] == pytest.approx(stand[joint], abs=1e-9)


def test_refining_keeps_every_original_keyframe() -> None:
    """The gait is what makes the clip walk. Bookending it must not edit it."""
    raw = _cyberbotics("TurnLeft60.motion")
    refined = [p for _t, p in refine_clip(raw)]
    originals = [{j: p.get(j, standing_pose()[j]) for j in LEG_JOINTS} for _t, p in raw]
    for original in originals:
        assert any(all(abs(candidate[j] - original[j]) < 1e-9 for j in LEG_JOINTS)
                   for candidate in refined), "an original keyframe was lost"


def test_retiming_brings_a_clip_inside_the_motors() -> None:
    """Forwards50 asks LKneePitch for 102% of its rated speed. A motor that
    cannot reach a keyframe in time leaves the robot in a different pose from the
    one the clip was balanced for -- and it is latest exactly when the clip is
    fastest."""
    raw = _cyberbotics("Forwards50.motion")
    clip = refine_clip(raw)
    assert peak_velocity_fraction(clip) > 0.9, "fixture no longer outruns the motors"
    slowed = retime_for_velocity(clip, target_fraction=0.85)
    assert peak_velocity_fraction(slowed) <= 0.851
    assert slowed[-1][0] > clip[-1][0], "the clip was not actually stretched"


def test_retiming_preserves_the_poses_exactly() -> None:
    """Stretching changes WHEN, never WHAT -- which is what keeps a limit cycle
    a limit cycle."""
    clip = refine_clip(_cyberbotics("Forwards50.motion"))
    slowed = retime_for_velocity(clip, target_fraction=0.5)
    assert len(slowed) == len(clip)
    for (_ta, a), (_tb, b) in zip(clip, slowed, strict=True):
        assert a == b


# ---------------------------------------------------------------------------
# The built library on disk
# ---------------------------------------------------------------------------
BUILT = ["Squat", "RaiseLegLeft", "RaiseLegRight", "Forwards", "Forwards50",
         "Backwards", "TurnLeft40", "TurnLeft60", "TurnLeft180",
         "TurnRight40", "TurnRight60", "TurnRight180",
         "SideStepLeft", "SideStepRight"]


@pytest.mark.parametrize("name", BUILT)
def test_every_built_clip_is_legs_only(name) -> None:
    """A locomotion clip has no business touching the arms: the upper body must
    keep imitating straight through it."""
    from walk_motion import motion_joints
    path = os.path.join(MOTIONS, f"{name}.motion")
    if not os.path.isfile(path):
        pytest.skip("run scripts/build_motion_clips.py first")
    assert set(motion_joints(path)) <= set(LEG_JOINTS)


@pytest.mark.parametrize("name", BUILT)
def test_every_built_clip_starts_and_ends_standing(name) -> None:
    from walk_motion import motion_poses
    path = os.path.join(MOTIONS, f"{name}.motion")
    if not os.path.isfile(path):
        pytest.skip("run scripts/build_motion_clips.py first")
    poses = motion_poses(path)
    stand = standing_pose()
    for end in (poses[0][1], poses[-1][1]):
        for joint in LEG_JOINTS:
            assert end.get(joint, 0.0) == pytest.approx(stand[joint], abs=1e-4)


def test_the_continuous_walk_clip_is_still_cyclic() -> None:
    """Retiming and bookending must not cost the limit cycle -- without it
    walking degenerates back into 2.6 s bursts, each paying a restart, a ramp and
    a handover for 0.095 m of travel."""
    from walk_motion import gait_cycle
    path = os.path.join(MOTIONS, "Forwards50.motion")
    if not os.path.isfile(path):
        pytest.skip("run scripts/build_motion_clips.py first")
    cycle = gait_cycle(path)
    assert cycle is not None, "the gait cycle was lost"
    assert cycle.period_s > 0.3
    assert cycle.speed_mps > 0.03


def test_the_mirrored_turn_declares_its_angle() -> None:
    """plan_action picks the largest turn clip that will not overshoot, so a clip
    whose nominal angle is unreadable is a clip it will fire at any error."""
    from walk_motion import motion_nominal_yaw
    path = os.path.join(MOTIONS, "TurnRight180.motion")
    if not os.path.isfile(path):
        pytest.skip("run scripts/build_motion_clips.py first")
    assert math.degrees(motion_nominal_yaw(path)) == pytest.approx(-180.0, abs=1.0)


def test_the_library_covers_every_action_the_planner_knows() -> None:
    from walk_motion import KNOWN_MOTIONS, find_motion_files
    if not os.path.isdir(MOTIONS):
        pytest.skip("run scripts/build_motion_clips.py first")
    found = find_motion_files([MOTIONS])
    missing = [a for a in KNOWN_MOTIONS if a not in found]
    assert not missing, f"no clip built for: {missing}"


# ---------------------------------------------------------------------------
# Action -> clip. The half of the decision that lives on the robot.
# ---------------------------------------------------------------------------
ALL_CLIPS = {k: "/fake.motion" for k in
             ("forward", "backward", "side_left", "side_right",
              "squat", "leg_raise_left", "leg_raise_right",
              "turn_left", "turn_right")}


@pytest.mark.parametrize(("verdict", "clip"), [
    ("walk_forward", "forward"),
    ("walk_backward", "backward"),
    ("step_left", "side_left"),
    ("step_right", "side_right"),
    ("squat", "squat"),
    ("raise_left", "leg_raise_left"),
    ("raise_right", "leg_raise_right"),
])
def test_each_action_reaches_its_clip(verdict, clip) -> None:
    from walk_motion import plan_action
    plan = plan_action(yaw_error_rad=0.0, available=ALL_CLIPS,
                       action={"action": verdict, "conf": 0.9})
    assert plan.action == clip, plan.reason


def test_an_unseen_human_stands_rather_than_falling_through() -> None:
    """The failure this exists to stop: a gait cue with warm evidence happily
    keeps the robot walking at a human who has left the frame -- measured once as
    51.4 s of false march in a 273 s session. 'unknown' must stop the decision
    here, not merely fail to contribute to it."""
    from walk_motion import plan_action
    warm_gait = {"state": "march", "cadence_hz": 1.0, "conf": 1.0}
    plan = plan_action(yaw_error_rad=0.0, available=ALL_CLIPS,
                       action={"action": "unknown", "conf": 0.0}, gait=warm_gait)
    assert plan.action is None, plan.reason


def test_a_still_human_stands_and_a_marching_one_walks() -> None:
    """``idle`` means the PELVIS is still, not the human.

    This used to assert that an idle verdict stands even against a marching
    gait cue, on the theory that the gait cue's warm evidence was the less
    trustworthy witness. The 2026-09-17 session showed the opposite failure:
    the action cue's walk witnesses are absolute speeds fitted to a faster
    walker, so a subject marching in place or walking at 0.07-0.10 m/s read as
    idle for 75% of 59 s of visible marching -- and "stand" handed the legs to
    per-joint pose imitation of a walking human, which is what fell over. Both
    cues are witnesses to one question now: the pelvis travels, or the legs
    cycle. A human who has LEFT the frame is still stopped by 'unknown' (see the
    test above); a human standing with still legs is still stopped here.
    """
    from walk_motion import plan_action
    still_legs = {"state": "idle", "cadence_hz": 0.0, "conf": 1.0}
    plan = plan_action(yaw_error_rad=0.0, available=ALL_CLIPS,
                       action={"action": "idle", "conf": 1.0}, gait=still_legs)
    assert plan.action is None, plan.reason
    marching = {"state": "march", "cadence_hz": 1.0, "conf": 1.0}
    plan = plan_action(yaw_error_rad=0.0, available=ALL_CLIPS,
                       action={"action": "idle", "conf": 1.0}, gait=marching)
    assert plan.action == "forward", plan.reason


def test_a_low_confidence_verdict_is_not_acted_on() -> None:
    from walk_motion import plan_action
    plan = plan_action(yaw_error_rad=0.0, available=ALL_CLIPS,
                       action={"action": "walk_forward", "conf": 0.05})
    assert plan.action is None, plan.reason


def test_turning_still_outranks_the_action_classifier() -> None:
    """Heading is a CLOSED loop on the robot's own yaw; the action verdict is an
    open-loop read of the human. Walking off along the wrong heading is much
    harder to undo than a slightly late departure, so the yaw keeps priority."""
    from walk_motion import plan_action
    plan = plan_action(yaw_error_rad=math.radians(90.0), available=ALL_CLIPS,
                       action={"action": "walk_forward", "conf": 1.0})
    assert plan.is_turn, plan.reason


def test_an_action_with_no_clip_says_so_rather_than_guessing() -> None:
    from walk_motion import plan_action
    plan = plan_action(yaw_error_rad=0.0, available={"forward": "/f.motion"},
                       action={"action": "squat", "conf": 1.0})
    assert plan.action is None
    assert "squat" in plan.reason


# ---------------------------------------------------------------------------
# Turning as an aimed motion rather than a fixed gesture
#
# A turn clip's filename says how far it turns; these ask the KEYFRAMES, because
# the whole point of aiming a turn is that the clip is stopped part-way and the
# filename says nothing about what any earlier moment is worth.
# ---------------------------------------------------------------------------
def _turn_clip(name: str) -> str:
    path = os.path.join(MOTIONS, name)
    if not os.path.isfile(path):
        pytest.skip(f"{name} is not installed")
    return path


def test_clip_yaw_odometry_agrees_with_the_filename() -> None:
    """The yaw the keyframes command must match the angle the clip is named for.

    This is the calibration that everything else here rests on: if the odometry
    and the filename disagreed, aiming a turn would be aiming with a broken
    ruler. Webots' clips overshoot their own names slightly (184 deg for a clip
    called 180), which is itself worth pinning -- it is real rotation the robot
    performs and the closed loop has to absorb.
    """
    from balance import clip_torso_yaw
    from walk_motion import motion_nominal_yaw, motion_poses

    for name in ("TurnLeft40.motion", "TurnRight40.motion",
                 "TurnLeft180.motion", "TurnRight180.motion"):
        path = _turn_clip(name)
        measured = clip_torso_yaw(motion_poses(path))[-1]
        nominal = motion_nominal_yaw(path)
        assert measured * nominal > 0.0, f"{name} turns the wrong way"
        assert abs(measured - nominal) <= math.radians(6.0), (
            f"{name} is named {math.degrees(nominal):+.0f} deg but its keyframes "
            f"command {math.degrees(measured):+.0f} deg")


def test_a_walk_clip_has_no_turn_schedule() -> None:
    """Straight walking must not be mistaken for a turn, or the heading loop
    would try to aim it."""
    from walk_motion import turn_schedule

    for name in ("Forwards.motion", "Forwards50.motion", "Backwards.motion"):
        path = os.path.join(MOTIONS, name)
        if os.path.isfile(path):
            assert turn_schedule(path) is None, name


def test_the_big_turn_clip_turns_faster_than_the_small_one() -> None:
    """The premise of preferring the LARGEST turn clip we can stop part-way.

    It is counter-intuitive and it is the whole change, so it is measured rather
    than asserted: the small clip is not a finer tool, it is the same stepping
    turn with proportionally far more of its running time spent on its own
    start/stop transient.
    """
    from walk_motion import turn_schedule

    small = turn_schedule(_turn_clip("TurnLeft40.motion"))
    big = turn_schedule(_turn_clip("TurnLeft180.motion"))
    assert small is not None and big is not None
    assert big.rate_rad_s > 1.4 * small.rate_rad_s, (
        f"big {math.degrees(big.rate_rad_s):.1f} deg/s vs "
        f"small {math.degrees(small.rate_rad_s):.1f} deg/s")


def test_a_turn_clip_can_be_aimed_more_finely_than_it_can_be_named() -> None:
    """One clip, stopped at a certified keyframe, beats a whole small clip.

    Playing TurnLeft40 to its end leaves up to 20 deg of residual heading error
    -- half the only angle it can deliver. The ladder of stopping points inside
    TurnLeft180 is finer than that, which is what lets a single clip serve every
    turn instead of a chain of them.
    """
    from walk_motion import turn_schedule

    schedule = turn_schedule(_turn_clip("TurnLeft180.motion"))
    assert schedule is not None
    assert schedule.quantum_rad < math.radians(40.0), (
        f"deliverable angles are {math.degrees(schedule.quantum_rad):.0f} deg "
        "apart, no better than playing the small clip whole")
    # Every angle a human is likely to turn through is reachable to within half
    # that gap, in ONE clip.
    for degrees in range(20, 181, 10):
        wanted = math.radians(degrees)
        rung = schedule.best_exit(wanted)
        assert rung is not None
        assert abs(rung[1] - wanted) <= schedule.quantum_rad, degrees


def test_a_turn_clip_is_entered_after_its_opening_crouch() -> None:
    """The prepare-ramp does the crouch better, so playback should skip it.

    Entry must still be a CERTIFIED keyframe (balance.safe_exit_times): the ramp
    has to be able to stop there and hold the robot while the clip starts.
    """
    from balance import clip_torso_yaw, safe_exit_times
    from walk_motion import motion_poses, turn_schedule

    path = _turn_clip("TurnLeft180.motion")
    poses = motion_poses(path)
    schedule = turn_schedule(path)
    assert schedule is not None
    assert schedule.entry_s > 0.5, "nothing was skipped"
    assert any(abs(schedule.entry_s - t) < 1e-6 for t in safe_exit_times(poses)), \
        "the robot would be parked somewhere nobody certified"
    # And nothing was skipped that the robot actually needed: no rotation has
    # happened yet at the entry point.
    yaw = clip_torso_yaw(poses)
    index = min(range(len(poses)), key=lambda i: abs(poses[i][0] - schedule.entry_s))
    assert abs(yaw[index]) < math.radians(2.0)
