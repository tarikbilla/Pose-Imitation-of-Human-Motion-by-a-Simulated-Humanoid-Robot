"""Tests for NAO walk-motion discovery and selection (main/libraries/walk_motion.py).

These cover the pure, Webots-free logic: finding motion files on disk (with
filename fallbacks and search-dir ordering) and mapping a gait command to a walk
action. The actual Webots Motion playback is exercised on the test machine.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import clip_torso_speeds  # noqa: E402
from conftest import cyclic_poses, write_clip, write_cyclic_clip  # noqa: E402
from walk_motion import (  # noqa: E402
    STAND,
    LocomotionParams,
    YawServo,
    default_motion_search_dirs,
    find_motion_files,
    gait_cycle,
    motion_joints,
    motion_nominal_yaw,
    motion_pose_at,
    motion_poses,
    plan_action,
    select_walk_clip,
    wrap_pi,
)

CLIPS = {
    "forward": "/w/Forwards.motion",
    "turn_left": "/w/TurnLeft60.motion",
    "turn_right": "/w/TurnRight60.motion",
}


def _touch(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#WEBOTS_MOTION,V1.0\n")


def test_find_motion_files_picks_first_existing_candidate(tmp_path) -> None:
    d = tmp_path / "motions"
    d.mkdir()
    _touch(str(d / "Forwards.motion"))
    _touch(str(d / "TurnLeft40.motion"))
    _touch(str(d / "TurnRight40.motion"))
    found = find_motion_files([str(d)])
    assert found["forward"].endswith("Forwards.motion")
    assert found["turn_left"].endswith("TurnLeft40.motion")
    assert found["turn_right"].endswith("TurnRight40.motion")
    # No SideStep / Backwards files present -> those actions are omitted.
    assert "backward" not in found
    assert "side_left" not in found


def test_find_motion_files_respects_search_dir_order(tmp_path) -> None:
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    _touch(str(d2 / "Forwards.motion"))
    _touch(str(d1 / "Forwards.motion"))
    found = find_motion_files([str(d1), str(d2)])
    # d1 comes first in the search order, so its file wins.
    assert found["forward"] == str(d1 / "Forwards.motion")


def test_find_motion_files_empty_when_nothing_present(tmp_path) -> None:
    assert find_motion_files([str(tmp_path)]) == {}


def test_default_search_dirs_include_webots_home(monkeypatch) -> None:
    monkeypatch.setenv("WEBOTS_HOME", "/opt/webots-test")
    dirs = default_motion_search_dirs()
    assert any(d.startswith("/opt/webots-test") and d.endswith("motions") for d in dirs)
    # extra dirs are searched first.
    dirs2 = default_motion_search_dirs(extra=["/repo/motions"])
    assert dirs2[0] == "/repo/motions"


def test_default_search_dirs_dedup() -> None:
    dirs = default_motion_search_dirs(extra=["/x", "/x"])
    assert dirs.count("/x") == 1


def _gait(state="march", cadence=1.0, conf=0.9, turn=0.0):
    return {"state": state, "cadence_hz": cadence, "conf": conf, "turn": turn}


def test_motion_nominal_yaw_reads_the_angle_off_the_filename() -> None:
    assert abs(motion_nominal_yaw("/w/TurnLeft60.motion") - math.radians(60)) < 1e-9
    assert abs(motion_nominal_yaw("/w/TurnRight40.motion") + math.radians(40)) < 1e-9
    # Unnumbered clips get a documented default rather than 0 (0 would disable
    # the overshoot guard entirely).
    assert motion_nominal_yaw("/w/TurnLeft.motion") > 0.0
    assert motion_nominal_yaw("/w/Forwards.motion") == 0.0
    assert motion_nominal_yaw(None) == 0.0


def test_plan_walks_forward_when_marching_and_aligned() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=gait, available=CLIPS).action == "forward"


def test_plan_stands_still_when_idle_and_aligned() -> None:
    assert plan_action(yaw_error_rad=0.0, gait=None, available=CLIPS) == STAND
    idle = {"state": "idle", "cadence_hz": 0.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=idle, available=CLIPS).action is None


def test_plan_turns_while_standing_still() -> None:
    """The whole point of the yaw servo: rotating in front of the camera has to
    move the robot's body, not just its head, with no marching involved."""
    assert plan_action(yaw_error_rad=1.2, gait=None, available=CLIPS).action == "turn_left"
    assert plan_action(yaw_error_rad=-1.2, gait=None, available=CLIPS).action == "turn_right"


def test_turning_takes_priority_over_walking_forward() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    plan = plan_action(yaw_error_rad=1.2, gait=gait, available=CLIPS)
    assert plan.action == "turn_left" and plan.is_turn


def test_plan_refuses_a_clip_that_would_overshoot() -> None:
    # A 60 deg clip must not be fired at a 15 deg error (it would leave a bigger
    # error, of the opposite sign, than it started with).
    plan = plan_action(yaw_error_rad=math.radians(15), gait=None, available=CLIPS)
    assert plan.action is None
    # A 28 deg error is past the entry gate, yet still too small for the 60 deg
    # clip -- and exactly right for a 40 deg one.
    small = dict(CLIPS, turn_left="/w/TurnLeft40.motion")
    assert plan_action(yaw_error_rad=math.radians(28), gait=None,
                       available=CLIPS).action is None
    assert plan_action(yaw_error_rad=math.radians(28), gait=None,
                       available=small).action == "turn_left"


def test_plan_never_asks_for_a_clip_that_is_not_on_disk() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=1.2, gait=gait, available={}).action is None
    only_turn = {"turn_left": "/w/TurnLeft60.motion"}
    assert plan_action(yaw_error_rad=0.0, gait=gait, available=only_turn).action is None


def test_plan_hysteresis_lowers_the_gate_once_turning() -> None:
    # A fine-grained clip, so the overshoot guard is not what decides this test.
    small = {"turn_left": "/w/TurnLeft20.motion"}
    p = LocomotionParams()
    err = 0.5 * (p.turn_stop_rad + p.turn_start_rad)   # between the two gates
    assert plan_action(yaw_error_rad=err, available=small, params=p).action is None
    # Mid-rotation the gate drops, so the robot finishes the turn instead of
    # stalling one clip short of facing the right way.
    assert plan_action(yaw_error_rad=err, available=small, params=p,
                       turning=True).action == "turn_left"


def test_plan_ignores_a_non_finite_yaw_error() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=float("nan"), gait=gait,
                       available=CLIPS).action == "forward"


def test_plan_requires_confident_marching_to_walk() -> None:
    weak = {"state": "march", "cadence_hz": 1.0, "conf": 0.2}
    assert plan_action(yaw_error_rad=0.0, gait=weak, available=CLIPS).action is None
    slow = {"state": "march", "cadence_hz": 0.01, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=slow, available=CLIPS).action is None


def _latch(servo, human, robot, *, t0=0.0, n=8, dt=0.15):
    """Feed the burst of frames YawServo needs before it latches a reference.

    The servo deliberately no longer latches on a single frame: body yaw is the
    noisiest cue in the pipeline, and taking one arbitrary frame as the origin is
    how the whole heading loop ends up with a fixed offset (measured: the hip-yaw
    bias pinned to one side in 81% of a recorded session). It latches the median
    of a short burst instead, so tests have to supply one.

    Returns the timestamp just after the latch, so callers can carry on.
    """
    t = t0
    for _ in range(n):
        servo.update(human_yaw=human, conf=1.0, robot_yaw=robot, now_s=t)
        t += dt
    assert servo.latched, "burst should have latched the reference"
    return t


def test_yaw_servo_tracks_a_relative_rotation() -> None:
    servo = YawServo()
    # Latching zeroes the error: the subject's and the robot's initial headings
    # are both arbitrary, so only the change matters.
    t = _latch(servo, 0.3, -2.0)
    assert abs(servo.error(-2.0)) < 1e-9
    # Human turns 0.5 rad -> the robot is asked to turn the same way.
    servo.update(human_yaw=0.8, conf=1.0, robot_yaw=-2.0, now_s=t)
    assert abs(servo.error(-2.0) - 0.5) < 1e-9
    # ... and the error closes as the robot actually gets there.
    assert abs(servo.error(-1.5)) < 1e-9


def test_yaw_servo_ignores_unusable_measurements() -> None:
    servo = YawServo()
    servo.update(human_yaw=None, conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=float("nan"), conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=0.5, conf=0.1, robot_yaw=0.0, now_s=0.0)
    assert not servo.latched
    assert servo.error(0.0) == 0.0


def test_yaw_servo_relatches_after_losing_the_subject() -> None:
    servo = YawServo(relatch_after_s=2.0)
    t = _latch(servo, 0.0, 0.0)
    servo.update(human_yaw=1.0, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0) - 1.0) < 1e-9
    # Subject walks off and comes back facing somewhere else entirely: chasing
    # the stale error would spin the robot for no reason. The gap drops the old
    # reference immediately; the new one comes from a fresh burst.
    servo.update(human_yaw=-1.0, conf=1.0, robot_yaw=0.0, now_s=t + 10.0)
    assert not servo.latched
    assert servo.error(0.0) == 0.0
    _latch(servo, -1.0, 0.0, t0=t + 10.0)
    assert abs(servo.error(0.0)) < 1e-9


def test_yaw_servo_sign_flips_the_mapping() -> None:
    servo = YawServo(sign=-1.0)
    t = _latch(servo, 0.0, 0.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0) + 0.4) < 1e-9


def test_yaw_servo_wraps_across_the_discontinuity() -> None:
    servo = YawServo()
    t = _latch(servo, 0.0, 3.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=3.0, now_s=t)
    # desired = 3.4 rad, which wraps past pi; the error must stay small and
    # correctly signed instead of demanding a near-full turn the other way.
    assert abs(servo.error(3.0) - 0.4) < 1e-9
    assert abs(servo.error(3.4 - 2 * math.pi)) < 1e-9


def test_wrap_pi() -> None:
    for angle in (3 * math.pi, -3 * math.pi, 5 * math.pi):
        assert abs(abs(wrap_pi(angle)) - math.pi) < 1e-9
    assert abs(wrap_pi(0.5) - 0.5) < 1e-12
    assert abs(wrap_pi(2 * math.pi + 0.25) - 0.25) < 1e-9
    assert -math.pi <= wrap_pi(123.456) < math.pi


def test_reset_unlatches() -> None:
    servo = YawServo()
    _latch(servo, 0.2, 0.0)
    servo.reset()
    assert not servo.latched and servo.error(1.0) == 0.0


# ---------------------------------------------------------------------------
# Turning through large angles, and noise immunity
# ---------------------------------------------------------------------------
def _converge(target_yaw, *, clip="/w/TurnLeft60.motion", steps=40, noise=0.0):
    """Settle facing forward, then have the human turn to ``target_yaw``.

    The settle phase matters: :class:`YawServo` latches the human/robot heading
    pair on first sight, so a *constant* offset is absorbed by the latch and is
    not an error at all -- only a change from the latched reference is. The robot
    is then turned by discrete clips of the size the filename implies.

    Returns ``(final |error| rad, clips played, left/right direction flips)``.
    """
    clips = {"turn_left": clip, "turn_right": clip.replace("Left", "Right")}
    nominal = abs(motion_nominal_yaw(clip))
    servo = YawServo()
    robot = 0.0
    played = flips = 0
    previous = None
    turning = False
    t = 0.0
    for _ in range(5):                      # settle: human faces forward
        servo.update(human_yaw=0.0, conf=1.0, robot_yaw=robot, now_s=t)
        t += 0.1
    for i in range(steps):                  # then they turn, with some wobble
        wobble = noise * (1 if i % 2 else -1)
        servo.update(human_yaw=target_yaw + wobble, conf=1.0,
                     robot_yaw=robot, now_s=t)
        t += 0.1
        plan = plan_action(yaw_error_rad=servo.error(robot),
                           available=clips, turning=turning)
        turning = plan.is_turn
        if plan.is_turn:
            played += 1
            if previous and previous != plan.action:
                flips += 1
            previous = plan.action
            robot += nominal * (1 if plan.action == "turn_left" else -1)
    return abs(wrap_pi(servo.error(robot))), played, flips


def test_the_servo_turns_the_robot_all_the_way_round() -> None:
    """Half a turn takes several 60 deg clips; it must converge, not stall."""
    for target in (math.radians(150), math.radians(180), -math.radians(170)):
        err, played, flips = _converge(target)
        assert played >= 2, target
        assert err <= math.radians(35), (target, math.degrees(err))
        assert flips == 0, (target, flips)


def test_the_servo_converges_for_any_target_heading() -> None:
    for degrees in range(-180, 181, 15):
        err, _, flips = _converge(math.radians(degrees))
        assert err <= math.radians(35), (degrees, math.degrees(err))
        assert flips <= 1, (degrees, flips)


def test_the_servo_takes_the_short_way_round() -> None:
    """A 170 deg target must not be chased the 190 deg way."""
    _, played_left, _ = _converge(math.radians(170))
    _, played_right, _ = _converge(-math.radians(170))
    assert played_left == played_right          # symmetric effort
    assert played_left <= 4                     # 170/60 -> 3 clips, not 4+


def test_measurement_noise_does_not_thrash_the_turn_direction() -> None:
    """The real failure: a jittery yaw made the planner alternate left/right,
    which starves forward walking and trips the locomotion failure backoff."""
    err, played, flips = _converge(0.0, noise=math.radians(12), steps=60)
    assert flips == 0
    assert played == 0            # 12 deg of wobble is below the turn gate
    assert err <= math.radians(15)


def test_a_small_turn_does_not_start_a_clip() -> None:
    """Shifting 20 deg is below the gate: firing a 60 deg clip would leave a
    bigger error, of the opposite sign, than it started with."""
    err, played, _ = _converge(math.radians(20), steps=80)
    assert played == 0
    assert err == pytest.approx(math.radians(20), abs=1e-6)


def test_a_constant_offset_is_absorbed_by_the_latch() -> None:
    """Standing habitually a little off-square is not an error to correct: the
    servo tracks rotation *relative* to where it first saw you."""
    servo = YawServo()
    for i in range(10):
        servo.update(human_yaw=math.radians(25), conf=1.0,
                     robot_yaw=0.0, now_s=i * 0.1)
    assert servo.error(0.0) == pytest.approx(0.0, abs=1e-9)
    assert plan_action(yaw_error_rad=servo.error(0.0),
                       available={"turn_left": "/w/TurnLeft60.motion"}).action is None


def test_yaw_servo_latch_ignores_a_spiking_frame() -> None:
    """The reason the latch is a median of a burst rather than one frame.

    The body-yaw cue is documented to spike to its +/-90 deg bound on a minority
    of frames while the subject stands square to the camera. If such a frame is
    the one that sets the origin, every subsequent error inherits the offset --
    which is exactly the fixed one-sided bias measured in a recorded session.
    """
    servo = YawServo()
    t = 0.0
    for i in range(9):
        # One frame in three is a bogus +90 deg spike; the truth is 0.0.
        human = math.pi / 2 if i % 3 == 0 else 0.0
        servo.update(human_yaw=human, conf=1.0, robot_yaw=0.0, now_s=t)
        t += 0.15
    assert servo.latched
    # The median rejected the spikes, so standing still is not a heading error.
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0)) < 1e-9


def test_yaw_servo_diagnostics_separate_offset_from_tracking() -> None:
    servo = YawServo()
    t = _latch(servo, 0.2, 1.0)
    servo.update(human_yaw=0.9, conf=1.0, robot_yaw=1.0, now_s=t)
    d = servo.diagnostics(1.0)
    assert d["latched"] == 1.0
    assert d["human_ref"] == pytest.approx(0.2)
    assert d["robot_ref"] == pytest.approx(1.0)
    assert d["human_now"] == pytest.approx(0.9)
    assert d["error"] == pytest.approx(0.7, abs=1e-9)


# ---------------------------------------------------------------------------
# Which joints a clip actually drives
# ---------------------------------------------------------------------------
def test_motion_joints_reads_the_clip_header(tmp_path) -> None:
    """Webots' NAO walk clips drive only the legs. Knowing that is what lets the
    controller keep the arms and head imitating while the robot walks, instead of
    suspending the whole body for the length of every step."""
    path = tmp_path / "Forwards.motion"
    path.write_text(
        "#WEBOTS_MOTION,V1.0,LHipYawPitch,LHipRoll,LHipPitch,LKneePitch,"
        "LAnklePitch,LAnkleRoll,RHipYawPitch,RHipRoll,RHipPitch,RKneePitch,"
        "RAnklePitch,RAnkleRoll\n"
        "00:00:000,Pose1,0,0.027,-0.505,1.042,-0.537,-0.027,0,0.027,-0.505,"
        "1.042,-0.537,-0.027\n",
        encoding="utf-8",
    )
    joints = motion_joints(str(path))
    assert len(joints) == 12
    assert "LHipPitch" in joints and "RAnkleRoll" in joints
    # Crucially, no arm or head joint is in there.
    assert not [j for j in joints if "Shoulder" in j or "Elbow" in j or "Head" in j]


def test_motion_joints_returns_empty_for_anything_it_cannot_read(tmp_path) -> None:
    """An empty list means "unknown", and the caller must then hand over the
    whole body -- handing over too little would fight the clip's keyframes."""
    assert motion_joints(None) == []
    assert motion_joints(str(tmp_path / "missing.motion")) == []
    junk = tmp_path / "junk.motion"
    junk.write_text("not a motion file\n", encoding="utf-8")
    assert motion_joints(str(junk)) == []
    bare = tmp_path / "bare.motion"
    bare.write_text("#WEBOTS_MOTION,V1.0\n", encoding="utf-8")
    assert motion_joints(str(bare)) == []


# ---------------------------------------------------------------------------
# Cyclic gait detection
# ---------------------------------------------------------------------------
def test_gait_cycle_finds_the_limit_cycle_in_a_walk_clip(tmp_path) -> None:
    """The whole point: a one-shot animation contains a repeatable stride.

    Cyberbotics' walk clips are authored as animations -- squat, accelerate,
    stride, decelerate, stand -- and played that way the transient dominates:
    it is 49% of the short clip's 2.60 s, and during the closing settle the torso
    travels BACKWARD. But the middle is a true limit cycle, so it can be rewound
    and the robot walks for as long as it is asked to, at the speed the stride is
    worth rather than the speed the transient averages down to.
    """
    clip = write_cyclic_clip(tmp_path / "Walk.motion", period=26, cycles=3,
                             lead=8, tail=10)
    cycle = gait_cycle(clip)
    assert cycle is not None
    # The detector must find the period the fixture was built with, not a
    # multiple of it: a tighter loop means finer control over when to leave.
    assert cycle.period_s == pytest.approx(26 * 0.04)
    assert cycle.loop_start_s == pytest.approx(8 * 0.04)
    assert cycle.loop_end_s == pytest.approx((8 + 26) * 0.04)
    # And it must be a cycle that goes somewhere.
    assert cycle.advance_m > 0.02
    assert cycle.speed_mps == pytest.approx(cycle.advance_m / cycle.period_s)


def test_gait_cycle_demands_a_seam_that_commands_no_motion(tmp_path) -> None:
    """A loop seam is a teleport executed in one 20 ms step with the velocity
    caps lifted, so the only acceptable seam is one where no joint moves.

    This is why the real short clip cannot be cycled: its best available seam is
    0.107 rad, which would be a 5.35 rad/s jolt once per stride. A drift of a
    thousandth of a radian per keyframe -- a gait that creeps rather than
    repeating, which is what most authored clips are -- is enough to disqualify
    one here, and that strictness is the safety property, not pedantry.

    Note what is NOT required: that the whole clip be uniformly periodic. Only
    the two ends of the loop window have to agree. Anything in between is simply
    part of the stride and gets replayed with it.
    """
    poses = cyclic_poses(period=26, cycles=3, lead=8, tail=10)
    assert gait_cycle(write_clip(tmp_path / "ok.motion", poses)) is not None

    # A single nudged keyframe does NOT disqualify the clip, and should not: it
    # is inside the window, so it repeats along with everything else and the seam
    # is untouched.
    poses[20][1]["LKneePitch"] += 0.001
    nudged = gait_cycle(write_clip(tmp_path / "nudged.motion", poses))
    assert nudged is not None
    assert nudged.period_s == pytest.approx(26 * 0.04)

    # A clip that creeps has no exact seam anywhere, and is refused.
    drifting = cyclic_poses(period=26, cycles=3, lead=8, tail=10)
    for index, (_t, angles) in enumerate(drifting):
        angles["LKneePitch"] += 0.001 * index
    assert gait_cycle(write_clip(tmp_path / "drift.motion", drifting)) is None


def test_gait_cycle_refuses_a_cycle_that_does_not_translate(tmp_path) -> None:
    """Periodicity alone is not a gait. Every clip has some -- a turn rotates
    through repeated steps, a side-step shuffles -- and looping those would spin
    or drift the robot forever with no way to reason about where it ends up.
    Turning is closed-loop on the heading and needs discrete, countable clips.
    """
    marching = write_cyclic_clip(tmp_path / "March.motion", translate=False)
    assert gait_cycle(marching) is None


def test_gait_cycle_skips_only_the_part_of_the_clip_that_goes_nowhere(tmp_path) -> None:
    """Playback starts at ``enter_s``, and the prepare-ramp does that prefix
    instead -- rate-limited and under balance supervision, which is strictly
    better than a clip commanding it with the caps lifted. So the prefix skipped
    must be one that translates the robot by essentially nothing; skipping real
    strides would enter a moving gait from a standstill, with no momentum where
    the clip assumes some.
    """
    clip = write_cyclic_clip(tmp_path / "Walk.motion", lead=8)
    cycle = gait_cycle(clip)
    assert cycle is not None
    assert 0.0 <= cycle.enter_s <= cycle.loop_start_s
    speeds = clip_torso_speeds(motion_poses(clip))
    skipped = sum(speeds[:int(round(cycle.enter_s / 0.04))]) * 0.04
    assert abs(skipped) < 0.002


def test_gait_cycle_leaves_through_the_clips_own_deceleration(tmp_path) -> None:
    """How it stops is what makes it safe rather than clever.

    Not by freezing mid-stride -- the legs stop and the body keeps its momentum,
    which walks the robot over. By jumping once, at the phase of the cycle where
    the jump costs nothing, into the clip's own closing deceleration, so
    Cyberbotics' own balanced feet-together settle brings the robot to rest.
    """
    clip = write_cyclic_clip(tmp_path / "Walk.motion", period=26, cycles=3,
                             lead=8, tail=10)
    cycle = gait_cycle(clip)
    assert cycle is not None
    # The jump is taken from inside the FIRST period, so its phase comes round
    # once per stride however long the robot has been walking.
    assert cycle.loop_start_s <= cycle.exit_from_s < cycle.loop_end_s
    # ...to a pose past the last full cycle, i.e. into the deceleration.
    assert cycle.exit_to_s >= cycle.loop_end_s
    assert cycle.tail_s > 0.0
    # ...and it commands no meaningful motion: one control step at 20 ms.
    assert cycle.exit_cost_rad / 0.02 < 1.0
    # Worst case: wait for the phase, then ride the tail out.
    assert cycle.stop_latency_s == pytest.approx(cycle.period_s + cycle.tail_s)


def test_gait_cycle_returns_none_for_clips_it_cannot_loop(tmp_path) -> None:
    """A clip that is not cyclic must be DETECTED as not cyclic, not looped on
    faith -- the fallback (one-shot playback) is always correct, so there is
    never a reason to guess."""
    assert gait_cycle(None) is None
    assert gait_cycle(str(tmp_path / "missing.motion")) is None
    short = write_clip(tmp_path / "short.motion", cyclic_poses(period=26, cycles=1,
                                                               lead=0, tail=0)[:6])
    assert gait_cycle(short) is None


def test_motion_pose_at_returns_the_keyframe_in_force_at_that_time(tmp_path) -> None:
    """The ramp target for a clip entered part-way in. It has to be the keyframe
    at or BEFORE the offset -- playback holds each keyframe until the next one,
    so that is the pose playback will actually start from."""
    poses = cyclic_poses(period=26, cycles=2, lead=4, tail=4)
    clip = write_clip(tmp_path / "Walk.motion", poses)
    assert motion_pose_at(clip, 0.0) == pytest.approx(poses[0][1])
    # Between keyframes: the earlier one is still in force.
    assert motion_pose_at(clip, 0.06) == pytest.approx(poses[1][1])
    assert motion_pose_at(clip, 0.08) == pytest.approx(poses[2][1])
    # Before the start and past the end, clamp rather than fail.
    assert motion_pose_at(clip, -1.0) == pytest.approx(poses[0][1])
    assert motion_pose_at(clip, 999.0) == pytest.approx(poses[-1][1])
    assert motion_pose_at(None, 0.5) == {}


# ---------------------------------------------------------------------------
# Walk clip selection
# ---------------------------------------------------------------------------
def test_the_long_clip_is_chosen_only_when_it_can_actually_be_cycled(tmp_path) -> None:
    """The trade is entirely conditional on cycling working.

    Played one-shot the long clip is strictly WORSE: it commits the robot to
    6.76 s and 0.46 m before it can be asked to stop. Cycled it is strictly
    better: the same stride without the transient between repetitions, and a stop
    latency shorter than the short clip's own length. So the choice is made by
    asking whether a cycle is really there.
    """
    short = write_clip(tmp_path / "Forwards.motion",
                       cyclic_poses(period=26, cycles=1, lead=8, tail=10,
                                    translate=False))
    cyclic = write_cyclic_clip(tmp_path / "Forwards50.motion")
    files, note = select_walk_clip({"forward": short, "forward_continuous": cyclic})
    assert files["forward"] == cyclic
    assert "continuous gait" in note

    # Same call, but the long clip marches in place instead of walking.
    dud = write_cyclic_clip(tmp_path / "Dud.motion", translate=False)
    files, note = select_walk_clip({"forward": short, "forward_continuous": dud})
    assert files["forward"] == short
    assert "no detectable gait cycle" in note


def test_walk_clip_selection_never_leaks_its_own_bookkeeping_key(tmp_path) -> None:
    """``forward_continuous`` exists only so both candidates get DISCOVERED. If
    it reached plan_action it would be a second, unplannable forward action."""
    cyclic = write_cyclic_clip(tmp_path / "Forwards50.motion")
    for available in ({"forward_continuous": cyclic},
                      {"forward": cyclic, "forward_continuous": cyclic},
                      {}):
        files, _note = select_walk_clip(available)
        assert "forward_continuous" not in files


def test_selection_falls_back_cleanly_with_no_continuous_clip(tmp_path) -> None:
    short = write_clip(tmp_path / "Forwards.motion", cyclic_poses(translate=False))
    files, note = select_walk_clip({"forward": short})
    assert files == {"forward": short}
    assert "no continuous walk clip" in note


# ---------------------------------------------------------------------------
# The turn gate
# ---------------------------------------------------------------------------
def test_a_turn_is_never_requested_that_no_clip_can_serve() -> None:
    """The gate that admits a turn and the test that picks a clip have to agree.

    They were allowed to disagree, and the gap between them was a trap:
    ``turn_start_rad`` 0.35 admits the request at 20 deg, but with
    ``overshoot_frac`` 0.65 the smallest clip on disk (40 deg = 0.698 rad) does
    not fit until 0.454 rad = 26 deg. In that band the controller announced a
    turn, spent 0.7 s ramping the legs down into the clip's opening crouch, then
    found nothing fitted and stood back up again. 29 prepares in the recorded
    session ramped and never played, 14 of them turn_right, with 13.8% of walking
    frames sitting in the band. It reads exactly like "it tries to step and
    falls".
    """
    available = {"turn_left": "/w/TurnLeft40.motion",
                 "turn_right": "/w/TurnRight40.motion"}
    params = LocomotionParams()
    for millideg in range(0, 90_000, 500):
        error = math.radians(millideg / 1000.0)
        for sign in (+1.0, -1.0):
            for turning in (False, True):
                plan = plan_action(yaw_error_rad=sign * error, available=available,
                                   params=params, turning=turning,
                                   yaw_trustworthy=True)
                if plan.action is None:
                    continue
                # Whatever was planned must be a clip that CONVERGES: firing a
                # clip of nominal N at error e leaves |e - N|, which is only an
                # improvement when e > N/2.
                nominal = abs(motion_nominal_yaw(available[plan.action]))
                assert error > nominal / 2.0, (
                    f"{plan.action} planned at {math.degrees(error):.1f} deg "
                    f"against a {math.degrees(nominal):.0f} deg clip: the turn "
                    f"would oscillate instead of converging")


def test_the_turn_gate_still_turns_once_a_clip_does_fit() -> None:
    """Closing the deadband must not close the door on turning altogether."""
    available = {"turn_left": "/w/TurnLeft40.motion",
                 "turn_right": "/w/TurnRight40.motion"}
    params = LocomotionParams()
    fits_at = params.overshoot_frac * math.radians(40.0)
    assert plan_action(yaw_error_rad=fits_at + 0.01, available=available,
                       params=params).action == "turn_left"
    assert plan_action(yaw_error_rad=-(fits_at + 0.01), available=available,
                       params=params).action == "turn_right"
    # And a coarse clip still wins when the error is big enough for it.
    available["turn_left_coarse"] = "/w/TurnLeft180.motion"
    assert plan_action(yaw_error_rad=math.radians(150.0), available=available,
                       params=params).action == "turn_left_coarse"


# ---------------------------------------------------------------------------
# What a cyclic clip asks of the hardware
# ---------------------------------------------------------------------------
def test_a_loop_seam_commands_less_motion_than_a_normal_keyframe(tmp_path) -> None:
    """The seam must be the QUIETEST moment in the stride, not just a quiet one.

    Rewinding is a teleport, executed in a single control step with the velocity
    caps already lifted, so if the seam commanded more motion than the clip's own
    keyframes do the loop would inject a jolt once per stride -- and a jolt once
    per stride is the "stepping, not walking" the whole change exists to remove.
    """
    poses = cyclic_poses(period=26, cycles=3, lead=8, tail=10)
    clip = write_clip(tmp_path / "Walk.motion", poses)
    cycle = gait_cycle(clip)
    assert cycle is not None
    lookup = {round(t, 3): angles for t, angles in poses}
    start = lookup[round(cycle.loop_start_s, 3)]
    end = lookup[round(cycle.loop_end_s, 3)]
    seam = max(abs(end[j] - start[j]) for j in start)
    # The largest step the clip itself takes between adjacent keyframes.
    normal = max(max(abs(b[j] - a[j]) for j in a)
                 for (_t1, a), (_t2, b) in zip(poses, poses[1:]))
    assert seam < normal / 100.0, (
        f"the seam moves {seam:.4f} rad against a normal keyframe step of "
        f"{normal:.4f} rad")


def test_the_cycled_window_stays_within_the_motors_declared_speed() -> None:
    """The clip is Cyberbotics', not ours, so what it demands is worth checking.

    Forwards50.motion asks for more than the declared 6.40 rad/s knee ceiling in
    5 of its 2028 joint-intervals, all at stance exchange, peaking at 6.55 rad/s
    (102.3%). Cycling does not inherit all of those: entering at 0.72 s and
    looping [1.80, 2.84] s plays exactly ONE of them per stride, at 6.425 rad/s
    = 100.4%, so the motor arrives 0.001 rad late once every 1.04 s. That is
    bounded and invisible, and it is strictly less than one-shot playback of the
    same clip would ask for -- but it is a real exceedance and it should be
    recorded rather than discovered later.

    Skipped where Webots is not installed; the shipped clips are the subject.
    """
    clips = os.path.join(
        "/snap/webots/current/usr/share/webots/projects/robots/softbank/nao/motions")
    path = os.path.join(clips, "Forwards50.motion")
    if not os.path.isfile(path):
        pytest.skip("Webots' NAO motion clips are not installed")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))
    from pose_control_utils import get_default_motor_configs

    caps = {name: cfg.max_velocity
            for name, cfg in get_default_motor_configs().items()}
    cycle = gait_cycle(path)
    assert cycle is not None
    window = [(t, a) for t, a in motion_poses(path)
              if cycle.loop_start_s - 1e-9 <= t <= cycle.loop_end_s + 1e-9]
    over = []
    for (t1, a), (t2, b) in zip(window, window[1:]):
        for joint, value in a.items():
            cap = caps.get(joint)
            if cap is None:
                continue
            speed = abs(b[joint] - value) / (t2 - t1)
            if speed > cap:
                over.append((speed, cap, joint, t2))
    # Exactly one, and only just over: a lag of a milliradian per stride.
    assert len(over) == 1, f"the loop window's demands changed: {over}"
    speed, cap, _joint, _t = over[0]
    assert speed / cap < 1.01
    assert (speed - cap) * 0.04 < 0.002


# ---------------------------------------------------------------------------
# 2026-09-10 live test: "the robot turned left fine, then never registered me
# turning right". Both defects were in the heading path.
# ---------------------------------------------------------------------------
def test_a_mid_turn_confidence_dropout_does_not_move_the_origin() -> None:
    """A yaw_conf dropout is CAUSED by turning, so it must not re-latch.

    Body yaw is recovered from shoulder-line foreshortening, which is exactly
    the measurement that degrades when the torso turns away from the camera. In
    the live test yaw_conf sat under conf_min for 4.6 s mid-turn; at the old
    relatch_after_s=2.0 the servo re-latched and the TURNED pose (-82.9 deg)
    became the new "facing forward". Standing square to the camera then read as
    a +36..+40 deg error, so the robot turned while the subject was still and
    ignored their next real turn -- and it compounds, because every turn moves
    the zero again.
    """
    servo = YawServo()
    assert servo.relatch_after_s >= 5.0, (
        "relatch_after_s must stay comfortably longer than a turn-induced "
        "yaw_conf dropout (measured: 4.6 s)")
    t = _latch(servo, 0.0, 0.0)
    origin = servo._human_ref

    # The subject turns 80 deg; confidence collapses for 4.6 s partway through.
    servo.update(human_yaw=-1.40, conf=1.0, robot_yaw=0.0, now_s=t + 0.5)
    for i in range(60):                       # 4.6 s of unusable yaw
        servo.update(human_yaw=-1.40, conf=0.1, robot_yaw=0.0,
                     now_s=t + 0.6 + 0.077 * i)
    servo.update(human_yaw=-1.40, conf=1.0, robot_yaw=0.0, now_s=t + 5.3)

    assert servo.latched, "the servo dropped its reference over a 4.6 s dropout"
    assert servo._human_ref == origin, (
        "the origin moved to the turned pose; facing the camera will now read "
        "as a large heading error")
    # Back square to the camera -> essentially nothing left to turn.
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=0.0, now_s=t + 6.0)
    assert abs(servo.error(0.0)) < 0.05, (
        f"square to the camera reads {math.degrees(servo.error(0.0)):+.1f} deg "
        f"of heading error")


def test_a_genuine_departure_still_relatches() -> None:
    """The longer timeout must not disable re-latching altogether."""
    servo = YawServo()
    t = _latch(servo, 0.0, 0.0)
    servo.update(human_yaw=1.0, conf=1.0, robot_yaw=0.0,
                 now_s=t + servo.relatch_after_s + 2.0)
    assert not servo.latched
    assert servo.error(0.0) == 0.0, "a stale error would spin the robot"


def test_a_steady_rotation_counts_as_a_stable_heading() -> None:
    """stable() must not reject the very motion that needs a turn.

    Peak-to-peak alone cannot separate a NOISY yaw estimate from a TURNING one,
    and rejecting both means refusing to turn while the subject rotates:
    stability_spread_rad 0.5 rad over a 1.0 s window rejects anything above
    29 deg/s, and a comfortable turn is faster. They are separable by SHAPE --
    a rotation is monotone, the cue's noise signature is scatter.
    """
    servo = YawServo()
    # 90 deg/s for a full window: way past the peak-to-peak limit, but a line.
    for i in range(21):
        t = 0.05 * i
        servo.update(human_yaw=math.radians(90.0) * t, conf=1.0,
                     robot_yaw=0.0, now_s=t)
    spread = math.radians(90.0) * 1.0
    assert spread > servo.stability_spread_rad, "precondition: spread is large"
    assert servo.stable(), "a steady rotation was rejected as an unstable heading"


def test_scatter_is_still_rejected_as_unstable() -> None:
    """The documented failure mode -- yaw spiking to its bounds -- must fail."""
    servo = YawServo()
    for i in range(21):
        t = 0.05 * i
        servo.update(human_yaw=(math.radians(90.0) if i % 2 else math.radians(-90.0)),
                     conf=1.0, robot_yaw=0.0, now_s=t)
    assert not servo.stable(), "alternating +/-90 deg scatter was called stable"


def test_a_noisy_ramp_is_rejected() -> None:
    """A rotation buried in large scatter is not a heading worth turning on."""
    servo = YawServo()
    for i in range(21):
        t = 0.05 * i
        jitter = math.radians(25.0) * (1 if i % 2 else -1)
        servo.update(human_yaw=math.radians(60.0) * t + jitter, conf=1.0,
                     robot_yaw=0.0, now_s=t)
    assert not servo.stable()


# ---------------------------------------------------------------------------
# Aimed turning: a clip that can be stopped part-way is planned differently
# ---------------------------------------------------------------------------
LADDER = {
    "turn_left": "/w/TurnLeft40.motion",
    "turn_right": "/w/TurnRight40.motion",
    "turn_left_coarse": "/w/TurnLeft180.motion",
    "turn_right_coarse": "/w/TurnRight180.motion",
}
STOPPABLE = frozenset(LADDER)


def test_an_interruptible_turn_prefers_the_largest_clip() -> None:
    """The inversion of the old rule, and the reason for it.

    "Do not overshoot" is right for a clip played whole and wrong for one that
    can be stopped at any of 133 certified keyframes. Once it can be stopped,
    the big clip wins on every axis: 24 deg/s against 9, a 16 deg ladder of
    deliverable angles against a single 40 deg step, and one clip instead of the
    three-with-two-settles that a 90 deg turn used to cost.
    """
    for degrees in (30, 45, 90, 150):
        plan = plan_action(yaw_error_rad=math.radians(degrees), available=LADDER,
                           interruptible=STOPPABLE)
        assert plan.action == "turn_left_coarse", degrees
    # ... and the old rule is untouched for a clip that must be played whole.
    assert plan_action(yaw_error_rad=math.radians(45),
                       available=LADDER).action == "turn_left"


def test_an_interruptible_turn_serves_errors_the_old_gate_refused() -> None:
    """The dead band this closes, in the robot's own terms.

    turn_start_rad admits a turn at 20 deg, but the smallest clip on disk could
    not be fired without overshooting until 26 deg -- so a heading error between
    the two was announced, ramped for, and then never served. That gate exists to
    stop a clip overshooting, and a clip we can stop part-way cannot overshoot,
    so it does not apply to one.
    """
    between = math.radians(23)
    assert plan_action(yaw_error_rad=between, available=LADDER).action is None
    assert plan_action(yaw_error_rad=between, available=LADDER,
                       interruptible=STOPPABLE).is_turn


def test_a_rotation_in_progress_is_not_handed_to_a_different_clip() -> None:
    """Swapping clips mid-turn means releasing the body, settling, re-preparing
    and starting again -- while half-turned. Whatever is turning us the right
    way keeps the body until the direction changes or the heading is served."""
    for degrees in (150, 90, 45, 20):
        plan = plan_action(yaw_error_rad=math.radians(degrees), available=LADDER,
                           interruptible=STOPPABLE, turning=True,
                           playing="turn_left_coarse")
        assert plan.action == "turn_left_coarse", degrees
    # Direction still beats continuity: the human turning back the other way
    # must not be answered by carrying on round.
    plan = plan_action(yaw_error_rad=-math.radians(45), available=LADDER,
                       interruptible=STOPPABLE, turning=True,
                       playing="turn_left_coarse")
    assert plan.action == "turn_right_coarse"


def test_a_play_whole_turn_keeps_its_overshoot_floor_at_both_gates() -> None:
    """The floor looks like lost responsiveness at the stop gate. It is not.

    A 60 deg clip asked to correct a 17 deg residual turns the robot to -43 deg,
    which is worse than where it started, and the next step asks for another
    clip: a limit cycle. For a clip that must be played whole the floor is what
    stops the loop at the finest error that clip can actually serve.
    """
    p = LocomotionParams()
    whole = {"turn_left": "/w/TurnLeft60.motion",
             "turn_right": "/w/TurnRight60.motion"}
    residual = math.radians(17)
    assert plan_action(yaw_error_rad=residual, available=whole, params=p,
                       turning=True).action is None
    # The same residual IS served when the clip can simply be stopped early.
    assert plan_action(yaw_error_rad=residual, available=LADDER, params=p,
                       interruptible=STOPPABLE, turning=True).is_turn


def test_aiming_a_turn_beats_chaining_whole_clips(tmp_path) -> None:
    """End to end, against the real clips: fewer clips, less time, less error.

    This is the claim the change is worth making, so it is measured rather than
    asserted -- replayed at the real 20 ms control step over the clips' own
    keyframe kinematics.
    """
    motions = os.path.join(os.path.dirname(__file__), "..", "main", "controllers",
                           "pose_imitation_controller", "motions")
    files = find_motion_files([motions])
    clips = {k: files[k] for k in
             ("turn_left", "turn_right", "turn_left_coarse", "turn_right_coarse")
             if k in files}
    if len(clips) < 4:
        pytest.skip("the turn clips are not installed")
    from walk_motion import turn_schedule
    schedules = {k: turn_schedule(v) for k, v in clips.items()}
    if any(s is None for s in schedules.values()):
        pytest.skip("no CoM model, so no clip can be aimed")
    stoppable = frozenset(schedules)

    def rotate(target_rad: float, *, aimed: bool) -> tuple[float, float, int]:
        robot = elapsed = 0.0
        played = 0
        action = schedule = None
        clock = base = start = 0.0
        while elapsed < 90.0:
            error = wrap_pi(target_rad - robot)
            if action is None:
                plan = plan_action(yaw_error_rad=error, available=clips,
                                   interruptible=stoppable if aimed else None)
                if not plan.is_turn:
                    break
                action, schedule = plan.action, schedules[plan.action]
                clock = schedule.entry_s if aimed else 0.0
                start, base = schedule.yaw_at(clock), robot
                elapsed += 0.76          # prepare ramp + settle between clips
                played += 1
                continue
            clock += 0.020
            elapsed += 0.020
            delivered = schedule.yaw_at(clock) - start
            robot = base + delivered
            error = wrap_pi(target_rad - robot)
            if aimed:
                ahead = [r for r in schedule.rungs if r[0] >= clock - 0.03]
                done = (not ahead) or min(
                    ahead, key=lambda r: abs((r[1] - start - delivered) - error)
                )[0] <= clock + 0.03
            else:
                at_rung = any(abs(clock - r[0]) <= 0.03 for r in schedule.rungs)
                done = at_rung and plan_action(
                    yaw_error_rad=error, available=clips, turning=True,
                ).action != action
            if clock >= schedule.duration_s:
                done = True
            if done:
                action = schedule = None
        return elapsed, abs(wrap_pi(target_rad - robot)), played

    for degrees in (45, 90, 120, 150, 180):
        want = math.radians(degrees)
        slow_t, slow_e, slow_n = rotate(want, aimed=False)
        fast_t, fast_e, fast_n = rotate(want, aimed=True)
        assert fast_n <= slow_n, degrees
        assert fast_e <= slow_e + 1e-9, (degrees, math.degrees(fast_e),
                                         math.degrees(slow_e))
        assert fast_e <= math.radians(10.0), (degrees, math.degrees(fast_e))
        if degrees >= 90:            # where chaining clips really hurt
            assert fast_t < 0.75 * slow_t, (degrees, fast_t, slow_t)


# ---------------------------------------------------------------------------
# Two witnesses to one question: the pelvis travels, OR the legs cycle
# ---------------------------------------------------------------------------
MARCH = {"state": "march", "cadence_hz": 0.6, "conf": 0.9}
WEAK_MARCH = {"state": "march", "cadence_hz": 0.6, "conf": 0.3}
STILL_LEGS = {"state": "idle", "cadence_hz": 0.0, "conf": 0.9}
WALK_CLIPS = {"forward": "/w/Forwards50.motion", "backward": "/w/Backwards.motion",
              "leg_raise_left": "/w/RaiseLegLeft.motion", "side_right": "/w/SideStepRight.motion"}


def test_idle_yields_to_a_confident_march() -> None:
    """The 2026-09-17 report, in one assertion.

    The action cue's walk witnesses are absolute speeds fitted to a faster
    walker; a subject marching in place or walking at 0.07-0.10 m/s reads as
    "idle" while the gait cue sees the legs cycling. Replayed over that session
    the old explicit stand answered 75% of 59 s of marching with per-joint pose
    imitation of a walking human -- which is what wobbled and fell.
    """
    idle = {"action": "idle", "conf": 0.4, "forward_mps": 0.02}
    plan = plan_action(yaw_error_rad=0.0, gait=MARCH, action=idle, available=WALK_CLIPS)
    assert plan.action == "forward", plan.reason
    # Standing still with still legs is still a stop, explicitly.
    plan = plan_action(yaw_error_rad=0.0, gait=STILL_LEGS, action=idle, available=WALK_CLIPS)
    assert plan.action is None and "standing still" in plan.reason


def test_a_march_the_gait_cue_is_unsure_of_does_not_override_idle() -> None:
    idle = {"action": "idle", "conf": 0.4, "forward_mps": 0.0}
    assert plan_action(yaw_error_rad=0.0, gait=WEAK_MARCH, action=idle,
                       available=WALK_CLIPS).action is None
    slow = dict(MARCH, cadence_hz=0.05)
    assert plan_action(yaw_error_rad=0.0, gait=slow, action=idle,
                       available=WALK_CLIPS).action is None


def test_nobody_observed_is_never_overridden_by_a_warm_gait_cue() -> None:
    """The one verdict a march never overrules: a cue with warm evidence must
    not keep the robot walking at a human who has left the frame (51.4 s of
    false march in one recorded session came from exactly that)."""
    gone = {"action": "unknown", "conf": 0.0, "forward_mps": 0.0}
    plan = plan_action(yaw_error_rad=0.0, gait=MARCH, action=gone, available=WALK_CLIPS)
    assert plan.action is None and plan.reason == "nobody observed"


def test_a_march_beats_a_one_leg_or_side_step_reading() -> None:
    """A foot caught at the top of its swing is not a one-leg stand, and the
    pelvis swaying with the stride is not a side-step. Each leak was a 3-5 s
    clip played at a walking human; measured at 6.1% of marching frames."""
    for verdict in ("raise_left", "step_right"):
        posed = {"action": verdict, "conf": 0.9, "forward_mps": 0.01}
        assert plan_action(yaw_error_rad=0.0, gait=MARCH, action=posed,
                           available=WALK_CLIPS).action == "forward", verdict
    # With still legs the same verdicts are honoured, as before.
    assert plan_action(yaw_error_rad=0.0, gait=STILL_LEGS,
                       action={"action": "raise_left", "conf": 0.9},
                       available=WALK_CLIPS).action == "leg_raise_left"
    assert plan_action(yaw_error_rad=0.0, gait=STILL_LEGS,
                       action={"action": "step_right", "conf": 0.9},
                       available=WALK_CLIPS).action == "side_right"


def test_a_march_drifting_backward_plays_the_backward_clip() -> None:
    p = LocomotionParams()
    back = {"action": "idle", "conf": 0.4, "forward_mps": -(p.march_backward_mps + 0.01)}
    assert plan_action(yaw_error_rad=0.0, gait=MARCH, action=back,
                       available=WALK_CLIPS, params=p).action == "backward"
    # A drift inside the noise band is still a forward march.
    slight = dict(back, forward_mps=-(p.march_backward_mps - 0.01))
    assert plan_action(yaw_error_rad=0.0, gait=MARCH, action=slight,
                       available=WALK_CLIPS, params=p).action == "forward"


def test_a_confident_walk_verdict_still_wins_outright() -> None:
    """The action cue keeps priority when it has a real opinion; the march is a
    second witness for the cases it misses, not a replacement."""
    walking = {"action": "walk_backward", "conf": 0.9, "forward_mps": -0.3}
    assert plan_action(yaw_error_rad=0.0, gait=MARCH, action=walking,
                       available=WALK_CLIPS).action == "backward"
    squat = {"action": "squat", "conf": 0.9, "forward_mps": 0.0}
    assert plan_action(yaw_error_rad=0.0, gait=MARCH, action=squat,
                       available=dict(WALK_CLIPS, squat="/w/Squat.motion")).action == "squat"
