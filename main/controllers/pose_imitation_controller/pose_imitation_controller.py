"""
Real-time Webots NAO pose imitation controller.

Receives human-pose frames via UDP from the Python pipeline (``src/``) and drives
the simulated NAO humanoid in real time: arms, head, legs, and genuine
locomotion (the robot's world coordinates change when you walk, and it turns to
face where you face).

Architecture: this file is the Webots glue and the **arbiter**. All the maths
lives in unit-tested, Webots-free libraries under ``main/libraries/``:

    UDP frame ──► arms + head        : nao_retarget.retarget_upper_body
                └► legs, one of:
                     1. locomotion   : walk_motion.plan_action  + Webots Motion
                                       clips  (real translation / turning)
                     2. march engine  : gait.GaitEngine
                                       (in-place, when no clips exist on disk)
                     3. pose imitation: lower_body.LowerBodyController
                                       (squat, single-leg lift, weight transfer)
                     4. stand         : balance.BalanceController only

Exactly ONE of those four commands the legs on any given step -- two of them at
once means the layers fight and the robot falls, which is the single most common
way a humanoid imitation controller breaks.

A fifth, always-on layer runs underneath the arbiter every tick regardless of
which of the four is active: a continuous tilt-risk EMA (_update_tilt_risk)
that makes _settled() progressively stricter about starting the next
locomotion clip after a wobble. It never touches an in-progress clip -- see
the MotionPlayer docstring for why -- only whether the *next* one is allowed
to start, so a rough patch degrades to march-in-place/pose-imitation (both
fall-safe by design) for a while instead of only reacting once a clip is
already failing.

Protocol (UDP, port 8765, JSON):
    {
      "timestamp_s": 1234567890.123,
      "frame_index": 45,
      "joint_angles_rad": {"LShoulderPitch": 0.5, ...},   # fallback
      "keypoints": {"left_shoulder": [x, y, z, visibility], ...},
      "gait": {"state": "march", "cadence_hz": 0.9, "body_yaw_rad": 0.4, ...}
    }
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import time

try:
    from controller import Motion, Robot  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

try:
    # Supervisor is a Robot subclass, so everything else in this file is
    # unaffected. It is needed only to RELOAD the world after a fall; detection
    # needs no special privileges. Absent on older builds, hence the guard.
    from controller import Supervisor  # type: ignore
except ImportError:  # pragma: no cover - depends on the Webots build
    Supervisor = None  # type: ignore

# Make the shared library importable regardless of Webots' working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402
from walk_motion import (  # noqa: E402
    LocomotionParams,
    YawServo,
    default_motion_search_dirs,
    find_motion_files,
    gait_cycle,
    motion_first_pose,
    motion_joints,
    motion_pose_at,
    motion_poses,
    plan_action,
    TURN_ACTIONS,
    select_walk_clip,
    turn_schedule,
)

# Every logical action that rotates the robot, flattened out of the direction
# table so "is this a turn?" is one membership test.
TURN_ACTION_NAMES = frozenset(
    action for actions in TURN_ACTIONS.values() for action in actions
)

# ===========================================================================
# Configuration
# ===========================================================================
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
SOCKET_RCVBUF = 1 << 16

# --- What drives the legs ---------------------------------------------------
# "auto"   (recommended) the full stack: Webots' pre-balanced NAO walk/turn clips
#          for REAL locomotion, the in-place march engine when no clips are found
#          on disk, and per-leg pose imitation (squat / single-leg lift with a
#          model-verified weight transfer) the rest of the time.
# "pose"   per-leg pose imitation only -- no locomotion, no marching.
# "engine" march engine + pose imitation, never the motion clips (use this to
#          keep the robot on the spot).
# "off"    legs held in the standing posture; upper body only.
# Default is "auto", the full stack. It was "pose" for a while, on the reasoning
# that a clip is a 2-3 second commitment during which the camera is ignored for
# the leg joints -- the opposite of real-time imitation. That reasoning is sound
# and it cost the project walking and turning entirely, because the alternative it
# left in place cannot do either: NAO has no torso-yaw joint, so a rotation can
# only be imitated by physically stepping round, and translating a free-standing
# NAO needs a balanced gait. The clips ARE the gait. The arms, head and torso keep
# imitating throughout playback (a clip declares only the 12 leg joints), so what
# is actually suspended for those 2-3 seconds is per-leg detail while the robot
# does the thing the human is doing: walking.
LEG_CONTROL = "auto"

DRIVE_HEAD = True         # head yaw/pitch follow the human head
SWAP_SIDES = False        # True = mirror-image mapping (robot's left <-> your right)
SMOOTHING_ALPHA = 0.4     # EMA factor for HEAD targets (0..1, higher = snappier)
VELOCITY_SCALE = 0.5      # fraction of each joint's hardware max velocity
LEG_VELOCITY_FACTOR = 0.5 # extra slow-down on leg joints when merely posturing

# --- Arm tracking (see pose_control_utils.ArmTracker) -----------------------
# The arms are NOT smoothed by SMOOTHING_ALPHA. That EMA was stepped once per
# camera frame, so its delay was whatever the frame rate happened to be --
# measured at 127 ms of the 180 ms between the pose log and the command on the
# 2026-09-16 session, and it would have doubled silently had the camera slowed.
# These two are in seconds instead, so the response is the same at any frame
# rate, and they are tuned as a pair: TAU buys smoothness, LEAD gives the delay
# back. Retune with scripts/tune_arm_tracking.py against a recorded session.
# Measured by scripts/tune_arm_tracking.py over run_20260916_110404 (5473
# frames, 411 s, gap-aware), against the linearly-interpolated retarget target.
# This pair is not a trade-off against the old EMA -- it is better on every
# axis. Mean over the eight solved arm joints:
#
#   EMA alpha=0.4 (old)   lag 108 ms   jitter 1.00x   err rms 5.50d   p99 20.29d
#   tau=0.07 lead=0.00    lag  80 ms   jitter 0.39x   err rms 4.65d   p99 13.58d
#   tau=0.07 lead=0.20    lag  38 ms   jitter 0.56x   err rms 5.21d   p99 15.32d
#   tau=0.07 lead=0.32    lag  32 ms   jitter 0.63x   err rms 5.93d   p99 17.54d
#
# LEAD is where the lag actually goes: 0 leaves 80 ms on the table. 0.32 buys
# only 6 ms more than 0.20 while costing more error than the EMA it replaced on
# both measures, because past that the prediction overshoots every direction
# reversal. 0.20 is the fast end of the range that still tracks more accurately
# than what it replaced.
#
# Note the jitter column is below 1.00x even at LEAD=0: that part is not the
# prediction at all, it is `tick_arms` filling in between camera frames instead
# of holding. The old command moved in ~12 steps a second.
ARM_TAU_S = 0.07          # smoothing time constant (was ~0.17 s, frame-rate-set)
ARM_LEAD_S = 0.20         # velocity extrapolation: cancels the residual delay
ARM_VELOCITY_FACTOR = 1.8 # multiplies VELOCITY_SCALE; product capped at hardware max
STALE_AFTER_S = 0.5       # hold pose if no command for this long

# --- Fall recovery ---------------------------------------------------------
# When the robot goes down it stays down: every layer correctly stands itself
# down, and the rest of a test session is then spent driving a robot lying on the
# floor. A recorded session lost 170 of its 178 seconds that way.
#
# The test is the one that needs no conventions: how far the head sits above the
# soles, measured along the world vertical. Forward kinematics gives both in the
# torso frame and the (calibrated) InertialUnit gives the vertical, so no
# Supervisor is needed to DETECT a fall -- only to recover from one.
#
# The threshold separates cleanly from every legitimate posture, because NAO's
# crouch keeps the torso vertical and so barely lowers the head:
#
#     standing                0.460 m       tipped 30 deg      0.398 m
#     base crouch  u=0.10     0.459 m       tipped 60 deg      0.228 m
#     DEEPEST squat u=0.70    0.412 m       on its side       -0.000 m
#                                           face down          0.035 m
#
# 0.25 m is ~57 deg of tilt: far past any recoverable posture, and 40% below the
# deepest squat the robot will ever be asked for. Negative values -- feet above
# head -- are covered by the same test.
AUTO_RELOAD_ON_FALL = True
# 0.30, not 0.25: a recorded fall came to rest with the head at 0.255 m (soles
# carrying 0.04 N, i.e. the robot on its arms) and sat there, unrecovered, for
# the rest of the episode. The deepest squat is 0.41 m, so 0.30 still clears every
# legitimate posture by 110 mm.
FALL_HEAD_HEIGHT_M = 0.30
FALL_CONFIRM_S = 1.0        # must hold this long: no reloading on a transient
FALL_RELOAD_COOLDOWN_S = 10.0
FALL_MAX_RELOADS = 20       # a broken setup must not reload forever
# Two further ways to be "down" that the head-height test misses, both recorded:
#   * STUCK: the torso tilted 0.37 rad for 40 s, both CoM shifters at their
#     clamps, the soles carrying 0.4 and 5 N -- the robot propped on an arm,
#     head still 0.35 m up. Nothing the balance loop can do from there.
#   * UNLOADED: soles carrying almost nothing for seconds on end -- the robot is
#     resting on something other than its feet. A hop or a rocking foot unloads
#     for a fraction of a second; single support keeps ~50 N on the stance foot.
STUCK_TILT_RAD = 0.35       # rad; past every stand-down gate, short of a topple
STUCK_CONFIRM_S = 3.0       # s; the balance loop gets this long to recover
FALL_UNLOADED_N = 8.0       # N; total sole load below this = not standing on the feet
FALL_UNLOADED_S = 2.0       # s

# --- IMU tilt zero ---------------------------------------------------------
# The InertialUnit's roll/pitch are used as "how tipped over is the robot", which
# assumes they read zero when it stands upright. On this NAO model they do not:
# measured on a standing robot at rest, roll reads +1.618 rad (93 deg) while the
# foot sensors carry its full 50 N of body weight and the gyro sits at
# 0.007 rad/s. The sensor frame is mounted rotated; the robot is fine.
#
# Every lower-body symptom on this project traced back to that one number:
#   * balance displaces the CoM by tilt_weight * height * roll = 291 mm of
#     phantom lateral error, against an 88 mm support half-width, so the loop
#     leans the robot onto one foot permanently (measured 47.6 N vs 2.8 N);
#   * lower_body's tilt_abort_rad (0.28) can never be satisfied, so leg imitation
#     stands down every frame and the legs stay bit-identical;
#   * _falling() is permanently true, so no walk clip or march ever starts.
#
# So the zero is LEARNED at startup instead of assumed. The world file always
# spawns the robot standing, and the foot sensors confirm it independently -- soles
# carrying roughly body weight mean "standing", whatever the IMU claims. Samples
# are only accepted while that holds, so a controller restarted on an
# already-fallen robot will not latch the fall as its new upright.
IMU_AUTO_ZERO = True
IMU_CALIBRATION_S = 1.0        # settle window at startup
IMU_CALIBRATION_MIN_SAMPLES = 20
# Soles must carry at least this much IN TOTAL for a sample to count as
# "standing". NAO weighs ~5.2 kg, so a loaded pair reads ~50 N.
IMU_CALIBRATION_MIN_LOAD_N = 20.0
# ...and EACH sole must carry at least this much. The total alone is not
# evidence of standing: a fallen robot resting on one foot logged 25.85 N on the
# right against 0.72 N on the left and passed the total check, latching a zero
# 17.4 deg out. A standing robot splits its weight (measured 25.26/25.26 N), so
# 10 N per foot passes every genuine stance and rejects a one-footed sprawl.
IMU_CALIBRATION_MIN_SOLE_LOAD_N = 10.0
# ...and the head must be about where a standing robot's head is, by forward
# kinematics with tilt taken as zero (so it does not depend on the estimate being
# calibrated). Standing reads ~0.459 m and the deepest squat 0.41; every bad
# calibration in the logs read 0.049-0.205 m.
IMU_CALIBRATION_MIN_HEAD_M = 0.40

# --- Where the torso attitude comes from -----------------------------------
# NOT from the InertialUnit's pitch. That device is unusable for pitch on this
# robot, and the reason is in the proto:
#
#     DEF INERTIAL_UNIT InertialUnit { rotation 1 0 0 1.5708   yAxis FALSE }
#
# The mount (+90 deg about x) is only an offset, which IMU_AUTO_ZERO below
# handles. ``yAxis FALSE`` is the problem. It is meant to disable the yaw output,
# and Webots implements it by converting the sensor's attitude to axis-angle,
# ZEROING the world-z component of the axis, and re-deriving roll/pitch/yaw from
# the result (WbInertialUnit::computeValue). On a sensor mounted upright that is
# harmless. On one rolled 90 deg it mangles the very axes it is not supposed to
# touch. Composing R_world_sensor = R_torso * Rx(pi/2) and running Webots' own
# ENU formula through that zeroing gives, for a pure torso pitch b:
#
#     reported pitch = b / 2        reported yaw = b / 2
#
# and for a pure torso YAW psi, with no tilt at all:
#
#     reported pitch = psi / 2      reported yaw = psi / 2
#
# So the pitch channel reads HALF the real pitch and cannot be told apart from a
# body rotation, and the yaw channel is not a heading at all -- it is a copy of
# the same half-pitch. Both predictions are confirmed on the recorded sessions:
# over 702,719 standing frames imu_yaw and imu_pitch_raw correlate at +0.998 with
# slope +1.03 (they are the same number), and a regression of imu_pitch against
# the torso pitch implied by the loaded foot's own forward kinematics has slope
# +0.48. The roll channel is unaffected (slope +0.98).
#
# The consequences ran through everything that reads a tilt: the balance loop saw
# half the pitch it was correcting, the fall test built a world vertical from it,
# the tilt gates fired at twice their nominal angle -- and the heading servo,
# closing its loop on "yaw", was steering on the pitch signal, which is what put
# a phantom hip-yaw bias into the legs (see HEADING_FROM_IMU).
#
# The ACCELEROMETER has no such problem: all three axes are enabled, its mount is
# a plain 180 deg about x, and gravity is an absolute reference that needs no
# learned zero and cannot be confused with a heading. Webots documents the
# convention exactly ("an Accelerometer at rest with earth's gravity will
# indicate 1 g along the vertical axis"; the source computes -gravity), so for an
# upright torso the device reads (0, 0, -9.81) through its mount. Undoing the
# mount gives the world-up direction in torso coordinates,
#
#     u = (a_x, -a_y, -a_z) = (-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch))
#     pitch = atan2(-u_x, hypot(u_y, u_z))        roll = atan2(u_y, u_z)
#
# with the same sign conventions the balance model uses (see balance.py:
# +pitch = forward, +roll = right).
#
# Low-passed, because an accelerometer measures the robot's own acceleration as
# well as gravity. The time constant is deliberately short: 0.08 s is four
# control steps, enough to reject the spikes a weight shift puts on the sensor,
# and it has to stay well under the 0.17 s time constant of the pendulum it is
# helping to control. What lag remains is paid back by the gyro lead the balance
# loop already applies (BalanceParams.tilt_lead_s, 0.12 s) -- together they are a
# poor man's complementary filter: gravity for the slow truth, the rate gyro for
# the fast term.
#
# The InertialUnit's ROLL is kept as an independent witness: it is trustworthy on
# that axis, so it is compared against the gravity-derived roll every step, and a
# sustained disagreement disables the accelerometer path and says so rather than
# quietly trusting a reading nobody has checked.
#
# AND IT IS OFF, because the live session of 2026-09-04 says so. The derivation
# above is right -- with the robot quiet, the gravity-derived roll matched the
# InertialUnit's to four decimals -- but an accelerometer does not measure
# attitude, it measures gravity PLUS the robot's own acceleration, and this loop
# is what turns that into a fall:
#
#     the loop shifts the pelvis -> the torso accelerates sideways -> the
#     accelerometer reports that acceleration as tilt -> the loop shifts further
#
# An accelerometer inside a position loop is a second derivative fed back as a
# position, which is 180 degrees of phase error, and 0.08 s of low-pass is
# nowhere near enough to suppress it (ACC_SNAP_RAD makes it worse: a transient
# large enough to matter goes straight through). Measured, in log
# webots_joint_trajectory_1788521290 -- ten episodes, nobody in front of the
# camera in any of them, so the legs were not imitating anything:
#
#     episodes 0-6   attitude from gravity   fell within ~2 s, six times over,
#                                            median head height 0.19 m
#     episodes 7-9   attitude from the IMU    stood 40 s and 298 s, head 0.46 m
#
# The cross-check below is what ended that: the roll disagreement latched and the
# path disabled itself, which is the only reason the session produced 298 seconds
# of standing rather than a tenth reload.
#
# What it needs before it can be switched on is the standard fix, not a longer
# filter: integrate the GYRO for the fast attitude and use gravity only to
# correct its slow drift (a complementary filter with a time constant of half a
# second or more). The gyro on this proto has both tilt axes enabled, so the
# parts are there. Until that exists and has been shown to hold a robot up, the
# InertialUnit -- with its pitch scaled, below -- is the honest choice.
# The torso attitude, in order of preference. "supervisor" is the truth and it is
# the default, because on this robot BOTH sensor channels are provably wrong and
# the wrongness is what has been felling the robot:
#
#   * the InertialUnit's pitch is half-scale AND carries a body rotation
#     one-for-one (yAxis FALSE; see below). Measured in log 1788768834, in an
#     episode that ran 3,735 s: the robot stood with hip+knee+ankle = 0.0000 (so
#     the soles were flat and the torso vertical), its head 0.439 m up, both feet
#     carrying 25.258 N and the model's own fore/aft margin at +0.062 m -- while
#     the InertialUnit reported -0.263 rad of pitch, with imu_yaw/imu_pitch_raw =
#     1.073, the yaw-leak signature exactly. The robot had shuffled about 30 deg
#     off its spawn heading and that heading was being read as a permanent
#     phantom pitch. The balance loop then held +0.18 rad of pelvis correction for
#     an hour against a tilt that did not exist, and _settled() -- which the clip
#     layer needs -- passed in 0.015% of that episode against 71.2% elsewhere. So
#     the phantom is not a small residual: it grows with however far the robot has
#     turned, without bound.
#   * the accelerometer measures gravity PLUS the robot's own acceleration, which
#     in a position loop is a second derivative fed back as a position. Tried
#     live: six falls in a row, ~2 s each. See ACC_SNAP_RAD.
#
# The Supervisor's node orientation is neither. It is the same call already made
# for the heading, it needs no learned zero (the world spawns the robot level),
# and it cannot be confused by a body rotation because it reports the rotation
# separately. From the row-major world<-torso matrix, with the ZYX convention
# Webots uses:
#
#     yaw   = atan2(R[3], R[0])
#     pitch = -asin(R[6])          + = nose-down = FORWARD  (matches balance.py)
#     roll  = atan2(R[7], R[8])    + = top toward -y = RIGHT (matches balance.py)
#
# A real NAO has no Supervisor, so both sensor paths stay in the code and in the
# log as the documented fallbacks -- and the balance ALGORITHM is unchanged and
# still model-based. What changes is only that it is now told the truth about
# which way up the robot is.
ATTITUDE_SOURCE = "supervisor"   # "supervisor" | "accel" | "imu"

TILT_FROM_ACCELEROMETER = False
ACC_TILT_TAU_S = 0.08          # low-pass time constant on the gravity angles
# ...but a change this big in one control step is not what the filter is for. The
# magnitude gate below has already thrown out any sample that is not gravity, so a
# surviving jump of a tenth of a radian in 20 ms is a real, fast tilt -- exactly
# the event a balance controller must not be told about three steps late. Filter
# the ripple, follow the lurch.
ACC_SNAP_RAD = 0.10
ACC_MAX_G_ERROR = 3.0          # m/s2 away from 9.81 before a sample is dropped
ACC_DISAGREE_RAD = 0.15        # roll disagreement with the InertialUnit...
ACC_DISAGREE_S = 2.0           # ...held this long -> fall back to the IMU

# The pitch the InertialUnit reports is half the real pitch (see above), so it is
# doubled before anything acts on it. That is a derivation, not a fit: the axis
# masking splits a torso pitch b into pitch b/2 and yaw b/2, exactly, for any b up
# to about 0.5 rad. Independently measured at 0.48 by regressing imu_pitch against
# the torso pitch implied by the loaded foot's own forward kinematics over 598,173
# quasi-static frames.
#
# Uncorrected, the balance loop sees half of every fore/aft error and pushes back
# less hard than it should. So the honest value is 2.0 -- and it ships at 1.0
# anyway, deliberately, because the one experiment available says the loop is
# closer to its stability limit than to its authority limit.
#
# The experiment: on 2026-09-04 the attitude came from gravity, which reports the
# full pitch (the two channels agree when the robot is quiet). That is the same
# doubling this constant would apply, and the fore/aft loop LIMIT CYCLED -- pitch
# swinging +/-0.25 rad within the first second of each episode, the correction
# pinned at its clamp, the robot down inside three seconds, six episodes in a row
# (log 1788521290, episodes 0-6). On the half-scale reading the same controller
# stood still for 900 seconds. Some of that was the reset bug fixed above and
# some was the accelerometer's own acceleration, but the direction of the
# evidence is unambiguous: more pitch sensitivity, less stability.
#
# Under-correcting is sluggish and stable; over-correcting is a fall. So the loop
# stays conservative until a session with a human actually in frame shows the
# fore/aft response is too slow -- and then this is the knob, with
# analyze_run.py's acc_pitch-on-imu_pitch slope as the measurement that says how
# far it can go (about +2). Raise it once, to 2.0, and watch the pitch trace.
#
# The caveat that comes with raising it: the same masking leaks a body rotation
# into this channel one-for-one, so a heading change psi reads as psi/2 of pitch
# and would be doubled back to psi. With the heading loop off the robot no longer
# yaws itself and the measured residual is small (imu_yaw within +/-0.03 rad p95
# while standing, so ~9 mm of modelled CoM error against a 25 mm deadband).
IMU_PITCH_SCALE = 1.0

# --- Heading ---------------------------------------------------------------
# Where the robot's own heading comes from. Turning NEEDS one: NAO has no
# torso-yaw joint, so "the human turned round" can only be imitated by stepping
# round, and stepping round without measuring the result is open-loop -- clip and
# human turn by different amounts and the error accumulates.
#
#   "supervisor" -- the truth, from Supervisor.getSelf().getOrientation(). The
#                   world already grants the Nao node `supervisor TRUE` (the
#                   controller uses simulationReset to stand the robot back up),
#                   so this costs one call per step and no setup. The robot's
#                   forward axis is +x in its own frame, so its world direction is
#                   the first column of the row-major orientation matrix and the
#                   heading is atan2(m[3], m[0]), increasing counter-clockwise
#                   about world +z, i.e. positive = toward the robot's own LEFT,
#                   which is the sign YawServo and TURN_SIGN already expect.
#   "imu"        -- NOT USABLE on this robot, and the reason is in the proto (see
#                   TILT_FROM_ACCELEROMETER): yAxis FALSE makes the InertialUnit's
#                   "yaw" a copy of its half-scale pitch. Kept only so the
#                   degradation is explicit.
#   "off"        -- no heading; turning is disabled and no hip-yaw bias is
#                   commanded. What the robot had before this change.
#
# A real NAO would take the heading from its own odometry or an external
# reference; this is the one place the project uses simulator ground truth, and
# it is used for STEERING, not for balance -- the balance loop stays model-based
# (see balance.py) precisely so it remains a real-robot algorithm.
HEADING_SOURCE = "supervisor"
# There is no heading measurement on this robot. The InertialUnit's yaw axis is
# disabled in the proto (and what it returns instead is the pitch channel, see
# above), and the Gyro's z axis is disabled too, so the yaw RATE is not available
# either. Nothing else observes the robot's rotation without a Supervisor.
#
# Closing the standing-turn loop on that number did real damage. The servo tracked
# the pitch signal, so simply standing still produced a heading "error" -- measured
# at a median 21 deg, agreeing with the human's yaw within 3 deg in 11% of frames --
# and the resulting hip-yaw bias tipped both soles (the axis is canted 45 deg, so
# it pitches the legs as well as yawing them). At the 0.112 rad bias recorded in
# one session that dropped the front sole corners 10.5 mm, the support polygon
# collapsed to the toe line, and the CoM compensation answered the phantom by
# driving the pelvis to its forward clamp with the robot standing dead level.
#
# So the heading loop is off until there is something real to close it on (a
# Supervisor read, or a proto with the yaw axes enabled). Turning is the one
# feature this costs, and it did not work anyway.
HEADING_FROM_IMU = False        # kept: the IMU's yaw is unusable, see HEADING_SOURCE

# --- Preparing to walk -----------------------------------------------------
# A pre-balanced clip opens in a deep, sole-flat crouch (Cyberbotics' walk and
# turn clips all start at knee 1.042 rad, about 0.51 rad of squat) while this
# controller stands at base_crouch_u 0.10 (knee 0.20). Playback commands its
# first keyframe on its very first step, with the velocity caps already lifted,
# so handing over from standing asks the knees for 0.84 rad in one 20 ms step.
# Instead the legs are ramped to the clip's own opening stance first (a posture
# from the statically-balanced crouch family, so the ramp itself is safe) and the
# clip is played only once the MEASURED joints are there.
#
# 2.5 s is generous: the ramp is 0.56 s of travel at the driver's 1.5 rad/s. If
# it has not arrived by then a joint is blocked or fighting something, and
# locomotion counts a failure rather than stalling forever.
CLIP_PREPARE_TIMEOUT_S = 2.5
# Ceiling on how long the legs may be held in a prepare ramp in TOTAL,
# across any number of changes of planned action. CLIP_PREPARE_TIMEOUT_S is
# per-action and _prepare_since restarts on every flip, so a planner that
# alternates forward/turn_left rewinds it forever: one measured episode
# ramped for 9.50 s, played nothing, and fell. Slightly above 2.5 s so a
# single honest retry on a second action is still allowed.
CLIP_PREPARE_RUN_TIMEOUT_S = 3.5

# --- Stopping a clip early -------------------------------------------------
# A clip used to be played to completion, on the sound reasoning that a clip
# BOUNDARY is a balanced double-support pose and therefore the only safe place to
# hand control back. The cost is that the clip's length becomes the latency of
# "stop walking" -- and that is what kept this controller on Cyberbotics' short
# 2.60 s walk, which is a single stride bracketed by a start and a stop transient
# and therefore both slow (0.036 m/s measured, against NAO's ~0.10) and visibly
# jerky when chained.
#
# The boundary is not actually special. Any keyframe in double support with both
# soles flat, the centre of mass inside the support polygon, nothing about to move
# fast AND the torso not still travelling has the same property, and a clip passes
# through many: 33 of Forwards.motion's 66 keyframes, 49 of TurnLeft40's 73 and
# 108 of TurnLeft180's 226 (see balance.safe_exit_times). The last of those gates
# is the one that matters and the one a static test misses -- freezing the legs
# does not freeze the robot, and 13 of the poses that pass the static tests are
# moving at up to 0.179 m/s, which throws the capture point 30 mm past the CoM
# against a 40-60 mm margin.
#
# For the TURN clips, which come to rest repeatedly, this is a good deal: the
# longest wait to a safe exit is 0.52 s. For a continuous WALK clip it is not --
# Forwards50.motion keeps only 38 of 170 keyframes and can make you wait 2.56 s,
# because a continuous walk is by design almost never standing still. The walk is
# served by GAIT_CYCLE below instead, and this remains its fallback.
CLIP_EXIT_TOLERANCE_S = 0.03    # how near a safe keyframe counts as being at one

# --- Cyclic gait -----------------------------------------------------------
# Play the walk clip as a GAIT GENERATOR rather than as a one-shot animation.
#
# Cyberbotics' clips are animations: squat, accelerate, stride, decelerate, stand.
# The stride is fine -- 0.051 m per half-step, peaking at 0.18 m/s -- but the
# transient around it is not: it is 49% of Forwards.motion's 2.60 s, and during
# the settle the torso actually travels 17 mm BACKWARD. Chaining that clip means
# paying the transient for every 0.095 m, which is why walking measured 0.036 m/s
# against NAO's documented ~0.10, and why it looks like stepping rather than
# walking.
#
# The middle of the long clip, though, is a true limit cycle, and exactly so:
# max|q(2.84 s) - q(1.80 s)| = 0.0000 rad over all 12 leg joints, holding to that
# precision from 1.80 s to 5.32 s. So the segment can be rewound with setTime()
# without commanding any joint motion at the seam, and one clip becomes a walk of
# unbounded length at 0.089 m/s by forward kinematics -- the speed the stride is
# actually worth, and 2.4x the 0.036 m/s measured today. (FK, not a measurement:
# it counts the ground the clip's own keyframes cover, so slip, servo lag and
# contact compliance are all unmodelled. Validate it on sv_com_x in a live log.)
# The
# transient is paid once per WALK instead of once per stride: for a 10 s walk that
# is 47 start/stop cycles reduced to 1.
#
# What makes this safe rather than clever is how it stops. Not by freezing
# mid-stride (see above), but by jumping once, at the single phase of the cycle
# where the jump is nearly free (0.0010 rad = 0.05 rad/s over one control step),
# into the clip's own deceleration -- so the robot is brought to rest by
# Cyberbotics' own balanced feet-together settle. Worst-case stop latency is one
# period plus that tail: 1.04 + 1.44 = 2.48 s, which is less than the 2.60 s the
# short clip commits to for one 0.095 m step.
#
# Set False to go back to one-shot playback of whichever clip is found; the
# detection is per-clip and refuses anything that is not periodic to a microradian
# and does not translate the robot, so of the clips Webots ships only the
# continuous walk qualifies (see walk_motion.gait_cycle).
GAIT_CYCLE = True

# --- Holding the walk intent ----------------------------------------------
# The cue flickers. Measured over the first session that ever walked
# (log 1788776922): gait_state "march" runs had a MEDIAN of 1.32 s -- shorter than
# one 2.60 s clip -- while the longest was 37.24 s, and 33 of the 65 idle runs
# were 1.2 s or shorter, totalling 18.2 s. The human was walking continuously;
# the cue was not. Each of those dropouts ended a walk and forced a fresh
# 0.6-1.0 s prepare ramp, which is the stutter.
#
# So the walk request is LATCHED: once the cue says the human is walking, the
# request survives this long after the cue drops. The cost is walking a little
# further than asked -- at 0.069 m/s a 1.2 s latch is 8 cm of overshoot -- and it
# is bounded, because the moment the latch expires the clip leaves the gait
# cycle (or, for a non-cyclic clip, exits at its next safe keyframe) rather than
# running to its end.
#
# Lowered 1.2 -> 0.6 on 2026-09-10. Two measurements from the 2026-09-08 session
# (238 s, log 1788865278) forced it:
#   * of the 34 gaps between marching bouts, only 13 were shorter than 1.2 s --
#     the real cue dropouts this latch exists to bridge -- and they totalled
#     7.26 s, a mean of 0.56 s each. The other 21 were genuine stops. So the
#     1.2 s setting was spending 21 x 1.2 = 25.2 s of trailing walk to buy
#     7.26 s of bridging, and that trailing walk IS the "robot keeps walking
#     after I stop" complaint.
#   * the dropouts themselves are now rarer: GaitCueExtractor no longer discards
#     its cadence evidence on a single low-confidence frame
#     (walk.cue_conf_grace_frames), which is what produced most of the
#     sub-second ones.
# 0.6 s still covers the mean dropout while halving the overshoot to 4 cm.
WALK_LATCH_RELEASE_S = 0.6

# --- Balance ---------------------------------------------------------------
# Model-based CoM feedback recovers the depth/balance information a 2D camera
# cannot give: forward kinematics + NAO link masses estimate the centre of mass
# each step, the InertialUnit supplies the gravity direction, and a
# Fibonacci-spiral search nudges ankles/hips to keep the CoM over the feet.
# Runs in a normal controller (no Supervisor). See main/libraries/balance.py.
ENABLE_BALANCE = True
INERTIAL_UNIT_NAME = "inertial unit"

# --- March engine ----------------------------------------------------------
# Tier A ("march") is a double-support weight-shift march: it never fully unloads
# a foot, so the symmetric balance loop stays valid and it cannot fall by design.
# Tier B ("step") is experimental single-support stepping. Real stepping is now
# better served by LEG_CONTROL="auto" (motion clips) or the pose-imitation
# sequencer, so leave this at "march".
WALK_TIER = "march"
GAIT_SMOOTHING_ALPHA = 0.7       # snappier than the arm EMA so gait/step survives
GAIT_LEG_VELOCITY_FACTOR = 0.85  # raised leg velocity while walking or stepping

# --- Locomotion (Webots .motion clips) -------------------------------------
# Turning is a closed loop on the InertialUnit heading, so it converges despite
# the clips being coarse: see walk_motion.YawServo / plan_action.
LOCOMOTION = LocomotionParams()
# Flip if the robot turns the wrong way for your camera setup. With the pipeline's
# default mirrored (selfie) preview the robot mirrors the on-screen figure, which
# is consistent with how the arms are mapped.
TURN_SIGN = 1.0
# Extra dirs searched (first) for NAO .motion files, in addition to $WEBOTS_HOME
# and the common install locations. A repo-local motions/ folder can go here.
MOTION_SEARCH_DIRS_EXTRA = [os.path.join(os.path.dirname(__file__), "motions")]

# --- Safety ----------------------------------------------------------------
# Emergency abort of a running motion clip. The gyro term is a lead compensator:
# a fall is visible in the tilt *rate* well before the tilt itself crosses a
# threshold, so predicting a quarter second ahead buys time to stop the clip and
# hand the body back to the balance loop.
TILT_ABORT_RAD = 0.40
TILT_RATE_LEAD_S = 0.25

# HARD WATCHDOG on motion playback. While a clip runs it owns the joints it
# declares -- for Webots' walk clips, the 12 leg joints -- and per-joint
# commanding is suspended for those, so anything that stops the clip from ever
# reporting "over" freezes the legs indefinitely. Webots' walk clips are a few
# seconds long, so any suspension beyond this is a bug, not a long clip: we take
# the joints back and stop trusting clips.
MOTION_WATCHDOG_S = 8.0
# Consecutive locomotion attempts that end badly (watchdog trip, tilt abort, or
# the robot still tipped when the clip finishes) before clips are abandoned for
# this session. Falling over repeatedly is worse than never walking.
MOTION_MAX_FAILURES = 3
# A clip is only started from a settled, upright robot: starting one mid-wobble
# is how a walk turns into a fall. This ceiling is not fixed: it shrinks
# continuously (see _tilt_risk / _settled) toward MOTION_START_MIN_TILT_RAD as
# recent tilt trends upward, so the controller keeps declining to start another
# clip for a while after a wobble instead of only reacting once mid-clip.
MOTION_START_MAX_TILT_RAD = 0.15
MOTION_START_MIN_TILT_RAD = 0.05
# EMA time constant for the continuous tilt-risk signal, updated every control
# step (independent of which leg-control layer is active) from the same
# predicted-tilt formula the hard abort below already trusts.
TILT_RISK_TAU_S = 1.5

# The control loop must survive a bad step. An exception used to end run(),
# which then called driver.stop() and set every motor velocity to zero -- a
# permanently dead robot from one transient error. Now each step is contained:
# we log, force the body back under control, and carry on.
MAX_CONSECUTIVE_ERRORS = 20

# --- Sensors ---------------------------------------------------------------
GYRO_NAME = "gyro"
ACCELEROMETER_NAME = "accelerometer"
# Webots' Nao.proto exposes one 3-axis force sensor per foot ("LFsr"/"RFsr");
# the other names are fallbacks for older/other NAO protos. Any that resolve are
# summed per foot; the rest are ignored and the CoM model gates stepping alone.
FSR_DEVICES = {
    "L": ["LFsr", "LFootFSR", "LFoot/Fsr", "force_sensor_left"],
    "R": ["RFsr", "RFootFSR", "RFoot/Fsr", "force_sensor_right"],
}

# --- Logging ---------------------------------------------------------------
# Commanded vs. achieved joint angles to <project>/logs/ for offline
# imitation-fidelity metrics (PRD FR-7 / US-3).
ENABLE_TRAJECTORY_LOG = True
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs"))
STATUS_EVERY = 100  # frames

# Per-step controller state written alongside the joint angles. A log of joint
# angles alone says what the body did but not what the controller believed, and
# every diagnosis on this project has needed both halves: the permanent lean was
# only identifiable because the roll channels could be compared, and its CAUSE
# needed the support margin, which was not recorded at all.
DIAGNOSTIC_COLUMNS = (
    "imu_roll", "imu_pitch", "imu_yaw",
    "imu_roll_raw", "imu_pitch_raw", "imu_zero_roll", "imu_zero_pitch",
    # The gravity-derived attitude and which source the controller acted on. Both
    # are logged so the InertialUnit's half-scale pitch stays visible instead of
    # being silently replaced (see TILT_FROM_ACCELEROMETER).
    "acc_roll", "acc_pitch", "tilt_source",
    # The simulator's own answers, logged for comparison with the model that
    # actually does the controlling (see _ground_truth): the true whole-body
    # centre of mass, whether the real CoM projects inside the convex hull of the
    # real contact points, and how many contact points there are.
    "sv_com_x", "sv_com_y", "sv_com_z", "sv_balanced", "sv_contacts", "sv_heading",
    # ...and the tilt the controller ACTUALLY acted on this step, which is not the
    # same as either channel: the InertialUnit's pitch is doubled on the way in
    # (IMU_PITCH_SCALE), and the source can change mid-run. Logging only the
    # sensor channels made a session ambiguous to read after the fact.
    "ctl_roll", "ctl_pitch",
    "gyro_roll_rate", "gyro_pitch_rate",
    "tilt_risk", "predicted_tilt",
    "leg_mode", "stale",
    "lb_mode", "lb_shift", "lb_lift", "lb_gate", "lb_stance_margin",
    "lb_lean_scale", "lb_crouch_u", "lb_crouch_cue", "lb_conf",
    "lb_lift_source", "lb_rejected", "lb_why",
    # Where every radian of pelvis shift came from: the lower body's feed-forward
    # term (ff), the balance loop's feedback term (fb), and the static margin the
    # sum achieved. Without these the 2026-09-03 falls could only be attributed by
    # arithmetic on the joint columns (both shifters at their clamps read as
    # HipPitch +0.45 / AnklePitch -0.65), which took a day to notice.
    "lb_ff_pitch", "lb_ff_roll", "lb_ff_width", "lb_fb_pitch", "lb_fb_roll",
    "lb_com_margin",
    # Lateral centre of pressure from the foot sensors, as the left foot's share
    # of the total load: the one direct measurement of where the weight really is.
    "cop_share_l",
    "support_margin_x", "support_margin_y", "head_height", "reloads",
    "clip_planned", "clip_status", "clips_available", "clip_time", "walk_latched",
    "early_exits", "tails_trimmed", "yaw_stable",
    # Cyclic gait (GAIT_CYCLE): how many strides this walk has repeated without
    # restarting the clip, the phase within the cycle, and what cycle_tick did.
    # A walk that is working shows clip_cycles climbing while clip_phase saws
    # between 0 and the period -- if clip_cycles stays 0 while the robot walks,
    # the loop is not engaging and every stride is paying the transient again.
    "clip_cycles", "clip_phase", "cycle_state",
    "yaw_error", "yaw_latched",
    "fsr_l", "fsr_r",
    "gait_state", "gait_cadence", "gait_conf", "body_yaw", "gait_cue_channel",
    # What the ACTION classifier said the human was doing, and the measurements
    # behind it. Logged next to clip_planned/clip_status so one row answers the
    # whole question: the human did X, we decided Y, we played Z, this long
    # after. Without these the decision can only be inferred from its
    # consequences, which is how the old cue's 18.8% of false march went
    # unnoticed for four sessions.
    "act_action", "act_conf", "act_forward_mps", "act_lateral_mps",
    "act_yaw_rate", "act_crouch", "act_lift", "act_observed", "act_reason",
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PoseController")


# ===========================================================================
# Webots Motion playback
# ===========================================================================
class MotionPlayer:
    """Plays Webots' pre-balanced NAO ``.motion`` clips, one at a time.

    Why clips at all: an online gait good enough to translate a free-standing
    NAO across the floor is a research project in itself, while Cyberbotics ship
    walk/turn clips that are already balanced for this exact robot. Playing them
    is what makes the robot *genuinely* move -- its world coordinates change --
    rather than marching on the spot.

    Playback comes in two flavours.

    **One-shot** is the conservative one and still the default: play the clip from
    end to end, because a clip boundary is a balanced double-support pose. A clip
    may also be released early, but only at a keyframe that is both statically
    holdable and not carrying momentum (``balance.safe_exit_times``), because
    stopping the legs does not stop the robot.

    **Cyclic** applies to a clip with a genuine limit cycle in it
    (``walk_motion.gait_cycle`` -- of the clips Webots ships, only the continuous
    walk). Playback starts past the opening squat, rewinds the periodic window
    with ``setTime`` for as long as the walk is wanted, and leaves through the
    clip's own deceleration. That turns one 6.76 s clip into a gait generator: the
    robot walks continuously at the speed its stride is worth instead of paying
    the start/stop transient for every 0.095 m.

    :meth:`abort` breaks either of them for the one case that justifies it: an
    incipient fall.
    """

    def __init__(self, files: dict[str, str],
                 log: object | None = None) -> None:
        self._files = dict(files)
        self._cache: dict[str, object] = {}
        self._joints: dict[str, list[str]] = {}
        self._first_pose: dict[str, dict[str, float]] = {}
        self._safe_exits: dict[str, list[float]] = {}
        self._cycles: dict[str, object] = {}
        self._turns: dict[str, object] = {}
        # Cleared whenever a clip is dropped; see the `interruptible` property.
        self._interruptible: frozenset[str] | None = None
        self._entry_pose: dict[str, dict[str, float]] = {}
        self._log = log or (lambda *_a, **_k: None)
        self.action: str | None = None
        self._motion: object | None = None
        # Cyclic playback state, all reset by _clear().
        self._cycle: object | None = None
        self._leaving = False       # the stop jump has been taken
        self._previous_time: float | None = None
        self.cycles_done = 0

    @property
    def available(self) -> dict[str, str]:
        return dict(self._files)

    @property
    def active(self) -> bool:
        return self._motion is not None

    def _load(self, action: str) -> object | None:
        """The Motion object for ``action``, or None (and the clip is dropped).

        Validity is judged by PARSING THE FILE, not by ``Motion.isValid()``. That
        method cannot answer the question: the R2025a Python binding implements it
        as ``self._ref != ctypes.c_void_p(0)``, comparing two freshly constructed
        ctypes pointers, which are never equal -- so it returns True even for a
        file Webots failed to load. The clip's own header is the honest test, and
        the NULL check below catches a load Webots refused.
        """
        if action in self._cache:
            return self._cache[action]
        path = self._files.get(action)
        if path is None:
            return None
        if not motion_joints(path):
            self._log("Motion '%s' has no readable joint header (%s); dropping it",
                      action, path)
            self._files.pop(action, None)
            return None
        try:
            motion = Motion(path)
        except Exception as exc:  # noqa: BLE001
            self._log("Motion '%s' unusable (%s); dropping it", action, exc)
            self._files.pop(action, None)
            return None
        # wbu_motion_new returns NULL when it cannot load the file, which the
        # binding stores as a c_void_p whose .value is None.
        ref = getattr(motion, "_ref", None)
        if ref is not None and getattr(ref, "value", 1) in (None, 0):
            self._log("Motion '%s' was rejected by Webots' loader; dropping it",
                      action)
            self._files.pop(action, None)
            return None
        self._cache[action] = motion
        return motion

    def start(self, action: str, at_s: float = 0.0) -> bool:
        """Begin ``action``; returns False if the clip could not be started.

        THE ONE LINE THAT KEPT THIS ROBOT FROM EVER WALKING. Webots' R2025a
        Python binding is::

            def play(self):
                wb.wbu_motion_play(self._ref)      # no return statement

        so ``play()`` evaluates to None, and the old ``if not motion.play():
        return False`` therefore returned False on every call ever made. The
        locomotion layer never once became active -- ``leg_mode`` is "pose" in
        100.0% of the frames of every recorded session -- and the controller
        logged it as "start REFUSED by Webots", which read like a Webots problem
        rather than an inverted truth test.

        Worse than not walking: ``wbu_motion_play`` HAD started the clip, and the
        Webots controller library applies a playing clip's keyframes on every
        step. So the clip drove the 12 leg joints while this class believed
        nothing was playing, per-joint commanding was never suspended, and the two
        fought -- and because the arbiter retries a planned clip every step, the
        clip was stopped, rewound and replayed every 20 ms, pinning it to its
        first keyframe. Measured in log 1788428293: 964 frames of "REFUSED" with
        LKneePitch MEASURED at the clip's own first keyframe (1.042 rad) while the
        controller commanded 0.20-0.52.

        So playback is confirmed by asking the clip whether it is running, which
        is a question the API does answer.

        ``at_s`` starts playback part-way in, which is how a cyclic clip skips its
        opening squat. The caller is responsible for having put the legs in the
        pose at that offset first (``walk_motion.motion_pose_at``).
        """
        motion = self._load(action)
        if motion is None:
            return False
        cycle = self.cycle(action)
        at_s = max(0.0, float(at_s))
        try:
            motion.setLoop(False)
            # A clip that already ran must be rewound, or play() resumes at its
            # end and returns immediately. stop() + setTime() covers both the
            # "interrupted" and the "finished" case.
            motion.stop()
            try:
                motion.setTime(int(round(at_s * 1000.0)))
            except Exception:  # noqa: BLE001 - older API without setTime
                if at_s > 0.0:
                    self._log("Motion '%s': no setTime(), so it cannot be entered "
                              "at %.2fs or cycled; playing it one-shot.",
                              action, at_s)
                    cycle = None
            motion.play()               # returns None -- see the docstring
            if bool(motion.isOver()):   # ...so THIS is the test that works
                self._log("Motion '%s' reported itself finished the moment it was "
                          "played; dropping it", action)
                self._files.pop(action, None)
                self._cache.pop(action, None)
                return False
        except Exception as exc:  # noqa: BLE001
            self._log("Could not play motion '%s': %s", action, exc)
            return False
        self.action = action
        self._motion = motion
        self._cycle = cycle
        self._leaving = False
        self._previous_time = None
        self.cycles_done = 0
        return True

    def time_s(self) -> float | None:
        """How far into the clip playback is, in seconds, or None."""
        if self._motion is None:
            return None
        try:
            ms = float(self._motion.getTime())
        except Exception:  # noqa: BLE001
            return None
        return ms / 1000.0 if math.isfinite(ms) else None

    def safe_exits(self, action: str | None = None) -> list[float]:
        """Times (s) at which this clip may be stopped safely; [] if unknown.

        Computed once per clip from its own keyframes (see
        ``balance.safe_exit_times``). Needs the CoM model, so without NumPy the
        list is empty and the caller plays the clip to completion, which is the
        behaviour this replaces.
        """
        target = action or self.action
        if target is None:
            return []
        if target not in self._safe_exits:
            times: list[float] = []
            path = self._files.get(target)
            if path is not None:
                try:
                    from balance import safe_exit_times

                    times = safe_exit_times(motion_poses(path))
                except Exception as exc:  # noqa: BLE001
                    self._log("No early-exit points for '%s' (%s); it will be "
                              "played to completion.", target, exc)
                    times = []
            self._safe_exits[target] = times
        return self._safe_exits[target]

    def at_safe_exit(self, tolerance: float = 0.03) -> bool:
        """Is playback at (or just past) a keyframe it can be stopped at?"""
        now = self.time_s()
        exits = self.safe_exits()
        if now is None or not exits:
            return False
        return any(abs(now - t) <= tolerance for t in exits)

    @property
    def interruptible(self) -> frozenset[str]:
        """Actions whose clips can be stopped part-way, at a point WE choose.

        Handed to :func:`walk_motion.plan_action`, which plans turning very
        differently for these (see ``_pick_turn``). Derived from the clips
        themselves rather than declared, so a Webots release whose clip does not
        come to rest, or a machine with no NumPy and therefore no CoM model, is
        *detected* as un-stoppable and falls back to whole-clip playback instead
        of being stopped somewhere nobody certified.

        A TURN clip has to clear a higher bar than the rest: a schedule, not just
        somewhere safe to stop. The difference is the whole safety argument.
        Dropping the overshoot floor for a turn is only sound because the clip is
        then AIMED -- stopped at the rung that best serves the heading error. A
        clip with safe keyframes but no measurable rotation gives the caller
        nowhere to aim, so it would get the loosened gate and none of the
        aiming, and the loop would hunt below the finest error the clip can serve
        (a 60 deg clip asked for 17 deg turns to -43 deg, and asks again).
        """
        if self._interruptible is None:
            self._interruptible = frozenset(
                action for action in self._files
                if (self.turn(action) is not None
                    if action in TURN_ACTION_NAMES
                    else bool(self.safe_exits(action)))
            )
        return self._interruptible

    def turn(self, action: str | None = None):
        """The :class:`walk_motion.TurnSchedule` for ``action``, or None.

        Cached per clip: the schedule costs a forward-kinematics pass over every
        keyframe, which is far too slow to repeat per control step. The safe-exit
        list it needs is the one already cached for this clip, so asking for the
        schedule does not recompute it.
        """
        target = action or self.action
        if target is None:
            return None
        if target not in self._turns:
            found = None
            path = self._files.get(target)
            if path is not None:
                try:
                    found = turn_schedule(path, self.safe_exits(target))
                except Exception as exc:  # noqa: BLE001
                    self._log("No turn schedule for '%s' (%s); it will be played "
                              "whole.", target, exc)
                    found = None
                if found is not None:
                    self._log(
                        "Clip '%s' turns %+.0f deg in %.2fs: enter at %.2fs "
                        "(skipping its opening crouch), then %d certified places "
                        "to stop, %.1f deg/s sustained, worst gap between "
                        "deliverable angles %.0f deg.",
                        target, math.degrees(found.total_rad), found.duration_s,
                        found.entry_s, len(found.rungs),
                        math.degrees(found.rate_rad_s),
                        math.degrees(found.quantum_rad))
            self._turns[target] = found
        return self._turns[target]

    def cycle(self, action: str | None = None):
        """The :class:`walk_motion.GaitCycle` for ``action``, or None.

        Cached per clip -- the detection walks every keyframe through forward
        kinematics, which is far too slow to repeat per control step.
        """
        target = action or self.action
        if target is None:
            return None
        if target not in self._cycles:
            found = None
            path = self._files.get(target)
            if path is not None and GAIT_CYCLE:
                try:
                    found = gait_cycle(path)
                except Exception as exc:  # noqa: BLE001
                    self._log("No gait cycle for '%s' (%s); it will be played "
                              "one-shot.", target, exc)
                    found = None
                if found is not None:
                    self._log(
                        "Clip '%s' is cyclic: enter at %.2fs, loop [%.2f, %.2f]s "
                        "(%.2fs, %+.0f mm => %.3f m/s), leave %.2f->%.2fs at a "
                        "cost of %.4f rad. Stop latency at most %.2fs.",
                        target, found.enter_s, found.loop_start_s, found.loop_end_s,
                        found.period_s, found.advance_m * 1000.0, found.speed_mps,
                        found.exit_from_s, found.exit_to_s, found.exit_cost_rad,
                        found.stop_latency_s)
            self._cycles[target] = found
        return self._cycles[target]

    @property
    def cyclic(self) -> bool:
        """Is playback of this clip being run to a cycle schedule?

        True from the moment a cyclic clip starts until it ends -- INCLUDING
        while it rides its own deceleration out after taking the exit jump. That
        tail is the schedule's last act and must be left alone: the whole reason
        for jumping into it is that Cyberbotics' balanced feet-together settle is
        what brings the robot to rest. Releasing the body part-way through it,
        which the early-exit path would happily do, throws away the deceleration
        that was just bought and hands back a robot mid-settle.
        """
        return self._motion is not None and self._cycle is not None

    @property
    def cycling(self) -> bool:
        """Is a cyclic clip playing, and still free to repeat?"""
        return self.cyclic and not self._leaving

    @property
    def leaving(self) -> bool:
        """Has the exit jump been taken (so the tail is playing out)?"""
        return self.cyclic and self._leaving

    def entry_pose(self, action: str | None = None) -> dict[str, float]:
        """The pose to ramp the legs to before playing ``action``.

        The clip's first keyframe normally, but the pose at the entry time for a
        clip whose opening crouch is skipped -- see :meth:`entry_time_s`.
        """
        target = action or self.action
        if target is None:
            return {}
        entry = self.entry_time_s(target)
        if entry <= 0.0:
            return self.first_pose(target)
        if target not in self._entry_pose:
            self._entry_pose[target] = motion_pose_at(
                self._files.get(target), entry)
        return dict(self._entry_pose[target])

    def entry_time_s(self, action: str | None = None) -> float:
        """Where playback of ``action`` should start, in seconds.

        Every one of Cyberbotics' locomotion clips opens with the same thing: a
        slow squat from the standing pose into the deep, sole-flat crouch it
        walks or turns in. That is a full second and more of clip during which
        the robot goes nowhere -- and the controller's own prepare-ramp has to
        put the legs in that crouch anyway before handing the clip over, because
        playback commands its first keyframe on its very first step with the
        velocity caps already lifted (see ``approach_leg_pose``).

        So the crouch is done once, by the ramp, rate-limited and under balance
        supervision, and playback starts after it. Measured: 1.38 s skipped on
        Forwards50.motion and 1.20 s on the turn clips, off the front of every
        walk and every turn.
        """
        target = action or self.action
        if target is None:
            return 0.0
        cycle = self.cycle(target)
        if cycle is not None:
            return float(cycle.enter_s)
        turn = self.turn(target)
        return 0.0 if turn is None else float(turn.entry_s)

    def cycle_tick(self, hold: bool) -> str:
        """Advance cyclic playback one control step. Returns a status word.

        Called on every step a cyclic clip is playing. ``hold`` is whether the
        walk is still wanted.

        The stride repeats either way -- the difference between holding and not
        is only whether the walk is looking for its exit. Waiting for the exit
        phase is done by walking, not by standing still mid-stride.

        * Holding, and the playhead has reached the end of the periodic window:
          rewind by exactly one period. The joints at the two ends of that window
          are identical to a microradian (``gait_cycle`` accepts nothing looser),
          so the seam commands no motion and the stride simply repeats. The
          overshoot within the step is carried across, which keeps the phase
          continuous instead of quantising it to the control period.
        * Not holding: wait until the playhead CROSSES the one phase of the cycle
          from which the clip's own deceleration is reachable for free, then jump
          there once. From that moment playback is ordinary one-shot: the tail
          plays out and :meth:`poll` reports it over, with the robot standing
          still and feet together, exactly as if the clip had been played end to
          end.

        Crossing, not "at or past". The cheap jump is cheap at ONE phase of the
        stride -- that is the whole basis for it being safe -- and the playhead
        spends the rest of the period past that phase, so a "past it" test fires
        at whatever moment the human happened to stop and lands wherever the
        joints happened to be. Measured on the test gait: the free exit costs
        0.0000 rad taken on the phase and up to 0.4 rad taken off it, which is a
        20 rad/s lurch. Crossings cannot be missed either, however the playhead
        is stepped, which a tolerance window can.
        """
        if self._motion is None or self._cycle is None:
            return "not cycling"
        now = self.time_s()
        if now is None:
            return "no clock"
        cycle = self._cycle
        previous, self._previous_time = self._previous_time, now
        if self._leaving:
            return "leaving"
        if not hold and self._crossed(cycle.exit_from_s, previous, now, cycle):
            if self._set_time(cycle.exit_to_s):
                self._leaving = True
                self._log("Leaving the gait cycle after %d stride(s): "
                          "%.2fs -> %.2fs, then %.2fs of the clip's own "
                          "deceleration.", self.cycles_done,
                          now, cycle.exit_to_s, cycle.tail_s)
                return "leaving"
            # setTime failed, so this clip cannot be cycled at all; _set_time has
            # dropped the cycle and playback carries on as an ordinary one-shot.
        # Keep striding -- INCLUDING while waiting to leave. A robot that stopped
        # rewinding the moment it was asked to stop would run the playhead off the
        # end of the loop window into whatever the clip does next, which is not
        # the cycle and not the deceleration either. It would also never come back
        # round to the exit phase, so it would never leave; the watchdog would
        # eventually take the body back mid-stride and count a failure.
        if now + 1e-9 >= cycle.loop_end_s:
            if self._set_time(now - cycle.period_s):
                self.cycles_done += 1
                return "rewound" if hold else "stopping"
        return "cycling" if hold else "stopping"

    @staticmethod
    def _crossed(phase: float, previous: float | None, now: float, cycle) -> bool:
        """Did the playhead pass ``phase`` between ``previous`` and ``now``?

        The interval is normally [previous, now], but a rewind happened if the
        playhead went BACKWARD, and then the ground covered is [previous,
        loop_end) followed by [loop_start, now] -- so both pieces are tested.
        """
        if previous is None:
            return now + 1e-9 >= phase
        if now + 1e-9 >= previous:
            return previous < phase + 1e-9 <= now + 1e-9
        return phase + 1e-9 > previous or phase - 1e-9 <= now

    def _set_time(self, seconds: float) -> bool:
        if self._motion is None:
            return False
        try:
            self._motion.setTime(int(round(max(0.0, seconds) * 1000.0)))
        except Exception as exc:  # noqa: BLE001
            self._log("setTime(%.2fs) failed on '%s' (%s); this clip cannot be "
                      "cycled.", seconds, self.action, exc)
            self._cycle = None
            return False
        return True

    def first_pose(self, action: str | None = None) -> dict[str, float]:
        """The joint angles the clip for ``action`` opens on (see
        ``walk_motion.motion_first_pose``). Cached; {} if unknowable."""
        target = action or self.action
        if target is None:
            return {}
        if target not in self._first_pose:
            self._first_pose[target] = motion_first_pose(self._files.get(target))
        return dict(self._first_pose[target])

    def poll(self) -> bool:
        """True while the clip is still running; clears itself when it is over."""
        if self._motion is None:
            return False
        try:
            over = bool(self._motion.isOver())
        except Exception:  # noqa: BLE001
            over = True
        if over:
            self._clear()
            return False
        return True

    def duration_s(self) -> float | None:
        """Clip length in seconds, or None if Webots will not tell us."""
        if self._motion is None:
            return None
        try:
            ms = float(self._motion.getDuration())
        except Exception:  # noqa: BLE001
            return None
        return ms / 1000.0 if math.isfinite(ms) and ms > 0.0 else None

    def joints(self, action: str | None = None) -> list[str]:
        """Joints the clip for ``action`` drives, or [] if that is not knowable.

        Read from the clip's header (see ``walk_motion.motion_joints``) and cached,
        so the controller can suspend per-joint commanding for exactly those and
        leave the rest -- the arms and head -- imitating throughout the clip.
        """
        target = action or self.action
        if target is None:
            return []
        if target not in self._joints:
            self._joints[target] = motion_joints(self._files.get(target))
        return list(self._joints[target])

    def drop(self, action: str | None = None) -> None:
        """Stop using ``action`` (or every clip) for the rest of the session."""
        target = action or self.action
        self.abort()
        if target is None:
            self._files.clear()
            self._cache.clear()
            self._joints.clear()
            self._first_pose.clear()
            self._safe_exits.clear()
            self._cycles.clear()
            self._turns.clear()
            self._entry_pose.clear()
        else:
            self._files.pop(target, None)
            self._cache.pop(target, None)
            self._joints.pop(target, None)
            self._first_pose.pop(target, None)
            self._safe_exits.pop(target, None)
            self._cycles.pop(target, None)
            self._turns.pop(target, None)
            self._entry_pose.pop(target, None)
        self._interruptible = None

    def abort(self) -> None:
        """Stop mid-clip. Only for a safety abort -- see the class docstring."""
        if self._motion is None:
            return
        try:
            self._motion.stop()
        except Exception:  # noqa: BLE001
            pass
        self._clear()

    def _clear(self) -> None:
        self.action = None
        self._motion = None
        self._cycle = None
        self._leaving = False
        self._previous_time = None


# ===========================================================================
# Controller
# ===========================================================================
class PoseImitationController:
    """Webots glue and lower-body arbiter around :class:`NaoPoseDriver`."""

    def __init__(self) -> None:
        # A Supervisor when the world allows it (the Nao node needs
        # `supervisor TRUE`), so a fall can be recovered from automatically. It
        # behaves as a plain Robot for everything else, and falls back to one when
        # the class is unavailable.
        self.robot = Supervisor() if (AUTO_RELOAD_ON_FALL and Supervisor is not None) \
            else Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        logger.info("Initializing NAO pose controller (timestep: %dms)", self.timestep)
        if self.timestep > 24:
            logger.warning(
                "WorldInfo.basicTimeStep is %d ms. NAO leg control and the walk "
                "clips need <= 20 ms; expect poor balance.", self.timestep
            )

        self.leg_control = LEG_CONTROL if LEG_CONTROL in (
            "auto", "pose", "engine", "off"
        ) else "auto"
        drive_legs = self.leg_control in ("auto", "pose", "engine")
        enable_walk = self.leg_control in ("auto", "engine")

        self.driver = NaoPoseDriver(
            self.robot,
            drive_legs=drive_legs,
            drive_head=DRIVE_HEAD,
            swap_sides=SWAP_SIDES,
            smoothing_alpha=SMOOTHING_ALPHA,
            velocity_scale=VELOCITY_SCALE,
            arm_tau_s=ARM_TAU_S,
            arm_lead_s=ARM_LEAD_S,
            arm_velocity_factor=ARM_VELOCITY_FACTOR,
            leg_velocity_factor=LEG_VELOCITY_FACTOR,
            stale_after_s=STALE_AFTER_S,
            enable_balance=ENABLE_BALANCE,
            enable_walk=enable_walk,
            walk_tier=WALK_TIER,
            gait_smoothing_alpha=GAIT_SMOOTHING_ALPHA,
            gait_leg_velocity_factor=GAIT_LEG_VELOCITY_FACTOR,
            logger=logger.info,
        )
        self._init_imu()
        self._init_supervisor_sense()
        self._init_walk_sensors()
        self._init_locomotion()
        self._init_socket()

        self.trajectory_log = None
        if ENABLE_TRAJECTORY_LOG:
            self.trajectory_log = JointTrajectoryLogger(
                LOG_DIR, self.driver.logged_joints,
                diagnostics=DIAGNOSTIC_COLUMNS, logger=logger.info,
            )

        self.gait_cmd: dict | None = None
        self.action_cmd: dict | None = None
        self.leg_mode = "stand"
        self.frame_count = 0
        self._last_log_time = time.time()
        # True while a multi-clip rotation is still converging. It survives clip
        # boundaries on purpose: that is what lets plan_action use its tighter
        # gate to finish a turn instead of stalling one clip short.
        self._turning = False
        # Motion-playback watchdog state. Suspension hands the WHOLE body to a
        # clip, so it must always be bounded in time and in failure count.
        self._motion_started_at: float | None = None
        self._motion_deadline: float | None = None
        self._motion_failures = 0
        self._errors = 0
        # Continuous tilt-risk EMA (see _update_tilt_risk / _settled): runs every
        # tick regardless of which leg-control layer is active.
        self._tilt_risk = 0.0
        self._last_risk_update: float | None = None
        # Learned IMU tilt zero (see IMU_AUTO_ZERO). None until calibrated; tilt is
        # reported as level in the meantime so nothing aborts on a reading we do
        # not yet understand.
        self._imu_zero: tuple | None = None
        self._imu_cal: list[tuple] = []
        self._imu_cal_started: float | None = None
        if not IMU_AUTO_ZERO:
            self._imu_zero = (0.0, 0.0)
        # Gravity-derived attitude (see TILT_FROM_ACCELEROMETER): the low-passed
        # (roll, pitch), its learned zero, which source the last step used, and
        # since when it has disagreed with the InertialUnit's roll.
        self._acc_tilt: tuple | None = None
        self._acc_zero: tuple | None = None
        self._acc_usable = TILT_FROM_ACCELEROMETER
        self._acc_disagree_since: float | None = None
        self._acc_last_update: float | None = None
        self._tilt_source = "imu"
        # Fall detection / recovery state.
        self._fall_since: float | None = None
        self._stuck_since: float | None = None
        self._unloaded_since: float | None = None
        self._reloads = 0
        self._last_reload: float | None = None
        self._head_height: float | None = None
        # What the locomotion layer wanted and what became of it. Recorded because
        # "the robot never walks" has several indistinguishable causes -- no clips
        # on disk, a clip Webots refuses to load, the settle gate never opening, or
        # the turn servo starving forward motion -- and none of them are visible
        # from the joint angles.
        self._clip_planned = ""
        self._clip_status = "idle"
        # Ramp-to-clip-stance state (see CLIP_PREPARE_TIMEOUT_S).
        self._preparing: str | None = None
        self._prepare_since: float | None = None
        # Total time spent continuously ramping toward SOME clip stance,
        # immune to the planner changing its mind (see _prepare_for).
        self._prepare_run_since: float | None = None
        # Latched walk request (see WALK_LATCH_RELEASE_S): the last gait command
        # that asked for locomotion, and when the latch on it expires.
        self._walk_gait: dict | None = None
        self._walk_latch_until: float | None = None
        # How many clips were stopped early at a safe keyframe rather than played
        # to their end. Counted because "the robot stops promptly now" is exactly
        # the kind of claim that should be measurable after the fact.
        self._early_exits = 0
        # Clips whose closing stand-up was skipped because the settle had already
        # finished (see GaitCycle.rest_s). Counted separately from _early_exits:
        # an early exit cuts a clip short of what it was doing, whereas this one
        # let it finish and only declined to wait for the part that does nothing.
        # Conflating them would make "the walk was interrupted" unreadable.
        self._tails_trimmed = 0
        self._cycles_walked = 0
        self._cycle_state = ""
        self._report_startup()

    def _report_startup(self) -> None:
        """Print one block saying exactly what will and will not work.

        This exists because every "the robot does not move" report so far has had
        a cause that was visible at startup -- a missing device, a missing NumPy,
        no motion clips, a coarse timestep -- but was buried in the log. Saying
        it plainly up front turns a debugging session into a glance.
        """
        d = self.driver
        n_legs = sum(1 for n in d.motors if n.endswith(
            ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch", "AnklePitch", "AnkleRoll")))
        logger.info("=" * 68)
        logger.info("NAO pose imitation controller ready")
        logger.info("  timestep          : %d ms%s", self.timestep,
                    "" if self.timestep <= 20 else "   <-- TOO COARSE, use 20 ms")
        logger.info("  motors / sensors  : %d / %d  (%d leg joints)",
                    len(d.motors), len(d.sensors), n_legs)
        logger.info("  leg control       : %s", self.leg_control)
        logger.info("  arms + head       : ON")
        logger.info("  leg pose imitation: %s",
                    "ON (squat, single-leg lift)" if d.lower_body is not None
                    else "OFF  <-- legs will only hold a posture")
        logger.info("  CoM balance       : %s",
                    "ON" if d.balance is not None
                    else "OFF  <-- needs NumPy in Webots' Python")
        logger.info("  march engine      : %s",
                    "ON" if d.gait_engine is not None else "OFF")
        if self.leg_control == "auto":
            clips = sorted(self.motion.available)
            logger.info("  locomotion clips  : %s",
                        ", ".join(clips) if clips
                        else "NONE FOUND  <-- will march in place, not walk")
        logger.info("  torso attitude    : %s",
                    "SUPERVISOR node orientation (the truth; both sensor "
                    "channels are provably wrong on this proto -- see "
                    "ATTITUDE_SOURCE)"
                    if (ATTITUDE_SOURCE == "supervisor" and self.self_node is not None)
                    else "GRAVITY (accelerometer)"
                    if (TILT_FROM_ACCELEROMETER and self.accel is not None)
                    else f"InertialUnit, pitch x{IMU_PITCH_SCALE:.0f} (it reports "
                         f"half-scale on this proto)  <-- DEGRADED")
        logger.info("  heading feedback  : %s",
                    f"ON ({HEADING_SOURCE})" if self.heading_available
                    else f"OFF (source={HEADING_SOURCE})  <-- no heading, so "
                         f"turning is disabled. This proto's IMU and gyro yaw "
                         f"axes are switched off; 'supervisor' needs "
                         f"`supervisor TRUE` on the Nao node.")
        logger.info("  foot force sensors: %d",
                    len(self.fsr["L"]) + len(self.fsr["R"]))
        for reason in d.degraded:
            logger.error("DEGRADED: %s", reason)
        if d.degraded:
            logger.error(
                "The layer(s) above are NOT running. The usual cause is that "
                "Webots is launching this controller with an interpreter that "
                "has no NumPy -- check Tools > Preferences > Python command and "
                "point it at the project's environment. This message is repeated "
                "on the status line so it cannot scroll away."
            )
        logger.info("Waiting for pose commands on %s:%d ...", UDP_HOST, UDP_PORT)
        logger.info("=" * 68)

    # ---------------------------------------------------------------- devices
    def _init_imu(self) -> None:
        """Enable the InertialUnit: gravity direction for balance AND the robot's
        true heading, which the turn servo closes its loop on."""
        self.imu = None
        imu = self.robot.getDevice(INERTIAL_UNIT_NAME)
        if imu is None:
            logger.warning(
                "InertialUnit '%s' not found; balance runs CoM-only and turning "
                "is disabled (no heading feedback).", INERTIAL_UNIT_NAME
            )
            return
        try:
            imu.enable(self.timestep)
            self.imu = imu
            logger.info("InertialUnit enabled (balance + heading feedback)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable InertialUnit: %s", exc)

    def _init_supervisor_sense(self) -> None:
        """Grab our own scene-tree node, for the heading (see HEADING_SOURCE).

        Also the gateway to real ground truth -- getCenterOfMass() aggregates the
        descendant solids, getContactPoints(True) gives the actual sole/floor
        contacts and getStaticBalance() tests the real centre of mass against the
        convex hull of them -- which is logged for comparison with the model but
        deliberately NOT used for control; see _ground_truth.
        """
        self.self_node = None
        if HEADING_SOURCE != "supervisor" and ATTITUDE_SOURCE != "supervisor":
            return
        getter = getattr(self.robot, "getSelf", None)
        if getter is None:
            logger.warning(
                "HEADING_SOURCE is 'supervisor' but this controller is not a "
                "Supervisor, so there is no heading: turning will be disabled. "
                "Add `supervisor TRUE` to the Nao node in the world file."
            )
            return
        try:
            node = getter()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supervisor.getSelf() failed (%s); turning disabled", exc)
            return
        if node is None:
            logger.warning("Supervisor.getSelf() returned nothing; turning disabled")
            return
        self.self_node = node
        logger.info("Supervisor node acquired: attitude=%s, heading=%s",
                    ATTITUDE_SOURCE, HEADING_SOURCE)

    @property
    def heading_available(self) -> bool:
        """Is there a heading worth closing a loop on?"""
        if HEADING_SOURCE == "supervisor":
            return self.self_node is not None
        return HEADING_SOURCE == "imu" and HEADING_FROM_IMU

    def _heading(self, imu_yaw: float) -> float:
        """The robot's heading in the world frame, rad, CCW-positive (= its left).

        From the Supervisor's orientation matrix where available -- the robot's
        forward axis is +x in its own frame, so its world direction is the first
        COLUMN of the row-major 3x3, (m[0], m[3], m[6]). Falls back to the
        InertialUnit's yaw only if HEADING_SOURCE says to, which it should not:
        that channel is a copy of the half-scale pitch on this proto.
        """
        if HEADING_SOURCE == "supervisor" and self.self_node is not None:
            try:
                m = self.self_node.getOrientation()
            except Exception:  # noqa: BLE001
                return 0.0
            if m is not None and len(m) >= 9:
                try:
                    return math.atan2(float(m[3]), float(m[0]))
                except (TypeError, ValueError):
                    return 0.0
            return 0.0
        return imu_yaw if HEADING_FROM_IMU else 0.0

    def _ground_truth(self) -> dict[str, object]:
        """The simulator's own answers, for the log only -- never for control.

        The balance loop stays model-based (forward kinematics + link masses, see
        balance.py) so that it remains an algorithm a real NAO could run. But the
        simulator can be ASKED, and logging both is how the model gets validated
        instead of trusted: getCenterOfMass() aggregates descendant solids into
        the true whole-body CoM, getStaticBalance() projects it onto the convex
        hull of the real contact points, and getContactPoints(True) says which
        parts of the robot are actually touching the floor.
        """
        out: dict[str, object] = {}
        node = self.self_node
        if node is None:
            return out
        try:
            com = node.getCenterOfMass()
            if com is not None and len(com) >= 3 and math.isfinite(float(com[0])):
                out["sv_com_x"], out["sv_com_y"] = float(com[0]), float(com[1])
                out["sv_com_z"] = float(com[2])
        except Exception:  # noqa: BLE001 - a diagnostic must never break control
            pass
        try:
            out["sv_balanced"] = int(bool(node.getStaticBalance()))
        except Exception:  # noqa: BLE001
            pass
        try:
            points = node.getContactPoints(True)
            out["sv_contacts"] = 0 if points is None else len(points)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _imu_rpy(self) -> tuple:
        """(roll, pitch, yaw) of the torso in rad; (0, 0, 0) if unavailable."""
        if self.imu is None:
            return (0.0, 0.0, 0.0)
        try:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 0.0)
        if not all(math.isfinite(v) for v in (roll, pitch, yaw)):
            return (0.0, 0.0, 0.0)
        return (roll, pitch, yaw)

    def _calibrate_imu(self, now: float, raw_roll: float, raw_pitch: float,
                       fsr: dict[str, float] | None) -> None:
        """Learn what "upright" reads on this robot's InertialUnit.

        Only accepts samples while the foot sensors say the robot is standing, so
        the zero cannot be latched from a fallen pose. Where no foot sensors
        resolve it falls back to trusting the world file's spawn -- which does
        place the robot upright -- and the startup log says which happened.
        """
        if self._imu_zero is not None:
            return
        if not all(math.isfinite(v) for v in (raw_roll, raw_pitch)):
            return
        if fsr:
            left = float(fsr.get("L", 0.0))
            right = float(fsr.get("R", 0.0))
            # TOTAL load is not evidence of standing: measured on a fallen robot
            # after a reset, one foot alone carried 25.85 N against 0.72 N on the
            # other -- 26.6 N of "standing" that sailed past a 20 N total check
            # and latched a zero 17.4 deg out, worth ~90 mm of phantom CoM shift.
            # Both soles must be loaded, which a robot lying on its side is not.
            if min(left, right) < IMU_CALIBRATION_MIN_SOLE_LOAD_N:
                return
            if (left + right) < IMU_CALIBRATION_MIN_LOAD_N:
                return
        # Independent of the foot sensors: forward kinematics with the tilt taken
        # as zero gives the height the head WOULD be at if upright. A standing
        # robot reads ~0.459 m; every bad calibration in the logs read 0.049 to
        # 0.205 m, so this separates them with an enormous margin and does not
        # depend on the very tilt estimate being calibrated.
        height = self.head_height(0.0, 0.0)
        if height is not None and height < IMU_CALIBRATION_MIN_HEAD_M:
            return
        if self._imu_cal_started is None:
            self._imu_cal_started = now
        acc = self._acc_tilt_raw()
        self._imu_cal.append((raw_roll, raw_pitch,
                              None if acc is None else acc[0],
                              None if acc is None else acc[1]))
        if (now - self._imu_cal_started) < IMU_CALIBRATION_S:
            return
        if len(self._imu_cal) < IMU_CALIBRATION_MIN_SAMPLES:
            return
        rolls = sorted(v[0] for v in self._imu_cal)
        pitches = sorted(v[1] for v in self._imu_cal)
        mid = len(rolls) // 2
        samples = len(self._imu_cal)
        self._imu_zero = (rolls[mid], pitches[mid])
        # The gravity-derived attitude gets its own zero from the same standing
        # samples. Physically it should be about (0, 0) -- gravity needs no
        # calibration -- but taking it the same way absorbs a torso that does not
        # stand exactly axis-aligned, and keeps the two channels comparable so the
        # cross-check in _torso_tilt is meaningful.
        acc_rolls = sorted(v[2] for v in self._imu_cal if v[2] is not None)
        acc_pitches = sorted(v[3] for v in self._imu_cal if v[3] is not None)
        if len(acc_rolls) >= IMU_CALIBRATION_MIN_SAMPLES // 2:
            self._acc_zero = (acc_rolls[len(acc_rolls) // 2],
                              acc_pitches[len(acc_pitches) // 2])
            logger.info(
                "Gravity zero learned from %d standing samples: roll %+.3f, "
                "pitch %+.3f rad. %s",
                len(acc_rolls), self._acc_zero[0], self._acc_zero[1],
                "This is the control path; the InertialUnit's roll cross-checks it."
                if TILT_FROM_ACCELEROMETER else
                "Logged for comparison only -- the control path is the "
                "InertialUnit with its pitch doubled (see "
                "TILT_FROM_ACCELEROMETER for why gravity is not acted on).",
            )
        elif TILT_FROM_ACCELEROMETER:
            logger.error(
                "No usable accelerometer: tilt falls back to the InertialUnit, "
                "whose PITCH reads half the real pitch on this proto (see "
                "TILT_FROM_ACCELEROMETER). Fore/aft balance will be sluggish."
            )
        self._imu_cal.clear()
        magnitude = max(abs(self._imu_zero[0]), abs(self._imu_zero[1]))
        emit = logger.warning if magnitude > 0.05 else logger.info
        emit(
            "IMU tilt zero learned from %d standing samples: roll %+.3f, pitch "
            "%+.3f rad. Tilt is measured relative to this from now on.%s",
            samples, self._imu_zero[0], self._imu_zero[1],
            "" if magnitude <= 0.05 else
            f" That is {math.degrees(magnitude):.0f} deg, so this model's "
            f"InertialUnit is mounted rotated: without the correction every tilt "
            f"gate and the balance loop read a robot standing still as falling.",
        )

    # ------------------------------------------------------------ fall recovery
    def head_height(self, roll: float, pitch: float) -> float | None:
        """Height of the head above the soles along the WORLD vertical, in metres.

        Forward kinematics places the head and both soles in the torso frame; the
        calibrated InertialUnit supplies the vertical. Upright that is ~0.46 m and
        a deep squat only takes it to 0.41 m, because NAO's crouch keeps the torso
        vertical -- so it is a clean fall test rather than a proxy for one.

        Returns None when there is no CoM model to do the kinematics with (no
        NumPy in Webots' interpreter), in which case the caller falls back to tilt.
        """
        balance = self.driver.balance
        if balance is None:
            return None
        try:
            import numpy as np

            state = dict(self.driver.commanded)
            state.update(self.driver.measured)
            frames = balance.model.frames(state)
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            # Only the world-z row of the torso->world rotation is needed.
            rot_z = np.array([-sp * cr, sr, cr * cp])
            head_z = float(rot_z @ frames["HeadPitch"][:3, 3])
            soles = []
            for side in ("L", "R"):
                T = frames[f"{side}AnkleRoll"]
                sole = T[:3, :3] @ np.array([0.035, 0.0, -0.04519]) + T[:3, 3]
                soles.append(float(rot_z @ sole))
            return head_z - sum(soles) / len(soles)
        except Exception:  # noqa: BLE001 - a diagnostic must never break control
            return None

    def _fallen(self, now: float, roll: float, pitch: float,
                fsr: dict[str, float] | None = None) -> bool:
        """Has the robot been down for long enough to be worth recovering?

        Three independent tests, each with its own confirmation time so a stumble
        the balance loop catches is not treated as a fall:

        * the head is low (a topple; ``FALL_HEAD_HEIGHT_M`` / ``FALL_CONFIRM_S``),
        * the torso has been tilted past every stand-down gate for seconds
          (``STUCK_TILT_RAD`` / ``STUCK_CONFIRM_S``): the robot is propped on
          something, not standing, however high its head still is,
        * the soles have carried almost nothing for seconds
          (``FALL_UNLOADED_N`` / ``FALL_UNLOADED_S``).
        """
        if not self.attitude_ready:
            # Tilt is not yet meaningful, so neither is any test built on it.
            self._fall_since = self._stuck_since = self._unloaded_since = None
            return False
        height = self.head_height(roll, pitch)
        self._head_height = height
        tilt = max(abs(roll), abs(pitch))
        if height is not None:
            down = height < FALL_HEAD_HEIGHT_M
        else:
            # No kinematics available: tilt alone. Well past the abort limit, so it
            # cannot fire on a posture the balance loop might still save.
            down = tilt > 1.0

        confirmed = False
        if down:
            if self._fall_since is None:
                self._fall_since = now
                reason = (f"head only {height:.3f} m above the soles"
                          if height is not None else f"tilt {tilt:.2f} rad")
                logger.error("FALL DETECTED (%s). Confirming for %.1fs...",
                             reason, FALL_CONFIRM_S)
            confirmed |= (now - self._fall_since) >= FALL_CONFIRM_S
        else:
            self._fall_since = None

        if tilt > STUCK_TILT_RAD:
            if self._stuck_since is None:
                self._stuck_since = now
            elif (now - self._stuck_since) >= STUCK_CONFIRM_S:
                if not confirmed:
                    logger.error("STUCK: torso tilted %.2f rad for %.1fs; treating "
                                 "as a fall.", tilt, now - self._stuck_since)
                confirmed = True
        else:
            self._stuck_since = None

        total = None if not fsr else float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
        if total is not None and total < FALL_UNLOADED_N:
            if self._unloaded_since is None:
                self._unloaded_since = now
            elif (now - self._unloaded_since) >= FALL_UNLOADED_S:
                if not confirmed:
                    logger.error("OFF ITS FEET: soles carried %.1f N for %.1fs; "
                                 "treating as a fall.", total, now - self._unloaded_since)
                confirmed = True
        else:
            self._unloaded_since = None
        return confirmed

    def _recover_from_fall(self, now: float) -> bool:
        """Put the robot back on its feet by resetting the simulation.

        Returns True if a reset was issued. Rate-limited and capped: a setup that
        falls immediately every time must not reload in a loop, it must stop and
        say so, because an endless reload is harder to diagnose than a robot lying
        still.
        """
        if not AUTO_RELOAD_ON_FALL:
            return False
        # WALL time, not ``now``: ``now`` is robot.getTime(), which simulationReset()
        # rewinds to zero. Stamping the cooldown with it meant the next recovery had
        # to wait until the NEW episode's clock passed the OLD episode's reload time,
        # so the wait grew by 10 s every fall. The logs show it exactly -- resets at
        # sim 3.32, 16.82, 26.82, 36.82, each one 10 s after the previous stamp --
        # and a run that first fell at sim 46.60 needed sim > 56.6 to be picked up,
        # never reached it, and lay on the floor for the rest of the session. A rate
        # limit on a real-world action belongs on a clock that does not rewind.
        wall_now = time.time()
        if self._last_reload is not None and \
                (wall_now - self._last_reload) < FALL_RELOAD_COOLDOWN_S:
            return False
        if self._reloads >= FALL_MAX_RELOADS:
            if self._reloads == FALL_MAX_RELOADS:
                self._reloads += 1     # log this once, then stay quiet
                logger.error(
                    "Robot has fallen %d times; not reloading again. Something is "
                    "wrong that a reload will not fix -- run "
                    "scripts/analyze_run.py on this session's log.",
                    FALL_MAX_RELOADS,
                )
            return False

        self._reloads += 1
        self._last_reload = wall_now
        self._fall_since = self._stuck_since = self._unloaded_since = None
        logger.error("Reloading the simulation to stand the robot back up "
                     "(recovery %d/%d).", self._reloads, FALL_MAX_RELOADS)
        # Hand every layer back a clean slate first: whether or not Webots
        # restarts this controller, the state must not describe the fallen robot.
        self._reset_for_new_episode()
        for method in ("simulationReset", "worldReload"):
            call = getattr(self.robot, method, None)
            if call is None:
                continue
            try:
                call()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s() failed (%s); trying the next option.",
                               method, exc)
        logger.error(
            "Cannot reset the simulation: this controller is not a Supervisor. Add "
            "`supervisor TRUE` to the Nao node in the world file, or press "
            "Ctrl+Shift+R in Webots to reload by hand."
        )
        return False

    def _reset_for_new_episode(self) -> None:
        """Drop all state that describes the old, fallen robot."""
        self.motion.abort()
        self._abandon_prepare()
        self._drop_walk_latch()
        self.driver.reclaim_from_motion()
        # Return the COMMANDED pose to neutral. The reset puts the robot back
        # upright, but the driver's base_targets still held the collapsed pose it
        # fell in, so the first step of the new episode drove it straight back
        # into that shape -- which is what put the topple inside the IMU
        # calibration window and taught the balance loop a fallen "upright".
        self.driver.upper_body_stand_down()
        self.driver.lower_body_stand_down()
        if self.driver.lower_body is not None:
            self.driver.lower_body.reset()
        # ...and the balance FEEDBACK loop, which is an integrator and was the one
        # piece of state that survived a reset. The robot came back upright with
        # the pelvis still shifted to the clamp the old, fallen robot had needed:
        # measured on the first control step of a new episode, HipPitch -0.345 and
        # the centre of mass 40 mm off centre before anything had happened.
        self.driver.reset_balance()
        self.yaw_servo.reset()
        self._turning = False
        self._tilt_risk = 0.0
        self._last_risk_update = None
        # The tilt zero is KEPT. It measures how the InertialUnit is mounted on the
        # torso (Nao.proto: rolled +pi/2 about x), a property of the robot that a
        # reset cannot change. Re-learning it here is what produced 4-6 different
        # zeros per session, up to 2.6 deg apart (13 mm of phantom CoM error, with
        # a 39 mm fore/aft margin), each latched from a robot that had just been
        # stood back up and was still settling. The first zero of a session, taken
        # from the world file's clean spawn, is also the cleanest.
        self._imu_cal.clear()
        self._imu_cal_started = None

    def _corrected_tilt(self, raw_roll: float, raw_pitch: float) -> tuple:
        """(roll, pitch) from the INERTIAL UNIT, relative to the learned upright.

        (0, 0) until calibrated. Note the pitch here is half the real pitch on
        this proto -- see TILT_FROM_ACCELEROMETER -- so the control path uses
        :meth:`_torso_tilt`, and this stays the raw-sensor accessor.
        """
        if self._imu_zero is None:
            return (0.0, 0.0)
        return (raw_roll - self._imu_zero[0], raw_pitch - self._imu_zero[1])

    def _acc_tilt_raw_update(self, now: float) -> None:
        """Advance the low-passed gravity attitude without acting on it, so
        acc_roll/acc_pitch stay a live comparison in the log even when the
        Supervisor is the control source."""
        raw = self._acc_tilt_raw()
        if raw is None:
            return
        dt = 0.0 if self._acc_last_update is None else max(0.0, now - self._acc_last_update)
        self._acc_last_update = now
        if self._acc_tilt is None or dt <= 0.0 or \
                max(abs(n - p) for n, p in zip(raw, self._acc_tilt, strict=False)) > ACC_SNAP_RAD:
            self._acc_tilt = raw
        else:
            a = 1.0 - math.exp(-dt / ACC_TILT_TAU_S)
            self._acc_tilt = tuple(
                prev + a * (new - prev) for prev, new in zip(self._acc_tilt, raw, strict=False)
            )

    def _acc_tilt_raw(self) -> tuple | None:
        """(roll, pitch) of the torso from gravity, or None if unusable.

        Rejects samples while the robot is accelerating hard enough that the
        measured vector is not gravity: the magnitude then departs from 9.81 and
        the direction is not an attitude. See TILT_FROM_ACCELEROMETER for the
        mounting and the formulae.
        """
        if self.accel is None:
            return None
        try:
            values = self.accel.getValues()
        except Exception:  # noqa: BLE001
            return None
        if values is None or len(values) < 3:
            return None
        try:
            ax, ay, az = (float(v) for v in values[:3])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (ax, ay, az)):
            return None
        if abs(math.sqrt(ax * ax + ay * ay + az * az) - 9.81) > ACC_MAX_G_ERROR:
            return None
        # Undo the 180 deg mount: world-up in torso coordinates.
        ux, uy, uz = ax, -ay, -az
        if math.hypot(uy, uz) < 1e-6 and abs(ux) < 1e-6:
            return None
        pitch = math.atan2(-ux, math.hypot(uy, uz))
        roll = math.atan2(uy, uz)
        return (roll, pitch)

    def _supervisor_tilt(self) -> tuple | None:
        """(roll, pitch) of the torso from the scene tree, or None.

        See ATTITUDE_SOURCE for the derivation and the sign conventions.
        """
        if self.self_node is None:
            return None
        try:
            m = self.self_node.getOrientation()
        except Exception:  # noqa: BLE001
            return None
        if m is None or len(m) < 9:
            return None
        try:
            r20 = max(-1.0, min(1.0, float(m[6])))
            pitch = -math.asin(r20)
            roll = math.atan2(float(m[7]), float(m[8]))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            return None
        return (roll, pitch)

    @property
    def attitude_ready(self) -> bool:
        """Is the tilt the controller acts on meaningful yet?

        The Supervisor's is meaningful from the first step; the sensor paths have
        to learn their zero first (see IMU_AUTO_ZERO), and until they have, tilt
        is reported as level so nothing aborts on a reading we do not understand.
        """
        if ATTITUDE_SOURCE == "supervisor" and self.self_node is not None:
            return True
        return self._imu_zero is not None

    def _torso_tilt(self, now: float, imu_roll: float, imu_pitch: float,
                    fsr: dict[str, float] | None) -> tuple:
        """The (roll, pitch) the controller acts on, and the source it came from.

        Gravity where it is available and agrees with the InertialUnit's roll (the
        one axis that device reports honestly); the InertialUnit otherwise. The
        agreement test is only meaningful on a robot standing on its feet, so it
        is only evaluated there.
        """
        # The truth first, when the scene tree can be asked (see ATTITUDE_SOURCE).
        if ATTITUDE_SOURCE == "supervisor":
            supervised = self._supervisor_tilt()
            if supervised is not None:
                self._tilt_source = "supervisor"
                # Gravity is still computed and low-passed so acc_* stays a live
                # witness in the log, but it is not what we act on.
                self._acc_tilt_raw_update(now)
                return supervised

        self._tilt_source = "imu"
        imu_pitch *= IMU_PITCH_SCALE          # see IMU_PITCH_SCALE
        raw = self._acc_tilt_raw()
        if raw is None:
            return (imu_roll, imu_pitch)

        dt = 0.0 if self._acc_last_update is None else max(0.0, now - self._acc_last_update)
        self._acc_last_update = now
        if self._acc_tilt is None or dt <= 0.0 or \
                max(abs(n - p) for n, p in zip(raw, self._acc_tilt, strict=False)) > ACC_SNAP_RAD:
            self._acc_tilt = raw          # first sample, or a real lurch (ACC_SNAP_RAD)
        else:
            a = 1.0 - math.exp(-dt / ACC_TILT_TAU_S)
            self._acc_tilt = tuple(
                prev + a * (new - prev) for prev, new in zip(self._acc_tilt, raw, strict=False)
            )
        if not TILT_FROM_ACCELEROMETER or not self._acc_usable or self._acc_zero is None:
            return (imu_roll, imu_pitch)

        roll = self._acc_tilt[0] - self._acc_zero[0]
        pitch = self._acc_tilt[1] - self._acc_zero[1]

        # Cross-check against the InertialUnit's roll while the robot is standing.
        loaded = bool(fsr) and (float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))) > 20.0
        if loaded and abs(roll - imu_roll) > ACC_DISAGREE_RAD:
            if self._acc_disagree_since is None:
                self._acc_disagree_since = now
            elif (now - self._acc_disagree_since) >= ACC_DISAGREE_S:
                self._acc_usable = False
                logger.error(
                    "The accelerometer and the InertialUnit disagree about ROLL by "
                    "%.3f rad for %.1fs while the robot stands on its feet. The "
                    "InertialUnit is trustworthy on roll, so the gravity-derived "
                    "attitude is wrong (a mounting or sign assumption -- see "
                    "TILT_FROM_ACCELEROMETER) and is now disabled. Tilt falls back "
                    "to the InertialUnit, whose PITCH reads half-scale on this "
                    "proto, so expect sluggish fore/aft balance until this is "
                    "fixed.", roll - imu_roll, now - self._acc_disagree_since,
                )
                return (imu_roll, imu_pitch)
        else:
            self._acc_disagree_since = None
        self._tilt_source = "accel"
        return (roll, pitch)

    def _init_walk_sensors(self) -> None:
        """Enable the gyro/accelerometer and any foot force sensors.

        All best-effort and NaN-guarded: the march tier needs none of them, and
        the stepping gate falls back to the CoM model alone when they are absent.
        """
        self.gyro = None
        self.accel = None
        self.fsr: dict[str, list[object]] = {"L": [], "R": []}
        for name in (GYRO_NAME, ACCELEROMETER_NAME):
            dev = self.robot.getDevice(name)
            if dev is None:
                continue
            try:
                dev.enable(self.timestep)
            except Exception:  # noqa: BLE001
                continue
            if name == GYRO_NAME:
                self.gyro = dev
            else:
                self.accel = dev
        for side, names in FSR_DEVICES.items():
            for name in names:
                dev = self.robot.getDevice(name)
                if dev is None:
                    continue
                try:
                    dev.enable(self.timestep)
                except Exception:  # noqa: BLE001
                    continue
                self.fsr[side].append(dev)
        n_fsr = len(self.fsr["L"]) + len(self.fsr["R"])
        logger.info("Sensors: gyro=%s, foot-force sensors=%d",
                    self.gyro is not None, n_fsr)

    def _read_fsr(self) -> dict[str, float] | None:
        """Per-foot load ``{"L": n, "R": n}`` from the FSRs, or None.

        NAO's foot sensors are 3-axis ("force-3d") TouchSensors, so the value
        comes from ``getValues()``, not ``getValue()`` -- reading them as scalars
        is why an earlier version silently got no load information and the
        stepping gate never saw a weight transfer. Both APIs are handled so this
        also works with 1-axis protos.
        """
        if not self.fsr["L"] and not self.fsr["R"]:
            return None
        out: dict[str, float] = {}
        for side in ("L", "R"):
            total = 0.0
            for dev in self.fsr[side]:
                total += _sensor_magnitude(dev)
            out[side] = total
        return out

    def _tilt_rate(self) -> tuple:
        """(roll_rate, pitch_rate) in rad/s from the gyro; (0, 0) without one."""
        if self.gyro is None:
            return (0.0, 0.0)
        try:
            values = self.gyro.getValues()
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)
        if values is None or len(values) < 2:
            return (0.0, 0.0)
        rates = [float(v) if math.isfinite(float(v)) else 0.0 for v in values[:2]]
        return (rates[0], rates[1])

    def _init_locomotion(self) -> None:
        """Discover the walk/turn clips and build the yaw servo."""
        self.motion = MotionPlayer({}, log=logger.warning)
        self.yaw_servo = YawServo(sign=TURN_SIGN)
        if self.leg_control != "auto":
            logger.info("Locomotion clips disabled (LEG_CONTROL=%s)", self.leg_control)
            return
        dirs = default_motion_search_dirs(extra=MOTION_SEARCH_DIRS_EXTRA)
        files = find_motion_files(dirs)
        if GAIT_CYCLE:
            files, note = select_walk_clip(files)
        else:
            files.pop("forward_continuous", None)
            note = "GAIT_CYCLE is off; one-shot playback of the short clip"
        self.motion = MotionPlayer(files, log=logger.warning)
        if files:
            logger.info("Locomotion clips found: %s", ", ".join(sorted(files)))
            logger.info("Walk clip: %s", note)
            # Say how the robot will TURN, in the same breath as how it will
            # walk. Turning is the half of locomotion with no equivalent of
            # `note`, and "which clip, entered where, stoppable at which angles"
            # is exactly what has to be known to read a turn that went wrong.
            # Computing the schedules here also front-loads their cost (one
            # forward-kinematics pass per keyframe) into startup instead of
            # paying it on the first control step that wants to turn.
            aimable = sorted(self.motion.interruptible & TURN_ACTION_NAMES)
            if aimable:
                logger.info(
                    "Turning: %s can be aimed -- entered after the opening "
                    "crouch and stopped at the certified keyframe nearest the "
                    "heading error, instead of played whole.",
                    ", ".join(aimable))
            else:
                logger.info(
                    "Turning: no clip can be aimed (no rotation measurable in "
                    "the keyframes, or no CoM model). Turns will be played "
                    "whole, so the heading will settle within half a clip.")
        else:
            logger.warning(
                "No NAO .motion files found (searched %d dirs, e.g. %s). The robot "
                "will march in place instead of translating; set $WEBOTS_HOME or "
                "drop clips in %s to enable real locomotion.",
                len(dirs), dirs[0] if dirs else "-", MOTION_SEARCH_DIRS_EXTRA[0],
            )

    def _init_socket(self) -> None:
        logger.info("Opening UDP socket on %s:%d ...", UDP_HOST, UDP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    # ------------------------------------------------------------------- comms
    def _drain_latest_command(self) -> dict | None:
        """Return the most recent pose command, discarding any backlog.

        UDP can queue several frames between simulation steps. We only care
        about the freshest pose, so we drain the buffer and keep the last one
        (keeps end-to-end latency low -- PRD NFR-1).
        """
        latest: dict | None = None
        while True:
            try:
                data, _ = self.sock.recvfrom(SOCKET_RCVBUF)
            except (BlockingIOError, OSError):
                break
            try:
                latest = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return latest

    # ---------------------------------------------------------------- arbiter
    def _update_yaw_servo(self, now: float, robot_yaw: float) -> None:
        """Feed the heading servo. Only called when HEADING_FROM_IMU -- on this
        proto the InertialUnit's yaw axis is disabled and what it returns is the
        pitch channel, so there is no heading to close a loop on."""
        gait = self.gait_cmd or {}
        yaw = gait.get("body_yaw_rad")
        if yaw is None:
            return
        try:
            human_yaw = float(yaw)
        except (TypeError, ValueError):
            return
        self.yaw_servo.update(
            human_yaw=human_yaw,
            conf=float(gait.get("yaw_conf", gait.get("conf", 0.0)) or 0.0),
            robot_yaw=robot_yaw,
            now_s=now,
        )

    def _predicted_tilt_rad(self, roll: float, pitch: float) -> float:
        """Worst-axis tilt magnitude, predicted a short time ahead by the gyro.

        Shared by the hard mid-clip abort (_falling) and the continuous
        pre-clip risk signal (_update_tilt_risk) so there is one formula, not
        two definitions of "how tipped over are we" drifting apart.
        """
        d_roll, d_pitch = self._tilt_rate()
        roll_pred = roll + TILT_RATE_LEAD_S * d_roll
        pitch_pred = pitch + TILT_RATE_LEAD_S * d_pitch
        return max(abs(roll_pred), abs(pitch_pred))

    def _falling(self, roll: float, pitch: float) -> bool:
        """Tilt (predicted a short time ahead by the gyro) past the abort limit."""
        return self._predicted_tilt_rad(roll, pitch) > TILT_ABORT_RAD

    def _update_tilt_risk(self, now: float, roll: float, pitch: float) -> None:
        """Advance the continuous tilt-risk EMA. Called once per tick, before
        the arbiter picks a leg-control layer, so it tracks balance risk
        regardless of which layer is currently driving the legs -- the
        "always on" check that _settled() leans on to keep the robot from
        launching a new locomotion clip too soon after a wobble.
        """
        mag = self._predicted_tilt_rad(roll, pitch)
        dt = 0.0 if self._last_risk_update is None else max(0.0, now - self._last_risk_update)
        self._last_risk_update = now
        alpha = 1.0 - math.exp(-dt / TILT_RISK_TAU_S) if dt > 0 else 1.0
        self._tilt_risk += alpha * (mag - self._tilt_risk)

    def _latched_gait(self, now: float) -> dict | None:
        """The gait command the locomotion layer should act on.

        The live one while it asks for locomotion; the last one that did for
        WALK_LATCH_RELEASE_S after it stops asking. The cue flickers -- median
        "march" run 1.32 s against 33 sub-1.2 s dropouts in the first session that
        walked -- and without this every flicker ended a walk and paid for a fresh
        prepare ramp.
        """
        live = self.gait_cmd or {}
        if str(live.get("state", "idle")) == "march":
            self._walk_gait = dict(live)
            self._walk_latch_until = now + WALK_LATCH_RELEASE_S
            return self._walk_gait
        if self._walk_latch_until is not None and now < self._walk_latch_until:
            return self._walk_gait
        self._walk_gait = None
        self._walk_latch_until = None
        return live or None

    def _drop_walk_latch(self) -> None:
        """Forget the latched walk request (the robot must not resume on it)."""
        self._walk_gait = None
        self._walk_latch_until = None

    def _marching(self) -> bool:
        gait = self.gait_cmd or {}
        return (
            str(gait.get("state", "idle")) == "march"
            and float(gait.get("cadence_hz", 0.0) or 0.0) > 0.0
            and float(gait.get("conf", 0.0) or 0.0) >= LOCOMOTION.walk_conf_min
        )

    def _drive_legs(self, now: float, roll: float, pitch: float,
                    yaw: float, fsr: dict[str, float] | None = None) -> None:
        """Pick and run exactly one leg commander for this simulation step.

        ``roll``/``pitch`` are relative to the LEARNED upright (see
        :meth:`_corrected_tilt`), never the raw InertialUnit reading.
        """
        torso_rp = (roll, pitch)
        tilt_rate = self._tilt_rate()
        if fsr is None:
            fsr = self._read_fsr()
        falling = self._falling(roll, pitch)

        # What the locomotion layer is being asked for this step, with the walk
        # request latched across the cue's dropouts (see WALK_LATCH_RELEASE_S).
        locomotion_gait = self._latched_gait(now)

        # (1) A clip is playing. It owns the 12 leg joints -- but not
        #     unconditionally to the end of the clip. If nothing wants locomotion
        #     any more then a CYCLIC clip leaves its loop through the clip's own
        #     deceleration (see GAIT_CYCLE), and any other clip stops at the next
        #     keyframe it is safe to stop at (see CLIP_EXIT_TOLERANCE_S).
        if self.motion.active:
            action = self.motion.action
            if falling:
                logger.warning("Tilt abort (%.2f, %.2f rad): stopping motion '%s'",
                               roll, pitch, action)
                self._end_motion(action, ok=False, reason="tilt abort")
            elif self._motion_overran(now):
                # A clip that never reports "over" would keep the whole body
                # suspended forever, which reads as a totally dead robot.
                logger.error(
                    "Motion '%s' overran its watchdog (%.1fs); taking the body "
                    "back. This clip will not be used again.",
                    action, now - (self._motion_started_at
                                   if self._motion_started_at is not None else now),
                )
                self.motion.drop(action)
                self._end_motion(action, ok=False, reason="watchdog")
            elif self.motion.poll():
                # What the planner wants THIS step, recorded even though branch 2
                # is not running: clip_planned used to freeze at whatever was
                # planned when the clip started, which made it look as though the
                # planner still wanted a forward walk for the whole run. It was
                # stale, not agreeing -- and a stale diagnostic is worse than a
                # missing one, because it argues against the true explanation.
                self._clip_planned = self._wanted_action(locomotion_gait, yaw) or ""
                wanted = not self._clip_unwanted(locomotion_gait, roll, pitch, yaw)
                if self.motion.cyclic:
                    # A gait generator: repeat the stride while the walk is
                    # wanted, and when it is not, leave through the clip's own
                    # deceleration rather than freezing mid-stride. Either way
                    # the clip keeps the legs this step, right through the
                    # deceleration -- the early-exit path below must not get a
                    # look at it, or it would release the body mid-settle.
                    strides = self.motion.cycles_done
                    self._cycle_state = self.motion.cycle_tick(wanted)
                    if self.motion.cycles_done > strides:
                        self._cycles_walked += 1
                        # The clip cannot end while it is being rewound, so the
                        # watchdog has to be told that progress is being made or
                        # it would kill a perfectly healthy walk. Counted from
                        # the player's own tally rather than the status word,
                        # because a stride completed while the walk is looking
                        # for its exit is still a stride and still proof of life.
                        self._extend_motion_deadline(now)
                    # The settle is over: the clip has stopped travelling and
                    # everything left is it standing up out of its own walk
                    # crouch, which the lower-body layer does better and has to
                    # redo anyway. Riding it out is 1.28 s of a robot that will
                    # not answer the human -- on every stop, and again before it
                    # can be asked to walk a second time. See GaitCycle.rest_s.
                    if self.motion.leaving:
                        cycle = self.motion.cycle()
                        played = self.motion.time_s()
                        # Only when there is something to trim. A clip whose tail
                        # IS the settle (rest_s == duration_s) must be ridden to
                        # the end, and firing here on its last keyframe would
                        # steal the ordinary "clip finished" path for no gain.
                        if (cycle is not None and played is not None
                                and cycle.duration_s - cycle.rest_s
                                > CLIP_EXIT_TOLERANCE_S
                                and played >= cycle.rest_s):
                            logger.info(
                                "'%s' has come to rest at %.2fs; taking the legs "
                                "back rather than riding out the last %.2fs of "
                                "the clip standing itself up.",
                                action, played, cycle.duration_s - played)
                            self._tails_trimmed += 1
                            self._end_motion(action, ok=True,
                                             reason="settled, tail trimmed")
                            self._clip_status = "stopped, settled"
                            return
                    self._clip_status = f"cycling ({self._cycle_state})"
                    self.leg_mode = f"motion:{action}"
                    return
                if self._turning and self._turn_exit_due(yaw):
                    logger.info(
                        "Turn '%s' has delivered the heading (%.2fs of %.2fs, "
                        "%.0f deg still wanted); stopping here.", action,
                        self.motion.time_s() or 0.0,
                        self.motion.duration_s() or 0.0,
                        math.degrees(self.yaw_servo.error(yaw)))
                    self._early_exits += 1
                    self._turning = False
                    self._end_motion(action, ok=True, reason="turn aimed home")
                    self._clip_status = "turn complete"
                elif not wanted and self.motion.at_safe_exit(CLIP_EXIT_TOLERANCE_S):
                    logger.info(
                        "Nothing wants locomotion any more: stopping '%s' at a "
                        "safe keyframe (%.2fs of %.2fs).", action,
                        self.motion.time_s() or 0.0,
                        self.motion.duration_s() or 0.0)
                    self._early_exits += 1
                    self._end_motion(action, ok=True, reason="stopped early, safely")
                    self._clip_status = "stopped at a safe keyframe"
                else:
                    self.leg_mode = f"motion:{action}"
                    return
            else:
                # Finished normally -- but only counts as a success if the robot
                # is still upright, otherwise we are walking ourselves over.
                upright = abs(roll) < MOTION_START_MAX_TILT_RAD * 2.0 and \
                    abs(pitch) < MOTION_START_MAX_TILT_RAD * 2.0
                self._end_motion(action, ok=upright, reason="clip finished")

        if self.leg_control == "off":
            self.leg_mode = "stand"
            self.driver.balance_tick(torso_rp, tilt_rate=tilt_rate, now_s=now)
            return

        # (2) Real locomotion: walk/turn with a pre-balanced clip.
        #
        #     ``clip_declined`` records that this layer WANTED to act and could
        #     not. Branch 3 keys off it, which is the difference between "the
        #     robot marches in place while it waits to be steady enough to walk"
        #     and the old behaviour: it fell through to branch 4, which does
        #     nothing with the legs, so the robot stood motionless while the human
        #     marched at it and reported legs=pose with no error anywhere.
        clip_declined = False
        if self.leg_control == "auto" and not falling:
            # Without a heading the turn half of the planner is starved on purpose
            # (error 0, never trustworthy) while forward walking is untouched --
            # walking needs no heading. See HEADING_SOURCE.
            has_heading = self.heading_available
            plan = plan_action(
                yaw_error_rad=self.yaw_servo.error(yaw) if has_heading else 0.0,
                gait=locomotion_gait,
                action=self.action_cmd,
                available=self.motion.available,
                params=LOCOMOTION,
                turning=self._turning,
                yaw_trustworthy=has_heading and self.yaw_servo.stable(),
                interruptible=self.motion.interruptible,
                playing=self.motion.action,
            )
            self._clip_planned = plan.action or ""
            if plan.action is None:
                # Nothing left to correct: the rotation (if any) has converged.
                self._turning = False
                self._clip_status = "nothing planned"
                self._abandon_prepare()
            elif not self._settled(roll, pitch):
                # Starting a clip mid-wobble is how a walk becomes a fall; wait,
                # but let branch 3 keep the legs moving while we wait.
                clip_declined = True
                self._clip_status = "declined: not settled"
                self._abandon_prepare()
            elif not self._ready_to_play(now, plan.action):
                # Ramping the legs into the stance the clip opens in. THIS layer
                # is the leg commander while that happens -- see
                # CLIP_PREPARE_TIMEOUT_S and NaoPoseDriver.approach_leg_pose.
                self.leg_mode = f"prepare:{plan.action}"
                return
            elif self.motion.start(plan.action,
                                   self.motion.entry_time_s(plan.action)):
                cycle = self.motion.cycle(plan.action)
                if cycle is None:
                    logger.info("Locomotion: %s (%s)", plan.action, plan.reason)
                else:
                    logger.info(
                        "Locomotion: %s (%s) as a continuous gait -- %.3f m/s "
                        "for as long as you keep walking.",
                        plan.action, plan.reason, cycle.speed_mps)
                self._turning = plan.is_turn
                self._begin_motion(now)
                # Hand the clip only the joints it declares. Webots' walk clips
                # drive the 12 leg joints and nothing else, so the arms and head
                # keep following the human right through the step.
                self.driver.release_to_motion(self.motion.joints(plan.action))
                self.leg_mode = f"motion:{plan.action}"
                self._clip_status = "started"
                self._preparing = None
                self._prepare_since = None
                self._prepare_run_since = None
                return
            else:
                # The clip was planned but Webots would not play it. Silence here
                # made a rejected clip indistinguishable from a clip nobody asked
                # for, which is a long debugging session for a one-line cause.
                logger.warning(
                    "Locomotion clip '%s' was planned (%s) but would not start; "
                    "falling back to the march engine this step.",
                    plan.action, plan.reason,
                )
                clip_declined = True
                self._clip_status = "start REFUSED by Webots"
        else:
            self._turning = False

        # (3) The clip layer is not walking us: march in place instead.
        #     Never while going over -- the pose layer below is the better
        #     recovery, because its tilt gate ramps the asymmetric part of the
        #     posture out and returns the legs to the balanced symmetric crouch,
        #     with the CoM correction folded back in as soon as both feet are
        #     evenly loaded again.
        #
        #     The gate is "did the clip layer decline?", NOT "is a forward clip
        #     absent from disk?". The old disk test meant that installing Webots
        #     -- which ships Forwards.motion -- permanently disabled the march
        #     engine, so the fallback existed only on machines that could not run
        #     the robot in the first place.
        if (
            self.leg_control in ("auto", "engine")
            and not falling
            and self.driver.enable_walk
            and self._marching()
            and (clip_declined or "forward" not in self.motion.available)
        ):
            self.leg_mode = f"march:{WALK_TIER}"
            self.driver.gait_tick(now, torso_rp, fsr=fsr, tilt_rate=tilt_rate)
            return

        # (4) Default: per-leg pose imitation (squat, single-leg lift).
        if self.driver.lower_body is not None:
            # Give the standing turn a little immediate feedback via the shared
            # hip yaw while the (coarse) stepping turn has not fired yet.
            self.leg_mode = "pose"
            self.driver.lower_body_tick(
                now, torso_rp, fsr=fsr, tilt_rate=tilt_rate,
                # No heading, no bias: a bias derived from the pitch channel yaws
                # the legs for no reason, and because the HipYawPitch axis is
                # canted it tips both soles while doing it. See HEADING_SOURCE.
                yaw_bias=self.yaw_servo.error(yaw) if self.heading_available else 0.0,
            )
            return

        self.leg_mode = "stand"
        self.driver.balance_tick(torso_rp, tilt_rate=tilt_rate, now_s=now)

    def _turn_exit_due(self, yaw: float) -> bool:
        """Has the turn clip now delivered the rotation the heading asked for?

        This is what turns one coarse clip into an aimed turn. The clip knows how
        far it has rotated the robot at every certified stopping point
        (:class:`walk_motion.TurnSchedule`); the servo knows how much rotation is
        still wanted. So each step we ask which of the stopping points STILL
        AHEAD would leave the smallest heading error, and stop when the answer is
        "this one, now".

        Aiming rather than merely stopping when the error clears the deadband
        matters because the stopping points are up to 16 deg apart: a reactive
        "stop as soon as I am close enough" overshoots by however far the next
        one happens to be, whereas choosing the nearest rung to the error bounds
        the residual at half that gap -- 8 deg, against the 20 deg that playing
        even the small clip whole leaves behind.

        Re-evaluated every control step on the LIVE error, so a human who keeps
        turning simply moves the target rung further out and the robot keeps
        turning with them, in the same clip, with no restart.

        False whenever the heading is unusable or the clip has no schedule; the
        ordinary "the planner wants something else" exit then applies as before.
        """
        if not self.heading_available:
            return False
        schedule = self.motion.turn()
        now = self.motion.time_s()
        if schedule is None or now is None:
            return False
        delivered = schedule.yaw_at(now)
        error = self.yaw_servo.error(yaw)
        ahead = [rung for rung in schedule.rungs
                 if rung[0] >= now - CLIP_EXIT_TOLERANCE_S]
        if not ahead:
            return True            # the clip has no certified stop left; take it
        best = min(ahead, key=lambda rung: abs((rung[1] - delivered) - error))
        return best[0] <= now + CLIP_EXIT_TOLERANCE_S

    def _wanted_action(self, gait: dict | None, yaw: float) -> str | None:
        """What the locomotion layer would ask for right now, or None to stand.

        The planner's own answer, so there is exactly one place that decides what
        the robot should be doing -- whether or not a clip happens to be playing.
        """
        return plan_action(
            yaw_error_rad=self.yaw_servo.error(yaw) if self.heading_available else 0.0,
            gait=gait,
            action=self.action_cmd,
            available=self.motion.available,
            params=LOCOMOTION,
            turning=self._turning,
            yaw_trustworthy=self.heading_available and self.yaw_servo.stable(),
            interruptible=self.motion.interruptible,
            playing=self.motion.action,
        ).action

    def _clip_unwanted(self, gait: dict | None, roll: float, pitch: float,
                       yaw: float) -> bool:
        """Is the clip that is PLAYING no longer the one the planner wants?

        Asked every step while a clip plays, so a walk can end when the human
        stops rather than when the keyframes run out.

        The comparison has to be against the playing action, not against None.
        Asking only "does the planner want ANY clip?" conflated two different
        answers, and the difference is a robot that will not stop: plan_action
        considers turns FIRST and returns a turn action whenever the heading error
        clears the gate, so a pending turn kept reporting "yes, a clip is wanted"
        while the FORWARD clip was the one actually playing. Walking forward does
        not reduce a heading error, so the condition never cleared and the walk
        ran on until something else ended it.

        Measured in log 1788784412/1788784096 (the session the user reported):
        after the cue went idle and the walk latch expired, the forward clip kept
        cycling for a further 4.16 s -- four more strides, 0.37 m -- and replaying
        the real plan_action over the logged yaw_error for that window returns
        "turn_right" in 207 of its 422 frames. Trailing walk after the human
        stopped, across the four runs in that session: 3.36, 7.68, 4.40, 3.08 s.

        Returning True when a DIFFERENT clip is wanted is also what lets a turn
        interrupt a walk at all: the walk is released here, and branch 2 then
        plans and starts the turn on the next step.
        """
        if self.leg_control != "auto":
            return False
        return self._wanted_action(gait, yaw) != self.motion.action

    def _ready_to_play(self, now: float, action: str) -> bool:
        """Are the legs in the posture playback of ``action`` will start in?

        Ramps them there if not (see CLIP_PREPARE_TIMEOUT_S). Returns True when
        the clip may be played -- immediately, if the clip does not tell us what
        it opens in, because refusing to walk at all is worse than a jerky start.

        For a cyclic clip the target is the pose at ``enter_s`` rather than the
        first keyframe: this ramp IS the clip's opening squat, done better, so
        playback skips it. It is also a shallower crouch than the clip's own
        opening (knee 1.036 against 1.222 rad), so the handover is gentler than
        it was for one-shot playback, not harsher.
        """
        pose = self.motion.entry_pose(action)
        if not pose:
            return True
        if self._preparing != action:
            # Restart the per-ACTION clock, but NOT the clock that bounds how
            # long the legs may be held in a ramp overall. _prepare_since used
            # to be reset here too, which handed CLIP_PREPARE_TIMEOUT_S a clock
            # that a dithering planner could rewind forever: measured on the
            # 2026-09-08 session, one episode sat in prepare for 9.50 s -- 480
            # ticks, zero clips played, 6 flips between prepare:forward and
            # prepare:turn_left -- while the longest stretch on any SINGLE
            # action was 2.67 s, so the 2.5 s timeout never once fired. The legs
            # stood in a one-footed ramp for 9.5 s and the episode ended in a
            # fall. The run clock below is only cleared by _abandon_prepare or
            # by actually reaching the stance.
            self._preparing = action
            if self._prepare_run_since is None:
                self._prepare_run_since = now
            self._prepare_since = now
            self.driver.release_leg_pose()
            logger.info("Preparing to %s: ramping the legs into the stance the "
                        "clip is entered in (knee %.2f rad, at %.2fs into the "
                        "clip).", action, pose.get("LKneePitch", float("nan")),
                        self.motion.entry_time_s(action))
        if self.driver.approach_leg_pose(pose, now):
            self._clip_status = "ready (in the clip's stance)"
            self._prepare_run_since = None
            return True
        # ``x or now`` would be wrong here: these are TIMES, and 0.0 is falsy, so
        # a clock legitimately stamped at t=0 read as "unset" and the elapsed
        # time collapsed to zero -- disabling the timeout entirely. That is not
        # hypothetical: simulationReset() zeroes robot.getTime(), and this
        # controller resets on every fall (AUTO_RELOAD_ON_FALL), so the first
        # prepare after any fall had no timeout at all.
        waited = now - (self._prepare_since if self._prepare_since is not None else now)
        # Bound BOTH clocks: the per-action one (this clip is not reachable) and
        # the run one (we have been ramping the legs for too long in total, no
        # matter how many times the planner changed its mind).
        ramping_for = now - (
            self._prepare_run_since if self._prepare_run_since is not None else now)
        if waited >= CLIP_PREPARE_TIMEOUT_S or ramping_for >= CLIP_PREPARE_RUN_TIMEOUT_S:
            logger.warning(
                "Could not reach %s's opening stance (%.1fs on this action, "
                "%.1fs ramping in total); a leg joint is blocked or the planner "
                "is dithering. Abandoning this clip.", action, waited, ramping_for)
            self._abandon_prepare()
            self._end_motion(action, ok=False, reason="could not reach the stance")
            return False
        self._clip_status = f"preparing ({waited:.1f}s)"
        return False

    def _abandon_prepare(self) -> None:
        """Stop ramping toward a clip stance (nothing is being played)."""
        if self._preparing is None:
            return
        self._preparing = None
        self._prepare_since = None
        self._prepare_run_since = None
        self.driver.release_leg_pose()
        if self.driver.lower_body is not None:
            # The legs are wherever the ramp left them; let the crouch come back
            # down from there rather than snapping.
            seed = getattr(self.driver.lower_body, "seed_crouch_from", None)
            if callable(seed):
                seed(self.driver.measured)

    def _settled(self, roll: float, pitch: float) -> bool:
        """Is the robot upright and calm enough to hand over to a clip?

        The allowed tilt window is not fixed: it shrinks continuously from
        MOTION_START_MAX_TILT_RAD toward MOTION_START_MIN_TILT_RAD as recent
        tilt risk (self._tilt_risk) climbs toward TILT_ABORT_RAD, so the
        controller keeps declining to start another clip for a while after a
        wobble instead of only reacting once mid-clip.
        """
        risk_frac = max(0.0, min(1.0, self._tilt_risk / TILT_ABORT_RAD))
        ceiling = MOTION_START_MAX_TILT_RAD - risk_frac * (
            MOTION_START_MAX_TILT_RAD - MOTION_START_MIN_TILT_RAD
        )
        if abs(roll) > ceiling or abs(pitch) > ceiling:
            return False
        d_roll, d_pitch = self._tilt_rate()
        return abs(d_roll) < 1.0 and abs(d_pitch) < 1.0

    def _begin_motion(self, now: float) -> None:
        """Arm the watchdog for a clip we are about to hand the body to."""
        self._motion_started_at = now
        # Prefer the clip's own length (plus slack for Webots' interpolation);
        # fall back to the hard cap when the API will not tell us.
        # The clip's OWN length decides the budget when it is knowable, with no
        # ceiling: TurnLeft180 runs 9.0 s, and capping the budget at
        # MOTION_WATCHDOG_S guaranteed it overran, was dropped as broken and
        # counted a failure -- so the one clip that can turn the robot right
        # round in a single action could never be used. The fixed cap is for
        # clips whose duration Webots will not report.
        duration = self.motion.duration_s()
        budget = (duration * 1.5 + 1.0) if duration else MOTION_WATCHDOG_S
        self._motion_deadline = now + budget

    def _clip_phase(self) -> float | None:
        """How far into the current gait cycle playback is, in seconds."""
        cycle = self.motion.cycle()
        now = self.motion.time_s()
        if cycle is None or now is None:
            return None
        return max(0.0, now - cycle.loop_start_s)

    def _extend_motion_deadline(self, now: float) -> None:
        """Credit a cycling clip with another period's worth of watchdog.

        The watchdog exists to catch a clip that never reports itself over, which
        would leave the whole lower body suspended forever. A cyclic clip is in
        exactly that state BY DESIGN -- it is rewound before it can end -- so
        without this it would be killed mid-walk after its own length. Each
        completed stride is proof of life, and buys one more stride plus the
        deceleration tail; if rewinding ever stops happening, the deadline
        arrives as it always did.
        """
        if self._motion_deadline is None:
            return
        cycle = self.motion.cycle()
        if cycle is None:
            return
        self._motion_deadline = max(
            self._motion_deadline,
            now + cycle.period_s * 1.5 + cycle.tail_s + 1.0,
        )

    def _motion_overran(self, now: float) -> bool:
        return self._motion_deadline is not None and now > self._motion_deadline

    def _end_motion(self, action: str | None, *, ok: bool, reason: str) -> None:
        """Take the body back from a clip and update the locomotion health count."""
        self.motion.abort()
        self._motion_started_at = None
        self._motion_deadline = None
        # The cycle status belongs to a clip that is no longer playing. Leaving it
        # set made cycle_state read "leaving" in 92% of the frames of log
        # 1788784412 while a clip was playing in only 8% of them, which is a
        # diagnostic that actively misleads -- and analyze_run.py's own
        # check_gait_cycle reads this column.
        self._cycle_state = ""
        self._reclaim()
        if ok:
            self._motion_failures = 0
            return
        self._motion_failures += 1
        logger.warning("Locomotion attempt '%s' ended badly (%s): failure %d/%d",
                       action, reason, self._motion_failures, MOTION_MAX_FAILURES)
        if self._motion_failures >= MOTION_MAX_FAILURES:
            logger.error(
                "Disabling locomotion clips for this session after %d bad "
                "attempts. The robot will keep imitating your pose and will "
                "march in place instead of walking. Check WorldInfo "
                "contactProperties and basicTimeStep (see the controller README).",
                self._motion_failures,
            )
            self.motion.drop(None)

    def _reclaim(self) -> None:
        """Take the body back after a clip and reset the leg sequencers."""
        self.driver.reclaim_from_motion()
        if self.driver.lower_body is not None:
            self.driver.lower_body.reset()
            # AFTER the reset, which clears the crouch limiter: the clip left the
            # legs in its own ~0.51 rad squat and the standing depth is 0.10, so a
            # cleared limiter would snap the whole 0.41 rad of knee travel on the
            # first post-clip step. Seeding it makes that a ramp.
            seed = getattr(self.driver.lower_body, "seed_crouch_from", None)
            if callable(seed):
                seed(self.driver.measured)

    # ---------------------------------------------------------------- logging
    def _diagnostics(self, ctl_roll: float, ctl_pitch: float, yaw: float) -> dict[str, object]:
        """One row of controller state for the trajectory log.

        Cheap by design -- everything here is already computed for this step,
        except the support margin, which is one forward-kinematics pass and is the
        single most useful number for telling "the robot is standing badly" from
        "the robot is standing fine and the pose is wrong".
        """
        d_roll, d_pitch = self._tilt_rate()
        m = self.driver.lower_body_meta
        gait = self.gait_cmd or {}
        action = self.action_cmd or {}
        fsr = self._read_fsr() or {}
        raw_roll, raw_pitch, _ = self._imu_rpy()
        zero = self._imu_zero
        # imu_* is the INERTIAL UNIT's own zeroed reading; ctl_* is what the
        # controller acted on (see the DIAGNOSTIC_COLUMNS note).
        roll, pitch = self._corrected_tilt(raw_roll, raw_pitch)
        out: dict[str, object] = {
            "imu_roll": roll, "imu_pitch": pitch, "imu_yaw": yaw,
            "ctl_roll": ctl_roll, "ctl_pitch": ctl_pitch,
            "imu_roll_raw": raw_roll, "imu_pitch_raw": raw_pitch,
            "imu_zero_roll": None if zero is None else zero[0],
            "imu_zero_pitch": None if zero is None else zero[1],
            "acc_roll": None if self._acc_tilt is None else self._acc_tilt[0],
            "acc_pitch": None if self._acc_tilt is None else self._acc_tilt[1],
            "tilt_source": self._tilt_source,
            "gyro_roll_rate": d_roll, "gyro_pitch_rate": d_pitch,
            "tilt_risk": self._tilt_risk,
            "predicted_tilt": self._predicted_tilt_rad(ctl_roll, ctl_pitch),
            "leg_mode": self.leg_mode,
            "stale": int(bool(self.driver.stats.stale)),
            "lb_mode": m.get("mode"),
            "lb_shift": m.get("shift"),
            "lb_lift": m.get("lift"),
            "lb_gate": m.get("gate"),
            "lb_stance_margin": m.get("stance_margin"),
            "lb_lean_scale": m.get("lean_scale"),
            "lb_crouch_u": m.get("crouch_u"),
            "lb_crouch_cue": m.get("crouch_cue"),
            "lb_conf": m.get("confidence"),
            "lb_lift_source": m.get("lift_source"),
            "lb_rejected": m.get("rejected"),
            # Quoted-safe: the CSV writer escapes it, and it is the one field that
            # names the limiting factor in words.
            "lb_why": m.get("why"),
            "lb_ff_pitch": m.get("com_shift_pitch"),
            "lb_ff_roll": m.get("com_shift_roll"),
            "lb_ff_width": m.get("com_shift_width"),
            "lb_fb_pitch": m.get("com_fb_pitch"),
            "lb_fb_roll": m.get("com_fb_roll"),
            "lb_com_margin": m.get("com_margin"),
            "cop_share_l": (
                fsr["L"] / (fsr["L"] + fsr["R"])
                if fsr.get("L") is not None and fsr.get("R") is not None
                and (fsr["L"] + fsr["R"]) > 1.0 else None
            ),
            "head_height": self._head_height,
            "reloads": self._reloads,
            "clip_planned": self._clip_planned,
            "sv_heading": self._heading(yaw),
            "clip_status": ("playing" if self.motion.active else self._clip_status),
            "clips_available": len(self.motion.available),
            "clip_time": self.motion.time_s(),
            "walk_latched": int(self._walk_latch_until is not None),
            "early_exits": self._early_exits,
            "tails_trimmed": self._tails_trimmed,
            "clip_cycles": self._cycles_walked,
            "clip_phase": self._clip_phase(),
            "cycle_state": self._cycle_state,
            "yaw_stable": int(bool(self.yaw_servo.stable())),
            "yaw_error": self.yaw_servo.error(yaw),
            "yaw_latched": int(bool(self.yaw_servo.latched)),
            "fsr_l": fsr.get("L"), "fsr_r": fsr.get("R"),
            "act_action": action.get("action"),
            "act_conf": action.get("conf"),
            "act_forward_mps": action.get("forward_mps"),
            "act_lateral_mps": action.get("lateral_mps"),
            "act_yaw_rate": action.get("yaw_rate_dps"),
            "act_crouch": action.get("crouch"),
            "act_lift": action.get("lift"),
            "act_observed": action.get("observed"),
            "act_reason": action.get("reason"),
            "gait_state": gait.get("state"),
            "gait_cadence": gait.get("cadence_hz"),
            "gait_conf": gait.get("conf"),
            "body_yaw": gait.get("body_yaw_rad"),
            # Which cue declared the walk: "knee" (marching on the spot) or
            # "stride" (actually walking). Without it, "the robot did not walk"
            # cannot be told from "the human was not seen to walk".
            "gait_cue_channel": gait.get("cue_channel"),
        }
        out.update(self._ground_truth())
        balance = self.driver.balance
        if balance is not None:
            try:
                state = dict(self.driver.commanded)
                state.update(self.driver.measured)
                mx, my = balance.model.support_margins(state)
                out["support_margin_x"] = mx
                out["support_margin_y"] = my
            except Exception:  # noqa: BLE001 - diagnostics must never break control
                pass
        return out

    def _log_status(self) -> None:
        if self.frame_count % STATUS_EVERY != 0:
            return
        elapsed = time.time() - self._last_log_time
        fps = STATUS_EVERY / elapsed if elapsed > 0 else 0.0
        stats = self.driver.stats
        logger.info(
            "Frame %d | sim %.1f Hz | %s | legs=%s | %d joints applied",
            self.frame_count, fps,
            "STALE (holding)" if stats.stale else "tracking",
            self.leg_mode, stats.joints_last_applied,
        )
        if self.leg_mode == "pose":
            m = self.driver.lower_body_meta
            logger.info("  legs: %s", m.get("why", "?"))
            logger.info(
                "        mode=%s stance=%s shift=%.2f lift=%.2f gate=%.2f "
                "margin=%+.4fm crouch=%.2f lift-cue=%s",
                m.get("mode"), m.get("stance_side") or "-", float(m.get("shift", 0.0)),
                float(m.get("lift", 0.0)), float(m.get("gate", 0.0)),
                float(m.get("stance_margin", 0.0)), float(m.get("crouch_u", 0.0)),
                m.get("lift_source", "?"),
            )
        elif self.leg_mode.startswith("march"):
            m = self.driver.gait_meta
            logger.info(
                "  march amp=%.2f cadence=%.2fHz phase=%.2f single_support=%s",
                float(m.get("amp_gain", 0.0)), float(m.get("cadence", 0.0)),
                float(m.get("phase", 0.0)), m.get("single_support", False),
            )
        if self.motion.available:
            logger.info("  locomotion: %d clips | planned=%s | %s | yaw %s",
                        len(self.motion.available), self._clip_planned or "-",
                        self._clip_status,
                        "steady" if self.yaw_servo.stable() else "TOO NOISY to turn on")
            d = self.yaw_servo.diagnostics(self._imu_rpy()[2])
            # The reference pair is logged alongside the error on purpose: an
            # error alone cannot distinguish a loop that is converging from one
            # stuck at a fixed offset, and a fixed offset is what a recorded
            # session actually showed.
            logger.info(
                "  heading: error %+.0f deg | human %+.0f -> %+.0f | robot %+.0f "
                "-> %+.0f | servo %s",
                math.degrees(d["error"]),
                math.degrees(d["human_ref"]), math.degrees(d["human_now"]),
                math.degrees(d["robot_ref"]), math.degrees(d["robot_now"]),
                "latched" if self.yaw_servo.latched else "waiting to latch",
            )
        # Repeated, not printed once at startup: a silently absent balance layer
        # is the single most expensive failure mode this controller has.
        if self._imu_zero is None:
            logger.warning(
                "  IMU tilt zero not yet learned (%d standing samples, need %d): "
                "tilt is reported as level, so the balance loop and every tilt "
                "gate are idle. Is the robot standing with its feet loaded?",
                len(self._imu_cal), IMU_CALIBRATION_MIN_SAMPLES,
            )
        if self.driver.balance is not None:
            warning = self.driver.balance.diverging()
            if warning:
                logger.error("  %s", warning)
        if self._head_height is not None:
            logger.info("  head %.3f m above the soles (fall below %.2f); "
                        "recoveries this session: %d",
                        self._head_height, FALL_HEAD_HEIGHT_M, self._reloads)
        for reason in self.driver.degraded:
            logger.error("  DEGRADED: %s", reason)
        for name in self.driver.stuck_motors():
            logger.error(
                "Motor '%s' is not tracking its command (avg err %.3f rad). If "
                "this persists the joint is mechanically blocked -- most likely "
                "against the torso or the opposite limb.",
                name, self.driver.health.average_error(name),
            )
        self._last_log_time = time.time()

    # ------------------------------------------------------------------- loop
    def tick(self) -> None:
        """One control step: ingest, sense, arbitrate, log.

        Kept separate from :meth:`run` so the whole per-step path can be driven
        from a test harness without reimplementing it -- a duplicated loop body
        is a loop body that drifts out of sync with the real one.
        """
        now = self.robot.getTime()
        command = self._drain_latest_command()
        if command is not None:
            # Prefer full-body retargeting from raw landmarks; fall back to
            # pre-computed joint angles if only those were sent.
            keypoints = command.get("keypoints")
            if keypoints:
                self.driver.update_from_keypoints(keypoints, now_s=now)
            else:
                angles = command.get("joint_angles_rad", {})
                if angles:
                    self.driver.update(angles, now_s=now)
            self.gait_cmd = command.get("gait")
            # What the human is DOING, as opposed to the rhythm of a proxy
            # signal. See src/perception/action_cues.py; plan_action prefers it
            # over the gait cue when it has an opinion.
            self.action_cmd = command.get("action")
            self.driver.set_gait_command(self.gait_cmd)
        elif self.driver.check_stale(now):
            # Tracking lost: tell every layer to stand down. They ramp back to
            # the balanced crouch rather than freezing mid-step. The lower body
            # has to be told explicitly: it latches the last observation so it
            # can control at simulation rate between camera frames, and without
            # an expiry it would hold a one-legged stance long after the human
            # walked away.
            self.gait_cmd = None
            self.action_cmd = None
            self.driver.set_gait_command(None)
            self.driver.lower_body_stand_down()
            # The arms and head need telling too, or they hold the departed
            # human's last pose indefinitely (see upper_body_stand_down).
            self.driver.upper_body_stand_down()
            # A human who has walked out of frame is not walking, whatever the
            # last cue said. Without this the latch would keep the robot going.
            self._drop_walk_latch()

        # Advance the arms every step, not only on the steps that carried a
        # camera frame: the camera runs at ~12 Hz against the simulation's 50,
        # so the arm used to hold for four steps and jump on the fifth.
        self.driver.tick_arms(now)

        self.driver.read_feedback()
        raw_roll, raw_pitch, yaw = self._imu_rpy()
        fsr = self._read_fsr()
        self._calibrate_imu(now, raw_roll, raw_pitch, fsr)
        roll, pitch = self._torso_tilt(now, *self._corrected_tilt(raw_roll, raw_pitch), fsr)
        if self._fallen(now, roll, pitch, fsr) and self._recover_from_fall(now):
            return
        self._update_tilt_risk(now, roll, pitch)
        if self.driver.balance is not None:
            self.driver.balance.note_tilt(now, roll, pitch)
        heading = self._heading(yaw)
        if self.heading_available:
            self._update_yaw_servo(now, heading)
        self._drive_legs(now, roll, pitch, heading, fsr=fsr)

        if self.trajectory_log is not None:
            self.trajectory_log.record(
                now, self.frame_count, self.driver.commanded, self.driver.measured,
                self._diagnostics(roll, pitch, yaw),
            )
        self._log_status()
        self.frame_count += 1

    def run(self) -> None:
        """Step the simulation, containing errors so one bad step cannot end it.

        A raised exception used to break out of this loop straight into
        :meth:`_cleanup`, which sets every motor velocity to zero -- a single
        transient error left a permanently dead robot with no obvious cause. Now
        a failed step is logged, the body is forced back under our control (in
        case the failure happened mid-handover to a motion clip), and the loop
        carries on. Only a sustained run of failures gives up, and even then the
        robot is left standing rather than limp.
        """
        logger.info("Starting control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                try:
                    self.tick()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._errors += 1
                    logger.exception(
                        "Control step %d failed (%d consecutive): %s",
                        self.frame_count, self._errors, exc,
                    )
                    self._recover_from_error()
                    if self._errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            "Giving up after %d consecutive failed steps.",
                            self._errors,
                        )
                        break
                    self.frame_count += 1
                else:
                    self._errors = 0
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down")
        finally:
            self._cleanup()

    def _recover_from_error(self) -> None:
        """Best-effort return to a known-good state after a failed step."""
        try:
            if self.motion.active or self.driver.suspended:
                self._end_motion(self.motion.action, ok=False, reason="step error")
        except Exception:  # noqa: BLE001 - recovery must never raise
            logger.exception("Recovery itself failed; forcing control back")
            try:
                self.driver.reclaim_from_motion()
            except Exception:  # noqa: BLE001
                pass

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        try:
            self.motion.abort()
            self.driver.reclaim_from_motion()
            self.driver.stop()
        finally:
            if self.trajectory_log is not None:
                self.trajectory_log.close()
            if hasattr(self, "sock"):
                self.sock.close()
            logger.info("Controller stopped after %d frames", self.frame_count)


def _sensor_magnitude(device: object) -> float:
    """Per-foot load from a Webots TouchSensor, 3-axis or 1-axis, NaN-safe.

    NAO's ``LFsr``/``RFsr`` are ``TouchSensor`` nodes of type ``"force-3d"``, so
    they answer to ``getValues()`` and return ``[fx, fy, fz]``; ``getValue()``
    (singular) only supports the ``"bumper"``/``"force"`` types and Webots raises
    on it. That mismatch is why an earlier version silently got no load
    information at all and the step gate never saw a weight transfer.

    Of the three axes we take **fz**: the load bearing on a foot is the vertical
    force, and the shear components can be large during a walk and would inflate
    the reading exactly when the step gate needs it to be honest. Shorter vectors
    fall back to the norm, and 1-axis sensors to ``getValue()``, so other NAO
    protos still work.
    """
    getter = getattr(device, "getValues", None)
    if getter is not None:
        try:
            values = getter()
            if values is not None:
                fz = float(values[2])
                return abs(fz) if math.isfinite(fz) else 0.0
        except Exception:  # noqa: BLE001
            pass
    try:
        value = float(device.getValue())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0.0
    return abs(value) if math.isfinite(value) else 0.0


def main() -> None:
    try:
        PoseImitationController().run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
