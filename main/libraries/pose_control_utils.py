"""
Utilities for pose imitation control of the Webots NAO humanoid.

This module is the heart of the Webots side of the pipeline. It converts the
*generic* joint angles produced by the Python retargeting stage into commands
that are correct for the **NAO (H25)** robot, then drives the motors smoothly
and keeps the robot standing.

Why a mapping layer is needed
-----------------------------
The Python retargeting module (`src/retargeting/mapper.py`) emits angles using a
neutral convention that does *not* match NAO's joint conventions:

* NAO ``LElbowRoll`` is **negative** (-1.5446 .. -0.0349 rad) and ``RElbowRoll``
  is **positive** (0.0349 .. 1.5446 rad). The pipeline sends the opposite signs,
  so without correction ``Motor.setPosition`` clamps both elbows straight and
  they never bend.
* NAO ``ShoulderPitch`` uses arm-down = +1.57 rad; the pipeline sends arm-down
  ≈ -1.57 rad (inverted).
* NAO has **no** ``TorsoPitch`` motor, so that channel is dropped.

``NaoPoseDriver`` applies a per-joint affine correction (``scale``/``offset``),
clamps to the real NAO mechanical limits (FR-5), exponentially smooths the
targets to prevent oscillation (FR-6), and holds a stable standing posture so
the robot does not fall during upper-body imitation (NFR-4).

Who commands the legs
---------------------
Only ONE layer may command the 12 leg joints at a time, or they fight each other
and the robot falls. The driver exposes exactly one entry point per layer and the
controller picks between them each step:

* :meth:`NaoPoseDriver.lower_body_tick` -- per-leg pose imitation with the
  weight-shift/lift sequencer (``lower_body.LowerBodyController``). The default:
  this is what makes squatting and raising a single leg work.
* :meth:`NaoPoseDriver.gait_tick` -- the in-place march engine
  (``gait.GaitEngine``), used when the human is walking but no pre-balanced
  Webots walk clip is available to translate with.
* :meth:`NaoPoseDriver.balance_tick` -- CoM balance only (legs otherwise static).
* :meth:`NaoPoseDriver.release_to_motion` -- hands the WHOLE body to a Webots
  ``Motion`` clip: per-joint commanding stops and the motor velocity caps are
  lifted, because a velocity-capped motor cannot follow a motion clip's keyframes
  and the "walk" degenerates into a stumble.

This module deliberately does **not** import the Webots ``controller`` package,
so the math (limits, mapping, smoothing) stays unit-testable off-simulation.
Motor/sensor objects are passed in from the controller process.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# NAO H25 joint limits (radians)
#
# Source: Aldebaran/SoftBank NAO H25 joint documentation and the Webots
# Nao.proto RotationalMotor min/maxPosition values. These are the *hardware*
# ranges; Webots clamps setPosition() to them, so commanding outside the range
# silently saturates the joint.
# ---------------------------------------------------------------------------
NAO_JOINT_LIMITS: dict[str, MotorConfig] = {}


@dataclass
class MotorConfig:
    """Mechanical configuration for a single NAO motor."""
    name: str
    min_angle: float          # rad
    max_angle: float          # rad
    max_velocity: float       # rad/s (hardware ceiling)
    rest_angle: float = 0.0   # rad, neutral/standing default


def _deg(d: float) -> float:
    return math.radians(d)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def get_default_motor_configs() -> dict[str, MotorConfig]:
    """Return mechanical configs for every NAO joint we care about.

    ``rest_angle`` encodes a stable standing posture: legs straight (0 rad),
    arms hanging slightly away from the torso so they do not self-collide.
    """
    cfgs = [
        # Head
        MotorConfig("HeadYaw",        _deg(-119.5), _deg(119.5), 8.27, 0.0),
        MotorConfig("HeadPitch",      _deg(-38.5),  _deg(29.5),  7.19, 0.0),

        # Left arm
        MotorConfig("LShoulderPitch", _deg(-119.5), _deg(119.5), 8.27, _deg(85)),
        MotorConfig("LShoulderRoll",  _deg(-18.0),  _deg(76.0),  7.19, _deg(10)),
        MotorConfig("LElbowYaw",      _deg(-119.5), _deg(119.5), 8.27, _deg(-70)),
        MotorConfig("LElbowRoll",     _deg(-88.5),  _deg(-2.0),  7.19, _deg(-30)),
        MotorConfig("LWristYaw",      _deg(-104.5), _deg(104.5), 24.6, 0.0),

        # Right arm (note the mirrored roll signs)
        MotorConfig("RShoulderPitch", _deg(-119.5), _deg(119.5), 8.27, _deg(85)),
        MotorConfig("RShoulderRoll",  _deg(-76.0),  _deg(18.0),  7.19, _deg(-10)),
        MotorConfig("RElbowYaw",      _deg(-119.5), _deg(119.5), 8.27, _deg(70)),
        MotorConfig("RElbowRoll",     _deg(2.0),    _deg(88.5),  7.19, _deg(30)),
        MotorConfig("RWristYaw",      _deg(-104.5), _deg(104.5), 24.6, 0.0),

        # Fingers. Webots' NAO exposes eight phalanx motors per hand and no
        # single hand motor, so a grip is commanded by writing all eight. The
        # range below is the documented 0..0.96 rad, but it is NOT relied on:
        # _setup_devices asks each motor for its own getMinPosition() /
        # getMaxPosition() and adopts what it reports, because Nao.proto is an
        # EXTERNPROTO fetched at world-load time and is not on disk to check.
        # A model without fingers simply reports them missing and the grip
        # channel goes quiet.
        *[
            MotorConfig(f"{side}Phalanx{i}", 0.0, 0.96, 8.26, 0.96)
            for side in ("L", "R")
            for i in range(1, 9)
        ],

        # Left leg
        MotorConfig("LHipYawPitch",   _deg(-65.6),  _deg(42.4),  4.16, 0.0),
        MotorConfig("LHipRoll",       _deg(-21.7),  _deg(45.3),  4.16, 0.0),
        MotorConfig("LHipPitch",      _deg(-88.0),  _deg(27.7),  6.40, 0.0),
        MotorConfig("LKneePitch",     _deg(-5.3),   _deg(121.0), 6.40, 0.0),
        MotorConfig("LAnklePitch",    _deg(-68.2),  _deg(52.9),  6.40, 0.0),
        MotorConfig("LAnkleRoll",     _deg(-22.8),  _deg(44.1),  4.16, 0.0),

        # Right leg
        MotorConfig("RHipYawPitch",   _deg(-65.6),  _deg(42.4),  4.16, 0.0),
        MotorConfig("RHipRoll",       _deg(-45.3),  _deg(21.7),  4.16, 0.0),
        MotorConfig("RHipPitch",      _deg(-88.0),  _deg(27.7),  6.40, 0.0),
        MotorConfig("RKneePitch",     _deg(-5.9),   _deg(121.5), 6.40, 0.0),
        MotorConfig("RAnklePitch",    _deg(-67.9),  _deg(53.4),  6.40, 0.0),
        MotorConfig("RAnkleRoll",     _deg(-44.1),  _deg(22.8),  4.16, 0.0),
    ]
    configs = {c.name: c for c in cfgs}
    NAO_JOINT_LIMITS.clear()
    NAO_JOINT_LIMITS.update(configs)
    return configs


# Build the module-level table on import.
get_default_motor_configs()


# ---------------------------------------------------------------------------
# Pipeline -> NAO joint mapping
# ---------------------------------------------------------------------------
@dataclass
class JointMap:
    """Affine correction from a pipeline joint angle to a NAO motor target.

    ``nao_target = scale * pipeline_angle + offset`` (then clamped to limits).
    """
    nao_name: str
    scale: float = 1.0
    offset: float = 0.0
    is_leg: bool = False      # gated behind drive_legs for balance safety


# The pipeline emits these keys (see src/retargeting/mapper.py):
#   LShoulderPitch, RShoulderPitch, LElbowRoll, RElbowRoll,
#   LHipPitch, RHipPitch, TorsoPitch
#
# Corrections:
#   * ShoulderPitch: pipeline arm-down ≈ -1.57, NAO arm-down = +1.57  -> scale -1
#   * ElbowRoll: pipeline left is positive / right negative; NAO is the
#     opposite sign for each side                                     -> scale -1
#   * Hips: same axis sense, gated behind drive_legs                  -> scale +1
#   * TorsoPitch: no NAO motor                                        -> omitted
PIPELINE_TO_NAO: dict[str, JointMap] = {
    "LShoulderPitch": JointMap("LShoulderPitch", scale=-1.0),
    "RShoulderPitch": JointMap("RShoulderPitch", scale=-1.0),
    "LElbowRoll":     JointMap("LElbowRoll",     scale=-1.0),
    "RElbowRoll":     JointMap("RElbowRoll",     scale=-1.0),
    "LHipPitch":      JointMap("LHipPitch",      scale=1.0, is_leg=True),
    "RHipPitch":      JointMap("RHipPitch",      scale=1.0, is_leg=True),
}


# ---------------------------------------------------------------------------
# Joint limiting / smoothing helpers (pure math, unit-testable)
# ---------------------------------------------------------------------------
class JointLimiter:
    """Enforces joint angle limits."""

    def __init__(self, configs: dict[str, MotorConfig]) -> None:
        self.configs = configs

    def clamp_angle(self, joint_name: str, angle: float) -> float:
        cfg = self.configs.get(joint_name)
        if cfg is None:
            return angle
        return max(cfg.min_angle, min(cfg.max_angle, angle))

    def is_within_limits(self, joint_name: str, angle: float) -> bool:
        cfg = self.configs.get(joint_name)
        if cfg is None:
            return True
        return cfg.min_angle <= angle <= cfg.max_angle


class ExponentialSmoother:
    """Per-joint exponential moving average to damp jitter (FR-6).

    ``alpha`` in (0, 1]; higher = more responsive, lower = smoother. A per-call
    ``alpha`` override lets ONE smoother serve channels that want different
    responsiveness, which matters more than it sounds: the driver used to keep two
    smoothers holding independent state for the SAME twelve leg joints -- the
    balance path smoothed them at 0.4 and the gait/pose paths at 0.7 -- so every
    time the arbiter switched layer the leg targets jumped to whatever the other
    smoother happened to remember. On weight-bearing joints that is exactly the
    jolt ``leg_velocity_factor`` exists to prevent.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._state: dict[str, float] = {}

    def reset(self, joint_name: str, value: float) -> None:
        self._state[joint_name] = value

    def smooth(self, joint_name: str, target: float,
               alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else max(0.0, min(1.0, alpha))
        prev = self._state.get(joint_name)
        value = target if prev is None else prev + (target - prev) * a
        self._state[joint_name] = value
        return value


# Arm joints tracked by :class:`ArmTracker`: everything from the shoulder out.
# Explicitly NOT the head (its solve has its own per-subject calibration and its
# own neutral) and NOT the legs (they carry the robot's weight, and a predicted
# overshoot there is a fall, not a wobble).
TRACKED_ARM_JOINTS = (
    "LShoulderPitch", "RShoulderPitch",
    "LShoulderRoll", "RShoulderRoll",
    "LElbowYaw", "RElbowYaw",
    "LElbowRoll", "RElbowRoll",
    "LWristYaw", "RWristYaw",
)

# The two GRIP channels. These are not joints and not radians: each is a hand
# closure in [0, 1] (0 = open, 1 = fist) which the driver fans out across that
# hand's phalanx motors. Webots' NAO has no single "LHand" motor -- it exposes
# eight LPhalanx motors and eight RPhalanx ones -- so one commanded value per
# hand is the only sane interface for something the retargeter measures as a
# single thumb-to-finger distance.
HAND_GRIP_JOINTS = ("LHand", "RHand")
# Which end of a phalanx motor's travel is an OPEN hand. Webots' NAO documents
# 0.96 rad as open and 0 as closed, but Nao.proto is an EXTERNPROTO fetched at
# world-load time and cannot be read from disk here to confirm it. If the robot
# grips when the human opens their hand, flip this one flag -- the actual
# numbers come from the motor itself (see _adopt_reported_limits), so nothing
# else needs touching.
HAND_OPEN_AT_MAX = True
PHALANX_JOINTS: dict[str, tuple[str, ...]] = {
    "LHand": tuple(f"LPhalanx{i}" for i in range(1, 9)),
    "RHand": tuple(f"RPhalanx{i}" for i in range(1, 9)),
}


class ArmTracker:
    """Frame-rate-independent, velocity-predicted tracking for the arm joints.

    Replaces the plain per-frame EMA on the arms, which cost more delay than
    anything else in the chain. Two separate faults, both measured on the
    2026-09-16 session (run_20260916_110404 against
    webots_joint_trajectory_1789548982, 428 s overlap, 19441 non-stale ticks;
    lag by cross-correlation of the human's own elbow-bend signal against the
    logged command):

    **1. The EMA's delay was set by the camera's frame rate, not by a number
    anyone chose.** ``_apply_targets`` runs once per UDP frame, so ``alpha =
    0.4`` at the measured 11.8 Hz was ``(1 - a) / a / rate = 127 ms`` of the
    160-180 ms measured between the (already-smoothed) pose log and the
    commanded angle. Had the pipeline dropped to 6 FPS that would have silently
    doubled to 250 ms. So the smoothing is specified as a TIME CONSTANT here and
    the per-step factor is derived from the actual elapsed time,
    ``a = 1 - exp(-dt / tau)``, which holds the response fixed whatever the
    frame rate does.

    **2. Nothing moved between camera frames.** The simulation steps at 50 Hz
    and the arm target only changed at 11.8 Hz, so the arm advanced in ~12
    visible increments a second with a hold in between. :meth:`advance` is
    called every tick and keeps integrating the last known target VELOCITY, so
    the motion stays continuous at simulation rate instead of stepping.

    The same velocity also buys back the delay that is left. The command aims at
    where the joint is predicted to be ``lead_s`` from the last observation
    rather than where it was measured to be, so the residual pipeline delay is
    cancelled instead of merely reduced. That is only safe because it is bounded
    three ways:

    * the extrapolation term is clamped to ``max_lead_rad``, so a single noisy
      landmark cannot fling the arm across its range;
    * the extrapolation runs at full confidence only for as long as the next
      camera frame is still due -- the inter-frame period is measured, not
      assumed -- and then fades out over ``max_extrapolate_s``, so a dropped
      frame coasts briefly and settles onto the last pose the human was
      actually seen in rather than one that was only predicted;
    * once it has faded there is nothing left to integrate, so a human who
      leaves the frame leaves a still arm, not a drifting one.

    Prediction is deliberately built on the RETARGETED joint target rather than
    on the landmarks: joint space is where the limits live, so a clamp here is
    a clamp in the units the motor actually understands.
    """

    def __init__(
        self,
        *,
        tau_s: float = 0.07,
        lead_s: float = 0.20,
        max_lead_rad: float = 0.35,
        max_extrapolate_s: float = 0.20,
        vel_tau_s: float = 0.10,
    ) -> None:
        self.tau_s = max(0.0, tau_s)
        self.lead_s = max(0.0, lead_s)
        self.max_lead_rad = max(0.0, max_lead_rad)
        self.max_extrapolate_s = max(0.0, max_extrapolate_s)
        self.vel_tau_s = max(1e-3, vel_tau_s)
        self._target: dict[str, float] = {}
        self._vel: dict[str, float] = {}
        self._obs_t: dict[str, float] = {}
        self._pos: dict[str, float] = {}
        self._frame_dt: dict[str, float] = {}

    def reset(self, name: str, value: float) -> None:
        """Seed every state for ``name`` at ``value``, stationary."""
        self._target[name] = value
        self._vel[name] = 0.0
        self._pos[name] = value
        self._obs_t.pop(name, None)
        self._frame_dt.pop(name, None)

    def forget(self, name: str) -> None:
        """Drop the velocity estimate, keeping the position. Used when the
        target is being commanded by something other than the imitation (a
        stand-down ramp), where the old velocity describes a motion that is no
        longer happening."""
        self._vel[name] = 0.0
        self._obs_t.pop(name, None)

    def observe(self, name: str, target: float, now_s: float | None) -> None:
        """Record one camera frame's retargeted value for ``name``."""
        prev = self._target.get(name)
        prev_t = self._obs_t.get(name)
        self._target[name] = target
        if now_s is None:
            self._obs_t.pop(name, None)
            return
        self._obs_t[name] = now_s
        if prev is None or prev_t is None:
            self._vel.setdefault(name, 0.0)
            return
        dt = now_s - prev_t
        # A zero or negative dt means two frames landed on one simulation step
        # (or the clock went backwards across a reset): there is no velocity to
        # read from it, and dividing by it would manufacture an enormous one.
        if dt <= 1e-6:
            return
        raw = (target - prev) / dt
        blend = 1.0 - math.exp(-dt / self.vel_tau_s)
        self._vel[name] = self._vel.get(name, 0.0) + (raw - self._vel.get(name, 0.0)) * blend
        # How long until the next frame is due. Measured rather than assumed:
        # the camera rate is adaptive (AdaptiveFPSController), so a fixed guess
        # would either cut the extrapolation short at low frame rates or keep
        # extrapolating through a real dropout at high ones.
        known = self._frame_dt.get(name)
        self._frame_dt[name] = dt if known is None else known + (dt - known) * 0.2

    def advance(self, name: str, now_s: float | None, dt_s: float) -> float | None:
        """Command for ``name`` this step, or ``None`` if it has never been seen."""
        target = self._target.get(name)
        if target is None:
            return None
        goal = target
        obs_t = self._obs_t.get(name)
        if now_s is not None and obs_t is not None:
            age = max(0.0, now_s - obs_t)
            # One expected frame period of full-confidence coasting. Inside it
            # the extrapolation is not a guess about a missing frame at all --
            # it is the continuation between two frames that are both going to
            # arrive, which is what keeps the arm moving at simulation rate.
            # Fading during THAT window would just re-introduce the lag the
            # lead exists to remove (measured: 22 mrad of residual error on a
            # 1 rad/s ramp, a fifth of the delay handed straight back).
            hold = self._frame_dt.get(name, 0.0)
            if age <= hold or self.max_extrapolate_s <= 0.0:
                fade = 1.0
            else:
                fade = _clamp(1.0 - (age - hold) / self.max_extrapolate_s, 0.0, 1.0)
            reach = min(age, hold + self.max_extrapolate_s)
            step = self._vel.get(name, 0.0) * (self.lead_s + reach) * fade
            goal = target + _clamp(step, -self.max_lead_rad, self.max_lead_rad)
        pos = self._pos.get(name)
        if pos is None:
            self._pos[name] = goal
            return goal
        a = 1.0 if self.tau_s <= 0.0 or dt_s <= 0.0 else 1.0 - math.exp(-dt_s / self.tau_s)
        pos += (goal - pos) * _clamp(a, 0.0, 1.0)
        self._pos[name] = pos
        return pos


class MotorHealthMonitor:
    """Tracks position-tracking error and flags stuck motors."""

    def __init__(self, max_position_error: float = 0.1, window: int = 100) -> None:
        self.max_position_error = max_position_error
        self.window = window
        self.position_errors: dict[str, list[float]] = {}

    def record(self, joint_name: str, target: float, current: float) -> float:
        error = abs(target - current)
        errs = self.position_errors.setdefault(joint_name, [])
        errs.append(error)
        if len(errs) > self.window:
            errs.pop(0)
        return error

    def average_error(self, joint_name: str) -> float:
        errs = self.position_errors.get(joint_name, [])
        return sum(errs) / len(errs) if errs else 0.0

    def is_stuck(self, joint_name: str) -> bool:
        errs = self.position_errors.get(joint_name, [])[-10:]
        if not errs:
            return False
        return (sum(errs) / len(errs)) > self.max_position_error * 2


def map_pipeline_angles(
    incoming: dict[str, float],
    *,
    drive_legs: bool = False,
    limiter: JointLimiter | None = None,
) -> dict[str, float]:
    """Convert pipeline joint angles to clamped NAO motor targets.

    Pure function (no Webots dependency) so the mapping is unit-testable.
    Unknown joints and (when ``drive_legs`` is False) leg joints are dropped.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    out: dict[str, float] = {}
    for src, value in incoming.items():
        spec = PIPELINE_TO_NAO.get(src)
        if spec is None:
            continue
        if spec.is_leg and not drive_legs:
            continue
        target = spec.scale * float(value) + spec.offset
        out[spec.nao_name] = limiter.clamp_angle(spec.nao_name, target)
    return out


# ---------------------------------------------------------------------------
# Standing posture
# ---------------------------------------------------------------------------
# Joints that imitation actively drives. Everything else is held at its
# rest_angle for a stable, natural-looking standing pose.
DRIVEN_ARM_JOINTS = ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "RElbowRoll")
# Lower body: the symmetric, statically-balanced crouch axis (hip/knee/ankle
# pitch). This is the posture every leg layer decays back to; see
# ``nao_retarget.crouch_posture``.
DRIVEN_LEG_JOINTS = (
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
)
# Every joint the lower-body layers may command. Used for the gentler leg
# velocity cap: these carry the robot's weight, so a jolt here is a fall.
ALL_LEG_JOINTS = DRIVEN_LEG_JOINTS + (
    "LHipYawPitch", "RHipYawPitch",
    "LHipRoll", "RHipRoll",
    "LAnkleRoll", "RAnkleRoll",
)


def standing_posture() -> dict[str, float]:
    """Return the neutral standing target for every NAO joint (radians).

    Legs are kept straight (0 rad) and stiff so the robot stays balanced
    during upper-body imitation (NFR-4).
    """
    return {name: cfg.rest_angle for name, cfg in NAO_JOINT_LIMITS.items()}


def _null_logger(_msg: str) -> None:  # pragma: no cover - default sink
    pass


# ---------------------------------------------------------------------------
# NaoPoseDriver — owns the Webots motors and applies pose frames
# ---------------------------------------------------------------------------
@dataclass
class DriverStats:
    frames_applied: int = 0
    joints_last_applied: int = 0
    stale: bool = False


class NaoPoseDriver:
    """Drives the NAO motors from pipeline pose frames.

    The driver is constructed with a live Webots ``Robot`` instance. It does
    the device lookups, applies the standing posture, then on each ``update``
    maps -> clamps -> smooths -> commands the motors. It also reads the
    position sensors (``<name>S``) for health monitoring.

    Parameters
    ----------
    robot:
        Webots ``Robot`` instance.
    drive_legs:
        If True, build the per-leg lower-body layer
        (``lower_body.LowerBodyController``): squat, leg abduction, and
        single-leg lift with a model-verified weight transfer. If False the legs
        are held in the neutral standing posture and only the upper body imitates.
    smoothing_alpha:
        EMA factor for joint targets in (0, 1].
    velocity_scale:
        Fraction of each joint's hardware max velocity used as the motion cap.
    leg_velocity_factor:
        Extra multiplier (0..1) applied on top of ``velocity_scale`` for the leg
        joints only, so a weight-bearing crouch/lean eases in slowly and does not
        jolt the centre of mass off the feet (NFR-4). While stepping or marching
        the (higher) ``gait_leg_velocity_factor`` is used instead, because there
        the leg has to actually keep up with the motion.
    stale_after_s:
        If no command arrives within this many seconds, the driver is marked
        stale (the robot simply holds its last commanded pose).
    """

    def __init__(
        self,
        robot,
        *,
        drive_legs: bool = False,
        drive_head: bool = True,
        swap_sides: bool = False,
        smoothing_alpha: float = 0.4,
        velocity_scale: float = 0.5,
        leg_velocity_factor: float = 0.5,
        stale_after_s: float = 0.5,
        enable_balance: bool = False,
        enable_walk: bool = False,
        walk_tier: str = "march",
        gait_smoothing_alpha: float = 0.7,
        gait_leg_velocity_factor: float = 0.85,
        arm_tau_s: float = 0.07,
        arm_lead_s: float = 0.16,
        arm_velocity_factor: float = 1.0,
        gait_params: object | None = None,
        lower_body_params: object | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.drive_legs = drive_legs
        self.drive_head = drive_head
        self.swap_sides = swap_sides
        self.velocity_scale = max(0.05, min(1.0, velocity_scale))
        self.leg_velocity_factor = max(0.05, min(1.0, leg_velocity_factor))
        self.gait_leg_velocity_factor = max(0.05, min(1.0, gait_leg_velocity_factor))
        self.stale_after_s = stale_after_s
        self.enable_walk = enable_walk
        self.walk_tier = walk_tier
        self.log = logger or _null_logger

        self.configs = get_default_motor_configs()
        self.limiter = JointLimiter(self.configs)
        # ONE smoother for the whole robot, with a per-joint alpha. The legs still
        # get the snappier factor -- a walking waveform double-attenuated by the
        # arm/head EMA collapses into a shuffle (FR-6) -- but they now get it from
        # a single piece of state, so switching leg-control layer no longer jumps
        # the targets. See ExponentialSmoother.
        self.smoother = ExponentialSmoother(smoothing_alpha)
        self.leg_alpha = max(0.0, min(1.0, gait_smoothing_alpha))
        # The arms do not go through `smoother`: they are tracked in continuous
        # time and predicted forward, because the plain per-frame EMA was the
        # single largest delay in the imitation chain. See ArmTracker.
        self.arm_tracker = ArmTracker(tau_s=arm_tau_s, lead_s=arm_lead_s)
        # >1 is meaningful here and 1.0 is not the ceiling: it multiplies
        # `velocity_scale` (0.5 by default), and the product is what gets capped
        # at the hardware maximum.
        self.arm_velocity_factor = max(0.05, min(4.0, arm_velocity_factor))
        self._last_arm_tick: float | None = None
        self.health = MotorHealthMonitor()

        self.motors: dict[str, object] = {}
        self.sensors: dict[str, object] = {}
        self.commanded: dict[str, float] = {}
        self.measured: dict[str, float] = {}
        # True while a Webots Motion clip owns the whole body (see
        # release_to_motion / reclaim_from_motion).
        # Joints a Webots Motion clip currently owns. A SET rather than a bool:
        # Webots' NAO walk clips drive only the 12 leg joints (checked against the
        # header of Forwards.motion, which lists exactly those), so suspending the
        # whole body handed the clip joints it never commands and froze arm and
        # head imitation for the clip's whole duration -- and since a marching
        # human restarts the clip immediately, that read as the upper body dying.
        self._suspended: set = set()
        # The pose imitation wants this leg posture; the balance loop adds small
        # corrections on top of it each control step.
        self.base_targets: dict[str, float] = {}
        self.stats = DriverStats()
        self._last_command_time: float | None = None
        self._last_balance_time: float | None = None
        # Set when a clip hands the body back; starts the leg-target rate
        # cap on the next applied frame (see _handover_leg_step).
        self._handover_pending: bool = False
        self._handover_since: float | None = None
        self._last_leg_pose_time: float | None = None

        # Model-based CoM balance feedback (Option 2: FK + known link masses).
        # Imported lazily and guarded so the driver still runs if numpy/balance
        # is unavailable.
        # Layers that asked to be built and could not be. Kept as data rather than
        # only logged once at startup: a missing NumPy in Webots' interpreter
        # silently removes the entire model-based balance layer AND the CoM step
        # gate, and the only symptom is a robot that never leans. The controller
        # repeats this on the status line so it cannot scroll away unnoticed.
        self.degraded: list[str] = []

        self.balance = None
        if enable_balance:
            try:
                from balance import BalanceController, NaoCoMModel
                self.balance = BalanceController(NaoCoMModel())
                self.log("Balance feedback ON (model-based CoM, Fibonacci search)")
            except Exception as exc:  # noqa: BLE001
                self.degraded.append(f"CoM balance OFF ({exc})")
                self.log(f"Balance feedback OFF ({exc})")

        # Walk engine (gait command -> balance-stable leg motion). When enabled,
        # the gait path is the SOLE commander of the legs and the retargeter's
        # crouch is skipped (gait owns the lower body). Imported lazily because
        # ``gait`` imports this module.
        self.gait_engine = None
        self._gait_meta: dict[str, object] = {"single_support": False, "amp_gain": 0.0}
        if enable_walk:
            try:
                from gait import GaitEngine
                com_model = self.balance.model if self.balance is not None else None
                self.gait_engine = GaitEngine(
                    params=gait_params, com_model=com_model, limiter=self.limiter
                )
                self.log(f"Walk engine ON (tier={walk_tier})")
            except Exception as exc:  # noqa: BLE001
                self.enable_walk = False
                self.degraded.append(f"march engine OFF ({exc})")
                self.log(f"Walk engine OFF ({exc})")

        # Per-leg pose imitation + weight-shift/lift sequencer. This is the layer
        # that makes a squat and a single-leg lift actually reach the robot; it
        # needs the CoM model to decide when unloading a foot is safe, and
        # degrades to a hard-capped lift without it (never to "no motion at all",
        # which is what made the legs look dead before).
        # Self-calibrating neutral for the head-pitch solve (nao_retarget's
        # HeadGeometry). Held here so it persists across frames -- the whole
        # point is that it converges on THIS subject's neck over a run; built
        # lazily because ``nao_retarget`` imports this module.
        self.head_geom = None

        self.leg_retargeter = None
        self.lower_body = None
        if drive_legs:
            try:
                from lower_body import LowerBodyController
                from nao_retarget import LowerBodyRetargeter
                com_model = self.balance.model if self.balance is not None else None
                self.leg_retargeter = LowerBodyRetargeter()
                self.lower_body = LowerBodyController(
                    params=lower_body_params, com_model=com_model, limiter=self.limiter
                )
                self.log(
                    "Lower-body pose imitation ON"
                    f" (CoM-gated stepping: {com_model is not None})"
                )
            except Exception as exc:  # noqa: BLE001
                self.degraded.append(f"leg pose imitation OFF ({exc})")
                self.log(f"Lower-body pose imitation OFF ({exc})")
            else:
                if com_model is None:
                    self.degraded.append(
                        "leg step gate UNGATED (no CoM model: lift capped, not verified)"
                    )
        self._lb_meta: dict[str, object] = {"mode": "off", "balance_ok": True}

        self._setup_devices()
        self.apply_standing_posture()

    @property
    def suspended(self) -> bool:
        """True while a motion clip owns ANY joint (the leg layers key off this)."""
        return bool(self._suspended)

    def _is_suspended(self, name: str) -> bool:
        return name in self._suspended

    # -- device setup -------------------------------------------------------
    def _setup_devices(self) -> None:
        found, missing = 0, []
        for name in self.configs:
            motor = self.robot.getDevice(name)
            if motor is None:
                missing.append(name)
                continue
            self.motors[name] = motor
            found += 1
            if "Phalanx" in name:
                self._adopt_reported_limits(name, motor)
            sensor = self.robot.getDevice(name + "S")
            if sensor is not None:
                try:
                    sensor.enable(self.timestep)
                except Exception:  # noqa: BLE001 - some devices may not be sensors
                    pass
                else:
                    self.sensors[name] = sensor
        self.log(f"Motors found: {found}/{len(self.configs)}; sensors: {len(self.sensors)}")
        if missing:
            self.log(f"Motors not present on this model: {', '.join(missing)}")
        grips = [g for g in HAND_GRIP_JOINTS
                 if any(p in self.motors for p in PHALANX_JOINTS[g])]
        if grips:
            sample = self.configs[PHALANX_JOINTS[grips[0]][0]]
            self.log(f"Grip available on {', '.join(grips)} "
                     f"(phalanx range {sample.min_angle:.3f}..{sample.max_angle:.3f} rad)")
        else:
            self.log("No phalanx motors on this model: grip will not be driven.")

    def _adopt_reported_limits(self, name: str, motor: object) -> None:
        """Replace a finger MotorConfig's range with what the motor reports.

        The hand is the one place this module cannot check its numbers against a
        file: Nao.proto is fetched over the network when the world loads. Rather
        than trust a documented 0..0.96 and drive the fingers to the wrong end of
        their travel, ask the device. Webots motors carry their own limits, and a
        PROTO that revises them stays correct here for free.
        """
        try:
            lo = float(motor.getMinPosition())
            hi = float(motor.getMaxPosition())
        except Exception:  # noqa: BLE001 - not every build exposes these
            return
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            return
        old = self.configs[name]
        self.configs[name] = MotorConfig(name, lo, hi, old.max_velocity,
                                         _clamp(old.rest_angle, lo, hi))

    def _set_motor(self, name: str, angle: float, velocity: float) -> None:
        motor = self.motors.get(name)
        if motor is None or self._is_suspended(name):
            # This joint belongs to a running Motion clip; commanding it now would
            # fight the clip's keyframes and break its balance. Joints the clip
            # does NOT drive stay ours, so the arms and head keep imitating.
            return
        angle = self.limiter.clamp_angle(name, angle)
        try:
            motor.setVelocity(max(0.01, velocity))
            motor.setPosition(angle)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to command {name}: {exc}")
            return
        self.commanded[name] = angle

    def _alpha_for(self, name: str) -> float:
        """Smoothing factor for one joint: snappier on the legs than the arms."""
        return self.leg_alpha if name in ALL_LEG_JOINTS else self.smoother.alpha

    def _smooth(self, name: str, target: float) -> float:
        return self.smoother.smooth(name, target, self._alpha_for(name))

    def _handover_leg_step(self, now_s: float | None) -> float | None:
        """Max leg-target change this tick while easing out of a clip, else None.

        Returns ``None`` once the blend window has expired, which restores full
        1:1 leg tracking -- this is a guard on the transition only, not a
        permanent speed limit on leg imitation.
        """
        if self._handover_since is None:
            if self._handover_pending and now_s is not None:
                self._handover_since = now_s
                self._handover_pending = False
            else:
                return None
        if now_s is None:
            return self.LEG_POSE_RATE * 0.02
        elapsed = now_s - self._handover_since
        if elapsed < 0.0 or elapsed >= self.HANDOVER_BLEND_S:
            self._handover_since = None
            return None
        dt = 0.02
        if self._last_command_time is not None:
            dt = max(0.0, min(0.1, now_s - self._last_command_time))
        return self.LEG_POSE_RATE * dt

    def _velocity_for(self, name: str) -> float:
        cfg = self.configs.get(name)
        ceiling = cfg.max_velocity if cfg else 4.0
        scale = self.velocity_scale
        # Legs carry the robot's weight: move them gently so a crouch/sway eases
        # in rather than jolting the centre of mass off the feet (NFR-4).
        if name in ALL_LEG_JOINTS:
            scale *= self.leg_velocity_factor
        elif name in TRACKED_ARM_JOINTS:
            # The arms hold nothing up, so the reason the legs are throttled
            # (NFR-4: a jolt moves the centre of mass off the feet) does not
            # apply to them. Measured on the 2026-09-16 session the arm command
            # demanded up to 3.2 rad/s against a 3.6 rad/s ceiling, so the cap
            # was clipping the fastest gestures -- exactly the ones that read as
            # "the robot did not follow".
            scale = min(1.0, scale * self.arm_velocity_factor)
        return ceiling * scale

    # -- posture ------------------------------------------------------------
    def apply_standing_posture(self) -> None:
        """Move every joint to its neutral standing target and seed smoothing."""
        posture = standing_posture()
        for name, angle in posture.items():
            self.smoother.reset(name, angle)
            if name in TRACKED_ARM_JOINTS:
                self.arm_tracker.reset(name, angle)
            self._set_motor(name, angle, self._velocity_for(name) * 0.6)
        self.log("Applied standing posture")

    # -- per-frame update ---------------------------------------------------
    def _apply_targets(self, targets: dict[str, float], now_s: float | None) -> int:
        """Smooth, command and bookkeep a set of NAO joint targets."""
        applied = 0
        # Rate ceiling on the LEG targets for a short window after a clip hands
        # the body back. Getting INTO a clip has always been ramped
        # (approach_leg_pose); coming out of one was not, and the clip leaves the
        # legs wherever its last keyframe put them while the human's pose asks
        # for something up to 0.5 rad away. Measured on the 2026-09-08 session:
        # 13 of 13 clip->pose handovers drove support_margin_x to between -0.046
        # and -0.091 m -- the centre of mass off the front of the feet -- and all
        # three falls began within 0.13-1.32 s of one. The cap is LEG_POSE_RATE,
        # the same rate the entry ramp already proves is safe.
        leg_step = self._handover_leg_step(now_s)
        for name, target in targets.items():
            if name not in self.motors and name not in HAND_GRIP_JOINTS:
                continue
            if self._is_suspended(name):
                # A suspended joint is the clip's; the rest are still ours. The
                # frame time is recorded either way so staleness detection keeps
                # working across a clip.
                continue
            if leg_step is not None and name in ALL_LEG_JOINTS:
                prev = self.base_targets.get(
                    name, self.measured.get(name, self.commanded.get(name, target)))
                target = prev + _clamp(target - prev, -leg_step, leg_step)
            self.base_targets[name] = target
            if name in HAND_GRIP_JOINTS:
                # A closure in [0, 1], not an angle: observed here, fanned out
                # across the phalanx motors by tick_arms.
                self.arm_tracker.observe(name, _clamp(target, 0.0, 1.0), now_s)
            elif name in TRACKED_ARM_JOINTS:
                # Record the observation; `tick_arms` turns it into a command
                # every simulation step, not just on the frames that carry one.
                self.arm_tracker.observe(name, target, now_s)
            else:
                smoothed = self._smooth(name, target)
                self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1

        self.tick_arms(now_s)
        if now_s is not None:
            self._last_command_time = now_s
        self.stats.frames_applied += 1
        self.stats.joints_last_applied = applied
        self.stats.stale = False
        return applied

    def tick_arms(self, now_s: float | None) -> int:
        """Re-command every tracked arm joint for this simulation step.

        Called once per control step (and once more whenever a camera frame
        lands, so a fresh observation reaches the motors without waiting for the
        next tick). The camera runs at ~12 Hz and the simulation at 50, so
        without this the arm held still for four steps out of five and then
        jumped -- visible as stepping rather than motion, and the reason the
        arms looked laggy even where the average delay was small.

        Returns the number of joints commanded.
        """
        dt = 0.001 * self.timestep
        if now_s is not None and self._last_arm_tick is not None:
            # Clamp: a simulation reset or a long stall must not hand the
            # smoother a dt so large that it snaps the arm to the goal.
            dt = _clamp(now_s - self._last_arm_tick, 0.0, 0.2)
        if now_s is not None:
            self._last_arm_tick = now_s
        applied = 0
        for name in TRACKED_ARM_JOINTS:
            if name not in self.motors or self._is_suspended(name):
                continue
            value = self.arm_tracker.advance(name, now_s, dt)
            if value is None:
                continue
            self._set_motor(name, value, self._velocity_for(name))
            applied += 1
        applied += self._tick_grip(now_s, dt)
        return applied

    def _tick_grip(self, now_s: float | None, dt: float) -> int:
        """Fan each hand's closure out across its eight phalanx motors."""
        applied = 0
        for grip_name in HAND_GRIP_JOINTS:
            closure = self.arm_tracker.advance(grip_name, now_s, dt)
            if closure is None:
                continue
            closure = _clamp(closure, 0.0, 1.0)
            for phalanx in PHALANX_JOINTS[grip_name]:
                if phalanx not in self.motors or self._is_suspended(phalanx):
                    continue
                cfg = self.configs[phalanx]
                # Interpolate between the motor's OWN reported ends, so which
                # end is "open" is a property of the model rather than a number
                # written here. HAND_OPEN_AT_MAX names the assumption.
                lo, hi = (cfg.max_angle, cfg.min_angle) if HAND_OPEN_AT_MAX \
                    else (cfg.min_angle, cfg.max_angle)
                self._set_motor(phalanx, lo + (hi - lo) * closure,
                                self._velocity_for(phalanx))
                applied += 1
        return applied

    def _balance_feedback(self, state: dict[str, float], torso_rp: tuple,
                          tilt_rate: tuple, now_s: float | None,
                          fsr: dict[str, float] | None = None) -> dict[str, float]:
        """One BalanceController cycle on the MEASURED posture; {} if off/failed.

        ``tilt_rate`` is the gyro (roll_rate, pitch_rate) the loop uses as lead
        compensation. The control period is taken from ``now_s`` so the loop's
        rate limit is honoured whatever the step size.
        """
        if self.balance is None:
            return {}
        dt = 0.02
        if now_s is not None:
            if self._last_balance_time is not None:
                dt = max(0.0, min(0.1, now_s - self._last_balance_time))
            self._last_balance_time = now_s
        contact = None
        # Only while the lower body believes both feet are down -- see the same
        # rule in lower_body.step. One tick of lag on the mode is immaterial for a
        # contact mask, and it keeps the loop from fighting a weight transfer.
        if self._lb_meta.get("mode", "double") == "double":
            try:
                from balance import contact_from_fsr
                contact = contact_from_fsr(fsr)
            except Exception:  # noqa: BLE001
                contact = None
        try:
            return self.balance.compute_correction(
                state, torso_rp, tilt_rate=tilt_rate, dt_s=dt, contact=contact
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Balance step failed, disabling ({exc})")
            self.balance = None
            return {}

    def balance_tick(self, torso_rp: tuple = (0.0, 0.0),
                     tilt_rate: tuple = (0.0, 0.0),
                     now_s: float | None = None) -> int:
        """Run one CoM balance cycle: re-command the legs as base + correction.

        Called every control step (not just on new pose frames) so balance is
        maintained continuously. ``torso_rp`` is the InertialUnit (roll, pitch)
        in rad and ``tilt_rate`` the gyro rates. Returns the number of joints
        nudged. No-op if balance is off.
        """
        if self.balance is None or self.suspended:
            return 0
        # Best estimate of the current pose: measured where available, else the
        # last commanded angle.
        state = dict(self.commanded)
        state.update(self.measured)
        corr = self._balance_feedback(state, torso_rp, tilt_rate, now_s)
        if not corr:
            return 0

        applied = 0
        for name, delta in corr.items():
            if name not in self.motors:
                continue
            base = self.base_targets.get(name, self.configs[name].rest_angle)
            smoothed = self._smooth(name, base + delta)
            self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1
        return applied

    # -- lower-body pose imitation -----------------------------------------
    def lower_body_tick(
        self,
        now_s: float,
        torso_rp: tuple = (0.0, 0.0),
        fsr: dict[str, float] | None = None,
        yaw_bias: float = 0.0,
        tilt_rate: tuple = (0.0, 0.0),
    ) -> int:
        """Advance the per-leg pose imitation one control step.

        This is the DEFAULT leg commander: it turns the latest camera
        observation into a safe leg posture via
        ``lower_body.LowerBodyController`` (symmetric crouch + authority-weighted
        per-leg deviation + a model-verified weight shift when the human lifts a
        foot), then folds in the symmetric CoM balance correction *only* while the
        sequencer says both feet are still evenly loaded -- once we are
        deliberately leaning onto one foot, "centre the CoM between the feet" is
        the wrong objective and would cancel the transfer.

        Called every simulation step rather than per camera frame, so the legs
        keep being controlled at sim rate even when pose packets stall. Returns
        the number of joints commanded (0 when the layer is off/suspended).
        """
        if self.lower_body is None or self.suspended:
            return 0
        state = dict(self.commanded)
        state.update(self.measured)

        # The balance FEEDBACK loop runs whenever a clip is not driving the legs,
        # in single support too: its objective is "keep the CoM inside whatever
        # support polygon the current stance actually has" (balance.support_margins
        # filters by foot contact). It is computed FIRST, from the measured posture,
        # the IMU tilt and the gyro, and then handed INTO the lower body, which
        # folds it into the same sole-flat pelvis shift as its own feed-forward
        # compensation -- summed, clamped and rate-limited in one place.
        #
        # It used to be added here, on top of the returned targets. That made two
        # independent controllers with independent clamps (0.30 + 0.25 rad) answer
        # the same tilt on the same joints: recorded as HipPitch +0.45 /
        # AnklePitch -0.65 -- both saturated -- before every pitch fall, and a
        # 0.4 rad lateral pelvis lurch inside 0.2 s before every lateral one.
        corr = self._balance_feedback(state, torso_rp, tilt_rate, now_s, fsr)
        feedback = (float(corr.get("LHipPitch", 0.0)), float(corr.get("LHipRoll", 0.0)))
        try:
            targets, meta = self.lower_body.step(
                now_s, torso_rp=torso_rp, fsr=fsr, measured=state,
                yaw_bias=yaw_bias, feedback=feedback,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Lower-body step failed, disabling ({exc})")
            self.lower_body = None
            return 0
        self._lb_meta = meta

        applied = 0
        for name, value in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = value
            # The legs get the snappier alpha here too: they must follow a step,
            # not lag it into a shuffle.
            smoothed = self._smooth(name, value)
            self._set_motor(name, smoothed, self._gait_velocity_for(name))
            applied += 1
        return applied

    # Rate-limited approach to a commanded leg posture (see approach_leg_pose).
    # 1.5 rad/s is under the driver's own leg velocity ceiling (0.85 * 0.5 * 6.40
    # = 2.7 rad/s on the knee) so the motors can actually follow it, and it takes
    # the 0.84 rad knee travel from the standing crouch to a walk clip's opening
    # stance in 0.56 s.
    LEG_POSE_RATE = 1.5      # rad/s
    LEG_POSE_TOL = 0.08      # rad; per-joint arrival tolerance on the MEASURED angle
    # How long the leg targets stay rate-capped after a clip hands the body
    # back. Long enough to walk 0.5 rad of gap in at LEG_POSE_RATE (0.33 s)
    # with margin; short enough that ordinary leg imitation is untouched.
    HANDOVER_BLEND_S = 0.6

    def approach_leg_pose(self, pose: dict[str, float], now_s: float | None = None,
                          rate: float | None = None,
                          tol: float | None = None) -> bool:
        """Drive the leg joints toward ``pose`` at a bounded rate. True once there.

        This is a leg commander in its own right -- while it is running, no other
        layer may command the legs -- and it exists for one job: getting the robot
        into the posture a pre-balanced motion clip opens in, BEFORE handing the
        clip the joints.

        Cyberbotics' walk and turn clips all begin in a deep, sole-flat crouch
        (knee 1.042 rad, about 0.51 rad of squat) while this controller stands at
        0.10. Playback commands its first keyframe on its very first step, and
        release_to_motion has by then lifted the velocity caps -- so handing over
        from the standing crouch asks the knees for 0.84 rad in one 20 ms step and
        the robot squats out from under itself. Arriving first turns the handover
        into a continuation.

        Arrival is judged on the MEASURED angles, not the commanded ones: the
        point is where the legs physically are. The smoother is reseeded as we go
        so its lag does not fight a deliberate ramp.
        """
        if self.suspended:
            return False
        rate = self.LEG_POSE_RATE if rate is None else rate
        tol = self.LEG_POSE_TOL if tol is None else tol
        dt = 0.02
        if now_s is not None:
            if self._last_leg_pose_time is not None:
                dt = max(0.0, min(0.1, now_s - self._last_leg_pose_time))
            self._last_leg_pose_time = now_s
        # Every joint arrives TOGETHER, by scaling its rate to its share of the
        # longest travel. A single flat rate is not a synchronised move: from the
        # standing crouch to a walk clip's opening stance the knee travels 0.842
        # rad while the hip travels 0.405 and the ankle 0.437, so at equal rates
        # the knee is still going when the others have stopped -- and the sole's
        # attitude is Hip + Knee + Ankle, which is what keeps the foot flat and
        # the torso vertical. Simulated against the repo's own CoM model, a flat
        # rate drives that sum to 0.405 rad (eight times the 0.05 sole-tilt
        # budget) and the fore/aft support margin to -0.064 m: the ramp added to
        # make the handover safe took the centre of mass off the feet on its own.
        # Scaled by travel it holds the sum at 0.0000 rad and the margin at
        # +0.0596 m, in the same 0.56 s.
        travels = {}
        for name, target in pose.items():
            if name not in self.motors:
                continue
            current = self.base_targets.get(
                name, self.measured.get(name, self.commanded.get(name, 0.0)))
            travels[name] = abs(float(target) - current)
        longest = max(travels.values(), default=0.0)

        arrived = True
        for name, target in pose.items():
            if name not in self.motors:
                continue
            target = self.limiter.clamp_angle(name, float(target))
            current = self.base_targets.get(
                name, self.measured.get(name, self.commanded.get(name, 0.0)))
            share = 1.0 if longest <= 1e-9 else travels.get(name, 0.0) / longest
            step = max(0.0, rate * dt) * share
            nxt = current + _clamp(target - current, -step, step)
            self.base_targets[name] = nxt
            self.smoother.reset(name, nxt)
            self._set_motor(name, nxt, self._gait_velocity_for(name))
            measured = self.measured.get(name)
            if measured is None or abs(measured - target) > tol:
                arrived = False
        # NOT _last_command_time: that clock measures how long since the CAMERA
        # last said anything, and this method is the robot commanding itself.
        # Touching it here made the driver never go stale while preparing, so a
        # human who walked out of frame never triggered the stand-down.
        self.stats.frames_applied += 1
        self.stats.joints_last_applied = sum(1 for n in pose if n in self.motors)
        return arrived

    def release_leg_pose(self) -> None:
        """Forget the approach's clock (call when it is abandoned)."""
        self._last_leg_pose_time = None

    def reset_balance(self) -> None:
        """Drop the balance loop's carried-over correction (see
        ``balance.BalanceController.reset``). Call after a fall recovery or a
        whole-body motion clip: the correction is an integrator, and the robot it
        described no longer exists."""
        if self.balance is not None:
            self.balance.reset()
        self._last_balance_time = None

    def set_lower_body_observation(self, obs: object | None) -> None:
        """Latch a fresh lower-body observation (no-op when the layer is off)."""
        if self.lower_body is not None and obs is not None:
            self.lower_body.set_observation(obs)

    def lower_body_stand_down(self) -> None:
        """Tell the lower body to forget the last observation (tracking lost)."""
        if self.lower_body is not None:
            self.lower_body.stand_down()

    # Arm/head joints that the upper-body stand-down returns to neutral.
    UPPER_BODY_JOINTS = (
        "LShoulderPitch", "RShoulderPitch",
        "LShoulderRoll", "RShoulderRoll",
        "LElbowYaw", "RElbowYaw",
        "LElbowRoll", "RElbowRoll",
        "LWristYaw", "RWristYaw",
        "HeadYaw", "HeadPitch",
        # The grip channels, so a departed human does not leave the robot
        # holding a fist. Named here rather than the sixteen phalanx motors:
        # tick_arms fans one closure out to all of them.
        *HAND_GRIP_JOINTS,
    )

    def upper_body_stand_down(self) -> int:
        """Ramp the arms and head back to the neutral standing posture.

        The counterpart of :meth:`lower_body_stand_down`, and it was missing. On
        staleness the legs correctly ramped to the balanced crouch while the arms
        and head simply stopped being updated, so they froze wherever they
        happened to be -- a recorded session ends with 80 s of *perfect* tracking
        of a pose belonging to a human who had left the frame, which reads as a
        crashed robot rather than an idle one.

        The existing smoother supplies the ramp, so this is a target change, not a
        jump. Deliberately does not touch ``_last_command_time`` or ``stats``:
        standing down is a consequence of staleness, not a refutation of it.
        """
        if self.suspended:
            return 0
        applied = 0
        for name in self.UPPER_BODY_JOINTS:
            if name in HAND_GRIP_JOINTS:
                # An open hand is the neutral, and there is no MotorConfig to
                # read it from: the grip is a closure in [0, 1], not an angle.
                self.base_targets[name] = 0.0
                self.arm_tracker.observe(name, 0.0, None)
                self.arm_tracker.forget(name)
                applied += 1
                continue
            if name not in self.motors:
                continue
            cfg = self.configs.get(name)
            if cfg is None:
                continue
            self.base_targets[name] = cfg.rest_angle
            if name in TRACKED_ARM_JOINTS:
                # Target the neutral and drop the velocity estimate: the old one
                # describes a gesture that is no longer being made, and
                # extrapolating it would send the arm PAST the rest pose on the
                # way to standing down.
                self.arm_tracker.observe(name, cfg.rest_angle, None)
                self.arm_tracker.forget(name)
            else:
                smoothed = self._smooth(name, cfg.rest_angle)
                self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1
        self.tick_arms(None)
        return applied

    @property
    def lower_body_meta(self) -> dict[str, object]:
        """Latest lower-body telemetry (mode, shift, lift, stance margin, ...)."""
        return dict(self._lb_meta)

    # -- whole-body Webots Motion clips ------------------------------------
    def release_to_motion(self, joints: Iterable[str] | None = None) -> None:
        """Hand the whole body to a Webots ``Motion`` clip.

        Two things have to happen, and missing either one is why "play a walk
        clip" usually looks broken:

        1. Stop commanding motors, so our targets do not fight the clip's
           keyframes.
        2. **Lift the velocity caps.** ``Motion`` playback works by calling
           ``setPosition`` on every joint each step; a motor still limited to
           ~25% of its maximum velocity simply cannot reach those keyframes, so
           the pre-balanced gait arrives late at every foot placement and the
           robot topples. Motors keep whatever velocity was last set, so the caps
           must be raised explicitly here.

        ``joints`` narrows the handover to the joints the clip actually drives,
        read from the clip's own header. Webots' NAO walk clips list exactly the
        12 leg joints, so passing them keeps arm and head imitation live for the
        clip's whole duration instead of freezing the upper body every time the
        human takes a step. Omit it to hand over everything (the safe default:
        handing over too much only costs expressiveness, handing over too little
        would let us fight a clip's keyframes).
        """
        if self.suspended:
            return
        # Default to the whole body when the clip will not say what it drives:
        # handing over too much is safe, handing over too little is not.
        names = [n for n in (joints or self.motors) if n in self.motors]
        if not names:
            names = list(self.motors)
        for name in names:
            cfg = self.configs.get(name)
            motor = self.motors[name]
            try:
                motor.setVelocity(cfg.max_velocity if cfg else 6.0)
            except Exception:  # noqa: BLE001
                pass
        self._suspended = set(names)

    def reclaim_from_motion(self) -> None:
        """Take the body back from a motion clip without a jolt.

        The smoothers still hold pre-clip values, and the robot is now somewhere
        else entirely, so they are reseeded from the position sensors before
        per-joint commanding resumes.
        """
        if not self.suspended:
            return
        # A clip has moved the whole body; the balance correction that suited the
        # pre-clip posture is meaningless against the new one.
        self.reset_balance()
        self.release_leg_pose()
        reclaimed = sorted(self._suspended)
        self._suspended = set()
        self.reseed_from_measured()
        # Ease the legs from where the clip left them to where the human
        # is, instead of commanding the whole gap on the next tick.
        self._handover_pending = True
        self._handover_since = None
        # Tell the lower body where the clip left the legs, so its own crouch
        # ramps DOWN from there instead of snapping. A walk clip ends in the same
        # 0.51 rad squat it started in, and the lower body's standing depth is
        # 0.10: without this the first post-clip step commands a 0.41 rad knee
        # jump, which is the handover jolt all over again, in reverse.
        if self.lower_body is not None:
            seed = getattr(self.lower_body, "seed_crouch_from", None)
            if callable(seed):
                try:
                    seed(self.measured)
                except Exception:  # noqa: BLE001 - never break the handover
                    pass
        for name in reclaimed:
            self._set_motor(name, self.measured.get(name, self.commanded.get(name, 0.0)),
                            self._velocity_for(name))

    # -- gait / walking -----------------------------------------------------
    def set_gait_command(self, gait: dict[str, object] | None) -> None:
        """Hand the latest gait command (from the Python cue extractor) to the
        walk engine. No-op when walking is disabled."""
        if self.gait_engine is not None:
            self.gait_engine.set_command(gait)

    def _gait_velocity_for(self, name: str) -> float:
        """Leg velocity while walking: a higher factor than the gentle crouch so
        the gait waveform actually moves (it would otherwise collapse to a
        shuffle under the crouch's slow leg velocity)."""
        cfg = self.configs.get(name)
        ceiling = cfg.max_velocity if cfg else 4.0
        return ceiling * self.velocity_scale * self.gait_leg_velocity_factor

    def gait_tick(self, now_s: float, torso_rp: tuple = (0.0, 0.0),
                  fsr: dict[str, float] | None = None,
                  tilt_rate: tuple = (0.0, 0.0)) -> int:
        """Advance the walk engine one step and command the legs.

        When walking is enabled this REPLACES ``balance_tick`` for the lower
        body and is the sole commander of the 12 leg joints: it asks the engine
        for the gait leg posture, folds in the symmetric CoM balance correction
        while in double support (Tier A, where that correction is valid), and
        commands the legs with the snappier gait smoother and raised leg
        velocity. In single support (Tier B) the gait owns the roll axis and the
        symmetric correction is skipped (the engine's IMU-tilt abort is the
        safety net). Returns the number of joints commanded; 0 if walk is off.
        """
        if self.gait_engine is None or self.suspended:
            return 0
        state = dict(self.commanded)
        state.update(self.measured)
        try:
            targets, meta = self.gait_engine.step(
                now_s, tier=self.walk_tier, torso_rp=torso_rp, fsr=fsr, measured=state
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Gait step failed, disabling walk ({exc})")
            self.gait_engine = None
            return 0
        self._gait_meta = meta

        if self.balance is not None and not meta.get("single_support", False):
            corr = self._balance_feedback(state, torso_rp, tilt_rate, now_s, fsr)
            for name, delta in corr.items():
                targets[name] = self.limiter.clamp_angle(
                    name, targets.get(name, 0.0) + delta
                )

        applied = 0
        for name, value in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = value
            smoothed = self._smooth(name, value)
            self._set_motor(name, smoothed, self._gait_velocity_for(name))
            applied += 1
        return applied

    @property
    def gait_meta(self) -> dict[str, object]:
        """Latest walk-engine telemetry (amp_gain, phase, cadence, single_support)."""
        return dict(self._gait_meta)

    def reseed_from_measured(self) -> None:
        """Reset the smoothers to the measured joint angles.

        Call this when handing control back from an external whole-body motion
        (e.g. a Webots walk clip) to per-joint imitation/gait, so targets ease
        from where the robot ACTUALLY is rather than from a stale smoother value
        — avoids a jolt at the motion->imitation transition.
        """
        for name in self.motors:
            val = self.measured.get(name)
            if val is None:
                continue
            self.smoother.reset(name, val)
            self.commanded[name] = val

    def update(self, incoming: dict[str, float], now_s: float | None = None) -> int:
        """Apply one frame of *pre-computed* pipeline joint angles (fallback).

        Returns the number of joints commanded.
        """
        targets = map_pipeline_angles(
            incoming, drive_legs=self.drive_legs, limiter=self.limiter
        )
        return self._apply_targets(targets, now_s)

    def update_from_keypoints(
        self, keypoints: dict[str, object], now_s: float | None = None
    ) -> int:
        """Apply one camera frame: arms/head now, legs via the lower-body layer.

        The arms and head are commanded here because they are safe to drive
        straight from the pose. The legs are NOT: their observation is only
        *latched* for whichever lower-body layer the controller is running this
        step, which then decides how much of it the robot can execute without
        losing balance. That split is what keeps a single commander on the legs.

        Returns the number of joints commanded. Imported lazily to avoid a
        circular import (``nao_retarget`` depends on this module).
        """
        from nao_retarget import HeadGeometry, retarget_upper_body

        if self.head_geom is None:
            self.head_geom = HeadGeometry()
        targets = retarget_upper_body(
            keypoints,
            drive_head=self.drive_head,
            swap_sides=self.swap_sides,
            limiter=self.limiter,
            head_geom=self.head_geom,
        )
        if self.leg_retargeter is not None:
            try:
                obs = self.leg_retargeter.observe(keypoints)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Leg retargeting failed, disabling ({exc})")
                self.leg_retargeter = None
            else:
                self.set_lower_body_observation(obs)
        return self._apply_targets(targets, now_s)

    def read_feedback(self) -> None:
        """Read position sensors and record tracking error for health checks."""
        for name, sensor in self.sensors.items():
            try:
                value = float(sensor.getValue())
            except Exception:  # noqa: BLE001
                continue
            if math.isnan(value):  # sensors read NaN until the first sim step
                continue
            self.measured[name] = value
            if name in self.commanded:
                self.health.record(name, self.commanded[name], value)

    def check_stale(self, now_s: float) -> bool:
        """Mark the driver stale if no command arrived recently."""
        if self._last_command_time is None:
            return False
        stale = (now_s - self._last_command_time) > self.stale_after_s
        self.stats.stale = stale
        return stale

    # Velocity used by :meth:`stop`. Deliberately NOT zero: a zero-velocity motor
    # cannot move at all, so if anything ever calls stop() while the controller
    # keeps running, the robot is bricked with no diagnostic. A small positive
    # value holds position just as well and stays recoverable.
    HOLD_VELOCITY = 0.2

    def stop(self) -> None:
        """Hold the current position (graceful shutdown)."""
        for name, motor in self.motors.items():
            try:
                motor.setVelocity(self.HOLD_VELOCITY)
                if name in self.measured:
                    motor.setPosition(self.measured[name])
            except Exception:  # noqa: BLE001
                pass

    def stuck_motors(self) -> list[str]:
        return [n for n in self.motors if self.health.is_stuck(n)]

    @property
    def logged_joints(self) -> list[str]:
        """Joints worth logging for fidelity metrics (driven joints only)."""
        joints = [
            "LShoulderPitch", "RShoulderPitch",
            "LShoulderRoll", "RShoulderRoll",
            "LElbowRoll", "RElbowRoll",
            # The roll joints and one representative finger per hand. Logged
            # because "is it tracking?" for these cannot be answered by looking
            # at the robot: a wrist that twists the wrong way and one that does
            # not twist at all are hard to tell apart in a video, and all
            # sixteen phalanx columns would say the same thing as the one.
            "LElbowYaw", "RElbowYaw",
            "LWristYaw", "RWristYaw",
            "LPhalanx1", "RPhalanx1",
        ]
        if self.drive_head:
            joints += ["HeadYaw", "HeadPitch"]
        if self.drive_legs or self.enable_walk:
            for side in ("L", "R"):
                joints += [
                    f"{side}HipYawPitch", f"{side}HipRoll", f"{side}HipPitch",
                    f"{side}KneePitch", f"{side}AnklePitch", f"{side}AnkleRoll",
                ]
        return [j for j in joints if j in self.motors]


# ---------------------------------------------------------------------------
# Trajectory logging (FR-7 / US-3)
# ---------------------------------------------------------------------------
class JointTrajectoryLogger:
    """Append-only CSV log of commanded vs. achieved joint angles.

    The Python pipeline logs the *commanded* angles upstream; only the Webots
    side can observe the robot's *achieved* angles (from the position sensors).
    Logging both here lets the evaluation step compute per-joint MAE between
    target and achieved motion (PRD US-3, NFR-3) and end-to-end timing.

    The logger is defensive by design: any I/O error disables logging rather
    than disturbing the real-time control loop.
    """

    def __init__(
        self,
        directory: str,
        joints: Iterable[str],
        *,
        filename: str | None = None,
        flush_every: int = 50,
        diagnostics: Iterable[str] = (),
        logger: Callable[[str], None] | None = None,
    ) -> None:
        import csv
        import os

        self.joints = list(joints)
        # Extra per-step columns for the controller's own state (IMU, which leg
        # layer ran, the balance margin, why the legs did or did not move).
        # Without them a log tells you what the joints did but not what the
        # controller believed, which is the half that explains the other half --
        # every diagnosis in this project so far has needed both.
        self.diagnostics = list(diagnostics)
        self.flush_every = max(1, flush_every)
        self.log = logger or _null_logger
        self._rows_since_flush = 0
        self._file = None
        self._writer = None

        try:
            os.makedirs(directory, exist_ok=True)
            if filename is None:
                filename = f"webots_joint_trajectory_{int(time_now())}.csv"
            path = os.path.join(directory, filename)
            self._file = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            header = ["wall_time_s", "sim_time_s", "frame_index"]
            header += list(self.diagnostics)
            for j in self.joints:
                header += [f"{j}_cmd_rad", f"{j}_meas_rad"]
            self._writer.writerow(header)
            self.path = path
            self.log(f"Trajectory log: {path}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Trajectory logging disabled ({exc})")
            self._file = None
            self._writer = None
            self.path = None

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def record(
        self,
        sim_time_s: float,
        frame_index: int,
        commanded: dict[str, float],
        measured: dict[str, float],
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        if self._writer is None:
            return
        try:
            row: list[object] = [round(time_now(), 6), round(sim_time_s, 6), frame_index]
            diag = diagnostics or {}
            for name in self.diagnostics:
                value = diag.get(name)
                if isinstance(value, float):
                    value = round(value, 6)
                row.append("" if value is None else value)
            for j in self.joints:
                cmd = commanded.get(j)
                meas = measured.get(j)
                row.append("" if cmd is None else round(cmd, 6))
                row.append("" if meas is None else round(meas, 6))
            self._writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self.flush_every:
                self._file.flush()
                self._rows_since_flush = 0
        except Exception as exc:  # noqa: BLE001
            self.log(f"Trajectory logging stopped ({exc})")
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
        self._file = None
        self._writer = None


def time_now() -> float:
    """Wall-clock seconds. Wrapped so it is trivial to stub in tests."""
    import time as _time

    return _time.time()
