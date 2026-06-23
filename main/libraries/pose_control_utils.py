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

This module deliberately does **not** import the Webots ``controller`` package,
so the math (limits, mapping, smoothing) stays unit-testable off-simulation.
Motor/sensor objects are passed in from the controller process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# NAO H25 joint limits (radians)
#
# Source: Aldebaran/SoftBank NAO H25 joint documentation and the Webots
# Nao.proto RotationalMotor min/maxPosition values. These are the *hardware*
# ranges; Webots clamps setPosition() to them, so commanding outside the range
# silently saturates the joint.
# ---------------------------------------------------------------------------
NAO_JOINT_LIMITS: Dict[str, "MotorConfig"] = {}


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


def get_default_motor_configs() -> Dict[str, MotorConfig]:
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
PIPELINE_TO_NAO: Dict[str, JointMap] = {
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

    def __init__(self, configs: Dict[str, MotorConfig]) -> None:
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

    ``alpha`` in (0, 1]; higher = more responsive, lower = smoother.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._state: Dict[str, float] = {}

    def reset(self, joint_name: str, value: float) -> None:
        self._state[joint_name] = value

    def smooth(self, joint_name: str, target: float) -> float:
        prev = self._state.get(joint_name)
        value = target if prev is None else prev + (target - prev) * self.alpha
        self._state[joint_name] = value
        return value


class MotorHealthMonitor:
    """Tracks position-tracking error and flags stuck motors."""

    def __init__(self, max_position_error: float = 0.1, window: int = 100) -> None:
        self.max_position_error = max_position_error
        self.window = window
        self.position_errors: Dict[str, List[float]] = {}

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
    incoming: Dict[str, float],
    *,
    drive_legs: bool = False,
    limiter: Optional[JointLimiter] = None,
) -> Dict[str, float]:
    """Convert pipeline joint angles to clamped NAO motor targets.

    Pure function (no Webots dependency) so the mapping is unit-testable.
    Unknown joints and (when ``drive_legs`` is False) leg joints are dropped.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    out: Dict[str, float] = {}
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
# Lower body: a single symmetric, statically-balanced crouch (hip/knee/ankle
# pitch only — no roll/lean, which would topple a free-standing NAO). See
# nao_retarget._lower_body.
DRIVEN_LEG_JOINTS = (
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
)


def standing_posture() -> Dict[str, float]:
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
        If True, hip-pitch channels from the pipeline drive the legs. Off by
        default because moving the legs without a balance controller makes NAO
        fall (NFR-4 / PRD risk table).
    smoothing_alpha:
        EMA factor for joint targets in (0, 1].
    velocity_scale:
        Fraction of each joint's hardware max velocity used as the motion cap.
    leg_velocity_factor:
        Extra multiplier (0..1) applied on top of ``velocity_scale`` for the
        leg joints only, so the weight-bearing crouch/sway eases in slowly and
        does not jolt the robot off balance (NFR-4).
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
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.drive_legs = drive_legs
        self.drive_head = drive_head
        self.swap_sides = swap_sides
        self.velocity_scale = max(0.05, min(1.0, velocity_scale))
        self.leg_velocity_factor = max(0.05, min(1.0, leg_velocity_factor))
        self.stale_after_s = stale_after_s
        self.log = logger or _null_logger

        self.configs = get_default_motor_configs()
        self.limiter = JointLimiter(self.configs)
        self.smoother = ExponentialSmoother(smoothing_alpha)
        self.health = MotorHealthMonitor()

        self.motors: Dict[str, object] = {}
        self.sensors: Dict[str, object] = {}
        self.commanded: Dict[str, float] = {}
        self.measured: Dict[str, float] = {}
        # The pose imitation wants this leg posture; the balance loop adds small
        # corrections on top of it each control step.
        self.base_targets: Dict[str, float] = {}
        self.stats = DriverStats()
        self._last_command_time: Optional[float] = None

        # Model-based CoM balance feedback (Option 2: FK + known link masses).
        # Imported lazily and guarded so the driver still runs if numpy/balance
        # is unavailable.
        self.balance = None
        if enable_balance:
            try:
                from balance import BalanceController, NaoCoMModel
                self.balance = BalanceController(NaoCoMModel())
                self.log("Balance feedback ON (model-based CoM, Fibonacci search)")
            except Exception as exc:  # noqa: BLE001
                self.log(f"Balance feedback OFF ({exc})")

        self._setup_devices()
        self.apply_standing_posture()

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

    def _set_motor(self, name: str, angle: float, velocity: float) -> None:
        motor = self.motors.get(name)
        if motor is None:
            return
        angle = self.limiter.clamp_angle(name, angle)
        try:
            motor.setVelocity(max(0.01, velocity))
            motor.setPosition(angle)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to command {name}: {exc}")
            return
        self.commanded[name] = angle

    def _velocity_for(self, name: str) -> float:
        cfg = self.configs.get(name)
        ceiling = cfg.max_velocity if cfg else 4.0
        scale = self.velocity_scale
        # Legs carry the robot's weight: move them gently so a crouch/sway eases
        # in rather than jolting the centre of mass off the feet (NFR-4).
        if name in DRIVEN_LEG_JOINTS:
            scale *= self.leg_velocity_factor
        return ceiling * scale

    # -- posture ------------------------------------------------------------
    def apply_standing_posture(self) -> None:
        """Move every joint to its neutral standing target and seed smoothing."""
        posture = standing_posture()
        for name, angle in posture.items():
            self.smoother.reset(name, angle)
            self._set_motor(name, angle, self._velocity_for(name) * 0.6)
        self.log("Applied standing posture")

    # -- per-frame update ---------------------------------------------------
    def _apply_targets(self, targets: Dict[str, float], now_s: Optional[float]) -> int:
        """Smooth, command and bookkeep a set of NAO joint targets."""
        applied = 0
        for name, target in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = target
            smoothed = self.smoother.smooth(name, target)
            self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1

        if now_s is not None:
            self._last_command_time = now_s
        self.stats.frames_applied += 1
        self.stats.joints_last_applied = applied
        self.stats.stale = False
        return applied

    def balance_tick(self, torso_rp: tuple = (0.0, 0.0)) -> int:
        """Run one CoM balance cycle: re-command the legs as base + correction.

        Called every control step (not just on new pose frames) so balance is
        maintained continuously. ``torso_rp`` is the InertialUnit (roll, pitch)
        in rad. Returns the number of joints nudged. No-op if balance is off.
        """
        if self.balance is None:
            return 0
        # Best estimate of the current pose: measured where available, else the
        # last commanded angle.
        state = dict(self.commanded)
        state.update(self.measured)
        try:
            corr = self.balance.compute_correction(state, torso_rp)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Balance step failed, disabling ({exc})")
            self.balance = None
            return 0

        applied = 0
        for name, delta in corr.items():
            if name not in self.motors:
                continue
            base = self.base_targets.get(name, self.configs[name].rest_angle)
            smoothed = self.smoother.smooth(name, base + delta)
            self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1
        return applied

    def update(self, incoming: Dict[str, float], now_s: Optional[float] = None) -> int:
        """Apply one frame of *pre-computed* pipeline joint angles (fallback).

        Returns the number of joints commanded.
        """
        targets = map_pipeline_angles(
            incoming, drive_legs=self.drive_legs, limiter=self.limiter
        )
        return self._apply_targets(targets, now_s)

    def update_from_keypoints(
        self, keypoints: Dict[str, object], now_s: Optional[float] = None
    ) -> int:
        """Apply one frame by retargeting raw MediaPipe landmarks (full body).

        Returns the number of joints commanded. Imported lazily to avoid a
        circular import (``nao_retarget`` depends on this module).
        """
        from nao_retarget import retarget_full_body

        targets = retarget_full_body(
            keypoints,
            drive_legs=self.drive_legs,
            drive_head=self.drive_head,
            swap_sides=self.swap_sides,
            limiter=self.limiter,
        )
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

    def stop(self) -> None:
        """Hold current position with zero velocity (graceful shutdown)."""
        for name, motor in self.motors.items():
            try:
                motor.setVelocity(0.0)
                if name in self.measured:
                    motor.setPosition(self.measured[name])
            except Exception:  # noqa: BLE001
                pass

    def stuck_motors(self) -> List[str]:
        return [n for n in self.motors if self.health.is_stuck(n)]

    @property
    def logged_joints(self) -> List[str]:
        """Joints worth logging for fidelity metrics (driven joints only)."""
        joints = [
            "LShoulderPitch", "RShoulderPitch",
            "LShoulderRoll", "RShoulderRoll",
            "LElbowRoll", "RElbowRoll",
        ]
        if self.drive_head:
            joints += ["HeadYaw", "HeadPitch"]
        if self.drive_legs:
            for side in ("L", "R"):
                joints += [f"{side}HipPitch", f"{side}KneePitch", f"{side}AnklePitch"]
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
        filename: Optional[str] = None,
        flush_every: int = 50,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        import csv
        import os

        self.joints = list(joints)
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
        commanded: Dict[str, float],
        measured: Dict[str, float],
    ) -> None:
        if self._writer is None:
            return
        try:
            row: List[object] = [round(time_now(), 6), round(sim_time_s, 6), frame_index]
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
