"""Does this motion clip keep the robot on its feet? A certifier for ``.motion``.

Why this exists
---------------
Every locomotion clip the robot plays is an OPEN LOOP: for its whole duration the
legs follow keyframes and nothing is watching the centre of mass. Measured on the
2026-09-08 session, the robot spent 71.8% of a tracked window inside a clip, with
a mean commitment of 6.84 s. So a clip is not an animation asset here -- it is a
few seconds during which the balance controller has no say, and the only chance
to find out whether it falls over is BEFORE it ships.

Cyberbotics' own clips are animations, not gaits, and they were authored for a
robot standing on Cyberbotics' floor with Cyberbotics' timing. This module asks
the question their authors did not have to: given THIS robot's link masses, THIS
sole geometry and THIS motor's velocity ceiling, does the clip stay balanced at
every keyframe, and can the controller get into and out of it safely?

What it checks, and why each one is here
----------------------------------------
Each check below exists because its absence produced a specific measured failure.

``limits``      Every commanded angle inside its mechanical range. A clip that
                asks for more is silently clamped by ``JointLimiter``, so the
                robot plays a DIFFERENT pose than the clip's author drew -- and
                the difference shows up as an unbalanced one.

``velocity``    Keyframe-to-keyframe angular speed inside the motor's own
                ceiling. Exceed it and the joint simply arrives late; the clip's
                balance then depends on a pose the robot never actually reached.
                This is what makes a clip that "works" at one playback speed fall
                over at another.

``support``     Balance, judged differently for the two KINDS of clip, because
                one criterion cannot serve both:

                * ``kind="static"`` (squat, leg raise, turn in place) -- the CoM
                  itself projects inside the support polygon at every keyframe.
                  These clips have no business leaving it; if one does, it is
                  relying on momentum it was never designed to have.

                * ``kind="dynamic"`` (walking) -- the CAPTURE POINT does. A gait
                  is a controlled fall and the CoM leaves the polygon on purpose
                  during single support, so demanding static stability would
                  reject every real gait ever authored, this module's own
                  included. What must stay inside is the extrapolated CoM,
                  ``com + v / omega`` with ``omega = sqrt(g / h)`` -- the point
                  the robot would have to put a foot on to stop dead. Inside the
                  polygon means the robot can still stop; outside means it is
                  committed to a step it may not be able to take.

                The CoM velocity behind that is measured against the STANCE FOOT,
                not the torso: the torso is the coordinate frame, so its own
                motion is invisible in it, and only the planted sole is actually
                attached to the ground.

``momentum``    Whatever the kind, a clip that ENDS moving hands the controller a
                robot that is still going somewhere. 0.18 m/s of torso speed is
                30 mm of further travel against a 40-60 mm margin, and the
                balance loop inherits it mid-stride.

``posture``     The clip opens and closes near the controller's own standing
                pose. Every Webots walk clip opens in a crouch of u~0.51 (knee
                1.042 rad) while the controller stands at u~0.10 (knee 0.20), and
                bridging that gap is the ``prepare:`` ramp -- 0.7 s of squatting
                before every single action, and the handover back is where 13 of
                13 recorded transitions drove the support margin to between
                -0.046 and -0.091 m, with all three falls of that session landing
                within 0.13-1.32 s of one.

``legs_only``   The clip drives ONLY the 12 leg joints. Four of the clips Webots
                ships (Backwards, TurnLeft180, SideStepLeft, SideStepRight) also
                drive the arms, and Backwards drives the head -- and because
                ``release_to_motion`` suspends whatever a clip owns, playing one
                of those freezes the upper-body imitation for its whole duration.
                A locomotion clip has no business touching the arms.

What it deliberately does NOT check
-----------------------------------
This is a STATIC and QUASI-STATIC certificate, computed from forward kinematics
and known link masses. It models no contact compliance, no servo lag, no slip and
no true dynamics, so it cannot promise a clip will not fall -- it can only prove
that a clip which fails here *will*. A pass is a necessary condition and a
screening tool, not a guarantee; the live test is still the live test. Treating
it as more than that is how a certified clip ends up on the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pose_control_utils import get_default_motor_configs

# The 12 joints a locomotion clip is allowed to drive. Anything else belongs to
# the upper-body imitation, which must keep running THROUGH a clip.
LEG_JOINTS: tuple[str, ...] = tuple(
    f"{side}{joint}"
    for side in ("L", "R")
    for joint in ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch",
                  "AnklePitch", "AnkleRoll")
)

# Standing support margin is about 60 mm on this robot, so 20 mm of required
# margin keeps a third of it in hand for the modelling error this certificate
# openly does not cover (compliance, slip, servo lag).
MIN_SUPPORT_MARGIN_M = 0.020

# The same floor for a dynamic gait's CAPTURE POINT. Lower than the static one on
# purpose: a walking robot spends every step trading margin for progress, and a
# gait held to the standing figure would not walk. 8 mm still leaves the capture
# point inside the sole rather than on its edge.
MIN_CAPTURE_MARGIN_M = 0.008

# sqrt(g / h) for NAO at its standing CoM height (~0.28 m): the rate at which an
# inverted pendulum diverges, and so the constant that turns a velocity into the
# distance the robot will travel before it can stop.
GRAVITY = 9.81

# Torso speed permitted at the FIRST and LAST keyframe. Past this the robot is
# still travelling when the controller takes over and the capture point is
# already outside the foot. 0.05 m/s x (1/5.9 rad/s) = 8 mm of overshoot.
MAX_ENDPOINT_SPEED_MPS = 0.05

# How far a clip's opening/closing pose may sit from the controller's standing
# posture before the `prepare:` ramp becomes long enough to matter.
MAX_POSTURE_GAP_RAD = 0.35

# Fraction of a motor's rated maximum a clip may demand. Below 1.0 because the
# rating is a no-load figure and a leg carries the robot.
MAX_VELOCITY_FRACTION = 0.9


@dataclass
class Violation:
    """One specific thing wrong with a clip, at a specific place in it."""
    check: str
    time_s: float
    joint: str
    detail: str
    value: float
    limit: float

    def __str__(self) -> str:
        where = f"t={self.time_s:6.3f}s"
        who = f" {self.joint}" if self.joint else ""
        return f"[{self.check}] {where}{who}: {self.detail}"


@dataclass
class Certificate:
    """The verdict on one clip, plus the numbers behind it."""
    name: str
    passed: bool
    keyframes: int
    duration_s: float
    violations: list[Violation] = field(default_factory=list)
    min_support_margin_m: float = 0.0
    min_capture_margin_m: float = 0.0
    worst_margin_time_s: float = 0.0
    kind: str = "static"
    peak_velocity_fraction: float = 0.0
    start_speed_mps: float = 0.0
    end_speed_mps: float = 0.0
    posture_gap_rad: float = 0.0
    extra_joints: tuple[str, ...] = ()

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  {self.name:<22s} {self.kind:<7s} {self.keyframes:4d}kf "
            f"{self.duration_s:5.2f}s  CoM {self.min_support_margin_m*1000:+6.1f}mm  "
            f"cap {self.min_capture_margin_m*1000:+6.1f}mm  "
            f"vel {self.peak_velocity_fraction*100:5.1f}%  "
            f"ends {self.start_speed_mps:.3f}/{self.end_speed_mps:.3f}  "
            f"gap {self.posture_gap_rad:.2f}rad"
        )

    def by_check(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.check] = counts.get(v.check, 0) + 1
        return counts


def _standing_leg_pose() -> dict[str, float]:
    """The leg pose the controller holds between clips.

    Read from the same MotorConfig table the driver uses, so "where the robot
    stands" has one definition rather than a copy that can drift out of step.
    """
    configs = get_default_motor_configs()
    return {name: configs[name].rest_angle for name in LEG_JOINTS
            if name in configs}


def capture_points(poses: list[tuple[float, dict[str, float]]], model):
    """Per-keyframe ``(com_xy, capture_xy, com_height)`` in the torso frame.

    The capture point is ``com + v / omega``: where the centre of mass is
    *heading*, expressed as the spot the robot would have to stand on to come to
    rest. ``omega = sqrt(g / h)`` is the inverted pendulum's divergence rate.

    The velocity is taken relative to the STANCE SOLE rather than to the torso.
    That distinction is the whole point: these kinematics are expressed in the
    torso frame, so the torso's own travel is by construction invisible in them
    -- differencing the CoM directly would report a robot striding across the
    room as stationary. The planted sole is the only thing in the model actually
    attached to the ground, so it is the reference. Across a stance exchange the
    reference jumps, so those intervals contribute no velocity (the same rule
    ``balance.clip_torso_speeds`` follows for the same reason).
    """
    import numpy as np

    samples = []
    for t, angles in poses:
        leg = {j: a for j, a in angles.items() if j in LEG_JOINTS}
        frames = model.frames(leg)
        com = model.com(leg, frames)
        lows = {side: float(model.foot_corners(side, frames)[:, 2].min())
                for side in ("L", "R")}
        stance = "L" if lows["L"] <= lows["R"] else "R"
        sole = model.foot_sole_center(stance, frames)
        height = max(0.05, float(model.com_height(frames, com)))
        samples.append((t, leg, frames, np.asarray(com), np.asarray(sole),
                        stance, height))

    out = []
    for index, (_t, _leg, _frames, com, _sole, stance, height) in enumerate(samples):
        omega = math.sqrt(GRAVITY / height)
        velocity = np.zeros(2)
        lo, hi = max(0, index - 1), min(len(samples) - 1, index + 1)
        span = samples[hi][0] - samples[lo][0]
        if span > 1e-9 and samples[lo][5] == stance and samples[hi][5] == stance:
            rel_lo = samples[lo][3][:2] - samples[lo][4][:2]
            rel_hi = samples[hi][3][:2] - samples[hi][4][:2]
            velocity = (rel_hi - rel_lo) / span
        out.append((com[:2], com[:2] + velocity / omega, height))
    return out


def certify(
    name: str,
    poses: list[tuple[float, dict[str, float]]],
    *,
    kind: str = "static",
    min_margin_m: float = MIN_SUPPORT_MARGIN_M,
    min_capture_margin_m: float = MIN_CAPTURE_MARGIN_M,
    max_endpoint_speed: float = MAX_ENDPOINT_SPEED_MPS,
    max_posture_gap: float = MAX_POSTURE_GAP_RAD,
    max_velocity_fraction: float = MAX_VELOCITY_FRACTION,
    require_legs_only: bool = True,
    cyclic: bool = False,
) -> Certificate:
    """Check one clip against this robot's own geometry. See the module docstring.

    ``poses`` is ``[(seconds, {joint: radians})]`` -- the form
    ``walk_motion.motion_poses`` returns, so a clip on disk and a clip being
    generated are certified by exactly the same code path. That matters: a
    generator that certified its own output through a different route would be
    grading its own homework.

    ``cyclic`` relaxes the endpoint checks for a clip meant to be LOOPED rather
    than played once. A gait's opening and closing poses are mid-stride by
    design, so demanding they match the standing posture would reject exactly the
    clips that make continuous walking possible; the loop seam is checked instead.
    """
    try:
        from balance import NaoCoMModel, clip_torso_speeds
    except Exception as exc:  # noqa: BLE001 - numpy missing, etc.
        return Certificate(name=name, passed=False, keyframes=len(poses),
                           duration_s=poses[-1][0] if poses else 0.0,
                           violations=[Violation("model", 0.0, "",
                                                 f"cannot load the CoM model: {exc}",
                                                 0.0, 0.0)])

    if len(poses) < 2:
        return Certificate(name=name, passed=False, keyframes=len(poses),
                           duration_s=0.0,
                           violations=[Violation("format", 0.0, "",
                                                 "a clip needs at least 2 keyframes",
                                                 len(poses), 2)])

    configs = get_default_motor_configs()
    model = NaoCoMModel()
    violations: list[Violation] = []
    dynamic = kind == "dynamic"
    try:
        capture = capture_points(poses, model)
    except Exception as exc:  # noqa: BLE001
        violations.append(Violation("support", 0.0, "",
                                    f"capture point not computable: {exc}", 0.0, 0.0))
        capture = [(None, None, 0.28)] * len(poses)

    # --- legs only -------------------------------------------------------
    driven = set()
    for _t, angles in poses:
        driven.update(angles)
    extra = tuple(sorted(j for j in driven if j not in LEG_JOINTS))
    if extra and require_legs_only:
        violations.append(Violation(
            "legs_only", 0.0, "",
            f"also drives {len(extra)} non-leg joint(s): {', '.join(extra)} "
            "-- playing this clip suspends the upper-body imitation",
            len(extra), 0))

    # --- per-keyframe: limits, velocity, support -------------------------
    min_margin = math.inf
    worst_margin_t = 0.0
    peak_fraction = 0.0
    previous: tuple[float, dict[str, float]] | None = None
    min_capture = math.inf
    worst_capture_t = 0.0

    for index, (t, angles) in enumerate(poses):
        leg_angles = {j: a for j, a in angles.items() if j in LEG_JOINTS}

        for joint, angle in angles.items():
            cfg = configs.get(joint)
            if cfg is None:
                continue
            if not math.isfinite(angle):
                violations.append(Violation("limits", t, joint,
                                            "non-finite angle", angle, 0.0))
            elif angle < cfg.min_angle - 1e-6 or angle > cfg.max_angle + 1e-6:
                violations.append(Violation(
                    "limits", t, joint,
                    f"{math.degrees(angle):+.1f} deg is outside "
                    f"[{math.degrees(cfg.min_angle):+.1f}, "
                    f"{math.degrees(cfg.max_angle):+.1f}]",
                    angle, cfg.max_angle))

        if previous is not None:
            dt = t - previous[0]
            if dt > 1e-9:
                for joint, angle in angles.items():
                    cfg = configs.get(joint)
                    before = previous[1].get(joint)
                    if cfg is None or before is None:
                        continue
                    speed = abs(angle - before) / dt
                    fraction = speed / cfg.max_velocity if cfg.max_velocity else 0.0
                    peak_fraction = max(peak_fraction, fraction)
                    if fraction > max_velocity_fraction:
                        violations.append(Violation(
                            "velocity", t, joint,
                            f"needs {speed:.2f} rad/s = {fraction*100:.0f}% of "
                            f"the motor's {cfg.max_velocity:.2f} rad/s",
                            speed, cfg.max_velocity * max_velocity_fraction))

        # Support. Computed on the leg pose alone: the arms are a small fraction
        # of the mass and are driven by the imitation rather than by the clip, so
        # a clip cannot make promises about where they are.
        #
        # BOTH numbers are always measured; only which one GATES depends on the
        # kind. A static clip that needs its capture point, or a gait whose CoM
        # never leaves the polygon, are both worth being able to see.
        com_xy, capture_xy, _h = capture[index]
        try:
            frames = model.frames(leg_angles)
            margin = float(model.support_margin(leg_angles, frames))
            capture_margin = (margin if capture_xy is None else
                              float(model.support_margin(leg_angles, frames,
                                                         com_xy=capture_xy)))
        except Exception as exc:  # noqa: BLE001
            violations.append(Violation("support", t, "",
                                        f"margin not computable: {exc}", 0.0, 0.0))
        else:
            if margin < min_margin:
                min_margin, worst_margin_t = margin, t
            if capture_margin < min_capture:
                min_capture, worst_capture_t = capture_margin, t
            if dynamic:
                if capture_margin < min_capture_margin_m:
                    violations.append(Violation(
                        "support", t, "",
                        f"capture point {capture_margin*1000:+.1f} mm inside the "
                        f"support polygon, under the "
                        f"{min_capture_margin_m*1000:.0f} mm floor -- the robot "
                        "could not stop here",
                        capture_margin, min_capture_margin_m))
            elif margin < min_margin_m:
                violations.append(Violation(
                    "support", t, "",
                    f"centre of mass {margin*1000:+.1f} mm inside the support "
                    f"polygon, under the {min_margin_m*1000:.0f} mm floor",
                    margin, min_margin_m))
        previous = (t, angles)

    # --- momentum at the ends -------------------------------------------
    try:
        speeds = clip_torso_speeds(poses, model)
    except Exception:  # noqa: BLE001
        speeds = [0.0] * len(poses)
    start_speed = abs(speeds[0]) if speeds else 0.0
    end_speed = abs(speeds[-1]) if speeds else 0.0
    if not cyclic:
        for label, speed, t in (("opens", start_speed, poses[0][0]),
                                ("closes", end_speed, poses[-1][0])):
            if speed > max_endpoint_speed:
                violations.append(Violation(
                    "momentum", t, "",
                    f"{label} at {speed:.3f} m/s; the capture point is "
                    f"{speed/5.9*1000:.0f} mm beyond the centre of mass when the "
                    "controller takes over",
                    speed, max_endpoint_speed))

    # --- posture at the ends ---------------------------------------------
    standing = _standing_leg_pose()
    gap = 0.0
    for _label, (_t, angles) in (("first", poses[0]), ("last", poses[-1])):
        for joint, rest in standing.items():
            if joint in angles:
                gap = max(gap, abs(angles[joint] - rest))
    if not cyclic and gap > max_posture_gap:
        violations.append(Violation(
            "posture", 0.0, "",
            f"opens/closes {gap:.2f} rad from the standing pose; the controller "
            f"has to ramp into and out of that on every play",
            gap, max_posture_gap))

    # --- the loop seam, for a clip meant to be cycled ---------------------
    if cyclic:
        seam = max(abs(poses[-1][1].get(j, 0.0) - poses[0][1].get(j, 0.0))
                   for j in LEG_JOINTS)
        if seam > 1e-3:
            violations.append(Violation(
                "seam", poses[-1][0], "",
                f"first and last keyframe differ by {seam:.4f} rad, so looping "
                "it steps the joints at the seam",
                seam, 1e-3))

    return Certificate(
        name=name,
        passed=not violations,
        keyframes=len(poses),
        duration_s=poses[-1][0] - poses[0][0],
        violations=violations,
        min_support_margin_m=0.0 if min_margin is math.inf else min_margin,
        min_capture_margin_m=0.0 if min_capture is math.inf else min_capture,
        worst_margin_time_s=worst_capture_t if dynamic else worst_margin_t,
        kind=kind,
        peak_velocity_fraction=peak_fraction,
        start_speed_mps=start_speed,
        end_speed_mps=end_speed,
        posture_gap_rad=gap,
        extra_joints=extra,
    )


def certify_file(path: str, **kwargs) -> Certificate:
    """Certify a ``.motion`` file on disk."""
    import os

    from walk_motion import motion_poses

    return certify(os.path.basename(path), motion_poses(path), **kwargs)
