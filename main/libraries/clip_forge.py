"""Generate NAO locomotion clips from the robot's own geometry, not by hand.

Why generate rather than author
-------------------------------
The clips Webots ships are animations: keyframes drawn for a robot, at a fixed
speed, with no record of why any pose is where it is. Certified against this
robot's own mass model (see :mod:`clip_safety`), all ten of them leave the
support polygon -- centre of mass by 8-23 mm, capture point by 24-55 mm -- and
all ten open and close about 1.05 rad away from the pose the controller stands
in, which is the gap the ``prepare:`` ramp and the clip-to-pose handover have to
bridge on every single play. Four of them also drive the arms, which suspends the
upper-body imitation for the clip's whole duration.

None of that is fixable by editing keyframes, because there is nothing in a
``.motion`` file that says what the author was trying to achieve. Generating the
clips instead means the intent is the code: a squat is "crouch to depth u and
come back", and the keyframes fall out of NAO's link lengths.

The design rule every clip here follows
---------------------------------------
**The robot must be able to stop at any instant.** Concretely: the capture point
stays inside the support polygon for the whole clip, so freezing the legs at any
keyframe leaves a robot that settles rather than one that keeps going. That is a
strictly stronger promise than the shipped clips make, and it is what makes an
early exit safe -- which in turn is what lets the controller stop following a
human who has stopped, instead of committing to a fixed 2.6 s of animation.

Where a motion can be done quasi-statically (squat, leg raise, turning in place)
this module goes further and keeps the CENTRE OF MASS itself inside the polygon,
not merely the capture point. Those clips are then stable at any playback speed,
including stopped.

Solving rather than deriving
-----------------------------
The lateral weight shift needed to stand on one foot is not written out
algebraically here. It is SOLVED, per pose, against :class:`balance.NaoCoMModel`
-- the same model the certifier and the live balance loop use. Hand-derived
kinematics is exactly the kind of thing that is right in the docstring and wrong
in the code, and a search over one scalar costs nothing offline.
"""
from __future__ import annotations

import math

from pose_control_utils import get_default_motor_configs

# Joints a generated clip drives. Leg only, always -- the arms belong to the
# imitation and must keep tracking straight through a clip.
LEG_JOINTS: tuple[str, ...] = tuple(
    f"{side}{joint}"
    for side in ("L", "R")
    for joint in ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch",
                  "AnklePitch", "AnkleRoll")
)

# Webots' .motion keyframes are 40 ms apart. Matching that is not cosmetic: the
# player interpolates between keyframes at the simulation step, so a coarser clip
# is a coarser motion, and the certifier's velocity check reads the same spacing
# the robot will actually execute.
KEYFRAME_DT = 0.04

Pose = dict[str, float]
Clip = list[tuple[float, Pose]]


def standing_pose() -> Pose:
    """The leg pose the controller holds between clips.

    Every generated clip starts and ends here, which is the whole point: the
    ``prepare:`` ramp shrinks to nothing and the handover back to pose-imitation
    has no posture jump to absorb. The shipped clips sit 1.05 rad away from it.
    """
    configs = get_default_motor_configs()
    return {name: configs[name].rest_angle for name in LEG_JOINTS}


def crouch(u: float) -> Pose:
    """Symmetric sole-flat crouch of depth ``u`` (rad of hip pitch).

    ``Hip = -u, Knee = +2u, Ankle = -u``. Because NAO's thigh and shank are
    within 3 mm of the same length, this keeps the ankle under the hip -- and so
    the centre of mass over the feet -- at ANY depth, with the torso vertical and
    both soles flat. Measured against the mass model it holds +57.6 mm of fore/aft
    margin at u=0 and +64.5 mm at u=0.5, i.e. crouching actually improves it.
    The same relation ``nao_retarget.crouch_posture`` uses.
    """
    pose = dict.fromkeys(LEG_JOINTS, 0.0)
    for side in ("L", "R"):
        pose[f"{side}HipPitch"] = -u
        pose[f"{side}KneePitch"] = 2.0 * u
        pose[f"{side}AnklePitch"] = -u
    return pose


def lateral_shift(pose: Pose, phi: float) -> Pose:
    """Lean the body sideways by ``phi`` while keeping both soles flat.

    Rolling the hips and counter-rolling the ankles by the same angle leaves the
    torso-to-foot rotation at zero, so both soles stay flat while the body
    translates sideways over them. Flat matters: a foot up on its edge has almost
    no support polygon left, which is what the live balance loop's sole-tilt
    budget exists to prevent.

    **Positive ``phi`` moves the mass toward the robot's RIGHT foot**, measured
    rather than derived -- at ``phi = -0.30`` the centre of mass sits +34.8 mm
    inside the LEFT foot's rectangle, against -12.0 mm for both feet at
    ``phi = 0``. Reasoning it out from the joint sign conventions gives the
    opposite answer, because in the torso frame it is the feet that appear to
    swing while the body stays put. Do not "fix" this sign from first
    principles; :func:`solve_weight_shift` reads the model, not this comment.
    """
    out = dict(pose)
    for side in ("L", "R"):
        out[f"{side}HipRoll"] = pose.get(f"{side}HipRoll", 0.0) + phi
        out[f"{side}AnkleRoll"] = pose.get(f"{side}AnkleRoll", 0.0) - phi
    return out


def solve_weight_shift(base: Pose, side: str, model=None,
                       search_rad: float = 0.45, steps: int = 91,
                       min_margin: float | None = None) -> float:
    """Lateral lean that puts the mass over ``side``'s foot.

    With ``min_margin`` set, returns the SMALLEST lean that achieves that much
    stance margin instead of the one that maximises it. That distinction is not
    a refinement, it is what makes a side step possible at all: both the lean and
    the leg abduction are paid for out of the same ``AnkleRoll`` budget
    (-22.8..+44.1 deg on the left), and the maximising lean spends 18.3 deg of it
    on margin nobody asked for. Stacking a 17 deg abduction on top then lands the
    ankle at -35.5 deg, outside its limit -- so the clip is silently clamped and
    the robot plays a pose whose balance was never computed.

    Asking for enough margin rather than the most margin leaves the rest of the
    budget for the step.

    Solved against the real mass model rather than derived: the answer depends on
    hip spacing, sole geometry and every link mass above them, and a closed form
    for it would be a second, divergent copy of :mod:`balance`. One scalar, swept
    at ~0.01 rad, costs microseconds offline and cannot drift out of step with
    the model the robot actually balances against.

    Returns the lean in radians, signed as :func:`lateral_shift` expects.
    """
    from balance import NaoCoMModel

    model = model or NaoCoMModel()
    best_phi, best_margin = 0.0, -math.inf
    cheapest: tuple[float, float] | None = None
    for index in range(steps):
        phi = -search_rad + 2.0 * search_rad * index / (steps - 1)
        candidate = lateral_shift(base, phi)
        if not _within_limits(candidate):
            continue
        try:
            margin = float(model.stance_margin(candidate, side))
        except Exception:  # noqa: BLE001
            continue
        if margin > best_margin:
            best_phi, best_margin = phi, margin
        if min_margin is not None and margin >= min_margin:
            if cheapest is None or abs(phi) < abs(cheapest[0]):
                cheapest = (phi, margin)
    if min_margin is not None:
        # Falling back to the maximising lean when nothing reaches the target is
        # deliberate: the caller asked to stand on that foot, and the best
        # available answer beats refusing to move.
        return cheapest[0] if cheapest else best_phi
    return best_phi


def _within_limits(pose: Pose) -> bool:
    configs = get_default_motor_configs()
    for joint, angle in pose.items():
        cfg = configs.get(joint)
        if cfg is None:
            continue
        if not math.isfinite(angle) or angle < cfg.min_angle or angle > cfg.max_angle:
            return False
    return True


def blend(a: Pose, b: Pose, s: float) -> Pose:
    """Linear interpolation between two poses at ``s`` in [0, 1]."""
    return {j: a.get(j, 0.0) + (b.get(j, 0.0) - a.get(j, 0.0)) * s
            for j in set(a) | set(b)}


def ease(s: float) -> float:
    """Smoothstep: zero velocity at both ends of a segment.

    Every segment boundary in these clips is eased rather than linear, and that
    is a balance property rather than a cosmetic one. A linear ramp starts and
    stops with a velocity STEP, which the certifier's capture-point test sees as
    the robot acquiring momentum instantaneously -- and the motor sees as a
    demand it cannot meet, so the pose the clip's balance was computed for is not
    the pose the robot reaches.
    """
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def segment(start: Pose, end: Pose, duration_s: float,
            dt: float = KEYFRAME_DT, easing: bool = True) -> list[Pose]:
    """Poses stepping from ``start`` to ``end`` over ``duration_s`` (end excluded)."""
    count = max(1, int(round(duration_s / dt)))
    out = []
    for index in range(count):
        s = index / count
        out.append(blend(start, end, ease(s) if easing else s))
    return out


def timed(poses: list[Pose], dt: float = KEYFRAME_DT) -> Clip:
    """Attach timestamps to a pose sequence."""
    return [(round(index * dt, 3), pose) for index, pose in enumerate(poses)]


# ---------------------------------------------------------------------------
# The clips
# ---------------------------------------------------------------------------
def make_squat(depth_u: float = 0.75, down_s: float = 1.2, hold_s: float = 0.4,
               up_s: float = 1.2) -> Clip:
    """Crouch to ``depth_u`` and stand back up.

    Quasi-static throughout: the crouch relation keeps the mass over the feet at
    every depth, so this clip is stable stopped at any keyframe.

    ``depth_u`` is hip-pitch radians, and it has to be larger than intuition
    suggests. NAO's thigh is 100 mm and its tibia 103 mm, so hip height goes as
    ``0.203 * cos(u)`` -- a 0.45 rad crouch lowers the mass by only 16 mm, which
    on a 58 cm robot is invisible. 0.75 rad gives about 54 mm and still leaves
    room: it puts the hip at -0.75 against a -0.88 limit and the knee at 1.5
    against 2.11.
    """
    stand = standing_pose()
    bottom = crouch(depth_u)
    poses = segment(stand, bottom, down_s)
    poses += [bottom] * max(1, int(round(hold_s / KEYFRAME_DT)))
    poses += segment(bottom, stand, up_s)
    poses.append(stand)
    return timed(poses)


def make_leg_raise(side: str, lift_rad: float = 0.50, knee_rad: float = 0.90,
                   shift_s: float = 1.0, lift_s: float = 0.8, hold_s: float = 0.6,
                   model=None) -> Clip:
    """Stand on one foot and raise the other knee.

    Three phases, and the first one is the whole trick: **shift the weight before
    lifting anything**. Lifting first and shifting after is how a robot falls
    sideways, and it is what the live per-leg imitation was doing -- a one-leg
    read leaning the hip to -0.92 rad while the mass was still between the feet.

    ``side`` is the leg that LIFTS, so the weight goes to the other one. The lean
    is solved against the mass model (:func:`solve_weight_shift`), not assumed.
    """
    stance = "R" if side == "L" else "L"
    stand = standing_pose()
    # A slight crouch first: bending the stance knee lowers the mass and gives
    # the ankle roll something to work against. Standing bolt upright on one
    # locked leg is the least controllable way to do this.
    base = crouch(0.25)
    phi = solve_weight_shift(base, stance, model)
    shifted = lateral_shift(base, phi)

    lifted = dict(shifted)
    lifted[f"{side}HipPitch"] = shifted.get(f"{side}HipPitch", 0.0) - lift_rad
    lifted[f"{side}KneePitch"] = shifted.get(f"{side}KneePitch", 0.0) + knee_rad
    # Keep the raised sole level with the floor rather than dangling toes-down:
    # a level foot lands flat if the clip is cut short.
    lifted[f"{side}AnklePitch"] = -(lifted[f"{side}HipPitch"]
                                    + lifted[f"{side}KneePitch"])

    poses = segment(stand, base, 0.5)
    poses += segment(base, shifted, shift_s)
    poses += segment(shifted, lifted, lift_s)
    poses += [lifted] * max(1, int(round(hold_s / KEYFRAME_DT)))
    poses += segment(lifted, shifted, lift_s)
    poses += segment(shifted, base, shift_s)
    poses += segment(base, stand, 0.5)
    poses.append(stand)
    return timed(poses)


def write_motion(path: str, clip: Clip, joints: list[str] | None = None) -> str:
    """Write ``clip`` as a Webots ``.motion`` file. Returns the path."""
    names = joints or list(LEG_JOINTS)
    lines = ["#WEBOTS_MOTION,V1.0," + ",".join(names)]
    for index, (t, pose) in enumerate(clip):
        ms = int(round(t * 1000.0))
        stamp = f"{ms // 60000:02d}:{ms // 1000 % 60:02d}:{ms % 1000:03d}"
        lines.append(f"{stamp},Pose{index + 1},"
                     + ",".join(f"{pose.get(j, 0.0):.6f}" for j in names))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Working from an existing clip
# ---------------------------------------------------------------------------
# Walking and turning are genuinely dynamic: the gait's balance comes from
# momentum and foot placement, not from a pose that is stable standing still, and
# re-deriving that from scratch is a research project rather than an afternoon.
# Cyberbotics' clips DO walk the robot, so the honest move is to keep their
# dynamics and fix what is demonstrably wrong with them as assets:
#
#   * they drive the arms (4 of 10), which suspends the upper-body imitation;
#   * they open and close ~1.05 rad from the pose the controller stands in, and
#     that gap is what the `prepare:` ramp and the handover have to absorb -- the
#     handover being where 13 of 13 recorded transitions pushed the support
#     margin negative, with every fall of that session inside 1.32 s of one.
#
# Both are properties of the file, not of the gait, so both can be fixed without
# touching a single mid-clip keyframe.

# Rate the bookend ramps run at, in rad/s of the fastest-moving joint.
#
# This was 1.0, and 1.0 was too slow twice over. Cyberbotics' clips open about
# 1.05 rad from the standing pose, so each ramp took 1.05 s and a clip carried
# 2.1 s of pure approach: SideStepLeft went 4.92 s -> 7.04 s and SideStepRight
# 5.76 s -> 8.08 s, which is most of the "the side step is very slow" report.
#
# The second cost was worse and less obvious. A slow ramp is SLOW MOTION, and
# ``balance.safe_exit_times`` marks slow keyframes as safe places to abandon a
# clip -- so the ramp handed the arbiter 30 legal exits inside the first 1.2 s
# against 19-20 for the un-bookended original. The arbiter takes the first exit
# it is offered once a clip is no longer wanted, so clips were being dropped
# INSIDE their own approach, before reaching a single keyframe that did
# anything: measured live, 10 of 99 plays ran to completion, side_right ran
# 0.26 s of 8.08 s and moved the robot 19.5 mm.
#
# 2.0 rad/s halves both. It is not free-for-all: ``segment`` eases the ramp, so
# the PEAK rate is 1.5x this, and the slowest leg motor (AnkleRoll, 4.16 rad/s)
# then sees 3.0 rad/s = 72% of its rating -- inside the certifier's 90% ceiling
# with room to spare. Faster than this starts failing certification on the ankle
# rather than on the knee, which is the sort of thing that looks fine until the
# one clip that leans hardest gets rejected.
RAMP_RATE_RAD_S = 2.0

# Floor on a ramp's duration. A ramp shorter than this is a step change however
# small the travel, and the certifier's momentum check reads it as the robot
# being handed over already moving.
RAMP_FLOOR_S = 0.25


def _ramp_duration(a: Pose, b: Pose, max_rate: float = RAMP_RATE_RAD_S,
                   floor_s: float = RAMP_FLOOR_S) -> float:
    """Time to move between two poses without exceeding ``max_rate`` rad/s."""
    travel = max((abs(b.get(j, 0.0) - a.get(j, 0.0)) for j in LEG_JOINTS),
                 default=0.0)
    return max(floor_s, travel / max_rate)


def refine_clip(clip: Clip, *, max_rate: float = RAMP_RATE_RAD_S,
                dt: float = KEYFRAME_DT) -> Clip:
    """Strip a clip to the legs and bookend it with ramps from/to standing.

    The gait itself is untouched -- every original keyframe survives, at its
    original spacing, so whatever made the clip walk still makes it walk. What
    changes is the two ends and the joint set.

    Bookending here rather than at runtime is the point. The controller's
    `prepare:` state already ramps into a clip's opening pose, but it does so
    blind: it cannot ease, it cannot be certified, and it is charged fresh on
    every play. Baked into the clip, the same approach is eased (zero velocity at
    both ends), is checked by :mod:`clip_safety` along with everything else, and
    leaves a clip whose first and last keyframe ARE the standing pose -- so there
    is no posture step left for the handover to absorb.
    """
    stand = standing_pose()
    core = [{j: pose.get(j, stand[j]) for j in LEG_JOINTS} for _t, pose in clip]
    if not core:
        return timed([stand, stand], dt)

    lead = segment(stand, core[0], _ramp_duration(stand, core[0], max_rate), dt)
    tail = segment(core[-1], stand, _ramp_duration(core[-1], stand, max_rate), dt)
    return timed(lead + core + tail + [stand], dt)


# Joints whose sign flips when a pose is mirrored across the sagittal plane.
#
# Mirroring maps a rotation ``R(a, theta)`` to ``R(M a, -theta)`` where ``M`` is
# the reflection. So whether a joint's angle negates depends entirely on how its
# AXIS sits relative to the mirror:
#
#   HipRoll, AnkleRoll   axis (1,0,0) on both legs; M leaves it alone, so the
#                        angle negates.
#   Hip/Knee/AnklePitch  axis (0,1,0); M maps it to its own negative, and the
#                        two sign flips cancel -- the angle does NOT negate.
#   HipYawPitch          the subtle one. Its axis is the 45-degree
#                        (0, .707, -.707*sign) from Nao.urdf, which is ALREADY
#                        opposite between the legs -- ``M a_left == -a_right``.
#                        That built-in flip cancels the mirror's own, so this
#                        angle does NOT negate either.
#
# Getting that last one wrong is not a small error and it is not obvious in the
# output: negating HipYawPitch mirrored TurnLeft180 into a clip whose centre of
# mass sat -58.6 mm outside the support polygon against the original's -18.0 mm,
# while the opening and closing keyframes -- which carry no hip yaw -- mirrored
# perfectly and looked fine. Pinned by a symmetry test rather than by this
# comment: NAO is symmetric, so a mirrored pose must report the same margins.
_MIRROR_NEGATED = ("HipRoll", "AnkleRoll")


def mirror_pose(pose: Pose) -> Pose:
    """Left-right mirror of one pose."""
    out: Pose = {}
    for joint, angle in pose.items():
        side = joint[0]
        if side not in ("L", "R"):
            out[joint] = angle
            continue
        rest = joint[1:]
        flipped = ("R" if side == "L" else "L") + rest
        out[flipped] = -angle if rest in _MIRROR_NEGATED else angle
    return out


def mirror_clip(clip: Clip) -> Clip:
    """Left-right mirror of a whole clip.

    Turns TurnLeft180 into the TurnRight180 Webots does not ship -- and the
    asymmetry mattered: ``plan_action`` picks the LARGEST turn clip that will not
    overshoot, so without a coarse right turn an about-face to the right had to
    be served by repeated 60 deg clips, each with its own ramp, settle and
    handover. A mirrored clip is balanced exactly as well as its original,
    because NAO is symmetric about the same plane.
    """
    return [(t, mirror_pose(pose)) for t, pose in clip]


def peak_velocity_fraction(clip: Clip) -> float:
    """Highest fraction of any motor's rated speed the clip demands."""
    configs = get_default_motor_configs()
    peak = 0.0
    for (t0, a), (t1, b) in zip(clip, clip[1:], strict=False):
        dt = t1 - t0
        if dt <= 1e-9:
            continue
        for joint, angle in b.items():
            cfg = configs.get(joint)
            if cfg is None or cfg.max_velocity <= 0 or joint not in a:
                continue
            peak = max(peak, abs(angle - a[joint]) / dt / cfg.max_velocity)
    return peak


def retime_for_velocity(clip: Clip, target_fraction: float = 0.85) -> Clip:
    """Stretch a clip in time until no motor is asked for more than it has.

    ``Forwards50`` -- the only clip on this install with a real limit cycle in
    it, and so the only one that can walk continuously rather than in 2.6 s
    bursts -- demands 6.55 rad/s of ``LKneePitch`` against a 6.40 rad/s motor.
    Only 2% over, and the temptation is to wave it through.

    It should not be waved through, and the reason is not the 2%. A motor that
    cannot reach a keyframe in the time allowed simply arrives late, so the robot
    is in a DIFFERENT pose from the one the clip's author balanced -- and it is
    late by the most at exactly the moment the clip is moving fastest, which is
    when it can least afford to be somewhere else. Stretching time is the honest
    repair: every pose is still reached, still in the same order, just at a speed
    the hardware actually has. The clip gets slower; it does not get wrong.

    Uniform stretching also leaves any limit cycle intact -- ``gait_cycle``
    matches on the joint values, which are untouched -- so a cyclic clip stays
    cyclic and simply cycles more slowly.
    """
    peak = peak_velocity_fraction(clip)
    if peak <= target_fraction or peak <= 0.0:
        return clip

    # Iterate rather than solve in one step. A .motion timestamp is whole
    # milliseconds, so the stretched clip is QUANTISED -- at 40 ms keyframes that
    # is 2.5% of granularity, which is easily enough to land a clip computed to
    # sit exactly on the target just above it. Re-measuring the rounded result
    # and nudging again converges in a step or two and cannot be fooled by the
    # rounding, which a closed-form factor can.
    stretch = peak / target_fraction
    for _attempt in range(8):
        candidate = [(round(t * stretch, 3), pose) for t, pose in clip]
        if peak_velocity_fraction(candidate) <= target_fraction:
            return candidate
        stretch *= 1.01
    return candidate


def abduct(pose: Pose, side: str, width: float) -> Pose:
    """Swing one leg out sideways by ``width`` rad, keeping its sole flat.

    Rolling the hip out and the ankle back by the same angle leaves the foot
    parallel to the floor while the leg travels, which is what lets the foot be
    PLANTED at the end of the swing instead of landed on its edge.
    """
    out = dict(pose)
    sign = 1.0 if side == "L" else -1.0
    out[f"{side}HipRoll"] = pose.get(f"{side}HipRoll", 0.0) + sign * width
    out[f"{side}AnkleRoll"] = pose.get(f"{side}AnkleRoll", 0.0) - sign * width
    return out


def make_side_step(side: str, width_rad: float = 0.18, lift_rad: float = 0.22,
                   crouch_u: float = 0.25, stance_margin_m: float = 0.012,
                   model=None) -> Clip:
    """Step sideways toward ``side``, quasi-statically.

    Webots' own SideStep clips do work, but they are dynamic animations and they
    cost more than they look. Certified against this robot they leave the support
    polygon by 20 mm (centre of mass) and 48 mm (capture point), they run 4.9-5.8
    s before any approach is added, and -- measured live -- they were abandoned
    after 0.26 s having moved the robot 19.5 mm, because a clip the arbiter can
    legally exit early WILL be exited early.

    This one is built instead, and the difference that matters is not the length:
    it is **quasi-static**. The centre of mass stays inside the support polygon
    at every keyframe, so stopping it anywhere leaves a robot standing up. A clip
    that is safe to abandon cannot be ruined by being abandoned, which turns the
    arbiter's early exit from a failure mode into a feature.

    Six phases, and the ORDER is the whole thing -- each weight transfer finishes
    before the foot it frees is allowed to leave the floor:

        1. sink into a shallow crouch (gives the ankles authority)
        2. shift the mass onto the TRAILING foot          <- solved, not assumed
        3. lift the leading foot, swing it out, plant it
        4. shift the mass onto the now-leading foot       <- solved
        5. lift the trailing foot, close the stance, plant it
        6. rise back to standing

    ``side`` is the direction travelled: "L" steps to the robot's left, so the
    LEFT leg leads and the right follows.
    """
    lead = side
    trail = "R" if side == "L" else "L"
    stand = standing_pose()
    base = crouch(crouch_u)

    # Each lean is solved for the pose it will actually hold, INCLUDING whatever
    # the swinging leg is doing at the time. Solving it on the neutral pose and
    # abducting afterwards is the obvious order and it is wrong: the swing leg is
    # about a sixth of the robot's mass, so carrying it 0.3 rad outward moves the
    # centre of mass after the balance was computed. Measured, that error was the
    # whole margin -- +11 mm at a 0.12 rad step became -19 mm at 0.30 rad, purely
    # from the mass the solve had not been told about.
    def swing(pose: Pose, leg: str) -> Pose:
        """``pose`` with ``leg`` lifted clear of the floor, sole level."""
        out = dict(pose)
        out[f"{leg}HipPitch"] = pose.get(f"{leg}HipPitch", 0.0) - lift_rad
        out[f"{leg}KneePitch"] = pose.get(f"{leg}KneePitch", 0.0) + 2.0 * lift_rad
        out[f"{leg}AnklePitch"] = -(out[f"{leg}HipPitch"] + out[f"{leg}KneePitch"])
        return out

    def lean_for(pose: Pose, foot: str) -> float:
        return solve_weight_shift(pose, foot, model, min_margin=stance_margin_m)

    # ONE lean per support phase, held while the foot moves.
    #
    # The obvious construction re-solves the lean for every keyframe, which means
    # the swing foot descends WHILE the body is already shifting toward it -- and
    # for the moments before touchdown the support polygon is still one foot
    # while the mass is on its way to the other. Measured, that transition was
    # the only negative margin in the whole clip: -2.5 mm at t=1.72 s, against
    # +13.5 mm held comfortably through both single-support phases either side of
    # it. Planting the foot first and shifting afterwards costs a quarter of a
    # second and removes it.
    wide = abduct(base, lead, width_rad)
    lead_airborne = abduct(swing(base, lead), lead, width_rad)
    on_trail = lean_for(lead_airborne, trail)
    on_lead = lean_for(wide, lead)

    # Phase 2-3: shift onto the trailing foot, lift, carry out, plant -- all at
    # the SAME lean, so only one thing is moving at a time.
    onto_trail = lateral_shift(swing(base, lead), on_trail)
    lead_up = onto_trail
    lead_out = lateral_shift(lead_airborne, on_trail)
    lead_down = lateral_shift(wide, on_trail)

    # Phase 4: both feet planted and wide; now, and only now, the mass crosses.
    onto_lead = lateral_shift(wide, on_lead)

    # Phase 5: the trailing foot is free -- bring it across to close the stance.
    # Phase 5: the trailing foot is free -- bring it across to close the stance,
    # again at one held lean. Closing gives the lead leg's abduction back,
    # carried now by the trailing leg: the feet end together, one step over.
    trail_airborne = swing(wide, trail)
    on_lead_swinging = lean_for(trail_airborne, lead)
    trail_up = lateral_shift(trail_airborne, on_lead_swinging)
    closed_up = lateral_shift(abduct(trail_airborne, lead, -width_rad),
                              on_lead_swinging)
    closed = lateral_shift(abduct(wide, lead, -width_rad), on_lead_swinging)

    poses = segment(stand, base, 0.35)
    poses += segment(base, onto_trail, 0.45)
    poses += segment(onto_trail, lead_up, 0.25)
    poses += segment(lead_up, lead_out, 0.30)
    poses += segment(lead_out, lead_down, 0.30)
    poses += segment(lead_down, onto_lead, 0.45)
    poses += segment(onto_lead, trail_up, 0.25)
    poses += segment(trail_up, closed_up, 0.30)
    poses += segment(closed_up, closed, 0.25)
    poses += segment(closed, base, 0.35)
    poses += segment(base, stand, 0.35)
    poses.append(stand)
    return timed(poses)


def lateral_travel(clip: Clip, model=None) -> float:
    """Net sideways ground covered by ``clip``, in metres (+ = the robot's left).

    Odometry, exactly as ``balance.clip_torso_speeds`` does it fore/aft: the
    planted sole is the only thing attached to the floor, so the body moves
    against it by however much forward kinematics says the hips travelled over
    it. Only intervals with the SAME stance foot contribute -- across a stance
    exchange the reference jumps and the difference means nothing.

    This exists because a side-step clip that does not actually go sideways is
    perfectly safe and completely useless, and the certifier cannot tell the
    difference: standing still passes every balance check there is.
    """
    from balance import NaoCoMModel

    model = model or NaoCoMModel()
    total = 0.0
    previous = None
    last_stance = None
    for _t, angles in clip:
        legs = {j: a for j, a in angles.items() if j in LEG_JOINTS}
        frames = model.frames(legs)
        lows = {s: float(model.foot_corners(s, frames)[:, 2].min()) for s in ("L", "R")}
        stance = "L" if lows["L"] <= lows["R"] else "R"
        sole = float(model.foot_sole_center(stance, frames)[1])
        if last_stance == stance and previous is not None:
            total += -(sole - previous)
        previous = sole
        last_stance = stance
    return total
