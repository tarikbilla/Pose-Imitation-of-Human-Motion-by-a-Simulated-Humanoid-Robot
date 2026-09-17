"""
Full-body retargeting: MeTRAbs 3D landmarks -> NAO joint angles.

The Python pipeline streams MeTRAbs pose landmarks: absolute METRIC
coordinates in MILLIMETERS, in the camera's coordinate frame (``x`` right,
``y`` down, ``z`` forward/away from the camera -- a standard pinhole-camera
convention), plus a visibility PROXY in [0,1] (MeTRAbs has no true per-joint
confidence; see ``src/perception/pose_estimator.py``). This module converts
them into NAO joint targets for the whole body.

This replaces an earlier version of this file built around MediaPipe, which
gave only a 2D image-plane projection plus an unreliable depth channel, and
so had to *reconstruct* a plausible 3D pose via a frontal-projection
assumption and a self-calibration scheme that learned the subject's
foreshortened proportions over time. With MeTRAbs' real 3D, that
reconstruction is no longer needed -- the angle math below is an exact
closed-form decomposition of a real 3D bone direction, not an approximation
recovered from partial information.

Why retarget here instead of upstream
--------------------------------------
Driving the robot's *full* pose needs the actual limb geometry, not just a
handful of pre-baked angles. Keeping the kinematics next to the robot means
the controller owns everything NAO-specific (joint axes, signs, limits) and
the Python side stays a generic pose source.

Torso-local frame
------------------
NAO's joint angles are defined relative to its OWN body (e.g. "raise the arm
forward" means forward relative to the torso, not relative to the camera).
The old MediaPipe-era code implicitly assumed the subject stood frontal to
the camera, so "camera right" could stand in for "subject's right". With real
depth, the subject may face any direction, so each frame we build an
orthonormal **torso-local basis** (``right``, ``up``, ``forward``) from the
shoulder line and the hip-to-shoulder line:

    right   = normalize(right_shoulder - left_shoulder)
    up_raw  = normalize(mid_shoulder - mid_hip)
    up      = normalize(up_raw - (up_raw . right) * right)      # orthogonalize
    forward = normalize(right x up)                              # faces the camera when frontal

Any bone vector (e.g. hip->knee) is projected onto this basis to get its
(lateral, vertical, forward) components *relative to the subject's own body*,
independent of which way they face the camera.

Swing-twist decomposition
--------------------------
Each 2-DOF joint (shoulder, hip) is solved as two sequential rotations about
fixed local axes -- a closed-form inverse of exactly the kind of forward
kinematics chain NAO's own joints implement. Given a bone's UNIT direction in
the torso-local frame, decomposed into a "swept" component ``a``, a
reference-axis component ``b`` (1.0 at zero rotation), and a "sign" component
``c`` (picks up magnitude only once the second rotation tilts the bone out of
the ``a``/``b`` plane)::

    d              = clamp(b, -1, 1)
    first_angle    = atan2(a, d)
    second_angle   = +-acos(clamp(d / cos(first_angle), -1, 1)), sign from c

For the **legs** (hip/knee), the reference direction is straight down
(``b = -up``), the first angle is HipRoll (``a = lateral, outward-positive``,
about the forward axis) and the second is HipPitch (sign from ``fwd``). This
is the same closed-form relationship the old frontal-projection code used
(see git history) -- the difference is that ``fwd`` is now a real measurement
instead of a noisy sign hint.

For the **arms** (shoulder), the reference direction is straight forward
(``b = fwd``), the first angle is ShoulderPitch (``a = -up``, about the
lateral axis) and the second is ShoulderRoll (sign from the outward-signed
lateral component). Elbow flexion needs no such decomposition -- it is simply
the angle between the upper arm and forearm vectors, which was already exact
even in the old code (a plain dot-product angle, coordinate-frame-agnostic).

Lift / crouch / ground-line
----------------------------
Foot-lift and crouch detection stay in CAMERA-frame vertical (``y``, assuming
a roughly level camera) rather than the torso-local frame: the question
"which foot is on the ground" is about real-world verticality, and answering
it from the torso's own up axis would make a forward lean look like a foot
lift. This mirrors what the old image-``y``-based code effectively assumed
(a level camera), just in real millimeters instead of a normalized image
fraction -- the lift/crouch tuning constants are already expressed as ratios
of leg/torso length, so they carry over unchanged.

Self-calibration
------------------
Segment lengths (thigh, shank, torso) are simply measured directly each frame
in millimeters and lightly EMA-smoothed for jitter. The old MediaPipe-era
``PeakHold`` scheme existed only because a 2D projection can never overstate a
segment's true length (foreshortening only ever makes it look shorter), so
scale had to be learned as a running maximum; MeTRAbs' distances are already
metric, so there is nothing to learn.

Everything is still gated on landmark ``visibility`` so out-of-frame joints
are simply not commanded and the driver holds their last pose. The *safety*
of a leg pose (may the robot actually unload a foot right now?) is
deliberately NOT decided here -- that needs the robot's own CoM/force state
and lives in ``lower_body.LowerBodyController``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from pose_control_utils import JointLimiter, get_default_motor_configs

# Visibility below which a landmark is considered unreliable and its dependent
# joints are skipped.
VIS_THRESHOLD = 0.5

# Tuning gains (kept gentle; joint limits clamp the rest).
HEAD_YAW_GAIN = 1.0
HEAD_PITCH_GAIN = 1.6
HEAD_PITCH_BASELINE = 0.9  # nose sits ~0.9 shoulder-widths above shoulder line

# Per-subject calibration of that baseline (see :class:`HeadGeometry`). A neck
# longer or shorter than the population average shifts the neutral, and the
# robot then holds a permanent nod while the human looks straight ahead.
HEAD_BASELINE_WARMUP = 45    # frames (~1.5 s at 30 FPS) averaged as "neutral"
HEAD_BASELINE_DECAY = 0.002  # afterwards: slow drift, so a NEW subject re-calibrates

# ---------------------------------------------------------------------------
# Lower-body tuning (unchanged from the MediaPipe-era version: these are all
# already expressed as ratios of leg/torso length, so real metric lengths
# carry the same tuning over unchanged).
# ---------------------------------------------------------------------------
LIFT_DEADBAND = 0.030
LIFT_FULL = 0.260
LIFT_FULL_KNEE = 0.190
CROUCH_FULL_DROP = 0.28
KNEE_STRAIGHT_DEADZONE = 0.20  # rad of knee bend treated as "standing straight"
KNEE_BEND_RANGE = 1.30         # rad of human knee bend mapped to full crouch

# --- Elbow: a human elbow out-bends NAO's by 60 deg ------------------------
# A human elbow flexes to about 150 deg; NAO's ElbowRoll stops at 88.5 deg. The
# bend was handed over 1:1, so every bend past 88.5 deg clamped to the stop --
# and a clamped joint has stopped imitating: it holds still through exactly the
# part of the motion the subject is putting the most travel into, which reads as
# "the arm is broken", not "the arm is at its limit". Measured over the
# 2026-09-10 live session (run_20260910_140624, 4653 frames with both arms
# visible): the human bend had a median of 24 deg but a p90 of 68 deg (left) and
# 132 deg (right), putting 8.0% of left-elbow and 19.1% of right-elbow frames
# past the stop; on the robot side the commanded RElbowRoll sat pinned against
# its stop in 26.9% of non-stale frames and LElbowRoll in 9.6%.
#
# So the range is COMPRESSED instead, with a soft knee rather than a straight
# rescale. A straight rescale would fix the saturation by making every ordinary
# gesture smaller -- the median 24 deg bend would come out at 16 deg -- which
# trades a visible fault for a dull one. Below the knee the mapping stays exactly
# 1:1, so normal gestures are untouched; above it the remaining human travel is
# folded into the joint's remaining travel, so the response stays monotone all
# the way to a fully folded arm and never flatlines.
ELBOW_LINEAR_RAD = 1.047      # 60 deg: mapped 1:1 (covers the median and p75)
ELBOW_HUMAN_MAX_RAD = 2.618   # 150 deg: a fully folded human elbow
ELBOW_NAO_MAX_RAD = 1.545     # 88.5 deg: NAO's ElbowRoll mechanical stop
# Deepest symmetric crouch we ask for (hip 40 deg, knee 80 deg). Not a stability
# limit: because NAO's thigh and shank are within 3 mm of the same length, the
# Hip = -u / Knee = +2u / Ankle = -u posture keeps the ankle under the hip -- and
# so the CoM over the foot -- at ANY depth, with the torso vertical and the soles
# flat throughout. The real ceiling is the knee's own 121 deg range.
MAX_CROUCH = 0.70

# Below this the swing angle is undefined -- the bone lies along the second
# joint's own axis -- and only the signed zeros of atan2 would decide it.
SWING_DEGENERATE = 1e-9

# Below this |cos(first_angle)| the second angle is geometrically unobservable
# (the limb points nearly along the rotation axis of the first joint), so we
# report it as 0 rather than a noise-amplified value.
MIN_COS_ROLL = 0.30

# Per-frame EMA smoothing of the directly-measured segment lengths (mm).
GEOMETRY_ALPHA = 0.25

Vec = tuple[float, float, float]
Landmark = tuple[float, float, float, float]  # x, y, z (mm, camera frame), visibility


# ---------------------------------------------------------------------------
# Small vector helpers (camera coords: x right, y down, z forward/away, mm)
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _acos(v: float) -> float:
    return math.acos(_clamp(v, -1.0, 1.0))


def _sub(a: Landmark, b: Landmark) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-9


def _normalize(v: Vec) -> Vec | None:
    n = _norm(v) - 1e-9
    if n < 1e-6:
        return None
    return _scale(v, 1.0 / n)


def _angle_between(a: Vec, b: Vec) -> float:
    dot = _dot(a, b)
    return math.acos(_clamp(dot / (_norm(a) * _norm(b)), -1.0, 1.0))


def _elbow_bend_to_nao(bend: float) -> float:
    """Human elbow flexion (rad, 0 = straight) -> NAO ElbowRoll MAGNITUDE.

    Identity below :data:`ELBOW_LINEAR_RAD`; above it the rest of the human
    range is compressed into the rest of NAO's, so the joint keeps responding
    instead of sitting on its stop. See the constants for the measurements that
    motivated it. The caller applies the per-side sign.
    """
    bend = max(0.0, bend)
    if bend <= ELBOW_LINEAR_RAD:
        return bend
    human_span = ELBOW_HUMAN_MAX_RAD - ELBOW_LINEAR_RAD
    nao_span = ELBOW_NAO_MAX_RAD - ELBOW_LINEAR_RAD
    if human_span <= SWING_DEGENERATE or nao_span <= 0.0:
        return min(bend, ELBOW_NAO_MAX_RAD)
    over = min(bend, ELBOW_HUMAN_MAX_RAD) - ELBOW_LINEAR_RAD
    return ELBOW_LINEAR_RAD + nao_span * (over / human_span)


# ---------------------------------------------------------------------------
# NAO arm chain (for the roll joints)
# ---------------------------------------------------------------------------
# From the kinematic model in balance.py, which is built from Nao.urdf:
#
#   ShoulderPitch about Y -> ShoulderRoll about Z -> ElbowYaw about X
#                         -> ElbowRoll about Z    -> WristYaw about X
#
# so the whole arm is Ry(sp).Rz(sr).Rx(ey).Rz(er).Rx(wy), each link lying along
# the local +X when its joints are zero.
#
# WHY THESE TWO JOINTS WERE NEVER TRACKED. A roll joint rotates about the axis
# its own bone lies along, so no accuracy on the bone's ENDPOINTS can reveal it:
# shoulder, elbow and wrist positions fix the arm's shape and say nothing about
# its twist. LElbowYaw and LWristYaw therefore sat at 0.0 rad for the project's
# whole life while every other arm joint tracked -- and nothing looked broken,
# because they had MotorConfigs and were in UPPER_BODY_JOINTS all along. The
# only cure is more landmarks, which is what the hand markers are for.
HAND_YAW_SIGN = 1.0
"""Which way the human's thumb maps onto NAO's hand-frame +Y.

The one number here that geometry cannot settle. It depends on which way the
palm faces at ``WristYaw = 0`` in ``Nao.proto``, and that proto is an
EXTERNPROTO fetched at world-load time -- it is not on disk to read (searched).
Everything else in this module is derived and round-trip tested. **If the
wrists rotate the wrong way, flip this one constant to -1.0**; it shifts both
WristYaw solves by pi and changes nothing else.
"""

# Below this |sin(ElbowRoll)| the forearm lies along the ElbowYaw axis and the
# yaw is geometrically unobservable -- every value of it puts the forearm in the
# same place, so a solve would be reading noise. ~0.20 rad of bend.
ELBOW_YAW_MIN_BEND = 0.20

# Minimum thumb offset from the forearm axis, as a fraction of the hand's own
# length, before the palm's orientation is believed. A hand seen end-on, or a
# fist with the thumb folded along the fingers, gives a near-zero lever arm and
# an angle that spins with the noise.
WRIST_YAW_MIN_SPAN = 0.06

# Grip: thumb-to-finger distance, as a fraction of FOREARM length. An open hand
# holds the thumb well clear of the fingers; a fist brings them together.
#
# Normalised against the forearm and not against the hand, which is the obvious
# choice and is self-defeating: the finger marker moves back toward the wrist as
# the fingers curl, so hand length shrinks along with the thumb-finger gap and
# their ratio barely moves. Measured on the synthetic hand it stayed above 0.83
# from open palm to closed fist -- a grip signal that could not tell them apart.
# The forearm is fixed by anatomy, so it makes an honest ruler, and dividing by
# any length at all is what keeps the reading free of camera distance.
#
# The two ends sit inside the anatomical extremes so an ordinary relaxed hand
# reaches fully open rather than sitting permanently half-shut. Re-measure them
# for a real subject with `scripts/tune_arm_tracking.py --grip` against a
# recorded session; these come from adult proportions (forearm 260 mm, hand
# 190 mm, thumb 110 mm) rather than from MeTRAbs output.
GRIP_OPEN_RATIO = 0.75
GRIP_CLOSED_RATIO = 0.25


def _rot_x(v: Vec, a: float) -> Vec:
    c, s = math.cos(a), math.sin(a)
    return (v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c)


def _rot_y(v: Vec, a: float) -> Vec:
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def _rot_z(v: Vec, a: float) -> Vec:
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2])


def _nao_torso(frame: TorsoFrame, v: Vec) -> Vec:
    """Torso-local vector -> NAO torso axes (+X chest, +Y left, +Z up).

    ``TorsoFrame.forward`` is ``right x up``, which exits the subject's BACK,
    and ``TorsoFrame.right`` is their anatomical right -- so both invert. The
    pre-existing shoulder solve does the same thing inline (it passes ``-up``
    and ``-fwd`` into the decomposition) for exactly this reason.
    """
    lat, up, fwd = _to_local(frame, v)
    return (-fwd, -lat, up)


def _elbow_yaw(fore_nao: Vec, shoulder_pitch: float, shoulder_roll: float,
               elbow_roll: float) -> float | None:
    """NAO ElbowYaw from the forearm's direction. ``None`` when unobservable.

    Undo the shoulder to put the forearm in the shoulder's own frame, where the
    remaining chain is just ``Rx(ey).Rz(er)`` acting on ``+X``::

        g = Rx(ey) . Rz(er) . xhat
          = (cos er,  cos(ey) sin(er),  sin(ey) sin(er))

    so the part of ``g`` off the shoulder's X axis is ``sin(er)`` times
    ``(cos ey, sin ey)`` and the yaw is a plain ``atan2`` of it -- with the sign
    of ``sin(er)`` divided out, which matters because NAO's left ElbowRoll is
    NEGATIVE by convention and would otherwise put every left-arm answer half a
    turn out.

    The same ``sin(er)`` is the lever arm: as the elbow straightens it vanishes
    and the forearm lies on the ElbowYaw axis itself, where the joint genuinely
    cannot be seen. Hence the gate rather than a noisy number.
    """
    if abs(math.sin(elbow_roll)) < math.sin(ELBOW_YAW_MIN_BEND):
        return None
    g = _rot_z(_rot_y(fore_nao, -shoulder_pitch), -shoulder_roll)
    sign = -1.0 if elbow_roll < 0.0 else 1.0
    if math.hypot(g[1], g[2]) < SWING_DEGENERATE:
        return None
    return math.atan2(sign * g[2], sign * g[1])


def _wrist_yaw(thumb_nao: Vec, shoulder_pitch: float, shoulder_roll: float,
               elbow_yaw: float, elbow_roll: float) -> float | None:
    """NAO WristYaw from the direction the thumb points. ``None`` when unseen.

    Undo the whole chain down to the forearm, then read where the thumb landed.
    The component ALONG the forearm is dropped first: WristYaw rotates about
    that axis, so it cannot move anything lying on it, and leaving it in would
    just dilute the part that carries the answer.

    Deliberately measured against a hand-fixed axis and NOT against the elbow's
    own bend plane, which is the obvious alternative and collapses exactly when
    the arm is straight -- and the arm is straight a lot. This version keeps
    working at any elbow angle; it needs only that the thumb is visibly off the
    forearm's axis.
    """
    u = _rot_z(_rot_x(_rot_z(_rot_y(thumb_nao, -shoulder_pitch), -shoulder_roll),
                      -elbow_yaw), -elbow_roll)
    span = math.hypot(u[1], u[2])
    if span < WRIST_YAW_MIN_SPAN:
        return None
    return math.atan2(HAND_YAW_SIGN * u[2], HAND_YAW_SIGN * u[1])


def _grip(kps: dict[str, Landmark], pre: str, forearm_mm: float) -> float | None:
    """Hand closure in [0, 1] (0 = open, 1 = fist), or ``None`` if unseen.

    Scale-free by construction: the same gesture reads the same whether the
    subject is one metre from the camera or four, and a small hand closes over
    the same range as a large one. See the GRIP_* constants for why the ruler is
    the forearm rather than the hand.
    """
    if not _visible(kps, pre + "thumb", pre + "finger"):
        return None
    if forearm_mm < 1e-6:
        return None
    ratio = _dist3(kps[pre + "thumb"], kps[pre + "finger"]) / forearm_mm
    span = GRIP_OPEN_RATIO - GRIP_CLOSED_RATIO
    if span <= 0.0:
        return None
    return _clamp((GRIP_OPEN_RATIO - ratio) / span, 0.0, 1.0)


def _dist3(a: Landmark, b: Landmark) -> float:
    return _norm(_sub(a, b))


def _lift_fraction(rise: float, leg_length: float, full: float) -> float:
    """Normalize a landmark's rise above the reference into a [0, 1] lift."""
    return _clamp((rise / leg_length - LIFT_DEADBAND) / full, 0.0, 1.0)


def _shoulder_angles(lat_outward: float, up: float, fwd: float) -> tuple[float, float]:
    """Unit upper-arm direction (torso-local) -> NAO (ShoulderPitch, roll-outward).

    The shoulder is NOT solved with :func:`_swing_twist`, and the difference is
    not cosmetic. NAO's shoulder is ``v = Ry(pitch) . Rz(roll) . xhat`` in torso
    coordinates (+X out of the chest, +Y left, +Z up), which expands to::

        v_x =  cos(roll) * cos(pitch)      # = -fwd   (TorsoFrame.forward
        v_y =  sin(roll)                   # = lat_outward    exits the BACK)
        v_z = -cos(roll) * sin(pitch)      # =  up

    so the roll falls straight out of ONE component -- ``roll = asin(v_y)`` --
    and is exact at every reachable pose. ``_swing_twist`` instead recovers it as
    ``acos(v_x / cos(pitch))``, which is the same number algebraically but
    divides by ``cos(pitch)``, and ``MIN_COS_ROLL`` then reports 0 wherever that
    divisor gets small.

    For the LEGS that guard costs nothing: there the divisor is ``cos(HipRoll)``
    and it only collapses with the leg held straight out sideways. For the ARM
    the divisor is ``cos(ShoulderPitch)``, which collapses for pitch in
    72.5..107.5 deg -- **an arm hanging at the side**, the single most common
    pose there is. Measured on the 2026-09-16 session (run_20260916_110404,
    19441 non-stale frames): the commanded ShoulderRoll was *exactly* 0.000 rad
    in 31.5% of frames on the left and 43.5% on the right, with a median
    ShoulderPitch of 80.8 deg sitting right inside the dead band. The robot
    could not lift an arm sideways from its side at all -- the roll was pinned
    to zero on the way up and only woke once the arm was already past 72.5 deg.
    That is the "range of motion is very limited" report.

    ``pitch`` still comes from ``atan2(-up, -fwd)``: dividing both arguments by
    the common ``cos(roll)`` leaves the ratio alone, and ``cos(roll) > 0`` for
    every roll inside NAO's own +-76 deg limit. With the arm straight out
    sideways (``|lat_outward| -> 1``) both arguments vanish together and the
    pitch is genuinely undefined; ``atan2(0, 0) == 0`` pins it to the identity
    and lets the roll carry the whole rotation, which is the same reachable
    choice :func:`_swing_twist` makes for its own degenerate case.
    """
    roll_outward = math.asin(_clamp(lat_outward, -1.0, 1.0))
    pitch = math.atan2(-up, -fwd)
    return pitch, roll_outward


def _swing_twist(a_signed: float, b_ref: float, c_signed: float) -> tuple[float, float]:
    """Closed-form inverse of a 2-DOF "first rotate about a fixed axis, then
    rotate about the resulting axis" joint -- see the module docstring's
    "Swing-twist decomposition" section for the derivation and how legs/arms
    each map their axes onto ``(a_signed, b_ref, c_signed)``.
    """
    d = _clamp(b_ref, -1.0, 1.0)
    # The bone lying exactly along the SECOND joint's axis makes the first
    # (swing) angle geometrically undefined: both arguments of the atan2 are
    # zero. That is not a corner case -- it is an arm raised straight out to the
    # side, and atan2's signed zeros decide the answer there, so `atan2(-0.0,
    # -0.0)` returns pi and the joint slams to its limit while `atan2(-0.0,
    # +0.0)` returns 0. Pin it to 0 (the identity) and let the second angle carry
    # the whole rotation, which is both stable and the reachable choice.
    if math.hypot(a_signed, d) < SWING_DEGENERATE:
        return 0.0, (math.pi / 2.0 if c_signed > 0.0 else -math.pi / 2.0)
    first = math.atan2(a_signed, d)
    cos_first = math.cos(first)
    if abs(cos_first) < MIN_COS_ROLL:
        return first, 0.0
    second_mag = _acos(_clamp(d / cos_first, -1.0, 1.0))
    second = second_mag if c_signed > 0.0 else -second_mag
    return first, second


# ---------------------------------------------------------------------------
# Landmark access
# ---------------------------------------------------------------------------
def _parse(keypoints: dict[str, Sequence[float]]) -> dict[str, Landmark]:
    """Normalize incoming landmark values to (x, y, z, visibility) tuples."""
    out: dict[str, Landmark] = {}
    for name, v in keypoints.items():
        try:
            x = float(v[0])
            y = float(v[1])
            z = float(v[2]) if len(v) > 2 else 0.0
            vis = float(v[3]) if len(v) > 3 else 1.0
        except (TypeError, IndexError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        out[name] = (x, y, z, vis)
    return out


def _visible(kps: dict[str, Landmark], *names: str, thr: float = VIS_THRESHOLD) -> bool:
    return all(n in kps and kps[n][3] >= thr for n in names)


def _mid_point(kps: dict[str, Landmark], *names: str) -> Landmark | None:
    """Midpoint of whichever of ``names`` are visible, or None if none are.

    Degrading to a single landmark (rather than requiring the pair) is what
    keeps the lower body alive when one hip or foot is briefly occluded.
    """
    pts = [kps[n] for n in names if _visible(kps, n)]
    if not pts:
        return None
    n = float(len(pts))
    return (
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
        sum(p[2] for p in pts) / n,
        min(p[3] for p in pts),
    )


def _side_sign(side: str) -> float:
    """Sign that turns a torso-local lateral component into an
    outward-positive quantity for ``side`` ("L"/"R"). ``right`` (the torso
    basis axis) points from the left shoulder to the right shoulder, i.e.
    toward the subject's own anatomical right -- so a leg/arm swinging
    outward on the LEFT side moves AWAY from ``right`` (negative dot
    product), while on the RIGHT side it moves WITH ``right`` (positive)."""
    return -1.0 if side == "L" else 1.0


# ---------------------------------------------------------------------------
# Torso-local reference frame
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TorsoFrame:
    right: Vec     # subject's own anatomical-right direction
    up: Vec        # subject's own up direction
    forward: Vec   # subject's own chest-facing direction
    origin: Vec    # mid-hip position (mm, camera frame) -- for height measurements


def _torso_frame(kps: dict[str, Landmark]) -> TorsoFrame | None:
    if not _visible(kps, "left_shoulder", "right_shoulder"):
        return None
    ls, rs = kps["left_shoulder"], kps["right_shoulder"]
    mid_sh = (
        (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0, (ls[2] + rs[2]) / 2.0,
    )
    right_raw = _sub(rs, ls)
    right_n = _normalize(right_raw)
    if right_n is None:
        return None

    mid_hip = _mid_point(kps, "left_hip", "right_hip")
    if mid_hip is not None:
        up_raw = _sub(mid_sh, mid_hip[:3])
        origin: Vec = mid_hip[:3]
    else:
        # Hips out of frame (e.g. a desk webcam framed from the waist up):
        # fall back to a camera-vertical "up" reference (assumes a roughly
        # level, roughly upright camera) so arms/head still track. Legs
        # cannot work at all without a hip anyway (see LowerBodyRetargeter),
        # so this fallback only ever affects the arm/head path.
        up_raw = (0.0, -1.0, 0.0)
        origin = mid_sh
    up_orth = _sub(up_raw, _scale(right_n, _dot(up_raw, right_n)))
    up_n = _normalize(up_orth)
    if up_n is None:
        return None

    forward_n = _normalize(_cross(right_n, up_n))
    if forward_n is None:
        return None
    # Re-orthogonalize to guarantee an exact right-handed orthonormal basis.
    up_n = _cross(forward_n, right_n)

    return TorsoFrame(right=right_n, up=up_n, forward=forward_n, origin=origin)


def _to_local(frame: TorsoFrame, v: Vec) -> Vec:
    return (_dot(v, frame.right), _dot(v, frame.up), _dot(v, frame.forward))


# ---------------------------------------------------------------------------
# Per-segment retargeting (upper body)
# ---------------------------------------------------------------------------
def _arm(kps: dict[str, Landmark], side: str, frame: TorsoFrame) -> dict[str, float]:
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "shoulder", pre + "elbow"):
        return {}

    s = kps[pre + "shoulder"]
    e = kps[pre + "elbow"]
    unit = _normalize(_sub(e, s))
    if unit is None:
        return {}
    lat, up, fwd = _to_local(frame, unit)
    lat_outward = _side_sign(side) * lat

    # NOTE the negated `fwd`. TorsoFrame.forward is right x up, which for any
    # real body in a right-handed frame points out of the subject's BACK, not
    # their chest (see the TorsoFrame docstring). The LEG solve wants exactly
    # that -- NAO's HipPitch is negative for a thigh swung forward -- but NAO's
    # ShoulderPitch is ZERO for an arm held forward, so the arm needs the
    # chest-facing axis, which is -forward.
    #
    # Without the negation an arm reaching straight at the camera solved to
    # atan2(0, -1) = 180 deg, which the +/-119.5 deg joint limit then clamped:
    # the arm slammed to its mechanical stop instead of pointing forward, and
    # every forward reach in between came out moving the wrong way. It was
    # invisible to the tests because figure() in tests/test_nao_retarget.py hangs
    # the arms straight down at z = 0, where the fore/aft term is 0 and its sign
    # therefore cannot matter.
    #
    # Only the pitch is affected by that sign: the roll reads `lat_outward`,
    # which the fore/aft axis does not enter at all. See _shoulder_angles for
    # why the shoulder does NOT go through _swing_twist like the legs do.
    pitch, roll_outward = _shoulder_angles(lat_outward, up, fwd)

    out: dict[str, float] = {}
    if side == "L":
        out["LShoulderPitch"] = pitch
        out["LShoulderRoll"] = +roll_outward   # NAO L: positive = outward
    else:
        out["RShoulderPitch"] = pitch
        out["RShoulderRoll"] = -roll_outward   # NAO R: negative = outward

    # Elbow flexion: angle between upper arm and forearm (0 = straight). Pure
    # dot-product angle -- exact regardless of coordinate frame.
    if not _visible(kps, pre + "wrist"):
        return out
    w = kps[pre + "wrist"]
    bend = _elbow_bend_to_nao(_angle_between(_sub(e, s), _sub(w, e)))
    if side == "L":
        elbow_roll = -bend                    # NAO L elbow bends negative
    else:
        elbow_roll = +bend                    # NAO R elbow bends positive
    out[f"{side}ElbowRoll"] = elbow_roll

    out.update(_hand(kps, side, frame, pitch, roll_outward, elbow_roll, e, w))
    return out


def _hand(kps: dict[str, Landmark], side: str, frame: TorsoFrame,
          pitch: float, roll_outward: float, elbow_roll: float,
          elbow: Landmark, wrist: Landmark) -> dict[str, float]:
    """The two roll joints and the grip, from the hand markers.

    Returns only what is actually observable this frame. Everything here is
    omitted rather than guessed when the geometry does not support it, so the
    driver holds the previous value instead of snapping a wrist to zero -- the
    same contract the rest of this module keeps for an invisible limb.

    ``ElbowYaw`` needs no hand landmarks at all -- the forearm's own direction
    fixes it once the shoulder and the elbow bend are known -- so it is solved
    under plain ``coco_19`` too. ``WristYaw`` and the grip DO need the hand
    markers, and those exist only under ``pose.skeleton: ""`` (MeTRAbs'
    122-joint superset).
    """
    pre = "left_" if side == "L" else "right_"
    out: dict[str, float] = {}

    fore_vec = _sub(wrist, elbow)
    grip = _grip(kps, pre, _norm(fore_vec))
    if grip is not None:
        out[f"{side}Hand"] = grip

    # The three outputs have three different evidence requirements, so they are
    # gated separately rather than behind one "is the hand visible" check.
    # ElbowYaw needs only the forearm, which the caller has already confirmed;
    # WristYaw additionally needs the thumb; the grip needs thumb and finger. An
    # earlier version gated all of them on the thumb, which threw away a
    # perfectly observable ElbowYaw whenever one marker dropped out.
    fore = _normalize(fore_vec)
    if fore is None:
        return out

    # NAO's ShoulderRoll sign is per-side but the kinematic chain is not: the
    # chain in balance.py is the same Ry.Rz.Rx.Rz.Rx on both arms, so the solve
    # below wants the angle as the JOINT carries it, not the outward-positive
    # convention the shoulder solve reports in.
    shoulder_roll = roll_outward if side == "L" else -roll_outward

    elbow_yaw = _elbow_yaw(_nao_torso(frame, fore), pitch, shoulder_roll, elbow_roll)
    if elbow_yaw is None:
        # WristYaw is solved IN the forearm frame, which is not known without
        # the elbow yaw, so it goes too.
        return out
    out[f"{side}ElbowYaw"] = elbow_yaw

    if not _visible(kps, pre + "hand_root", pre + "thumb"):
        return out
    thumb = _normalize(_sub(kps[pre + "thumb"], kps[pre + "hand_root"]))
    if thumb is None:
        return out
    wrist_yaw = _wrist_yaw(_nao_torso(frame, thumb), pitch, shoulder_roll,
                           elbow_yaw, elbow_roll)
    if wrist_yaw is not None:
        out[f"{side}WristYaw"] = wrist_yaw
    return out


@dataclass
class HeadGeometry:
    """Per-subject neutral for the head-pitch solve.

    :func:`_head` measures how far the nose sits above the shoulder line, in
    shoulder-width units, and reads the difference from a neutral as "how far
    from looking straight ahead". :data:`HEAD_PITCH_BASELINE` is a population
    average, so a subject whose neck is longer or shorter than average carries a
    constant offset: the robot holds a permanent nod while the human looks
    straight at the camera, and that offset eats the head's usable range at one
    end.

    Unlike the segment lengths in :class:`BodyGeometry` this cannot simply be
    measured -- a low nose is EITHER a short neck OR a subject looking down, and
    no single frame separates the two. Nor is it an extremum, so
    :class:`PeakHold` does not apply either: looking up and looking down move it
    in OPPOSITE directions, and a running maximum would latch onto the
    subject's most upward glance. It is therefore averaged over the opening
    frames -- someone who has just stepped in front of a camera is looking at it
    -- and afterwards held with a slow drift, so a different subject eventually
    re-calibrates instead of inheriting the first one's neck.
    """

    warmup: int = HEAD_BASELINE_WARMUP
    decay: float = HEAD_BASELINE_DECAY
    baseline: float = HEAD_PITCH_BASELINE
    _samples: int = field(default=0, init=False, repr=False)

    def update(self, nose_height: float) -> float:
        """Fold in one frame's nose height (shoulder-widths); return the neutral.

        Non-finite samples are ignored rather than allowed to poison the mean.
        """
        if not math.isfinite(nose_height):
            return self.baseline
        self._samples += 1
        if self._samples == 1:
            # Replace the population default outright: one real measurement of
            # THIS subject beats an average of everyone.
            self.baseline = nose_height
        elif self._samples <= self.warmup:
            self.baseline += (nose_height - self.baseline) / self._samples
        else:
            self.baseline += self.decay * (nose_height - self.baseline)
        return self.baseline

    @property
    def calibrated(self) -> bool:
        return self._samples >= self.warmup


def _head(kps: dict[str, Landmark], frame: TorsoFrame,
          geom: HeadGeometry | None = None) -> dict[str, float]:
    if not _visible(kps, "nose", "left_shoulder", "right_shoulder"):
        return {}
    nose = kps["nose"]
    ls = kps["left_shoulder"]
    rs = kps["right_shoulder"]
    mid_sh = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0, (ls[2] + rs[2]) / 2.0)
    shoulder_w = max(_dist3(ls, rs), 1e-4)

    nose_offset = _sub(nose, mid_sh)
    lat, up, _fwd = _to_local(frame, nose_offset)

    yaw = 0.0
    if _visible(kps, "left_ear", "right_ear"):
        # Same "rigid body-fixed segment, read its rotation from the plane it
        # sweeps" trick as the torso yaw in gait_cues.py, applied to the
        # ear-to-ear line: 0 when the head faces the same way as the torso.
        le, re = kps["left_ear"], kps["right_ear"]
        ear_vec = _sub(re, le)
        e_lat, _e_up, e_fwd = _to_local(frame, ear_vec)
        yaw = -math.atan2(e_fwd, e_lat) * HEAD_YAW_GAIN
    else:
        # Fall back to the nose's lateral offset from the shoulder midline.
        yaw = (lat / shoulder_w) * HEAD_YAW_GAIN

    # Pitch: nose vertical (torso-local "up") offset relative to its typical
    # above-shoulder-line height. Looking down brings the nose toward the
    # shoulders (nose_height shrinks) -> positive pitch. The neutral it is read
    # against is this subject's own once ``geom`` has seen enough frames,
    # falling back to the population average when no calibrator is supplied.
    nose_height = up / shoulder_w
    baseline = geom.update(nose_height) if geom is not None else HEAD_PITCH_BASELINE
    pitch = (baseline - nose_height) * HEAD_PITCH_GAIN
    return {"HeadYaw": yaw, "HeadPitch": pitch}


def _knee_bend(kps: dict[str, Landmark], side: str) -> float | None:
    """Human knee flexion (rad, 0 = straight) from hip-knee-ankle, or None."""
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "hip", pre + "knee", pre + "ankle"):
        return None
    thigh = _sub(kps[pre + "knee"], kps[pre + "hip"])
    shank = _sub(kps[pre + "ankle"], kps[pre + "knee"])
    return _angle_between(thigh, shank)


# ---------------------------------------------------------------------------
# Lower body: per-leg closed-form solve
# ---------------------------------------------------------------------------
@dataclass
class LegTarget:
    """One leg's retargeted NAO angles plus how far the human lifted that foot."""
    hip_pitch: float = 0.0
    hip_roll: float = 0.0
    knee_pitch: float = 0.0
    ankle_pitch: float = 0.0
    ankle_roll: float = 0.0
    lift: float = 0.0          # 0 = planted, 1 = knee-high lift
    confidence: float = 0.0    # 0..1 from landmark visibility

    def as_targets(self, side: str) -> dict[str, float]:
        """Expand to NAO joint names for ``side`` in ("L", "R")."""
        return {
            f"{side}HipPitch": self.hip_pitch,
            f"{side}HipRoll": self.hip_roll,
            f"{side}KneePitch": self.knee_pitch,
            f"{side}AnklePitch": self.ankle_pitch,
            f"{side}AnkleRoll": self.ankle_roll,
        }


@dataclass
class LowerBodyObservation:
    """Everything the lower-body controller needs from one camera frame."""
    left: LegTarget | None = None
    right: LegTarget | None = None
    crouch_u: float = 0.0        # rad; symmetric squat amplitude (0 = upright)
    stance_side: str = ""        # "L" / "R" / "" (both feet down)
    confidence: float = 0.0      # 0..1 overall lower-body confidence
    valid: bool = False
    # Which landmarks produced the lift signal: "feet", "knees" or "none".
    lift_source: str = "none"

    def leg(self, side: str) -> LegTarget | None:
        return self.left if side == "L" else self.right


class PeakHold:
    """Running maximum of a signal: rises quickly, decays very slowly.

    Unlike segment lengths (now measured directly in real mm, see
    :class:`BodyGeometry`), a subject's own "standing tall" hip-height/torso
    ratio genuinely has to be LEARNED from the stream -- there is no way to
    know it a priori, real 3D or not. An instantaneous ratio is always <= the
    true standing ratio (crouching can only bring the hips closer to the
    ground line, never past standing), so the running peak converges on it;
    the slow decay lets the estimate follow a genuinely different subject
    instead of latching forever.
    """

    __slots__ = ("value", "rise", "decay")

    def __init__(self, rise: float = 0.35, decay: float = 0.004) -> None:
        self.value: float = 0.0
        self.rise = rise
        self.decay = decay

    def update(self, sample: float) -> float:
        if not math.isfinite(sample) or sample <= 0.0:
            return self.value
        if self.value <= 0.0:
            self.value = sample
        elif sample > self.value:
            self.value += self.rise * (sample - self.value)
        else:
            self.value += self.decay * (sample - self.value)
        return self.value


@dataclass
class BodyGeometry:
    """Directly-measured (mm) segment lengths, lightly EMA-smoothed to reduce
    per-frame jitter. Unlike the old MediaPipe-era self-calibration (a
    running-maximum learned over time, needed because a 2D projection could
    only ever look shorter than the truth), MeTRAbs gives true metric
    distances directly -- so there is nothing to learn, only smooth."""
    alpha: float = GEOMETRY_ALPHA
    torso: float = 0.0
    thigh: float = 0.0
    shank: float = 0.0
    hip_height_ratio: PeakHold = field(default_factory=PeakHold)
    _has_torso: bool = field(default=False, init=False, repr=False)
    _has_thigh: bool = field(default=False, init=False, repr=False)
    _has_shank: bool = field(default=False, init=False, repr=False)

    def _ema(self, prev: float, sample: float, has_prev: bool) -> float:
        return sample if not has_prev else prev + self.alpha * (sample - prev)

    def update_torso(self, kps: dict[str, Landmark]) -> float:
        ls, rs = kps["left_shoulder"], kps["right_shoulder"]
        mid_sh = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5, (ls[2] + rs[2]) * 0.5, 1.0)
        mid_hip = _mid_point(kps, "left_hip", "right_hip")
        if mid_hip is None:
            return self.torso
        span = max(_dist3(ls, rs), 1e-4)
        # Shoulder span is a useful floor: it keeps the scale sane when the
        # subject leans and the torso projects short.
        raw = max(_dist3(mid_sh, mid_hip), 0.6 * span, 1e-4)
        self.torso = self._ema(self.torso, raw, self._has_torso)
        self._has_torso = True
        return self.torso

    def update_thigh(self, sample: float) -> None:
        self.thigh = self._ema(self.thigh, sample, self._has_thigh)
        self._has_thigh = True

    def update_shank(self, sample: float) -> None:
        self.shank = self._ema(self.shank, sample, self._has_shank)
        self._has_shank = True

    @property
    def shank_or_thigh(self) -> float:
        # No ankle has ever been seen -- the usual case for someone standing
        # close to a webcam, cropped at the shins. Thigh and shank are within
        # a few percent of each other in both the human and NAO (0.100 m vs
        # 0.1029 m), so borrowing the thigh keeps the leg scale usable.
        return self.shank if self._has_shank else self.thigh

    @property
    def leg_length(self) -> float:
        return max(self.thigh + self.shank_or_thigh, 1e-4)

    @property
    def calibrated(self) -> bool:
        # Only the torso scale and the thigh are required: the shank falls
        # back to the thigh (see above), and the shank length is only ever
        # used on frames where the ankle IS visible.
        return self.torso > 1e-4 and self._has_thigh


class LowerBodyRetargeter:
    """Stateful per-leg retargeting of MeTRAbs landmarks to NAO leg angles.

    Stateful only for the light EMA smoothing in :class:`BodyGeometry`;
    apart from that it is a pure function of the landmark stream -- no
    Webots, no camera, no RNG -- so it is unit-testable off-simulation.
    """

    def __init__(self) -> None:
        self.geom = BodyGeometry()

    # -- public ------------------------------------------------------------
    def observe(self, keypoints: dict[str, Sequence[float]]) -> LowerBodyObservation:
        """Retarget one frame of landmarks into a :class:`LowerBodyObservation`."""
        return self.observe_parsed(_parse(keypoints))

    def observe_parsed(self, kps: dict[str, Landmark]) -> LowerBodyObservation:
        frame = _torso_frame(kps)
        if frame is None:
            return LowerBodyObservation()

        self.geom.update_torso(kps)
        self._calibrate_segments(kps)
        if not self.geom.calibrated:
            return LowerBodyObservation()

        ground_y = self._ground_line(kps)
        lifts, lift_source = self._lifts(kps)
        left = self._leg(kps, "L", frame, lifts["L"])
        right = self._leg(kps, "R", frame, lifts["R"])
        if left is None and right is None:
            return LowerBodyObservation(lift_source=lift_source)

        legs = [lg for lg in (left, right) if lg is not None]
        confidence = sum(lg.confidence for lg in legs) / len(legs)
        # Two legs seen is materially more trustworthy than one: a single-leg
        # read cannot tell a lift from the other foot leaving the frame.
        if len(legs) < 2:
            confidence *= 0.5

        stance = ""
        if left is not None and right is not None:
            if left.lift > right.lift + 0.08:
                stance = "R"
            elif right.lift > left.lift + 0.08:
                stance = "L"

        return LowerBodyObservation(
            left=left,
            right=right,
            crouch_u=self._crouch(kps, ground_y),
            stance_side=stance,
            confidence=_clamp(confidence, 0.0, 1.0),
            valid=True,
            lift_source=lift_source,
        )

    # -- internals ---------------------------------------------------------
    def _calibrate_segments(self, kps: dict[str, Landmark]) -> None:
        """Directly measure this frame's thigh/shank lengths (mm), EMA-smoothed."""
        for side in ("L", "R"):
            pre = "left_" if side == "L" else "right_"
            if _visible(kps, pre + "hip", pre + "knee"):
                self.geom.update_thigh(_dist3(kps[pre + "knee"], kps[pre + "hip"]))
            if _visible(kps, pre + "knee", pre + "ankle"):
                self.geom.update_shank(_dist3(kps[pre + "ankle"], kps[pre + "knee"]))

    def _foot_height(self, kps: dict[str, Landmark], side: str) -> float | None:
        """Camera-frame vertical (mm) of a foot. MeTRAbs' coco_19 skeleton has
        no separate heel landmark (unlike MediaPipe's 33-point set), so this
        is the ankle alone."""
        pre = "left_" if side == "L" else "right_"
        if not _visible(kps, pre + "ankle"):
            return None
        return kps[pre + "ankle"][1]

    def _ground_line(self, kps: dict[str, Landmark]) -> float | None:
        """Camera-frame vertical (mm) of the ground: the LOWER of the two feet.

        This is the trick that makes lift detection calibration-free -- whichever
        foot is planted defines the floor, so the other foot's rise above it is
        the lift, with no need to know where the real floor is in the image.
        """
        feet = (self._foot_height(kps, "L"), self._foot_height(kps, "R"))
        ys = [y for y in feet if y is not None]
        return max(ys) if ys else None

    def _lifts(self, kps: dict[str, Landmark]) -> tuple[dict[str, float], str]:
        """Per-side foot-lift fraction in [0, 1], and which landmarks gave it.

        Falls back to the KNEES (visible whenever the hips are) when the feet
        are out of frame -- the usual case for someone standing close to a
        webcam. Knees are also the signal the Python-side gait detector uses,
        for the same robustness reason.
        """
        leg_len = self.geom.leg_length

        feet = {side: self._foot_height(kps, side) for side in ("L", "R")}
        if all(v is not None for v in feet.values()):
            ground = max(feet.values())          # camera y grows downward
            return ({s: _lift_fraction(ground - feet[s], leg_len, LIFT_FULL)
                     for s in ("L", "R")}, "feet")

        knees: dict[str, float | None] = {}
        for side in ("L", "R"):
            pre = "left_" if side == "L" else "right_"
            knees[side] = kps[pre + "knee"][1] if _visible(kps, pre + "knee") else None
        if all(v is not None for v in knees.values()):
            ref = max(knees.values())
            return ({s: _lift_fraction(ref - knees[s], leg_len, LIFT_FULL_KNEE)
                     for s in ("L", "R")}, "knees")

        return ({"L": 0.0, "R": 0.0}, "none")

    def _leg(
        self, kps: dict[str, Landmark], side: str, frame: TorsoFrame, lift: float
    ) -> LegTarget | None:
        pre = "left_" if side == "L" else "right_"
        if not _visible(kps, pre + "hip", pre + "knee"):
            return None
        hip = kps[pre + "hip"]
        knee = kps[pre + "knee"]
        ankle = kps[pre + "ankle"] if _visible(kps, pre + "ankle") else None

        thigh_unit = _normalize(_sub(knee, hip))
        if thigh_unit is None:
            return None
        lat, up, fwd = _to_local(frame, thigh_unit)
        lat_outward = _side_sign(side) * lat
        roll_outward, hip_pitch = _swing_twist(lat_outward, -up, fwd)

        total = hip_pitch          # theta_h + theta_k, defaults to knee straight
        if ankle is not None:
            shank_unit = _normalize(_sub(ankle, knee))
            if shank_unit is not None:
                _lat_s, up_s, fwd_s = _to_local(frame, shank_unit)
                # The shank shares the hip roll (the knee has no roll DOF): reuse
                # roll_outward (rather than re-solving it from the shank alone)
                # and read out theta_h + theta_k = total directly.
                d_s = _clamp(-up_s, -1.0, 1.0)
                cos_roll = math.cos(roll_outward)
                if abs(cos_roll) >= MIN_COS_ROLL:
                    total_mag = _acos(_clamp(d_s / cos_roll, -1.0, 1.0))
                    total_signed = total_mag if fwd_s > 0.0 else -total_mag
                    # Knees do not hyperextend: of the two sign branches keep
                    # the one that yields a non-negative KneePitch.
                    total = total_signed if (total_signed - hip_pitch) >= 0.0 else -total_signed
        knee_pitch = max(0.0, total - hip_pitch)
        total = hip_pitch + knee_pitch

        # NAO roll signs: LHipRoll positive = left leg outward, RHipRoll
        # negative = right leg outward. AnkleRoll cancels it so the sole
        # stays level.
        hip_roll = roll_outward if side == "L" else -roll_outward

        names = [pre + "hip", pre + "knee"] + ([pre + "ankle"] if ankle else [])
        conf = sum(kps[n][3] for n in names) / len(names)

        return LegTarget(
            hip_pitch=hip_pitch,
            hip_roll=hip_roll,
            knee_pitch=knee_pitch,
            ankle_pitch=-total,       # level the sole (all three rotate about y)
            ankle_roll=-hip_roll,     # level the sole laterally
            lift=lift,
            confidence=_clamp(conf, 0.0, 1.0),
        )

    def _crouch(self, kps: dict[str, Landmark], ground_y: float | None) -> float:
        """Symmetric squat amplitude u (rad) for the balanced crouch posture.

        Two independent cues must agree before the robot squats: the hips
        actually dropped toward the ground line, AND the knees are actually bent.
        """
        bends = [b for b in (_knee_bend(kps, "L"), _knee_bend(kps, "R")) if b is not None]
        if not bends:
            return 0.0
        # The STRAIGHTER knee, not the average: while one leg is lifted its own
        # deep knee fold says nothing about how low the body is.
        knee_cue = _clamp(
            (min(bends) - KNEE_STRAIGHT_DEADZONE) / KNEE_BEND_RANGE, 0.0, 1.0
        )

        height_cue = 1.0
        mid_hip = _mid_point(kps, "left_hip", "right_hip")
        if ground_y is not None and mid_hip is not None and self.geom.torso > 1e-4:
            ratio = (ground_y - mid_hip[1]) / self.geom.torso
            ref = self.geom.hip_height_ratio.update(ratio)
            if ref > 1e-4:
                height_cue = _clamp((1.0 - ratio / ref) / CROUCH_FULL_DROP, 0.0, 1.0)

        return min(knee_cue, height_cue) * MAX_CROUCH


# ---------------------------------------------------------------------------
# Legacy symmetric crouch (kept for the walk engine's idle posture)
# ---------------------------------------------------------------------------
def crouch_posture(u: float) -> dict[str, float]:
    """Symmetric, statically-balanced crouch: hip -u, knee +2u, ankle -u.

    ``HipPitch + KneePitch + AnklePitch == 0`` keeps the torso vertical and the
    feet flat, and thigh ~= shank length keeps the hip over the ankle, so the
    centre of mass stays inside the foot polygon and NAO holds the squat
    statically. This is the posture every lower-body mode decays back to.
    """
    return {
        "LHipPitch": -u, "RHipPitch": -u,
        "LKneePitch": 2.0 * u, "RKneePitch": 2.0 * u,
        "LAnklePitch": -u, "RAnklePitch": -u,
        "LHipRoll": 0.0, "RHipRoll": 0.0,
        "LAnkleRoll": 0.0, "RAnkleRoll": 0.0,
        "LHipYawPitch": 0.0, "RHipYawPitch": 0.0,
    }


def _swap_sides(targets: dict[str, float]) -> dict[str, float]:
    """Swap L<->R joints for a mirror-image mapping."""
    swapped: dict[str, float] = {}
    for name, value in targets.items():
        if name.startswith("L"):
            swapped["R" + name[1:]] = value
        elif name.startswith("R"):
            swapped["L" + name[1:]] = value
        else:
            swapped[name] = value
    return swapped


# ---------------------------------------------------------------------------
# Public entry point (upper body + head; legs go through LowerBodyRetargeter)
# ---------------------------------------------------------------------------
def retarget_upper_body(
    keypoints: dict[str, Sequence[float]],
    *,
    drive_head: bool = True,
    swap_sides: bool = False,
    limiter: JointLimiter | None = None,
    head_geom: HeadGeometry | None = None,
) -> dict[str, float]:
    """Map MeTRAbs landmarks to clamped NAO arm/head targets (radians).

    Only joints whose source landmarks are visible are returned; everything
    else is omitted so the caller can hold the previous pose.

    ``head_geom`` is the caller's :class:`HeadGeometry`, held across frames so
    the head-pitch neutral calibrates to the subject; omit it for a one-shot
    solve against the population default.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    kps = _parse(keypoints)

    targets: dict[str, float] = {}
    frame = _torso_frame(kps)
    if frame is not None:
        targets.update(_arm(kps, "L", frame))
        targets.update(_arm(kps, "R", frame))
        if drive_head:
            targets.update(_head(kps, frame, head_geom))

    if swap_sides:
        targets = _swap_sides(targets)
    return {name: limiter.clamp_angle(name, value) for name, value in targets.items()}


def retarget_full_body(
    keypoints: dict[str, Sequence[float]],
    *,
    drive_legs: bool = False,
    drive_head: bool = True,
    swap_sides: bool = False,
    limiter: JointLimiter | None = None,
    retargeter: LowerBodyRetargeter | None = None,
    head_geom: HeadGeometry | None = None,
) -> dict[str, float]:
    """Arms + head, plus (when ``drive_legs``) the raw per-leg leg solve.

    NOTE: the leg angles returned here are the *human's* pose, with no balance
    safety applied. Production control goes through
    ``lower_body.LowerBodyController``, which gates and blends them against the
    robot's own CoM/force state. This entry point exists for the joint-angle
    fallback path and for tests.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    targets = retarget_upper_body(
        keypoints, drive_head=drive_head, swap_sides=False, limiter=limiter,
        head_geom=head_geom,
    )

    if drive_legs:
        obs = (retargeter or LowerBodyRetargeter()).observe(keypoints)
        if obs.valid:
            for side in ("L", "R"):
                leg = obs.leg(side)
                if leg is not None:
                    targets.update(leg.as_targets(side))

    if swap_sides:
        targets = _swap_sides(targets)
    return {name: limiter.clamp_angle(name, value) for name, value in targets.items()}


def retargetable_joints(drive_legs: bool = False, drive_head: bool = True) -> list[str]:
    """The set of NAO joints this module can drive (for logging headers)."""
    joints = [
        "LShoulderPitch", "RShoulderPitch",
        "LShoulderRoll", "RShoulderRoll",
        "LElbowRoll", "RElbowRoll",
        "LElbowYaw", "RElbowYaw",
        "LWristYaw", "RWristYaw",
        "LHand", "RHand",
    ]
    if drive_head:
        joints += ["HeadYaw", "HeadPitch"]
    if drive_legs:
        for side in ("L", "R"):
            joints += [
                f"{side}HipPitch", f"{side}HipRoll", f"{side}KneePitch",
                f"{side}AnklePitch", f"{side}AnkleRoll",
            ]
    return joints
