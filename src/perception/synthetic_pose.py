"""Anatomically-consistent synthetic human poses, in MeTRAbs' output format.

Why this exists
---------------
PRD US-2 asks for deterministic replay "for evaluation", and acceptance
criterion 3 asks that a reference motion reproduce equivalent robot motion. Both
were impossible to satisfy: there was no way to drive the robot except by
standing in front of a camera, which is neither deterministic nor available on a
CI machine, a machine with no webcam, or at 3am the night before a demo.

These generators emit exactly what ``PoseEstimator`` emits -- ``coco_19``
landmarks as absolute metric 3D in millimetres, camera frame (x right, y DOWN,
z forward/away from the camera) -- so everything downstream (smoothing, gait
cues, retargeting, the UDP bridge, the Webots controller) runs its real code
path and cannot tell the difference.

They are NOT a substitute for testing with a real human: MeTRAbs' noise,
occlusion and depth error are exactly what these lack. They are a way to ask
"given a KNOWN human motion, does the robot do the right thing", which is the
question the fidelity metrics (NFR-3) need answered and the one a live camera
can never answer exactly, because the ground truth is unknown.

Segment lengths are held constant across every frame by construction (limbs are
placed by forward kinematics from joint angles, never by interpolating
endpoints), so a retargeter's own limb-length calibration sees a rigid body --
as it would with a real person.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.type_defs import Keypoint, PoseFrame

# Adult male 50th-percentile-ish segment lengths (mm). Only the RATIOS matter to
# the retargeting maths -- every threshold in nao_retarget.py is expressed as a
# fraction of torso or leg length -- but real magnitudes keep the logged numbers
# readable and let the metric-scale code paths run for real.
SHOULDER_HALF_WIDTH = 190.0
HIP_HALF_WIDTH = 100.0
TORSO = 500.0          # shoulder midpoint -> hip midpoint
UPPER_ARM = 300.0
FOREARM = 260.0
THIGH = 430.0
SHANK = 420.0
HAND = 190.0           # wrist -> middle fingertip (adult 50th percentile)
THUMB = 110.0          # wrist -> thumb tip, out to the thumb side of the palm
NECK_TO_NOSE = 240.0
HEAD_HALF_WIDTH = 75.0  # ear to ear / 2
EYE_HALF_WIDTH = 32.0

STANDING_DISTANCE = 2500.0  # subject stands this far from the camera (mm)
GROUND_Y = 850.0            # ankle height below the hips when standing tall

# Which side of the image a FRONT-FACING subject's anatomical left lands on.
#
# When you face someone, your left is on their right -- so a subject facing the
# camera has their left shoulder at POSITIVE x. Landmark names are anatomical
# (COCO convention), not screen-side, so this is a fact about human beings and
# not about any particular estimator.
#
# Mirroring the input (configs/default.yaml input.flip_horizontal) does NOT
# change it: the flip moves the shoulder to the other side of the image AND
# makes the estimator label it as the other shoulder, and the two swaps cancel.
#
# It matters because src/perception/gait_cues.py derives torso yaw from
# atan2(-depth, left.x - right.x): get the sign backwards and a subject standing
# still reads as turned 180 degrees, so the controller demands turn clips forever
# and never walks.
LEFT_SIDE_SIGN = +1.0


def _rot_y(x: float, z: float, angle: float) -> tuple[float, float]:
    """Rotate a point about the vertical (y) axis -- the subject turning."""
    c, s = math.cos(angle), math.sin(angle)
    return x * c + z * s, -x * s + z * c


@dataclass(frozen=True)
class BodyState:
    """One instant of a human, as joint angles rather than point positions.

    Angles are radians. Arm elevations are measured from straight-down, so 0 is
    a hanging arm and pi/2 is horizontal; ``*_arm_side`` swings the arm out to
    the side and ``*_arm_fwd`` swings it toward the camera. Both may be non-zero
    at once (the arm then rises diagonally), which is what makes a natural-looking
    wave possible.
    """

    left_arm_side: float = 0.0
    left_arm_fwd: float = 0.0
    left_elbow: float = 0.0      # flexion, 0 = straight
    right_arm_side: float = 0.0
    right_arm_fwd: float = 0.0
    right_elbow: float = 0.0
    crouch: float = 0.0          # 0 = standing tall, 1 = deepest squat
    left_hip_flex: float = 0.0   # lifted-leg thigh angle from vertical
    left_knee_flex: float = 0.0
    right_hip_flex: float = 0.0
    right_knee_flex: float = 0.0
    body_yaw: float = 0.0        # subject rotating on the spot
    head_yaw: float = 0.0
    head_pitch: float = 0.0      # + = looking down
    # Hand. ``*_wrist_roll`` rotates the palm about the forearm's own long axis
    # (0 = thumb up when the arm hangs), which is the ONLY thing that moves
    # NAO's WristYaw -- shoulder, elbow and wrist positions are all invariant
    # under it, which is why the joint went untracked for the project's whole
    # life. ``*_grip`` closes the fingers: 0 = open hand, 1 = fist.
    left_wrist_roll: float = 0.0
    right_wrist_roll: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0


MAX_CROUCH_DROP = 320.0   # how far the hips sink at crouch = 1


def _leg_chain(
    hip: tuple[float, float, float],
    hip_flex: float,
    knee_flex: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Knee and ankle from a hip position and two flexion angles (forward kinematics).

    Doing it this way rather than interpolating an ankle position is what keeps
    the thigh and shank exactly THIGH/SHANK long in every frame; a retargeter
    that calibrates its own limb lengths would otherwise see the subject's bones
    change length mid-motion and quietly mis-scale everything after it.
    """
    knee = (
        hip[0],
        hip[1] + THIGH * math.cos(hip_flex),
        hip[2] - THIGH * math.sin(hip_flex),
    )
    shank_angle = hip_flex - knee_flex
    ankle = (
        knee[0],
        knee[1] + SHANK * math.cos(shank_angle),
        knee[2] - SHANK * math.sin(shank_angle),
    )
    return knee, ankle


def _planted_leg(
    hip: tuple[float, float, float], ankle_y: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Knee/ankle for a foot that stays on the ground while the hips sink.

    The knee is the intersection of the thigh circle (about the hip) and the
    shank circle (about the fixed ankle), taken on the forward side -- which is
    simply how a human squats.
    """
    ankle = (hip[0], ankle_y, hip[2])
    # Clamp the hip-ankle span to the range the two segments can actually span.
    # Clamping to exactly THIGH + SHANK (rather than a hair under it) matters:
    # standing tall IS the fully-extended case, and nudging it by an epsilon
    # leaves the knee a hair off the hip-ankle line, so the shank comes out
    # 420.000001 mm instead of 420. That is physically nothing, but it means the
    # subject's bones are not bit-identical between frames -- and "the limbs are
    # rigid" is the one guarantee this module exists to provide.
    d = ankle[1] - hip[1]
    d = min(max(d, abs(THIGH - SHANK) + 1e-9), THIGH + SHANK)
    # Distance along the hip->ankle line to the knee's projection.
    a = (THIGH * THIGH - SHANK * SHANK + d * d) / (2.0 * d)
    forward = math.sqrt(max(THIGH * THIGH - a * a, 0.0))
    knee = (hip[0], hip[1] + a, hip[2] - forward)
    return knee, ankle


def _limb_direction(side_sign: float, elevation_side: float, elevation_fwd: float
                    ) -> tuple[float, float, float]:
    """Unit direction of a limb hanging from a joint, after two swings.

    Starts pointing straight down (0, 1, 0) -- y is DOWN -- then rotates about
    the forward axis by ``elevation_side`` (arm goes out to the side) and about
    the lateral axis by ``elevation_fwd`` (arm goes toward the camera, -z).

    Composed as two ROTATIONS rather than by setting components independently and
    solving for the third. The independent form is only unit-length while
    sin^2(side) + sin^2(fwd) <= 1; past that -- a raised arm with a bent elbow,
    i.e. an ordinary wave -- it silently clamped the remaining component to zero
    and returned a vector 1.25 long, stretching the subject's forearm from 260 mm
    to 326 mm mid-motion.
    """
    x = side_sign * math.sin(elevation_side)
    y = math.cos(elevation_side)
    return (x, y * math.cos(elevation_fwd), -y * math.sin(elevation_fwd))


def _arm_chain(
    shoulder: tuple[float, float, float],
    side_sign: float,
    elevation_side: float,
    elevation_fwd: float,
    elbow_flex: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Elbow and wrist from a shoulder and the arm's two elevation angles.

    Flexing the elbow swings the forearm further forward than the upper arm,
    which is the direction a human elbow actually bends.
    """
    ux, uy, uz = _limb_direction(side_sign, elevation_side, elevation_fwd)
    elbow = (
        shoulder[0] + UPPER_ARM * ux,
        shoulder[1] + UPPER_ARM * uy,
        shoulder[2] + UPPER_ARM * uz,
    )
    fx, fy, fz = _limb_direction(side_sign, elevation_side, elevation_fwd + elbow_flex)
    wrist = (
        elbow[0] + FOREARM * fx,
        elbow[1] + FOREARM * fy,
        elbow[2] + FOREARM * fz,
    )
    return elbow, wrist


def _hand_points(shoulder, elbow, wrist, side_sign: float, roll: float, grip: float):
    """Finger and thumb markers for one hand.

    Built as a frame carried along the forearm, because that is what makes the
    result mean anything: the FINGER marker continues the forearm's own
    direction (shortening as the fingers curl into a fist), and the THUMB
    marker sits off to one side of it, rotated about the forearm axis by
    ``roll``. So the thumb's position is the only carrier of the palm's
    orientation -- exactly the relationship ``nao_retarget`` has to invert to
    recover WristYaw, and a synthetic hand that got it wrong would let a broken
    solve pass.
    """
    fx, fy, fz = _limb_direction_between(wrist, elbow)
    ax, ay, az = _thumb_reference(_limb_direction_between(elbow, shoulder),
                                 (fx, fy, fz))
    bx, by, bz = fy * az - fz * ay, fz * ax - fx * az, fx * ay - fy * ax
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    # Thumb side flips with the hand: the thumbs of two relaxed arms point at
    # each other, not the same way.
    tx = side_sign * (ax * cos_r + bx * sin_r)
    ty = side_sign * (ay * cos_r + by * sin_r)
    tz = side_sign * (az * cos_r + bz * sin_r)
    reach = HAND * (1.0 - 0.6 * max(0.0, min(1.0, grip)))
    finger = (wrist[0] + reach * fx, wrist[1] + reach * fy, wrist[2] + reach * fz)
    # A closing hand brings the thumb across the palm toward the fingers, so the
    # thumb-to-finger distance is what reads as grip.
    lean = 0.5 * max(0.0, min(1.0, grip))
    thumb = (
        wrist[0] + THUMB * (tx * (1.0 - lean) + fx * lean),
        wrist[1] + THUMB * (ty * (1.0 - lean) + fy * lean),
        wrist[2] + THUMB * (tz * (1.0 - lean) + fz * lean),
    )
    return finger, thumb


def _limb_direction_between(tip, root) -> tuple[float, float, float]:
    dx, dy, dz = tip[0] - root[0], tip[1] - root[1], tip[2] - root[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    return dx / n, dy / n, dz / n


def _thumb_reference(upper, fore) -> tuple[float, float, float]:
    """Unit vector perpendicular to the forearm that ``wrist_roll`` measures from.

    Referenced to the UPPER ARM -- which is how a real forearm's twist is
    anatomically defined, about the elbow -- rather than to a world axis. That
    choice is what makes it continuous through every pose a scripted motion
    actually holds. Its only pole is the forearm lying along the upper arm, i.e.
    a perfectly straight elbow, and that pole is harmless because it is the SAME
    configuration where ``nao_retarget`` declines to solve the roll joints at all
    (``ELBOW_YAW_MIN_BEND``): past it the retargeter emits nothing, so nothing
    downstream can see the fixture's reference swing.

    Two earlier versions both picked a world axis and both teleported the thumb
    180 degrees mid-motion: "whichever axis the forearm is least aligned with"
    flipped when a bending elbow crossed |y| = 0.9, and a fixed forward axis
    flipped at elbow 55 deg of an ordinary forward reach, where the forearm
    points straight at the camera. Measured: a 156 mm jump between adjacent
    one-degree steps, with the wrist held still. A body that teleports is
    exactly what this module's docstring promises not to build.
    """
    dot = sum(upper[i] * fore[i] for i in range(3))
    perp = [upper[i] - dot * fore[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in perp))
    if n < 1e-6:
        # Straight elbow: no angle to measure from. Any perpendicular keeps the
        # geometry valid, and nothing reads it here.
        perp = [-fore[1], fore[0], 0.0]
        n = math.sqrt(sum(c * c for c in perp)) or 1.0
    return tuple(c / n for c in perp)


def build_pose(state: BodyState, timestamp_s: float, frame_index: int) -> PoseFrame:
    """One ``PoseFrame`` of body + hand landmarks for the given body state."""
    z0 = STANDING_DISTANCE
    hip_y = state.crouch * MAX_CROUCH_DROP
    shoulder_y = hip_y - TORSO

    pelvis = (0.0, hip_y, z0)
    neck = (0.0, shoulder_y, z0)
    left_hip = (LEFT_SIDE_SIGN * HIP_HALF_WIDTH, hip_y, z0)
    right_hip = (-LEFT_SIDE_SIGN * HIP_HALF_WIDTH, hip_y, z0)
    left_shoulder = (LEFT_SIDE_SIGN * SHOULDER_HALF_WIDTH, shoulder_y, z0)
    right_shoulder = (-LEFT_SIDE_SIGN * SHOULDER_HALF_WIDTH, shoulder_y, z0)

    # Legs: a flexed hip means the foot has left the ground, otherwise it is planted.
    if state.left_hip_flex > 1e-6:
        left_knee, left_ankle = _leg_chain(left_hip, state.left_hip_flex, state.left_knee_flex)
    else:
        left_knee, left_ankle = _planted_leg(left_hip, GROUND_Y)
    if state.right_hip_flex > 1e-6:
        right_knee, right_ankle = _leg_chain(right_hip, state.right_hip_flex, state.right_knee_flex)
    else:
        right_knee, right_ankle = _planted_leg(right_hip, GROUND_Y)

    # "Out to the side" is +x for the left arm and -x for the right, matching
    # which side of the image each shoulder is on.
    left_elbow, left_wrist = _arm_chain(
        left_shoulder, LEFT_SIDE_SIGN,
        state.left_arm_side, state.left_arm_fwd, state.left_elbow
    )
    right_elbow, right_wrist = _arm_chain(
        right_shoulder, -LEFT_SIDE_SIGN,
        state.right_arm_side, state.right_arm_fwd, state.right_elbow
    )

    # Head: the nose sits above the shoulder line, swinging with head yaw/pitch.
    nose_up = NECK_TO_NOSE * math.cos(state.head_pitch)
    nose_fwd = -NECK_TO_NOSE * math.sin(state.head_pitch)
    nose = (
        math.sin(state.head_yaw) * 40.0,
        shoulder_y - nose_up,
        z0 + nose_fwd,
    )
    ear_y = shoulder_y - nose_up * 0.75
    lx, lz = _rot_y(LEFT_SIDE_SIGN * HEAD_HALF_WIDTH, 0.0, state.head_yaw)
    rx, rz = _rot_y(-LEFT_SIDE_SIGN * HEAD_HALF_WIDTH, 0.0, state.head_yaw)
    left_ear = (lx, ear_y, z0 + lz)
    right_ear = (rx, ear_y, z0 + rz)
    ex, ez = _rot_y(LEFT_SIDE_SIGN * EYE_HALF_WIDTH, 0.0, state.head_yaw)
    left_eye = (ex, shoulder_y - nose_up * 1.05, z0 + ez)
    ex, ez = _rot_y(-LEFT_SIDE_SIGN * EYE_HALF_WIDTH, 0.0, state.head_yaw)
    right_eye = (ex, shoulder_y - nose_up * 1.05, z0 + ez)

    points = {
        "nose": nose,
        "left_eye": left_eye, "right_eye": right_eye,
        "left_ear": left_ear, "right_ear": right_ear,
        "neck": neck, "pelvis": pelvis,
        "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
        "left_elbow": left_elbow, "right_elbow": right_elbow,
        "left_wrist": left_wrist, "right_wrist": right_wrist,
        "left_hip": left_hip, "right_hip": right_hip,
        "left_knee": left_knee, "right_knee": right_knee,
        "left_ankle": left_ankle, "right_ankle": right_ankle,
    }

    # Hands. left_hand_root is the hand frame's own origin; on a real subject it
    # is MeTRAbs' H36M wrist marker, which sits a little off the cmu_panoptic
    # wrist the arm chain uses. Here they coincide, which is the honest choice
    # for a synthetic body: the solve only ever reads differences within the
    # hand triad, so an invented offset would test nothing but itself.
    for side, shoulder, elbow, wrist, sign, roll, grip in (
        ("left", left_shoulder, left_elbow, left_wrist, LEFT_SIDE_SIGN,
         state.left_wrist_roll, state.left_grip),
        ("right", right_shoulder, right_elbow, right_wrist, -LEFT_SIDE_SIGN,
         state.right_wrist_roll, state.right_grip),
    ):
        finger, thumb = _hand_points(shoulder, elbow, wrist, sign, roll, grip)
        points[f"{side}_hand_root"] = wrist
        points[f"{side}_finger"] = finger
        points[f"{side}_thumb"] = thumb

    # Whole-body yaw last, about the vertical axis through the subject's centre,
    # so turning does not also translate them across the frame.
    if abs(state.body_yaw) > 1e-9:
        rotated = {}
        for name, (x, y, z) in points.items():
            rx_, rz_ = _rot_y(x, z - z0, state.body_yaw)
            rotated[name] = (rx_, y, rz_ + z0)
        points = rotated

    return PoseFrame(
        timestamp_s=timestamp_s,
        frame_index=frame_index,
        keypoints={
            name: Keypoint(x=x, y=y, z=z, visibility=1.0)
            for name, (x, y, z) in points.items()
        },
    )
