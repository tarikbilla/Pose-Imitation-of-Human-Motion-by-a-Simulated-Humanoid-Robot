"""
Model-based centre-of-mass (CoM) balance feedback for the NAO.

Why this exists
---------------
The pose comes from a single 2D camera, so depth — and therefore the human's
true balance — is lost. A free-standing NAO that blindly copies a depth-free
pose drifts its CoM off its small feet and topples. This module recovers the
missing information from the *robot's own model* instead of the camera:

  Option 2 (forward kinematics + known link masses).  NAO's link masses and
  local CoM offsets are fixed and documented. Every cycle we run forward
  kinematics from the measured joint angles, place each link's CoM in a common
  frame, and take the mass-weighted average to get the whole-body CoM:

      for each link:
          T_torso_link = forwardKinematics(jointAngles)
          com_torso    = T_torso_link * localCOM
          weightedSum += mass * com_torso
          totalMass   += mass
      robotCoM = weightedSum / totalMass

  This needs no Supervisor and runs in the normal controller. The InertialUnit
  (also readable in normal mode) supplies the gravity direction the camera
  cannot, so we can project the CoM onto the ground and compare it to the foot
  support polygon — a true balance error.

Closing the loop: Fibonacci-spiral, model-predictive search
-----------------------------------------------------------
A fixed-sign PD on that error is fragile: get a sign wrong (easy without a real
robot to test on) and the feedback becomes *positive* and tips faster. Instead
we *search*: candidate ankle/hip corrections are sampled on a golden-angle
(Fibonacci) spiral expanding around the last applied correction (the "past
fixed position"), the CoM model predicts the balance of each candidate, and we
apply the one that best re-centres the CoM. The model scores every move, so the
loop can only pick corrections it predicts will help — robust to sign and easy
to reason about.

Pure NumPy, no Webots import, so the kinematics/CoM math stays unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# NAO H25 kinematic constants (metres). Leg values match Nao.urdf; arm/head/
# torso values are the documented Aldebaran NAO H25 offsets.
# ---------------------------------------------------------------------------
NECK_OFFSET_Z = 0.1265
SHOULDER_OFFSET_Y = 0.098
SHOULDER_OFFSET_Z = 0.100
ELBOW_OFFSET_Y = 0.015
UPPER_ARM_LENGTH = 0.105
LOWER_ARM_LENGTH = 0.05595
HAND_OFFSET_X = 0.05775
HIP_OFFSET_Y = 0.050
HIP_OFFSET_Z = 0.085
THIGH_LENGTH = 0.100
TIBIA_LENGTH = 0.1029
FOOT_HEIGHT = 0.04519

# Foot sole rectangle in the AnkleRoll frame (metres): NAO foot is ~ -0.03 .. 0.10
# fore/aft and ~ +/-0.038 lateral, sole FOOT_HEIGHT below the ankle.
FOOT_X_BACK, FOOT_X_FRONT = -0.030, 0.100
FOOT_HALF_WIDTH = 0.038
# Contact tolerances. Two of them, because one number cannot do both jobs and
# trying to make it caused a real misdiagnosis:
#
#   * A sole tilted enough to matter lifts its outer corner only a little --
#     0.05 rad lifts it 1.9 mm -- so detecting sole tilt needs a TIGHT tolerance.
#   * A whole foot can sit a few millimetres high without leaving the floor: the
#     march's knee bob shortens one leg by 3.2 mm, and the robot answers that by
#     tipping its pelvis slightly, not by lifting the foot. Calling that foot
#     airborne collapses the support polygon to the other foot alone and reports a
#     negative margin for a robot standing squarely on both feet.
#
# So: whether a FOOT is on the floor is judged loosely, against the other foot;
# which of its CORNERS carry load is judged tightly, against that foot's own
# lowest corner.
FOOT_LIFT_TOL = 0.006      # m; a foot higher than this above the other is airborne
FOOT_CONTACT_TOL = 0.002   # m; within a grounded foot, which corners touch

# ---------------------------------------------------------------------------
# Gravity: how a measured torso tilt moves the CoM's ground projection.
#
# These live here, at module level, because TWO layers need them and they must
# agree: the balance feedback loop (BalanceController) and the feed-forward CoM
# compensation in lower_body. A sign that differs between the two would have one
# layer undoing the other.
#
# COM_HEIGHT_M is the lever arm -- a CoM at height h tilted by theta moves its
# ground projection by h*sin(theta). NAO's CoM stands about 0.30 m above the soles.
#
# The two signs are DERIVED from how the sensor is mounted and then checked
# against the logs -- they are not tuning knobs.
#
#   * Nao.proto (Webots R2025a) mounts the InertialUnit with
#     ``rotation 1 0 0 1.5708``: rolled +90 deg about the torso's x axis. That is
#     why its raw roll reads +pi/2 on an upright robot (the value the controller
#     measures and zeroes at startup).
#   * Webots' ENU API decomposes the sensor's world attitude as
#     Z(yaw) Y(pitch) X(roll)  (src/controller/c/inertial_unit.c), so
#         R_world_sensor = R_world_torso * Rx(pi/2) = Rz(yaw) Ry(pitch) Rx(roll+pi/2)
#     and the reported PITCH is the torso's own rotation about its +y (left) axis.
#     By the right-hand rule a positive rotation about +y tips the nose DOWN: a
#     FORWARD tilt reads POSITIVE. The zeroed ROLL is the torso's rotation about
#     +x (forward): positive tips the top of the robot toward -y, to the RIGHT.
#   * The Gyro node is mounted UNROTATED (torso frame), so it is an independent
#     witness: over the 2026-09-03 sessions d(imu_pitch)/dt agrees in sign with
#     gyro_y in 81% of 13,759 samples (r = +0.57) and d(imu_roll)/dt with gyro_x
#     in 85% of 9,527 (r = +0.39).
#   * The falls agree too: at 19 of 20 pitch-fall onsets the model's own FK centre
#     of mass sat at the TOE edge of the sole while imu_pitch grew POSITIVE.
#
# A forward tilt moves the real CoM forward (+x), so the model displaces it by
# +sin(pitch); a right tilt moves it to -y, so by -sin(roll).
#
# History: TILT_PITCH_SIGN used to be -1, inferred from one session whose fall
# direction was assumed rather than measured. Inverted, the term put the modelled
# CoM BEHIND the feet whenever the robot tilted forward, so both CoM shifters
# pushed the pelvis further FORWARD -- positive feedback. Every pitch fall in the
# 2026-09-03 logs carries its signature: HipPitch commanded to +0.45 and
# AnklePitch to -0.65 (both shifters at their clamps, CoM as far forward as they
# can put it) while the pitch grows from +0.2 to +0.6 rad.
#
# tests/test_balance.py::test_pitch_sign_matches_webots_convention_for_the_mounted_sensor
# re-derives both signs from the mounting and the API's formula, so they cannot
# silently regress.
COM_HEIGHT_M = 0.30
TILT_PITCH_SIGN = 1.0
TILT_ROLL_SIGN = 1.0


def _T(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = (tx, ty, tz)
    return M


def _rot(axis: Sequence[float], angle: float) -> np.ndarray:
    """Homogeneous rotation of ``angle`` rad about a (possibly non-unit) axis."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4)
    x, y, z = a / n
    c, s, C = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    M = np.eye(4)
    M[:3, :3] = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    return M


@dataclass(frozen=True)
class Joint:
    """A revolute joint: fixed offset from its parent frame, then rotation."""
    name: str
    parent: str
    offset: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class Link:
    """A point mass at ``local_com`` (m) in frame ``frame`` weighing ``mass`` kg."""
    frame: str
    mass: float
    local_com: tuple[float, float, float]


def _hip_axis(sign: float) -> tuple[float, float, float]:
    # 45-degree HipYawPitch axis (matches Nao.urdf): left (0, .707, -.707).
    return (0.0, 0.707107, -0.707107 * sign)


def _build_model() -> tuple[dict[str, Joint], list[Link]]:
    joints: list[Joint] = [
        # Head
        Joint("HeadYaw", "Torso", (0.0, 0.0, NECK_OFFSET_Z), (0, 0, 1)),
        Joint("HeadPitch", "HeadYaw", (0.0, 0.0, 0.0), (0, 1, 0)),
    ]
    links: list[Link] = [
        Link("Torso", 1.0496, (-0.004, 0.0, 0.043)),
        Link("HeadYaw", 0.0594, (0.0, 0.0, 0.030)),
        Link("HeadPitch", 0.5205, (0.0, 0.0, 0.053)),
    ]

    for side, sy in (("L", 1.0), ("R", -1.0)):
        # --- arm chain ---
        joints += [
            Joint(f"{side}ShoulderPitch", "Torso",
                  (0.0, sy * SHOULDER_OFFSET_Y, SHOULDER_OFFSET_Z), (0, 1, 0)),
            Joint(f"{side}ShoulderRoll", f"{side}ShoulderPitch", (0.0, 0.0, 0.0), (0, 0, 1)),
            Joint(f"{side}ElbowYaw", f"{side}ShoulderRoll",
                  (UPPER_ARM_LENGTH, sy * ELBOW_OFFSET_Y, 0.0), (1, 0, 0)),
            Joint(f"{side}ElbowRoll", f"{side}ElbowYaw", (0.0, 0.0, 0.0), (0, 0, 1)),
            Joint(f"{side}WristYaw", f"{side}ElbowRoll", (LOWER_ARM_LENGTH, 0.0, 0.0), (1, 0, 0)),
        ]
        links += [
            Link(f"{side}ShoulderPitch", 0.090, (0.0, 0.0, 0.0)),
            Link(f"{side}ShoulderRoll", 0.1577, (0.5 * UPPER_ARM_LENGTH, 0.0, 0.0)),
            Link(f"{side}ElbowYaw", 0.0648, (0.0, 0.0, 0.0)),
            Link(f"{side}ElbowRoll", 0.0777, (0.5 * LOWER_ARM_LENGTH, 0.0, 0.0)),
            Link(f"{side}WristYaw", 0.185, (HAND_OFFSET_X, 0.0, 0.0)),
        ]
        # --- leg chain (offsets from Nao.urdf) ---
        joints += [
            Joint(f"{side}HipYawPitch", "Torso",
                  (0.0, sy * HIP_OFFSET_Y, -HIP_OFFSET_Z), _hip_axis(sy)),
            Joint(f"{side}HipRoll", f"{side}HipYawPitch", (0.0, 0.0, 0.0), (1, 0, 0)),
            Joint(f"{side}HipPitch", f"{side}HipRoll", (0.0, 0.0, 0.0), (0, 1, 0)),
            Joint(f"{side}KneePitch", f"{side}HipPitch", (0.0, 0.0, -THIGH_LENGTH), (0, 1, 0)),
            Joint(f"{side}AnklePitch", f"{side}KneePitch", (0.0, 0.0, -TIBIA_LENGTH), (0, 1, 0)),
            Joint(f"{side}AnkleRoll", f"{side}AnklePitch", (0.0, 0.0, 0.0), (1, 0, 0)),
        ]
        links += [
            Link(f"{side}HipYawPitch", 0.07, (0.0, 0.0, 0.0)),
            Link(f"{side}HipRoll", 0.13, (0.0, 0.0, 0.0)),
            Link(f"{side}HipPitch", 0.39, (0.0, 0.0, -0.5 * THIGH_LENGTH)),
            Link(f"{side}KneePitch", 0.29, (0.0, 0.0, -0.5 * TIBIA_LENGTH)),
            Link(f"{side}AnkleRoll", 0.16, (0.020, 0.0, -0.6 * FOOT_HEIGHT)),
        ]

    return {j.name: j for j in joints}, links


_JOINTS, _LINKS = _build_model()


class NaoCoMModel:
    """Forward kinematics + mass-weighted whole-body CoM in the Torso frame."""

    def __init__(self) -> None:
        self.joints = _JOINTS
        self.links = _LINKS
        self.total_mass = sum(link.mass for link in self.links)

    def frames(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        """4x4 transforms of every frame in the Torso frame for ``angles`` (rad)."""
        out: dict[str, np.ndarray] = {"Torso": np.eye(4)}
        # _JOINTS is built parent-before-child, so a single pass resolves the tree.
        for name, j in self.joints.items():
            parent = out.get(j.parent)
            if parent is None:  # parent not yet placed; shouldn't happen with our order
                continue
            local = _T(*j.offset) @ _rot(j.axis, float(angles.get(name, 0.0)))
            out[name] = parent @ local
        return out

    def com(self, angles: dict[str, float],
            frames: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Whole-body CoM (x, y, z) in the Torso frame."""
        frames = frames if frames is not None else self.frames(angles)
        acc = np.zeros(3)
        for link in self.links:
            T = frames[link.frame]
            acc += link.mass * (T[:3, :3] @ np.asarray(link.local_com) + T[:3, 3])
        return acc / self.total_mass

    def foot_sole_center(self, side: str,
                         frames: dict[str, np.ndarray]) -> np.ndarray:
        """Centre of a foot's sole (x, y, z) in the Torso frame."""
        T = frames[f"{side}AnkleRoll"]
        sole_local = np.array([0.5 * (FOOT_X_BACK + FOOT_X_FRONT), 0.0, -FOOT_HEIGHT])
        return T[:3, :3] @ sole_local + T[:3, 3]

    def foot_corners(self, side: str,
                     frames: dict[str, np.ndarray]) -> np.ndarray:
        """The four sole corners of one foot as (x, y, z) in the Torso frame."""
        T = frames[f"{side}AnkleRoll"]
        out = []
        for x in (FOOT_X_BACK, FOOT_X_FRONT):
            for y in (-FOOT_HALF_WIDTH, FOOT_HALF_WIDTH):
                out.append(T[:3, :3] @ np.array([x, y, -FOOT_HEIGHT]) + T[:3, 3])
        return np.array(out)

    def support_margin(self, angles: dict[str, float],
                       frames: dict[str, np.ndarray] | None = None,
                       com_xy: np.ndarray | None = None,
                       contact: dict[str, bool] | None = None) -> float:
        """Signed distance (m) of the CoM *inside* the DOUBLE-support region.

        Positive means the centre of mass projects inside the area bounded by both
        feet, which is the actual condition for standing up; negative means it has
        left it and the robot is going over.

        This is the objective :class:`BalanceController` minimises, and it replaces
        "distance from the midpoint of the two foot centres". That earlier measure
        was not a balance test at all: the sole rectangle runs from 30 mm behind
        the ankle to 100 mm in front of it, so its centroid is 35 mm FORWARD of the
        ankle, while a standing NAO rests with its CoM about 7 mm forward of the
        ankle. The old cost therefore read a perfectly balanced robot as 28 mm out
        of balance -- above its own 20 mm action threshold -- so the loop pushed
        continuously, saturated the ankle-roll correction against its clamp, and
        the sole-tilt limiter then dragged both hips over to protect sole contact.
        The measured result was a constant 0.300 rad lean in 98% of frames.

        The region is taken as the axis-aligned bounding box of all eight sole
        corners. For a normal stance the feet are near-parallel so that is the true
        convex hull; with the feet yawed it is slightly optimistic at the corners,
        which is acceptable here because the hip-yaw authority is small and the
        IMU-tilt term below is deliberately pessimistic.
        """
        return min(self.support_margins(angles, frames, com_xy, contact))

    def tilted_com_xy(self, angles: dict[str, float],
                      torso_rp: tuple[float, float],
                      frames: dict[str, np.ndarray] | None = None,
                      weight: float = 1.0) -> np.ndarray:
        """Ground projection of the CoM, corrected for measured torso tilt.

        The kinematics are expressed in the TORSO frame, so they cannot know that
        the torso itself is tilted -- and a tilted torso means the true ground
        projection of the mass above it is displaced. This is where the
        InertialUnit supplies the gravity information the camera cannot.

        ``weight`` scales the lever arm; 1.0 is the physical value.
        """
        frames = frames if frames is not None else self.frames(angles)
        com = self.com(angles, frames)
        roll, pitch = torso_rp
        k = weight * self.com_height(frames, com)
        return np.array([
            com[0] + k * TILT_PITCH_SIGN * math.sin(pitch),
            com[1] - k * TILT_ROLL_SIGN * math.sin(roll),
        ])

    def com_height(self, frames: dict[str, np.ndarray], com: np.ndarray) -> float:
        """Lever arm of the tilt term: height of the CoM above the lowest sole (m).

        Read off the same forward kinematics as everything else rather than fixed
        at COM_HEIGHT_M, so a deep squat -- which brings the CoM 40 mm closer to
        the floor -- is not charged the standing lever arm. A 0.1 rad pelvis shift
        moves the CoM about 16 mm, so overstating this by 40 mm at 0.3 rad of tilt
        is a 12 mm error the loop would then chase. Falls back to the nominal value
        if the geometry is degenerate (a fallen robot, no feet under the body).
        """
        soles = [float(self.foot_sole_center(side, frames)[2]) for side in ("L", "R")]
        height = float(com[2]) - min(soles)
        return height if 0.10 < height < 0.50 else COM_HEIGHT_M

    def support_margins_tilted(self, angles: dict[str, float],
                               torso_rp: tuple[float, float],
                               frames: dict[str, np.ndarray] | None = None,
                               weight: float = 1.0,
                               contact: dict[str, bool] | None = None) -> tuple[float, float]:
        """``support_margins`` with the gravity correction applied. One call, so
        the two layers that need it cannot disagree about the signs."""
        frames = frames if frames is not None else self.frames(angles)
        com_xy = self.tilted_com_xy(angles, torso_rp, frames, weight)
        return self.support_margins(angles, frames, com_xy, contact)

    def support_margins(self, angles: dict[str, float],
                        frames: dict[str, np.ndarray] | None = None,
                        com_xy: np.ndarray | None = None,
                        contact: dict[str, bool] | None = None) -> tuple[float, float]:
        """``(fore_aft, lateral)`` margins inside the double-support region.

        Reported per axis rather than only as their minimum, because the two are
        not comparable: NAO's sole runs 30 mm behind the ankle and 100 mm in front,
        so standing neutral leaves about 39 mm of fore/aft margin but 88 mm of
        lateral. Collapsing them with ``min`` means the fore/aft term is *always*
        the smaller one, and a loop that optimises the minimum would never correct
        a sideways disturbance at all -- it would keep working on the axis that was
        already the tightest. Each axis is instead defended against its own
        threshold (see BalanceParams.desired_margin_x / _y).

        ``contact`` -- ``{"L": bool, "R": bool}`` from the foot force sensors
        (see :func:`contact_from_fsr`) -- says which feet actually carry load.
        The kinematic contact test below can only judge the COMMANDED geometry:
        a pose whose two soles are both nominally flat is counted as double
        support however the robot is really standing. An offline replay of a
        recorded lateral fall found this model reporting 30-110 mm of margin
        while the sensors read 55 N on one foot and 1 N on the other: the robot
        was on one sole's edge and the polygon still spanned both feet. A foot
        the sensors say is unloaded does not bound the support region.
        """
        frames = frames if frames is not None else self.frames(angles)
        if com_xy is None:
            com_xy = self.com(angles, frames)[:2]
        # Only corners actually touching the floor bound the support region, judged
        # in two stages (see FOOT_LIFT_TOL / FOOT_CONTACT_TOL).
        #
        # This is load-bearing rather than a refinement: the balance loop's main
        # lever is ankle roll, so a polygon built from all eight corners regardless
        # of height would GROW as the ankles rolled -- rewarding the loop for
        # standing the robot on the edges of its feet, which is the one thing the
        # sole-tilt budget exists to prevent.
        sides = [s for s in ("L", "R") if contact is None or contact.get(s, True)]
        if not sides:
            sides = ["L", "R"]
        feet = {side: self.foot_corners(side, frames) for side in sides}
        floor = min(float(c[:, 2].min()) for c in feet.values())
        contact = []
        for corners in feet.values():
            foot_low = float(corners[:, 2].min())
            if foot_low > floor + FOOT_LIFT_TOL:
                continue                     # this foot is off the ground
            contact.extend(corners[corners[:, 2] <= foot_low + FOOT_CONTACT_TOL])
        if len(contact) < 2:                 # cannot bound an area from one point
            contact = [c for corners in feet.values() for c in corners]
        contact = np.asarray(contact)
        lo = contact[:, :2].min(axis=0)
        hi = contact[:, :2].max(axis=0)
        return (float(min(com_xy[0] - lo[0], hi[0] - com_xy[0])),
                float(min(com_xy[1] - lo[1], hi[1] - com_xy[1])))

    def stance_margin(self, angles: dict[str, float], side: str,
                      frames: dict[str, np.ndarray] | None = None) -> float:
        """Signed distance (m) of the whole-body CoM *inside* one foot's support
        rectangle — positive when the CoM projects within the foot, negative when
        it has left it.

        This is the single-support balance test the symmetric double-support
        :class:`BalanceController` cannot provide: it scores the CoM against the
        STANCE foot alone (not the two-foot midpoint). The walk engine uses it as
        a hard gate — it refuses to lift the swing foot unless the CoM is safely
        over the stance foot. Model-based (forward kinematics), so it is testable
        off-simulation; an IMU-tilt refinement of the projection is future work.
        """
        frames = frames if frames is not None else self.frames(angles)
        com = self.com(angles, frames)
        T = frames[f"{side}AnkleRoll"]
        # Express the CoM in the foot's own (AnkleRoll) frame; its x/y are the
        # fore-aft and lateral offsets within the (near-horizontal) sole plane.
        com_local = T[:3, :3].T @ (com - T[:3, 3])
        x, y = float(com_local[0]), float(com_local[1])
        fore_aft = min(x - FOOT_X_BACK, FOOT_X_FRONT - x)
        lateral = min(y + FOOT_HALF_WIDTH, FOOT_HALF_WIDTH - y)
        return min(fore_aft, lateral)


def clip_torso_speeds(poses: Sequence[tuple[float, dict[str, float]]],
                      model: "NaoCoMModel | None" = None) -> list[float]:
    """Forward torso speed (m/s) the clip itself commands, per keyframe.

    Odometry, not a sensor: the stance foot is on the floor, so the torso moves
    backwards relative to it by exactly as much as forward kinematics says the
    hip has swung over it. The stance foot is whichever sole is lower, and only
    intervals with the SAME stance foot contribute -- across a stance exchange
    the reference jumps and the difference is meaningless.
    """
    model = model or NaoCoMModel()
    advance: list[float] = []
    total = 0.0
    previous: float | None = None
    last_stance: str | None = None
    for _t, angles in poses:
        frames = model.frames(angles)
        lows = {side: float(model.foot_corners(side, frames)[:, 2].min())
                for side in ("L", "R")}
        stance = "L" if lows["L"] <= lows["R"] else "R"
        transform = frames[f"{stance}AnkleRoll"]
        sole = transform[:3, :3] @ np.array([0.035, 0.0, -0.04519]) + transform[:3, 3]
        if last_stance == stance and previous is not None:
            total += -(float(sole[0]) - previous)
        previous = float(sole[0])
        last_stance = stance
        advance.append(total)
    speeds: list[float] = []
    for index, (t, _a) in enumerate(poses):
        lo = max(0, index - 1)
        hi = min(len(poses) - 1, index + 1)
        span = float(poses[hi][0]) - float(poses[lo][0])
        speeds.append(0.0 if span <= 0.0
                      else (advance[hi] - advance[lo]) / span)
    return speeds


def clip_torso_yaw(poses: Sequence[tuple[float, dict[str, float]]],
                   model: "NaoCoMModel | None" = None) -> list[float]:
    """Cumulative torso yaw (rad) the clip itself commands, per keyframe.

    The rotational twin of :func:`clip_torso_speeds`, and odometry for the same
    reason: the stance foot is planted, so whatever yaw forward kinematics puts
    between the torso and that sole is yaw the robot has actually turned
    through. Positive is the robot's own LEFT, matching
    ``walk_motion.motion_nominal_yaw``.

    This is what makes a turn clip measurable rather than merely named. The
    filename says ``TurnLeft180``; the keyframes say +184.0 deg, delivered at a
    steady 24 deg/s between t=1.2 s and t=8.8 s. Knowing the SHAPE of that -- not
    just the total -- is what lets a turn be stopped part-way at a chosen angle
    instead of being played whole, which is the difference between turning 90 deg
    in one clip and turning it in three.

    Only intervals with the SAME stance foot contribute, exactly as in
    :func:`clip_torso_speeds`: across a stance exchange the reference sole
    changes and the difference between the two yaw readings is a property of the
    robot's stance width, not of any rotation it performed.
    """
    model = model or NaoCoMModel()
    out: list[float] = []
    total = 0.0
    previous: float | None = None
    last_stance: str | None = None
    for _t, angles in poses:
        frames = model.frames(angles)
        lows = {side: float(model.foot_corners(side, frames)[:, 2].min())
                for side in ("L", "R")}
        stance = "L" if lows["L"] <= lows["R"] else "R"
        # frames[...] is the sole's pose in the TORSO frame, so its transpose is
        # the torso seen from the planted sole -- which is the world, for as long
        # as that sole stays put.
        rotation = frames[f"{stance}AnkleRoll"][:3, :3].T
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if last_stance == stance and previous is not None:
            total += float((yaw - previous + np.pi) % (2.0 * np.pi) - np.pi)
        previous = yaw
        last_stance = stance
        out.append(total)
    return out


def safe_exit_times(poses: Sequence[tuple[float, dict[str, float]]],
                    max_joint_speed: float = 3.0,
                    max_sole_spread: float = 0.004,
                    max_torso_speed: float = 0.05,
                    model: "NaoCoMModel | None" = None) -> list[float]:
    """Times in a motion clip at which playback may be stopped safely.

    A clip is normally played to completion because a clip BOUNDARY is a balanced
    double-support pose -- but that makes the clip's length the latency of "stop
    walking", which for Cyberbotics' 6.76 s continuous walk would be untenable.
    The boundary is not actually special, though: any keyframe with the same
    properties will do, and a walk clip passes through many of them. A pose
    qualifies when

      * both feet are on the floor (neither sole's lowest corner is more than
        FOOT_LIFT_TOL above the other's),
      * each sole is flat (its four corners within ``max_sole_spread``),
      * the centre of mass projects inside the contact-filtered support polygon
        on both axes, and
      * the clip is not about to move fast (the commanded joint speed to the next
        keyframe is under ``max_joint_speed`` rad/s), so stopping does not freeze
        the robot mid-lurch, and
      * the TORSO is not still travelling (``max_torso_speed``, m/s).

    That last gate is the one a purely static test misses, and it matters. Freezing
    the legs does not freeze the robot: the body keeps the momentum it had, and the
    capture point sits v/omega ahead of the centre of mass (omega = sqrt(g/h) ~=
    5.9 rad/s at this crouch). 13 of the 46 poses that pass the static tests in
    Forwards.motion are moving at up to 0.179 m/s, which puts the capture point
    30 mm beyond the CoM against a margin budget of only 40-60 mm -- so "stop
    here" would be handing back a robot that then walks itself over. At 0.05 m/s
    the excursion is 8.5 mm.

    Measured over the clips Webots ships, the momentum gate costs about a third
    of the candidates and most of the responsiveness: Forwards.motion keeps 33 of
    its 66 keyframes with a longest wait of 1.04 s (was 46 and 0.32 s), the turn
    clips keep 49/73 and 108/226 at 0.52 s, and Forwards50.motion keeps only
    38/170 at 2.56 s -- because a continuous walk is, by design, almost never
    standing still. That last number is why this is the FALLBACK path for the
    walk clip rather than the main one: see :func:`walk_motion.gait_cycle`, which
    leaves through the clip's own deceleration instead of freezing it mid-stride,
    and bounds the stop latency at 2.48 s without ever stopping on momentum. The
    turn clips, which do come to rest repeatedly, are served well by this path.

    Returns the times in seconds, ascending. An empty list means "no early exit
    is known" and the caller should play to completion, which is the old
    behaviour.
    """
    model = model or NaoCoMModel()
    torso = clip_torso_speeds(poses, model)
    out: list[float] = []
    for index, (t, angles) in enumerate(poses):
        if abs(torso[index]) > max_torso_speed:
            continue                                  # still carrying momentum
        frames = model.frames(angles)
        corners = {side: model.foot_corners(side, frames) for side in ("L", "R")}
        lows = {side: float(c[:, 2].min()) for side, c in corners.items()}
        floor = min(lows.values())
        if any(low > floor + FOOT_LIFT_TOL for low in lows.values()):
            continue                                  # a foot is in the air
        if any(float(c[:, 2].max()) - float(c[:, 2].min()) > max_sole_spread
               for c in corners.values()):
            continue                                  # a sole is on its edge
        speed = 0.0
        if index + 1 < len(poses):
            t2, nxt = poses[index + 1]
            dt = max(t2 - t, 1e-3)
            speed = max((abs(nxt.get(j, 0.0) - v) / dt for j, v in angles.items()),
                        default=0.0)
        if speed >= max_joint_speed:
            continue                                  # about to move fast
        if min(model.support_margins(angles, frames)) <= 0.0:
            continue                                  # not statically holdable
        out.append(float(t))
    return out


# ---------------------------------------------------------------------------
# Fibonacci / golden-angle spiral sampling
# ---------------------------------------------------------------------------
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996 rad


def fibonacci_spiral(n: int, scale: float) -> list[tuple[float, float]]:
    """``n`` points on a sunflower/Fibonacci spiral, radius growing as sqrt(k).

    Even, low-discrepancy coverage of a disc of radius ``scale`` — ideal for
    searching a 2D correction space outward from the centre (the last stable
    correction) without re-sampling the same direction.
    """
    pts: list[tuple[float, float]] = []
    for k in range(n):
        r = scale * math.sqrt((k + 0.5) / n)
        theta = (k + 1) * GOLDEN_ANGLE
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


# A foot carrying less than this share of the total sole load is not supporting
# the robot, whatever the commanded geometry says. Below ``min_total`` newtons
# the sensors are not reading a standing robot at all and the mask is left open.
CONTACT_SHARE_MIN = 0.08


def contact_from_fsr(fsr: dict[str, float] | None, min_share: float = CONTACT_SHARE_MIN,
                     min_total: float = 10.0) -> dict[str, bool] | None:
    """``{"L": bool, "R": bool}`` from per-foot vertical loads, or None if unknown."""
    if not fsr:
        return None
    left = float(fsr.get("L", 0.0) or 0.0)
    right = float(fsr.get("R", 0.0) or 0.0)
    total = left + right
    if total < min_total:
        return None
    return {"L": left / total >= min_share, "R": right / total >= min_share}


@dataclass
class BalanceParams:
    """Tuning for :class:`BalanceController`."""
    # Support margin (see NaoCoMModel.support_margin) the loop tries to keep. Every
    # posture the robot actually holds measures +0.037..+0.048 m, so at 0.025 the
    # loop is idle while standing and only engages once the margin has genuinely
    # degraded. The previous parameter was not a margin at all -- it thresholded
    # the CoM's distance from the midpoint of the two foot CENTROIDS, which is
    # 35 mm forward of the ankle, so a balanced robot measured 28 mm "off" and the
    # loop never stopped pushing.
    # Per axis, because the geometry is not symmetric: neutral standing leaves
    # ~39 mm of fore/aft margin (the sole is short behind the ankle) against
    # ~88 mm laterally. A single shared threshold would either sit permanently
    # tripped on fore/aft or permanently blind to roll.
    desired_margin_x: float = 0.025   # m, fore/aft (of ~39 mm available)
    desired_margin_y: float = 0.055   # m, lateral  (of ~88 mm available)
    # Clamps on the two correction parameters. Both move the pelvis while holding
    # the soles FLAT (see _apply_corr), so neither can tilt a sole however large it
    # gets -- the sole-tilt limiter downstream is a no-op on this path by
    # construction rather than by tuning.
    max_pitch_corr: float = 0.25     # rad; fore/aft pelvis shift
    max_roll_corr: float = 0.20       # rad; lateral pelvis shift
    # Penalty on the SIZE of the correction, in cost-units per radian. Without it
    # the search accepts any candidate that does not make the cost worse, and
    # because the spiral perturbs the pitch and roll axes together, a purely
    # lateral disturbance would drag the pitch correction to its clamp as a free
    # side effect -- NAO has 91 mm of sole ahead of the ankle, so moving there
    # costs nothing in the objective. A balance loop should use the smallest
    # correction that works: everything it spends is subtracted from the pose the
    # robot is supposed to be imitating.
    effort_weight: float = 0.010
    search_points: int = 24         # Fibonacci samples per cycle
    search_scale: float = 0.18      # rad; spiral radius in the correction space
    slew: float = 0.25              # per-cycle blend toward the chosen correction (0..1)
    # Fraction of the true lever arm used when converting torso tilt into a CoM
    # ground displacement. 1.0 is the physical value (h * sin(theta) for a CoM at
    # height h). It used to be 0.6, deliberately under-weighted so that a
    # mis-signed InertialUnit could not dominate the loop -- but under-weighting
    # does not fix a wrong sign, it just makes the loop lose more slowly, and it
    # also makes it too insensitive to recover a real lean: at 0.6 a 0.30 rad
    # forward tilt left the modelled CoM still inside the polygon, so the loop sat
    # idle while the robot toppled. The sign is now established (PITCH_SIGN) and a
    # divergence detector shouts if it is ever wrong again, so the honest value is
    # used.
    tilt_weight: float = 1.0
    # Lead compensation on the measured tilt: the model is fed
    # tilt + tilt_lead_s * gyro_rate instead of the raw tilt. An inverted pendulum
    # with NAO's CoM height diverges with a time constant of ~0.17 s
    # (omega = sqrt(g / 0.31 m) = 5.6 rad/s), while the command path -- EMA
    # smoothing, the motor velocity cap and Webots' position tracking -- adds
    # 60-100 ms of lag. A purely proportional loop on the tilt therefore acts on
    # where the robot WAS; the rate term makes it act on where it is going, which
    # is the damping every ankle-strategy balancer in the literature carries. The
    # controller's clip-abort logic already trusts the same formula with 0.25 s;
    # here 0.12 s is enough to lead the actuation lag without amplifying gyro
    # noise. Zero disables it.
    tilt_lead_s: float = 0.12
    # Fastest the correction may move, in rad/s of pelvis shift. The spiral +
    # slew above could otherwise step 0.045 rad per 20 ms tick (2.25 rad/s), which
    # the hip motors cannot follow (cap 1.77 rad/s): the command ran ahead of the
    # joint, the measured posture lagged by up to 0.2 rad, and the loop then
    # corrected the lag as if it were a new error. 0.8 rad/s is ~130 mm/s of CoM
    # travel -- faster than the pendulum grows for any recoverable disturbance and
    # still inside what the motors deliver.
    max_rate: float = 0.8


@dataclass
class BalanceController:
    """Model-predictive CoM balance via a Fibonacci-spiral correction search.

    Each :meth:`compute_correction` returns small additive joint deltas (rad)
    for the ankles (pitch+roll) and hips (pitch+roll) that the CoM model
    predicts will keep the projected CoM over the feet. Only the legs are
    touched; arms/head keep imitating.
    """
    model: NaoCoMModel
    params: BalanceParams = field(default_factory=BalanceParams)
    # Last applied correction = the spiral's centre next cycle ("past position").
    _state: dict[str, float] = field(default_factory=lambda: {
        "pitch": 0.0, "roll": 0.0,
    })
    _tilt_history: dict[str, list] = field(default_factory=dict)
    _divergence: str = ""

    CORRECTED_JOINTS = (
        "LAnklePitch", "RAnklePitch", "LAnkleRoll", "RAnkleRoll",
        "LHipPitch", "RHipPitch", "LHipRoll", "RHipRoll",
    )

    def reset(self) -> None:
        """Forget the correction and the tilt history.

        Call this whenever the robot is no longer the robot this state describes --
        after a fall recovery, or when a motion clip has moved the whole body. The
        correction is an integrator: it holds the last pelvis shift so the next
        cycle can search around it, which is right within an episode and wrong
        across one.

        Missing this was a fall in its own right. ``simulationReset`` drops the
        robot upright, but the loop kept whatever correction it had ended the
        previous episode with -- which, on a robot that had just fallen, was its
        clamp. Recorded on 2026-09-04 (log 1788521290): the first control step of
        a new episode commanded HipPitch -0.345 / AnklePitch +0.145, a pelvis
        already shifted 0.25 rad with the centre of mass 40 mm off centre, before
        the robot had taken a single step. Six of the ten episodes in that session
        began that way and none of them lasted three seconds.
        """
        for key in self._state:
            self._state[key] = 0.0
        self._tilt_history.clear()
        self._divergence = ""

    def _apply_corr(self, angles: dict[str, float], c: dict[str, float]) -> dict[str, float]:
        """A copy of ``angles`` with the correction applied, soles kept FLAT.

        Each axis moves the pelvis by rotating the hip one way and the ankle the
        other by the same amount, on both legs. Because a sole's attitude is
        ``Hip + Knee + Ankle`` and this adds ``+c`` and ``-c`` to that sum, the
        soles stay exactly as flat as they were while the body translates over
        them. That is the real ankle strategy, and getting it wrong is what made
        the previous version useless:

        it moved hip and ankle roll INDEPENDENTLY, so every correction tilted the
        soles. Once the support polygon is measured from the sole corners actually
        touching the floor, a 0.05 rad tilt cuts the lateral margin from 88 mm to
        14 mm -- so an independent correction destroyed support faster than it
        moved the centre of mass, every candidate scored worse than doing nothing,
        and the search sat still while the robot went over. A biped does not
        balance sideways by rolling its ankles off flat; it keeps its feet flat and
        moves its hips.
        """
        out = dict(angles)
        for side in ("L", "R"):
            hp, ap = f"{side}HipPitch", f"{side}AnklePitch"
            hr, ar = f"{side}HipRoll", f"{side}AnkleRoll"
            out[hp] = angles.get(hp, 0.0) + c["pitch"]
            out[ap] = angles.get(ap, 0.0) - c["pitch"]
            out[hr] = angles.get(hr, 0.0) + c["roll"]
            out[ar] = angles.get(ar, 0.0) - c["roll"]
        return out

    # InertialUnit sign conventions live at module level (TILT_PITCH_SIGN /
    # TILT_ROLL_SIGN, derived from the sensor mounting -- see the comment there)
    # because lower_body's feed-forward shift uses the same model helper. They are
    # mirrored here only for the divergence message.
    PITCH_SIGN = TILT_PITCH_SIGN
    ROLL_SIGN = TILT_ROLL_SIGN
    NOMINAL_COM_HEIGHT = COM_HEIGHT_M  # m; fallback lever arm (see NaoCoMModel.com_height)

    def _imbalance(self, angles: dict[str, float],
                   torso_rp: tuple[float, float],
                   contact: dict[str, bool] | None = None) -> tuple[float, float]:
        """Return ``(cost, margin)``. Cost 0 means "inside the support polygon
        with room to spare"; larger cost = closer to tipping.

        The objective is how far the centre of mass projects INSIDE the region
        bounded by both feet (``NaoCoMModel.support_margin``), not how close it is
        to any particular point. Standing balanced is a large family of postures,
        and asking the loop to centre the CoM on a single point instead made it
        fight a fixed 28 mm offset forever -- see support_margin's docstring.

        The InertialUnit contributes the gravity information the camera cannot:
        the kinematics are expressed in the TORSO frame, so a tilted torso means
        the true ground projection of the CoM is displaced from where they put it.
        Applying that as a displacement of the CoM (rather than as a separate
        additive penalty) keeps it *directional*, so the search can actually reduce
        it, while ``tilt_weight`` keeps a mis-signed IMU from dominating.
        """
        frames = self.model.frames(angles)
        mx, my = self.model.support_margins_tilted(
            angles, torso_rp, frames, self.params.tilt_weight, contact
        )
        # Sum of per-axis deficits, so a disturbance on either axis is answered on
        # that axis. Zero when both are comfortable, which is the state a standing
        # robot must sit in.
        cost = (max(0.0, self.params.desired_margin_x - mx)
                + max(0.0, self.params.desired_margin_y - my))
        return float(cost), float(min(mx, my))

    def _deficits(self, angles: dict[str, float], torso_rp: tuple[float, float],
                  contact: dict[str, bool] | None = None) -> tuple[float, float]:
        """How far each axis is short of its desired margin (>= 0 each)."""
        frames = self.model.frames(angles)
        mx, my = self.model.support_margins_tilted(
            angles, torso_rp, frames, self.params.tilt_weight, contact
        )
        return (max(0.0, self.params.desired_margin_x - mx),
                max(0.0, self.params.desired_margin_y - my))

    def _score(self, angles: dict[str, float], corr: dict[str, float],
               torso_rp: tuple[float, float],
               contact: dict[str, bool] | None = None) -> float:
        """Search objective: balance deficit plus a penalty on correction size."""
        dx, dy = self._deficits(self._apply_corr(angles, corr), torso_rp, contact)
        effort = sum(abs(v) for v in corr.values())
        return dx + dy + self.params.effort_weight * effort

    def compute_correction(self, angles: dict[str, float],
                           torso_rp: tuple[float, float] = (0.0, 0.0), *,
                           tilt_rate: tuple[float, float] = (0.0, 0.0),
                           dt_s: float = 0.02,
                           contact: dict[str, bool] | None = None) -> dict[str, float]:
        """Joint deltas (rad) keeping the CoM over the feet; all zero if safe.

        ``torso_rp`` is the torso tilt (roll, pitch); ``tilt_rate`` the gyro
        (roll_rate, pitch_rate) in rad/s, used as lead compensation (see
        BalanceParams.tilt_lead_s); ``dt_s`` the control period, which bounds how
        far the correction may move this call (BalanceParams.max_rate);
        ``contact`` which feet the force sensors say carry load (see
        :func:`contact_from_fsr`).
        """
        p = self.params
        roll, pitch = torso_rp
        d_roll, d_pitch = tilt_rate
        predicted = (roll + p.tilt_lead_s * d_roll, pitch + p.tilt_lead_s * d_pitch)
        step_cap = p.max_rate * max(0.0, dt_s)

        deficit_x, deficit_y = self._deficits(angles, predicted, contact)
        base_cost = deficit_x + deficit_y
        if base_cost <= 0.0:
            # Inside the support polygon with room to spare: relax any standing
            # correction back toward zero rather than holding a posture we no
            # longer need. This is the branch a STANDING robot should sit in, and
            # the reason the loop does nothing while the robot is upright.
            for k in self._state:
                self._state[k] = _toward(self._state[k],
                                         self._state[k] * (1.0 - p.slew), step_cap)
            return self._expand_state()

        # Search corrections on a Fibonacci spiral around the last one -- but only
        # along the axes that are actually short of margin. The spiral perturbs
        # pitch and roll together, and a replay of the recorded sessions found the
        # roll correction moving in 84% of its active frames with NO lateral
        # deficit at all: a candidate whose pitch component bought margin was
        # accepted with whatever roll component it happened to carry, since the
        # effort penalty (0.01 per rad) is smaller than the pitch gain. That roll
        # then stacked on the imitation's lean. An axis with nothing to fix stays
        # where it is and relaxes, as in the idle branch above.
        c0 = self._state
        best = dict(c0)
        best_cost = self._score(angles, c0, predicted, contact)
        move_pitch, move_roll = deficit_x > 0.0, deficit_y > 0.0
        for dx, dy in fibonacci_spiral(p.search_points, p.search_scale):
            cand = {
                "pitch": _clamp(c0["pitch"] + (dx if move_pitch else 0.0),
                                -p.max_pitch_corr, p.max_pitch_corr),
                "roll":  _clamp(c0["roll"] + (dy if move_roll else 0.0),
                                -p.max_roll_corr, p.max_roll_corr),
            }
            cost = self._score(angles, cand, predicted, contact)
            if cost < best_cost:
                best_cost, best = cost, cand
        if not move_pitch:
            best["pitch"] = c0["pitch"] * (1.0 - p.slew)
        if not move_roll:
            best["roll"] = c0["roll"] * (1.0 - p.slew)

        # Slew toward the best candidate so corrections ease in, and never faster
        # than the motors can follow (see max_rate).
        for k in self._state:
            target = self._state[k] + p.slew * (best[k] - self._state[k])
            self._state[k] = _toward(self._state[k], target, step_cap)
        return self._expand_state()

    # Divergence detector. A balance loop with an inverted sign does not fail
    # loudly: it corrects hard, at its clamp, in the wrong direction, and the robot
    # leans further and further over. That is exactly what happened here -- 70
    # seconds of monotonically growing forward tilt with the correction saturated
    # the whole time -- and nothing in the logs said so. Now it does.
    # How long tilt must keep growing to count as divergence, PER AXIS.
    #
    # These are not the same number because the two axes fail on completely
    # different timescales, and using the pitch figure for roll made the detector
    # structurally incapable of catching a lateral fall.
    #
    # The evidence: the fore/aft divergence that set PITCH_SIGN took 70 s to put
    # the robot down -- the sole is long in x, so the CoM has a long way to travel
    # and the loop has to push it the whole way. A lateral divergence observed on
    # 2026-09-03 (logs/webots_joint_trajectory_1788431474.csv, rows 2865-2925) went
    # from the first non-zero roll correction to both feet off the ground in 1.2 s,
    # because the foot is narrow in y and the CoM only has to cross ~30 mm. With a
    # 3.0 s window the detector needs the robot to keep diverging for more than
    # twice as long as the fall actually lasts, so it never fires and the one
    # instrument that exists for confirming ROLL_SIGN is silent exactly when it
    # matters.
    DIVERGENCE_S = {"pitch": 3.0, "roll": 0.6}
    # Rad the tilt must grow by over that window. Scaled with the window so both
    # axes describe a comparable rate rather than a comparable displacement.
    DIVERGENCE_GROWTH = {"pitch": 0.10, "roll": 0.05}
    _SATURATION_FRAC = 0.9      # "at the clamp" means this fraction of it

    def diverging(self) -> str:
        """A one-line warning if this loop looks like it is making things worse.

        Empty string when healthy. Checked by the controller, not by this class,
        so the message can be logged with the rest of the controller state.
        """
        return self._divergence

    def note_tilt(self, now_s: float, roll: float, pitch: float) -> None:
        """Feed the measured tilt in so divergence can be detected.

        Separate from compute_correction because the controller knows the time and
        this class deliberately does not keep a clock.
        """
        for axis, value, key in (("pitch", pitch, "pitch"), ("roll", roll, "roll")):
            window = self.DIVERGENCE_S[axis]
            history = self._tilt_history.setdefault(axis, [])
            history.append((now_s, abs(value)))
            while history and (now_s - history[0][0]) > window:
                history.pop(0)
            if len(history) < 3 or (history[-1][0] - history[0][0]) < window * 0.8:
                continue
            growth = history[-1][1] - history[0][1]
            clamp = (self.params.max_pitch_corr if axis == "pitch"
                     else self.params.max_roll_corr)
            saturated = abs(self._state[key]) >= self._SATURATION_FRAC * clamp
            if growth > self.DIVERGENCE_GROWTH[axis] and saturated:
                self._divergence = (
                    f"balance is DIVERGING on {axis}: |{axis}| grew "
                    f"{growth:+.3f} rad over {window:.1f}s while the "
                    f"correction sat at its clamp ({self._state[key]:+.3f}). That is "
                    f"the signature of an inverted sign -- check "
                    f"BalanceController.{axis.upper()}_SIGN against which way the "
                    f"robot actually tips."
                )
                return
        self._divergence = ""

    def _expand_state(self) -> dict[str, float]:
        """The two parameters as per-joint deltas. Hip and ankle get equal and
        opposite values so the soles stay flat -- see :meth:`_apply_corr`."""
        c = self._state
        return {
            "LHipPitch": c["pitch"], "RHipPitch": c["pitch"],
            "LAnklePitch": -c["pitch"], "RAnklePitch": -c["pitch"],
            "LHipRoll": c["roll"], "RHipRoll": c["roll"],
            "LAnkleRoll": -c["roll"], "RAnkleRoll": -c["roll"],
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _toward(current: float, target: float, max_delta: float) -> float:
    """``current`` moved toward ``target`` by at most ``max_delta`` (>= 0)."""
    return current + _clamp(target - current, -max_delta, max_delta)
