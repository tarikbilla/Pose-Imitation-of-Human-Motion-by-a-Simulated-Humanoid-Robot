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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
    offset: Tuple[float, float, float]
    axis: Tuple[float, float, float]


@dataclass(frozen=True)
class Link:
    """A point mass at ``local_com`` (m) in frame ``frame`` weighing ``mass`` kg."""
    frame: str
    mass: float
    local_com: Tuple[float, float, float]


def _hip_axis(sign: float) -> Tuple[float, float, float]:
    # 45-degree HipYawPitch axis (matches Nao.urdf): left (0, .707, -.707).
    return (0.0, 0.707107, -0.707107 * sign)


def _build_model() -> Tuple[Dict[str, Joint], List[Link]]:
    joints: List[Joint] = [
        # Head
        Joint("HeadYaw", "Torso", (0.0, 0.0, NECK_OFFSET_Z), (0, 0, 1)),
        Joint("HeadPitch", "HeadYaw", (0.0, 0.0, 0.0), (0, 1, 0)),
    ]
    links: List[Link] = [
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
        self.total_mass = sum(l.mass for l in self.links)

    def frames(self, angles: Dict[str, float]) -> Dict[str, np.ndarray]:
        """4x4 transforms of every frame in the Torso frame for ``angles`` (rad)."""
        out: Dict[str, np.ndarray] = {"Torso": np.eye(4)}
        # _JOINTS is built parent-before-child, so a single pass resolves the tree.
        for name, j in self.joints.items():
            parent = out.get(j.parent)
            if parent is None:  # parent not yet placed; shouldn't happen with our order
                continue
            local = _T(*j.offset) @ _rot(j.axis, float(angles.get(name, 0.0)))
            out[name] = parent @ local
        return out

    def com(self, angles: Dict[str, float],
            frames: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Whole-body CoM (x, y, z) in the Torso frame."""
        frames = frames if frames is not None else self.frames(angles)
        acc = np.zeros(3)
        for link in self.links:
            T = frames[link.frame]
            acc += link.mass * (T[:3, :3] @ np.asarray(link.local_com) + T[:3, 3])
        return acc / self.total_mass

    def foot_sole_center(self, side: str,
                         frames: Dict[str, np.ndarray]) -> np.ndarray:
        """Centre of a foot's sole (x, y, z) in the Torso frame."""
        T = frames[f"{side}AnkleRoll"]
        sole_local = np.array([0.5 * (FOOT_X_BACK + FOOT_X_FRONT), 0.0, -FOOT_HEIGHT])
        return T[:3, :3] @ sole_local + T[:3, 3]

    def stance_margin(self, angles: Dict[str, float], side: str,
                      frames: Optional[Dict[str, np.ndarray]] = None) -> float:
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


# ---------------------------------------------------------------------------
# Fibonacci / golden-angle spiral sampling
# ---------------------------------------------------------------------------
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996 rad


def fibonacci_spiral(n: int, scale: float) -> List[Tuple[float, float]]:
    """``n`` points on a sunflower/Fibonacci spiral, radius growing as sqrt(k).

    Even, low-discrepancy coverage of a disc of radius ``scale`` — ideal for
    searching a 2D correction space outward from the centre (the last stable
    correction) without re-sampling the same direction.
    """
    pts: List[Tuple[float, float]] = []
    for k in range(n):
        r = scale * math.sqrt((k + 0.5) / n)
        theta = (k + 1) * GOLDEN_ANGLE
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


@dataclass
class BalanceParams:
    """Tuning for :class:`BalanceController`."""
    safe_margin: float = 0.020      # m; CoM within this of support centre => no action
    max_ankle_corr: float = 0.30    # rad; clamp on |ankle correction|
    max_hip_corr: float = 0.18      # rad; clamp on |hip correction|
    search_points: int = 24         # Fibonacci samples per cycle
    search_scale: float = 0.18      # rad; spiral radius in the correction space
    slew: float = 0.25              # per-cycle blend toward the chosen correction (0..1)
    tilt_weight: float = 0.6        # how much measured torso tilt biases the target


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
    _state: Dict[str, float] = field(default_factory=lambda: {
        "ank_pitch": 0.0, "ank_roll": 0.0, "hip_pitch": 0.0, "hip_roll": 0.0,
    })

    CORRECTED_JOINTS = (
        "LAnklePitch", "RAnklePitch", "LAnkleRoll", "RAnkleRoll",
        "LHipPitch", "RHipPitch", "LHipRoll", "RHipRoll",
    )

    def _apply_corr(self, angles: Dict[str, float], c: Dict[str, float]) -> Dict[str, float]:
        """A copy of ``angles`` with a correction vector added symmetrically."""
        out = dict(angles)
        for j in ("LAnklePitch", "RAnklePitch"):
            out[j] = angles.get(j, 0.0) + c["ank_pitch"]
        for j in ("LHipPitch", "RHipPitch"):
            out[j] = angles.get(j, 0.0) + c["hip_pitch"]
        # Roll corrections lean the whole body the same world direction (both
        # legs same sign), so the robot shifts weight without splaying.
        out["LAnkleRoll"] = angles.get("LAnkleRoll", 0.0) + c["ank_roll"]
        out["RAnkleRoll"] = angles.get("RAnkleRoll", 0.0) + c["ank_roll"]
        out["LHipRoll"] = angles.get("LHipRoll", 0.0) + c["hip_roll"]
        out["RHipRoll"] = angles.get("RHipRoll", 0.0) + c["hip_roll"]
        return out

    # InertialUnit sign conventions: forward tilt -> +PITCH_SIGN*pitch adds
    # fore tipping; right tilt -> +ROLL_SIGN*roll adds lateral tipping. If the
    # robot fights the wrong way in your build, flip one of these.
    PITCH_SIGN = 1.0
    ROLL_SIGN = 1.0
    NOMINAL_COM_HEIGHT = 0.30  # m; lever arm used to convert tilt to a ground shift

    def _imbalance(self, angles: Dict[str, float],
                   torso_rp: Tuple[float, float]) -> Tuple[float, np.ndarray]:
        """Return (cost, com_xy_error). Lower cost = better balance.

        Primary term: the model CoM's horizontal offset from the support centre
        (sign-certain — it comes straight from the kinematics). Secondary,
        *bounded* term: the measured torso tilt times a nominal lever, scaled by
        ``tilt_weight`` — this is the InertialUnit supplying the gravity/vertical
        information the camera lost, without being allowed to dominate (so a
        mis-signed IMU can't drive the loop the wrong way).
        """
        frames = self.model.frames(angles)
        com = self.model.com(angles, frames)
        support = 0.5 * (self.model.foot_sole_center("L", frames)
                         + self.model.foot_sole_center("R", frames))
        d = (com - support)[:2]  # (fore/aft, lateral) model offset over the feet

        roll, pitch = torso_rp
        k = self.params.tilt_weight * self.NOMINAL_COM_HEIGHT
        err = np.array([
            d[0] + k * self.PITCH_SIGN * pitch,   # forward tilt -> fore tipping
            d[1] - k * self.ROLL_SIGN * roll,     # right tilt   -> lateral tipping
        ])
        cost = float(np.hypot(err[0], err[1]))
        return cost, err

    def compute_correction(self, angles: Dict[str, float],
                           torso_rp: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, float]:
        """Joint deltas (rad) keeping the CoM over the feet; {} if already safe."""
        base_cost, _ = self._imbalance(angles, torso_rp)
        if base_cost <= self.params.safe_margin:
            # Balanced: relax any standing correction back toward zero.
            for k in self._state:
                self._state[k] *= (1.0 - self.params.slew)
            return self._expand_state()

        # Search corrections on a Fibonacci spiral around the last one.
        c0 = self._state
        best = dict(c0)
        best_cost = self._imbalance(self._apply_corr(angles, c0), torso_rp)[0]
        # Spiral in ankle space; hips follow at a fraction (ankle strategy first).
        for dx, dy in fibonacci_spiral(self.params.search_points, self.params.search_scale):
            cand = {
                "ank_pitch": _clamp(c0["ank_pitch"] + dx, -self.params.max_ankle_corr, self.params.max_ankle_corr),
                "ank_roll":  _clamp(c0["ank_roll"] + dy, -self.params.max_ankle_corr, self.params.max_ankle_corr),
                "hip_pitch": _clamp(c0["hip_pitch"] + 0.4 * dx, -self.params.max_hip_corr, self.params.max_hip_corr),
                "hip_roll":  _clamp(c0["hip_roll"] + 0.4 * dy, -self.params.max_hip_corr, self.params.max_hip_corr),
            }
            cost = self._imbalance(self._apply_corr(angles, cand), torso_rp)[0]
            if cost < best_cost:
                best_cost, best = cost, cand

        # Slew toward the best candidate so corrections ease in (no jolt).
        s = self.params.slew
        for k in self._state:
            self._state[k] += s * (best[k] - self._state[k])
        return self._expand_state()

    def _expand_state(self) -> Dict[str, float]:
        c = self._state
        return {
            "LAnklePitch": c["ank_pitch"], "RAnklePitch": c["ank_pitch"],
            "LAnkleRoll": c["ank_roll"], "RAnkleRoll": c["ank_roll"],
            "LHipPitch": c["hip_pitch"], "RHipPitch": c["hip_pitch"],
            "LHipRoll": c["hip_roll"], "RHipRoll": c["hip_roll"],
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
