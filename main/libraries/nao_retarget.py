"""
Full-body retargeting: MediaPipe landmarks -> NAO joint angles.

The Python pipeline streams raw MediaPipe Pose landmarks (normalized image
coordinates: ``x`` right [0,1], ``y`` down [0,1], ``z`` depth, ``visibility``
[0,1]). This module converts them into NAO joint targets for the whole upper
body (and, optionally, the legs).

Why retarget here instead of upstream
-------------------------------------
Driving the robot's *full* pose needs the actual limb geometry, not just a
handful of pre-baked angles. Keeping the kinematics next to the robot means the
controller owns everything NAO-specific (joint axes, signs, limits) and the
Python side stays a generic pose source.

Shoulder model (the important bit)
----------------------------------
NAO's shoulder is 2-DOF: ``ShoulderPitch`` (raise the arm up/down in the
sagittal plane) and ``ShoulderRoll`` (abduct the arm sideways). From a frontal
camera the upper-arm direction projects onto the image plane as a vector with a
*vertical* part (up/down) and a *lateral* part (sideways). We decompose that
single observed direction into the two joints:

    lateral_unit = sideways component   ->  ShoulderRoll  = asin(lateral_unit)
    vertical_unit = up component        ->  ShoulderPitch = -asin(vertical_unit)

so arm-down -> pitch +90°, arm-up -> pitch -90°, arm-straight-out -> roll ±90°,
and diagonals split cleanly between the two. This needs no depth (which is
unreliable monocular), so it is robust for the standing-in-front-of-camera case.

Everything is gated on landmark ``visibility`` so out-of-frame joints (often the
legs) are simply not commanded — the driver then holds their last pose.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from pose_control_utils import JointLimiter, get_default_motor_configs

# Visibility below which a landmark is considered unreliable and its dependent
# joints are skipped.
VIS_THRESHOLD = 0.5

# Tuning gains (kept gentle; joint limits clamp the rest).
ROLL_GAIN = 1.0
PITCH_GAIN = 1.0
HEAD_YAW_GAIN = 2.2
HEAD_PITCH_GAIN = 1.6
HEAD_PITCH_BASELINE = 0.9  # nose sits ~0.9 shoulder-widths above shoulder line

Vec = Tuple[float, float, float]
Landmark = Tuple[float, float, float, float]  # x, y, z, visibility


# ---------------------------------------------------------------------------
# Small vector helpers (image coords: x right, y down, z depth)
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _asin(v: float) -> float:
    return math.asin(_clamp(v, -1.0, 1.0))


def _sub(a: Landmark, b: Landmark) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vec) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-9


def _angle_between(a: Vec, b: Vec) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.acos(_clamp(dot / (_norm(a) * _norm(b)), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Landmark access
# ---------------------------------------------------------------------------
def _parse(keypoints: Dict[str, Sequence[float]]) -> Dict[str, Landmark]:
    """Normalize incoming landmark values to (x, y, z, visibility) tuples."""
    out: Dict[str, Landmark] = {}
    for name, v in keypoints.items():
        try:
            x = float(v[0]); y = float(v[1])
            z = float(v[2]) if len(v) > 2 else 0.0
            vis = float(v[3]) if len(v) > 3 else 1.0
        except (TypeError, IndexError, ValueError):
            continue
        out[name] = (x, y, z, vis)
    return out


def _visible(kps: Dict[str, Landmark], *names: str, thr: float = VIS_THRESHOLD) -> bool:
    return all(n in kps and kps[n][3] >= thr for n in names)


# ---------------------------------------------------------------------------
# Per-segment retargeting
# ---------------------------------------------------------------------------
def _arm(kps: Dict[str, Landmark], side: str, mid_x: float) -> Dict[str, float]:
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "shoulder", pre + "elbow"):
        return {}

    s = kps[pre + "shoulder"]
    e = kps[pre + "elbow"]
    dx = e[0] - s[0]
    dy = e[1] - s[1]
    length = math.hypot(dx, dy) + 1e-9

    vertical_up = -dy / length                       # +1 elbow above shoulder
    out_dir = 1.0 if (s[0] - mid_x) >= 0.0 else -1.0  # image side -> "outward"
    lateral_out = (dx * out_dir) / length             # +1 arm abducted outward

    pitch = -_asin(vertical_up) * PITCH_GAIN          # +down, -up
    roll_mag = _asin(lateral_out) * ROLL_GAIN         # >=0 outward, <0 across body

    out: Dict[str, float] = {}
    if side == "L":
        out["LShoulderPitch"] = pitch
        out["LShoulderRoll"] = +roll_mag             # NAO L: positive = outward
    else:
        out["RShoulderPitch"] = pitch
        out["RShoulderRoll"] = -roll_mag             # NAO R: negative = outward

    # Elbow flexion: angle between upper arm and forearm (0 = straight).
    if _visible(kps, pre + "wrist"):
        w = kps[pre + "wrist"]
        bend = _angle_between(_sub(e, s), _sub(w, e))
        if side == "L":
            out["LElbowRoll"] = -bend                # NAO L elbow bends negative
        else:
            out["RElbowRoll"] = +bend                # NAO R elbow bends positive
    return out


def _head(kps: Dict[str, Landmark]) -> Dict[str, float]:
    if not _visible(kps, "nose", "left_shoulder", "right_shoulder"):
        return {}
    nose = kps["nose"]
    ls = kps["left_shoulder"]
    rs = kps["right_shoulder"]
    mid_x = (ls[0] + rs[0]) / 2.0
    mid_y = (ls[1] + rs[1]) / 2.0
    shoulder_w = abs(ls[0] - rs[0]) + 1e-6

    # Yaw: nose horizontal offset from the shoulder midline.
    yaw = ((nose[0] - mid_x) / shoulder_w) * HEAD_YAW_GAIN
    # Pitch: nose vertical offset relative to its typical above-shoulder height.
    # Looking down brings the nose lower (toward the shoulders) -> positive pitch.
    pitch_raw = (nose[1] - mid_y) / shoulder_w        # negative when nose is high
    pitch = (pitch_raw + HEAD_PITCH_BASELINE) * HEAD_PITCH_GAIN
    return {"HeadYaw": yaw, "HeadPitch": pitch}


def _leg(kps: Dict[str, Landmark], side: str) -> Dict[str, float]:
    """Experimental lower-body mapping (gated; NAO will need balance support)."""
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "hip", pre + "knee", pre + "ankle"):
        return {}
    h = kps[pre + "hip"]
    k = kps[pre + "knee"]
    a = kps[pre + "ankle"]

    thigh = _sub(k, h)
    shank = _sub(a, k)
    # Hip flexion: thigh deviation from straight-down (image down = +y).
    thigh_len = math.hypot(thigh[0], thigh[1]) + 1e-9
    flex = _asin(_clamp(-thigh[0] * 0.0 + thigh[1] / thigh_len, -1, 1))  # ~0 when vertical
    hip_pitch = -(math.pi / 2.0 - flex)               # negative = flexed forward
    # Knee bend: angle between thigh and shank (0 straight).
    knee_bend = _angle_between(thigh, shank)
    out = {
        f"{side}HipPitch": hip_pitch,
        f"{side}KneePitch": knee_bend,
        f"{side}AnklePitch": -_clamp(knee_bend * 0.5, 0.0, 0.9),  # keep foot flat-ish
    }
    return out


def _swap_sides(targets: Dict[str, float]) -> Dict[str, float]:
    """Swap L<->R joints for a mirror-image mapping."""
    swapped: Dict[str, float] = {}
    for name, value in targets.items():
        if name.startswith("L"):
            swapped["R" + name[1:]] = value
        elif name.startswith("R"):
            swapped["L" + name[1:]] = value
        else:
            swapped[name] = value
    return swapped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def retarget_full_body(
    keypoints: Dict[str, Sequence[float]],
    *,
    drive_legs: bool = False,
    drive_head: bool = True,
    swap_sides: bool = False,
    limiter: Optional[JointLimiter] = None,
) -> Dict[str, float]:
    """Map MediaPipe landmarks to clamped NAO joint targets (radians).

    Only joints whose source landmarks are visible are returned; everything
    else is omitted so the caller can hold the previous pose.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    kps = _parse(keypoints)

    targets: Dict[str, float] = {}
    if _visible(kps, "left_shoulder", "right_shoulder"):
        mid_x = (kps["left_shoulder"][0] + kps["right_shoulder"][0]) / 2.0
        targets.update(_arm(kps, "L", mid_x))
        targets.update(_arm(kps, "R", mid_x))
    if drive_head:
        targets.update(_head(kps))
    if drive_legs:
        targets.update(_leg(kps, "L"))
        targets.update(_leg(kps, "R"))

    if swap_sides:
        targets = _swap_sides(targets)

    return {name: limiter.clamp_angle(name, value) for name, value in targets.items()}


def retargetable_joints(drive_legs: bool = False, drive_head: bool = True) -> List[str]:
    """The set of NAO joints this module can drive (for logging headers)."""
    joints = [
        "LShoulderPitch", "RShoulderPitch",
        "LShoulderRoll", "RShoulderRoll",
        "LElbowRoll", "RElbowRoll",
    ]
    if drive_head:
        joints += ["HeadYaw", "HeadPitch"]
    if drive_legs:
        for side in ("L", "R"):
            joints += [f"{side}HipPitch", f"{side}KneePitch", f"{side}AnklePitch"]
    return joints
