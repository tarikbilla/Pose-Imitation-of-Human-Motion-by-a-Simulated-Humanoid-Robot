"""Tests for the synthetic pose generator (src/perception/synthetic_pose.py).

The generator's whole value is that it is a KNOWN human: if it silently produces
an impossible body (limbs changing length, feet through the floor, landmarks
missing) then every downstream measurement taken against it is meaningless, and
worse, meaningless in a way that looks like a robot bug.
"""
from __future__ import annotations

import math

import pytest

from src.perception.gait_cues import GaitCueExtractor
from src.perception.landmarks import POSE_LANDMARKS
from src.perception.synthetic_pose import (
    FOREARM,
    GROUND_Y,
    SHANK,
    THIGH,
    UPPER_ARM,
    BodyState,
    build_pose,
)

# Every scripted motion the replay driver can emit, sampled across a full cycle.
MOTIONS = [
    BodyState(),
    BodyState(right_arm_side=math.radians(115), right_elbow=math.radians(60)),
    BodyState(left_arm_fwd=math.radians(90), right_arm_fwd=math.radians(90)),
    BodyState(crouch=0.5),
    BodyState(crouch=1.0),
    BodyState(left_hip_flex=math.radians(65), left_knee_flex=math.radians(85)),
    BodyState(right_hip_flex=math.radians(65), right_knee_flex=math.radians(85)),
    BodyState(body_yaw=math.radians(75)),
    BodyState(body_yaw=math.radians(-75)),
    BodyState(head_yaw=math.radians(35), head_pitch=math.radians(25)),
]


def _dist(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


@pytest.mark.parametrize("state", MOTIONS)
def test_every_landmark_is_present_and_finite(state) -> None:
    kps = build_pose(state, 0.0, 0).keypoints
    assert set(kps) == set(POSE_LANDMARKS)
    for name, kp in kps.items():
        assert all(math.isfinite(v) for v in (kp.x, kp.y, kp.z)), name
        assert kp.visibility == 1.0


@pytest.mark.parametrize("state", MOTIONS)
def test_limb_lengths_are_rigid(state) -> None:
    """A retargeter calibrates its own limb-length scale from the stream. If the
    synthetic subject's bones changed length between frames it would mis-scale
    everything downstream and the fault would look like a robot bug."""
    kps = build_pose(state, 0.0, 0).keypoints
    for side in ("left", "right"):
        assert _dist(kps[f"{side}_shoulder"], kps[f"{side}_elbow"]) == pytest.approx(
            UPPER_ARM, abs=1e-6)
        assert _dist(kps[f"{side}_elbow"], kps[f"{side}_wrist"]) == pytest.approx(
            FOREARM, abs=1e-6)
        assert _dist(kps[f"{side}_hip"], kps[f"{side}_knee"]) == pytest.approx(
            THIGH, abs=1e-6)
        assert _dist(kps[f"{side}_knee"], kps[f"{side}_ankle"]) == pytest.approx(
            SHANK, abs=1e-6)


def test_limb_lengths_are_rigid_across_a_whole_squat() -> None:
    """The squat is the one motion that places the knee by circle intersection
    rather than forward kinematics, so it is the one that could drift."""
    for i in range(41):
        kps = build_pose(BodyState(crouch=i / 40.0), 0.0, i).keypoints
        for side in ("left", "right"):
            assert _dist(kps[f"{side}_hip"], kps[f"{side}_knee"]) == pytest.approx(
                THIGH, abs=1e-6)
            assert _dist(kps[f"{side}_knee"], kps[f"{side}_ankle"]) == pytest.approx(
                SHANK, abs=1e-6)


def test_planted_feet_stay_on_the_ground_through_a_squat() -> None:
    """Both feet are planted in a squat; a foot that sinks through the floor
    would read to the controller as a step."""
    for i in range(21):
        kps = build_pose(BodyState(crouch=i / 20.0), 0.0, i).keypoints
        assert kps["left_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)
        assert kps["right_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)


def test_a_squat_actually_lowers_the_hips() -> None:
    tall = build_pose(BodyState(crouch=0.0), 0.0, 0).keypoints
    deep = build_pose(BodyState(crouch=1.0), 0.0, 1).keypoints
    # y is DOWN, so sinking means a LARGER y.
    assert deep["pelvis"].y > tall["pelvis"].y + 100.0
    # ...and the knees must travel forward (toward the camera, -z) to allow it.
    assert deep["left_knee"].z < tall["left_knee"].z - 50.0


def test_a_leg_lift_raises_that_ankle_only() -> None:
    kps = build_pose(
        BodyState(left_hip_flex=math.radians(65), left_knee_flex=math.radians(85)), 0.0, 0
    ).keypoints
    assert kps["left_ankle"].y < GROUND_Y - 50.0    # lifted (y down)
    assert kps["right_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)   # still planted


def test_body_yaw_preserves_every_segment_length() -> None:
    """Turning must rotate the subject, not stretch them."""
    square = build_pose(BodyState(), 0.0, 0).keypoints
    turned = build_pose(BodyState(body_yaw=math.radians(75)), 0.0, 1).keypoints
    for a, b in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
                 ("neck", "pelvis")):
        assert _dist(turned[a], turned[b]) == pytest.approx(_dist(square[a], square[b]), abs=1e-6)


def test_body_yaw_changes_the_depth_spread_of_the_shoulders() -> None:
    """This is the signal the torso-yaw estimator actually reads."""
    square = build_pose(BodyState(), 0.0, 0).keypoints
    turned = build_pose(BodyState(body_yaw=math.radians(60)), 0.0, 1).keypoints
    assert abs(square["left_shoulder"].z - square["right_shoulder"].z) < 1e-6
    assert abs(turned["left_shoulder"].z - turned["right_shoulder"].z) > 100.0


def test_generation_is_deterministic() -> None:
    """The entire point: the same input must give byte-identical output, or
    'deterministic replay' (PRD US-2) is a lie."""
    a = build_pose(BodyState(crouch=0.3, body_yaw=0.2), 1.5, 9).keypoints
    b = build_pose(BodyState(crouch=0.3, body_yaw=0.2), 1.5, 9).keypoints
    for name in a:
        assert (a[name].x, a[name].y, a[name].z) == (b[name].x, b[name].y, b[name].z)


def test_the_arms_hang_by_default() -> None:
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    for side in ("left", "right"):
        # Elbow directly below the shoulder, a full upper-arm length down.
        assert kps[f"{side}_elbow"].x == pytest.approx(kps[f"{side}_shoulder"].x, abs=1e-6)
        assert kps[f"{side}_elbow"].y == pytest.approx(
            kps[f"{side}_shoulder"].y + UPPER_ARM, abs=1e-6)


# ------------------------------------------------------- left/right convention
def _settled_yaw(state: BodyState, frames: int = 40) -> float:
    """Body yaw after the extractor's EMA has settled, in degrees."""
    gait = GaitCueExtractor()
    cue = None
    for i in range(frames):
        cue = gait.update(build_pose(state, i / 30.0, i))
    return math.degrees(cue.body_yaw_rad)


def test_a_front_facing_subject_has_their_left_on_the_positive_x_side() -> None:
    """When you face someone, your left is on their right. Landmark names are
    ANATOMICAL, so a subject facing the camera has left_shoulder at positive x."""
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    assert kps["left_shoulder"].x > kps["right_shoulder"].x
    assert kps["left_hip"].x > kps["right_hip"].x
    assert kps["left_ear"].x > kps["right_ear"].x


def test_a_subject_facing_the_camera_reads_as_zero_yaw() -> None:
    """The check src/perception/gait_cues.py's docstring asks a human to perform
    ("stand facing the camera, confirm yaw reads ~0") and which had never been
    automated. Get the side convention backwards and someone standing perfectly
    still reads as turned 180 degrees, so the controller demands turn clips
    forever and never walks -- and nothing in the suite would have noticed."""
    assert abs(_settled_yaw(BodyState())) < 5.0


def test_turning_is_reported_with_the_right_sign_and_size() -> None:
    left = _settled_yaw(BodyState(body_yaw=math.radians(40.0)))
    right = _settled_yaw(BodyState(body_yaw=math.radians(-40.0)))
    assert left * right < 0, "turning each way must give opposite signs"
    assert 15.0 < abs(left) < 75.0, f"turn magnitude implausible: {left:.1f} deg"
    assert 15.0 < abs(right) < 75.0, f"turn magnitude implausible: {right:.1f} deg"


def test_the_side_convention_survives_a_mirrored_capture() -> None:
    """input.flip_horizontal mirrors the image before inference. That moves a
    shoulder to the other side AND makes the estimator label it as the other
    shoulder, so the two swaps cancel and the sign is unchanged. This pins that
    reasoning, because the alternative -- that the flip inverts the convention --
    is a very natural thing to conclude and would send someone inverting a
    correct sign in shipping robot code."""
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    mirrored = {}
    for name, kp in kps.items():
        if name.startswith("left_"):
            other = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            other = "left_" + name[len("right_"):]
        else:
            other = name
        mirrored[other] = (-kp.x, kp.y, kp.z)
    assert (mirrored["left_shoulder"][0] - mirrored["right_shoulder"][0]) == pytest.approx(
        kps["left_shoulder"].x - kps["right_shoulder"].x, abs=1e-9)


# ---------------------------------------------------------------------------
# Hands. These cross-check two INDEPENDENT constructions: this generator builds
# the hand frame from cross products against a world axis, while nao_retarget
# inverts NAO's own Ry.Rz.Rx.Rz.Rx arm chain. Neither knows about the other, so
# agreement between them is evidence about the convention and not just about the
# algebra -- which is what the round-trip test in test_nao_retarget.py, being
# the same maths forwards and backwards, cannot give.
# ---------------------------------------------------------------------------
# The elbows are BENT on purpose. With a straight arm the forearm lies on the
# ElbowYaw axis, so that joint is unobservable and WristYaw -- which is solved
# in the forearm frame -- goes with it. That is the documented behaviour and it
# is asserted separately in test_nao_retarget.py; here it would just mean the
# wrist tests silently measured nothing.
HAND_STATES = [
    BodyState(left_wrist_roll=r, right_wrist_roll=r,
              left_elbow=math.radians(70), right_elbow=math.radians(70),
              left_arm_fwd=math.radians(35), right_arm_fwd=math.radians(35))
    for r in (0.0, 0.3, 0.6, 0.9, 1.2)
]


def _retarget(state):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))
    from nao_retarget import retarget_upper_body

    kps = build_pose(state, 0.0, 0).keypoints
    return retarget_upper_body(
        {n: [kp.x, kp.y, kp.z, kp.visibility] for n, kp in kps.items()}
    )


def test_the_hand_keeps_its_size_however_the_wrist_turns() -> None:
    """Turning a wrist must not stretch the hand. The generator builds the
    thumb from a rotated frame, and a frame that was not orthonormal would show
    up here as a hand that grows as it rotates."""
    spans = []
    for state in HAND_STATES:
        kps = build_pose(state, 0.0, 0).keypoints
        spans.append(_dist(kps["left_hand_root"], kps["left_thumb"]))
    assert max(spans) - min(spans) < 1e-6, spans


def test_turning_the_wrist_moves_naos_wrist_and_nothing_else() -> None:
    """The defining property of a roll joint, and the reason it needed hand
    landmarks: rotating the palm about the forearm leaves every other joint --
    shoulder, elbow, the positions of every body landmark -- exactly where it
    was. If ShoulderPitch or ElbowRoll move here, the solve is reading the twist
    out of something that does not carry it."""
    baseline = _retarget(HAND_STATES[0])
    for state in HAND_STATES[1:]:
        t = _retarget(state)
        for joint in ("LShoulderPitch", "LShoulderRoll", "LElbowRoll",
                      "RShoulderPitch", "RShoulderRoll", "RElbowRoll"):
            assert t[joint] == pytest.approx(baseline[joint], abs=1e-6), joint


def _wrist_sweep(elbow_deg):
    states = [
        BodyState(left_wrist_roll=r, left_elbow=math.radians(elbow_deg),
                  left_arm_fwd=math.radians(35))
        for r in (0.0, 0.3, 0.6, 0.9, 1.2)
    ]
    angles = [_retarget(s).get("LWristYaw") for s in states]
    assert all(a is not None for a in angles), angles
    limit = math.radians(104.5)
    assert all(abs(a) < limit - 1e-3 for a in angles), (
        f"a clipped joint proves nothing about tracking: {angles}")
    return [a - s.left_wrist_roll for a, s in zip(angles, states, strict=True)]


def test_the_wrist_angle_tracks_the_palm_exactly() -> None:
    """A palm turning by X turns NAO's wrist by X -- and from the same zero.

    The offset being ~0 rather than merely CONSTANT is the interesting part, and
    it is not something either side was fitted to: this generator defines
    ``wrist_roll = 0`` as "thumb toward the upper arm", while nao_retarget
    inverts NAO's Ry.Rz.Rx.Rz.Rx chain and never sees the upper arm at all. They
    agree because both describe the same geometry.

    It is NOT a check on the real robot's palm. Which way NAO's palm faces at
    ``WristYaw = 0`` is set by Nao.proto, fetched over the network at world-load
    time and not readable here; ``nao_retarget.HAND_YAW_SIGN`` is the single
    constant that corrects it if the wrists come out half a turn round.

    Run below the elbow's compression knee so the arm the solve inverts is the
    arm the human is actually holding -- see the next test for what happens
    above it, which is a property of the compression and not of this solve.
    """
    offsets = _wrist_sweep(55)
    assert max(offsets) - min(offsets) < 1e-6, offsets
    assert abs(offsets[0]) < 1e-6, offsets


def test_a_compressed_elbow_costs_the_wrist_a_little_and_only_a_little() -> None:
    """Above ELBOW_LINEAR_RAD the robot's forearm is deliberately NOT where the
    human's is -- NAO's elbow stops 60 deg short, so the bend is compressed. The
    wrist is then solved against the robot's forearm rather than the human's,
    which is the right choice (it puts the palm where the human's is RELATIVE TO
    THE ARM, which is what an onlooker reads) but means the angle cannot also
    match in absolute terms. Measured: 0 mrad at the knee, 3.5 at 70 deg of
    bend, 32 at 90. Bounded and far below what MeTRAbs' own noise contributes --
    pinned here so a future change to the compression cannot quietly turn a
    third of a degree into thirty.
    """
    for bend, ceiling in ((60, 1e-6), (70, 0.006), (90, 0.05)):
        offsets = _wrist_sweep(bend)
        drift = max(offsets) - min(offsets)
        assert drift < ceiling, f"{bend} deg of bend drifted {drift*1000:.1f} mrad"


def test_the_grip_reads_a_closing_hand() -> None:
    grips = [_retarget(BodyState(left_grip=g, right_grip=g,
                                 left_elbow=math.radians(70))).get("LHand")
             for g in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(g is not None for g in grips), grips
    for earlier, later in zip(grips, grips[1:], strict=False):
        assert later > earlier - 1e-9, grips
    assert grips[0] < 0.2 and grips[-1] > 0.8, grips


def test_the_thumb_never_teleports_while_the_elbow_bends() -> None:
    """A synthetic body that jumps is worse than no synthetic body: every
    measurement taken against it reads the jump as a robot fault.

    The thumb's reference direction has to come from somewhere, and every
    candidate has a pole. Two world-axis choices were tried and both put theirs
    inside ordinary motion -- one flipped when a bending elbow crossed
    |y| = 0.9, the other at 55 degrees of a plain forward reach, each a 156 mm
    jump between adjacent one-degree steps with the wrist held still. Referencing
    the upper arm instead moves the pole onto a perfectly straight elbow, which
    is the one place ``nao_retarget`` already refuses to solve the roll joints
    (ELBOW_YAW_MIN_BEND), so nothing downstream can ever see it.
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))
    from nao_retarget import ELBOW_YAW_MIN_BEND

    floor = math.degrees(ELBOW_YAW_MIN_BEND)
    for arm_fwd in (0, 35, 70, 90):
        previous = None
        for step in range(int((145 - floor) * 4)):
            bend = floor + step * 0.25
            state = BodyState(left_wrist_roll=0.4,
                              left_elbow=math.radians(bend),
                              left_arm_fwd=math.radians(arm_fwd))
            kps = build_pose(state, 0.0, 0).keypoints
            wrist, thumb = kps["left_hand_root"], kps["left_thumb"]
            current = (thumb.x - wrist.x, thumb.y - wrist.y, thumb.z - wrist.z)
            if previous is not None:
                jump = math.dist(current, previous)
                assert jump < 5.0, (
                    f"thumb moved {jump:.1f} mm in a 0.25 deg elbow step at "
                    f"bend {bend:.2f}, arm_fwd {arm_fwd}")
            previous = current
