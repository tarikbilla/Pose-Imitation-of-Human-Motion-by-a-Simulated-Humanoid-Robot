from __future__ import annotations

import math

from src.retargeting.mapper import RetargetingMapper, default_joint_limits
from src.types import Keypoint, PoseFrame


def test_mapper_outputs_expected_joints() -> None:
    pose = PoseFrame(
        timestamp_s=0.0,
        frame_index=0,
        keypoints={
            "left_shoulder": Keypoint(0.4, 0.4),
            "right_shoulder": Keypoint(0.6, 0.4),
            "left_elbow": Keypoint(0.35, 0.5),
            "right_elbow": Keypoint(0.65, 0.5),
            "left_wrist": Keypoint(0.3, 0.6),
            "right_wrist": Keypoint(0.7, 0.6),
            "left_hip": Keypoint(0.45, 0.6),
            "right_hip": Keypoint(0.55, 0.6),
            "left_knee": Keypoint(0.45, 0.8),
            "right_knee": Keypoint(0.55, 0.8),
        },
    )

    mapper = RetargetingMapper(default_joint_limits())
    out = mapper.map_pose(pose)

    assert set(out.joint_angles_rad.keys()) == {
        "LShoulderPitch",
        "RShoulderPitch",
        "LElbowRoll",
        "RElbowRoll",
        "LHipPitch",
        "RHipPitch",
        "TorsoPitch",
    }


def test_mapper_clips_joint_limits() -> None:
    pose = PoseFrame(
        timestamp_s=0.0,
        frame_index=1,
        keypoints={
            "left_shoulder": Keypoint(0.5, 0.5),
            "right_shoulder": Keypoint(0.5, 0.5),
            "left_elbow": Keypoint(0.5, 0.1),
            "right_elbow": Keypoint(0.5, 0.1),
            "left_wrist": Keypoint(0.5, -0.4),
            "right_wrist": Keypoint(0.5, -0.4),
            "left_hip": Keypoint(0.5, 0.8),
            "right_hip": Keypoint(0.5, 0.8),
            "left_knee": Keypoint(0.5, 1.5),
            "right_knee": Keypoint(0.5, 1.5),
        },
    )

    mapper = RetargetingMapper(default_joint_limits())
    out = mapper.map_pose(pose)
    limits = default_joint_limits()

    for joint, angle in out.joint_angles_rad.items():
        assert limits[joint].min_rad <= angle <= limits[joint].max_rad

    assert math.isfinite(out.joint_angles_rad["TorsoPitch"])
