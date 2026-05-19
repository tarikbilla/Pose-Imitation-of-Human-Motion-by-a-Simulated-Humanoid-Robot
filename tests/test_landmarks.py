from __future__ import annotations

from src.perception.landmarks import MEDIAPIPE_POSE_LANDMARKS, NUM_LANDMARKS, landmark_id


def test_landmark_count_is_33() -> None:
    assert NUM_LANDMARKS == 33
    assert len(MEDIAPIPE_POSE_LANDMARKS) == 33


def test_landmark_id_round_trip() -> None:
    for idx, name in enumerate(MEDIAPIPE_POSE_LANDMARKS):
        assert landmark_id(name) == idx


def test_key_anatomy_present() -> None:
    expected = {
        "nose", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    }
    assert expected.issubset(set(MEDIAPIPE_POSE_LANDMARKS))
