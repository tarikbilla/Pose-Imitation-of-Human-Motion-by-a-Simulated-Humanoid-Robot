import numpy as np

A3_JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_index_mcp",
    "left_pinky_mcp",
    "right_index_mcp",
    "right_pinky_mcp",
    "neck",
    "hip_center",
)

A3_INDEX = {name: index for index, name in enumerate(A3_JOINTS)}
DERIVED = ("neck", "hip_center")

COCO_WHOLEBODY_FROM_A3 = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    "left_big_toe": 17, "left_small_toe": 18, "left_heel": 19,
    "right_big_toe": 20, "right_small_toe": 21, "right_heel": 22,
    "left_index_mcp": 91 + 5, "left_pinky_mcp": 91 + 17,
    "right_index_mcp": 112 + 5, "right_pinky_mcp": 112 + 17,
}

HALPE26_FROM_A3 = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    "left_big_toe": 20, "left_small_toe": 22, "left_heel": 24,
    "right_big_toe": 21, "right_small_toe": 23, "right_heel": 25,
}

SKELETON_LINKS = (
    ("neck", "nose"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index_mcp"),
    ("left_wrist", "left_pinky_mcp"),
    ("left_index_mcp", "left_pinky_mcp"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index_mcp"),
    ("right_wrist", "right_pinky_mcp"),
    ("right_index_mcp", "right_pinky_mcp"),
    ("neck", "hip_center"),
    ("hip_center", "left_hip"),
    ("hip_center", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
)

HAND_JOINTS = ("left_index_mcp", "left_pinky_mcp", "right_index_mcp", "right_pinky_mcp")
FOOT_JOINTS = ("left_big_toe", "left_small_toe", "left_heel",
               "right_big_toe", "right_small_toe", "right_heel")
CORE_JOINTS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle")


def schema_for(keypoint_count):
    if keypoint_count >= 133:
        return COCO_WHOLEBODY_FROM_A3
    if keypoint_count >= 26:
        return HALPE26_FROM_A3
    raise ValueError(f"unsupported keypoint count {keypoint_count}")


def to_a3(keypoints, scores):
    if keypoints is None:
        return None, None

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    mapping = schema_for(len(keypoints))

    out_kp = np.zeros((len(A3_JOINTS), 2), dtype=np.float32)
    out_sc = np.zeros(len(A3_JOINTS), dtype=np.float32)

    for name, source in mapping.items():
        target = A3_INDEX[name]
        out_kp[target] = keypoints[source]
        out_sc[target] = scores[source]

    left_shoulder = A3_INDEX["left_shoulder"]
    right_shoulder = A3_INDEX["right_shoulder"]
    left_hip = A3_INDEX["left_hip"]
    right_hip = A3_INDEX["right_hip"]

    neck = A3_INDEX["neck"]
    out_kp[neck] = (out_kp[left_shoulder] + out_kp[right_shoulder]) / 2.0
    out_sc[neck] = min(out_sc[left_shoulder], out_sc[right_shoulder])

    centre = A3_INDEX["hip_center"]
    out_kp[centre] = (out_kp[left_hip] + out_kp[right_hip]) / 2.0
    out_sc[centre] = min(out_sc[left_hip], out_sc[right_hip])

    return out_kp, out_sc


def has_hands(scores, threshold=0.3):
    if scores is None:
        return False
    return all(scores[A3_INDEX[name]] >= threshold for name in HAND_JOINTS)
