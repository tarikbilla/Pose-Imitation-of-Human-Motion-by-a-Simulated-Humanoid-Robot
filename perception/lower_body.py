import numpy as np

from .lift3d import H36M

MIN_LEG = 1e-6


def _leg_length(pose):
    thigh_l = np.linalg.norm(pose[H36M["left_knee"]] - pose[H36M["left_hip"]])
    shank_l = np.linalg.norm(pose[H36M["left_ankle"]] - pose[H36M["left_knee"]])
    thigh_r = np.linalg.norm(pose[H36M["right_knee"]] - pose[H36M["right_hip"]])
    shank_r = np.linalg.norm(pose[H36M["right_ankle"]] - pose[H36M["right_knee"]])
    return float((thigh_l + shank_l + thigh_r + shank_r) / 2.0)


class LowerBodyReference:
    """Task-space lower-body references from the lifted 3-D pose.

    Everything is normalised to the subject's leg length, so the values
    transfer to a robot with different proportions. The vertical axis of the
    lifter is image-down, hence the sign flips for heights.
    """

    def __init__(self, decay=0.999):
        self.decay = decay
        self._stand_height = None
        self._ground = None

    def reset(self):
        self._stand_height = None
        self._ground = None

    def __call__(self, pose):
        if pose is None:
            return None

        leg = _leg_length(pose)
        if leg < MIN_LEG:
            return None

        hip = pose[H36M["hip"]]
        left_ankle = pose[H36M["left_ankle"]]
        right_ankle = pose[H36M["right_ankle"]]

        # lifter y grows downwards; height is measured upwards from the feet
        foot_level = max(left_ankle[1], right_ankle[1])
        if self._ground is None:
            self._ground = foot_level
        else:
            self._ground = max(self._ground * self.decay, foot_level)

        hip_height = float((self._ground - hip[1]) / leg)
        if self._stand_height is None:
            self._stand_height = hip_height
        else:
            self._stand_height = max(self._stand_height * self.decay, hip_height)

        squat = float(np.clip(self._stand_height - hip_height, -0.6, 0.6))

        lateral = np.array([1.0, 0.0, 0.0])
        stance = float(abs(np.dot(left_ankle - right_ankle, lateral)) / leg)

        left_lift = float(np.clip((self._ground - left_ankle[1]) / leg, -0.1, 0.8))
        right_lift = float(np.clip((self._ground - right_ankle[1]) / leg, -0.1, 0.8))

        mid_foot = (left_ankle + right_ankle) / 2.0
        com_x = float((hip[0] - mid_foot[0]) / leg)
        com_y = float((hip[2] - mid_foot[2]) / leg)

        return {
            "hip_height": squat,
            "stance_width": stance,
            "left_foot_lift": left_lift,
            "right_foot_lift": right_lift,
            "com_offset_x": float(np.clip(com_y, -0.8, 0.8)),
            "com_offset_y": float(np.clip(com_x, -0.8, 0.8)),
        }
