import numpy as np

from .frames import BodyFrame
from .lift3d import Lifter3D
from .lower_body import LowerBodyReference
from .skeleton import A3_INDEX

ARM_JOINTS = ("shoulder", "elbow", "wrist")
MIN_SCORE = 0.35


class UpperBodyRetargeter:
    def __init__(self, min_score=MIN_SCORE, lifter=None, use_lifter=True):
        self.min_score = min_score
        self.body = BodyFrame()
        self.lifter = lifter if lifter is not None else (
            Lifter3D() if use_lifter else None)
        self.lower = LowerBodyReference()
        self.seq = 0

    def reset(self):
        self.body.reset()
        self.lower.reset()
        if self.lifter is not None:
            self.lifter.reset()
        self.seq = 0

    @property
    def lifter_ready(self):
        return self.lifter is not None and self.lifter.ready

    def _visible(self, scores, names):
        return all(scores[A3_INDEX[name]] >= self.min_score for name in names)

    def __call__(self, keypoints, scores, timestamp):
        self.seq += 1
        if keypoints is None:
            return None

        directions = {}
        scalars = {}
        confidences = []

        torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not self._visible(scores, torso_names):
            return None

        self.body.update_scale(keypoints, timestamp)
        yaw, yaw_ratio = self.body.torso_yaw(keypoints, scores)
        pitch, roll = self.body.torso_tilt(keypoints)
        scalars["torso_yaw"] = yaw
        scalars["torso_pitch"] = pitch
        scalars["torso_roll"] = roll
        confidences.append(float(np.mean([scores[A3_INDEX[n]] for n in torso_names])))

        head_names = ("nose", "left_ear", "right_ear")
        if self._visible(scores, head_names):
            head_yaw, head_pitch = self.body.head_angles(keypoints)
            scalars["head_yaw"] = head_yaw
            scalars["head_pitch"] = head_pitch
        else:
            scalars["head_yaw"] = 0.0
            scalars["head_pitch"] = 0.0

        for side, prefix in (("L", "left"), ("R", "right")):
            names = tuple(f"{prefix}_{joint}" for joint in ARM_JOINTS)
            if not self._visible(scores, names):
                continue
            upper, fore, ratio = self.body.arm_directions(keypoints, side)
            if upper is not None:
                directions[f"{prefix}_upper_arm"] = upper
            if fore is not None:
                directions[f"{prefix}_fore_arm"] = fore
            confidences.append(float(np.mean([scores[A3_INDEX[n]] for n in names])))

            hand_names = (f"{prefix}_index_mcp", f"{prefix}_pinky_mcp")
            if self._visible(scores, hand_names):
                normal = self.body.hand_normal(keypoints, side)
                if normal is not None:
                    directions[f"{prefix}_hand_normal"] = normal

        lower = None
        if self.lifter is not None:
            pose3d = self.lifter(keypoints, scores)
            if pose3d is not None and self.lifter.ready:
                lower = self.lower(pose3d)

        confidence = float(np.mean(confidences)) if confidences else 0.0
        return {
            "seq": self.seq,
            "t": timestamp,
            "directions": directions,
            "scalars": scalars,
            "lower": lower,
            "confidence": confidence,
            "valid": bool(directions),
        }
