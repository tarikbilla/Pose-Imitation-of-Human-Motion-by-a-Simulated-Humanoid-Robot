import math
import os

import numpy as np
import yaml

MODEL_YAML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "atlas_model.yaml")
)

CHAIN = {
    "BackLbz": None, "BackMby": "BackLbz", "BackUbx": "BackMby",
    "LArmUsy": "BackUbx", "LArmShx": "LArmUsy", "LArmEly": "LArmShx",
    "LArmElx": "LArmEly", "LArmUwy": "LArmElx", "LArmMwx": "LArmUwy",
    "RArmUsy": "BackUbx", "RArmShx": "RArmUsy", "RArmEly": "RArmShx",
    "RArmElx": "RArmEly", "RArmUwy": "RArmElx", "RArmMwx": "RArmUwy",
    "NeckAy": "BackUbx",
    "LLegUhz": None, "LLegMhx": "LLegUhz", "LLegLhy": "LLegMhx",
    "LLegKny": "LLegLhy", "LLegUay": "LLegKny", "LLegLax": "LLegUay",
    "RLegUhz": None, "RLegMhx": "RLegUhz", "RLegLhy": "RLegMhx",
    "RLegKny": "RLegLhy", "RLegUay": "RLegKny", "RLegLax": "RLegUay",
}

SEGMENT_OF = {
    "BackLbz": "LtorsoSolid", "BackMby": "MtorsoSolid", "BackUbx": "UtorsoSolid",
    "LArmUsy": "LClavSolid", "LArmShx": "LScapSolid", "LArmEly": "LUarmSolid",
    "LArmElx": "LLarmSolid", "LArmUwy": "LFarmSolid", "LArmMwx": "LHandSolid",
    "RArmUsy": "RClavSolid", "RArmShx": "RScapSolid", "RArmEly": "RUarmSolid",
    "RArmElx": "RLarmSolid", "RArmUwy": "RFarmSolid", "RArmMwx": "RHandSolid",
    "LLegUhz": "LUglutSolid", "LLegMhx": "LLglutSolid", "LLegLhy": "LUlegSolid",
    "LLegKny": "LLlegSolid", "LLegUay": "LTalusSolid", "LLegLax": "LFootSolid",
    "RLegUhz": "RUglutSolid", "RLegMhx": "RLglutSolid", "RLegLhy": "RUlegSolid",
    "RLegKny": "RLlegSolid", "RLegUay": "RTalusSolid", "RLegLax": "RFootSolid",
}

ROOT_SEGMENT = "PelvisSolid"
SOLE_OFFSET = np.array([0.048, 0.0, -0.076119])


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


class AtlasModel:
    def __init__(self, path=MODEL_YAML):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        self.joints = data["joints"]
        self.segments = data["segments"]
        self.names = [n for n in CHAIN if n in self.joints]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.limits = {n: (self.joints[n]["min"], self.joints[n]["max"])
                       for n in self.names}

        # NeckAy carries HeadMesh, which the Webots Atlas leaves massless.
        self.mass = {}
        self.com_local = {}
        for name in self.names:
            segment = self.segments.get(SEGMENT_OF.get(name))
            if segment is None:
                self.mass[name] = 0.0
                self.com_local[name] = np.zeros(3)
                continue
            self.mass[name] = float(segment["mass"])
            self.com_local[name] = np.asarray(
                segment.get("com", [0.0, 0.0, 0.0]), dtype=np.float64)

        root = self.segments[ROOT_SEGMENT]
        self.root_mass = float(root["mass"])
        self.root_com = np.asarray(root.get("com", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.total_mass = self.root_mass + sum(self.mass.values())

    def frames(self, angles):
        """World pose of every joint frame, pelvis at the origin."""
        poses = {None: (np.zeros(3), np.eye(3))}
        for name in self.names:
            parent = CHAIN[name]
            origin, rotation = poses[parent]
            spec = self.joints[name]
            anchor = np.asarray(spec["anchor"], dtype=np.float64)
            joint_origin = origin + rotation @ anchor
            local = rodrigues(spec["axis"], float(angles.get(name, 0.0)))
            poses[name] = (joint_origin, rotation @ local)
        return poses

    def com(self, angles, poses=None):
        poses = poses or self.frames(angles)
        total = self.root_mass * self.root_com.copy()
        for name in self.names:
            origin, rotation = poses[name]
            total += self.mass[name] * (origin + rotation @ self.com_local[name])
        return total / self.total_mass

    def com_jacobian(self, angles, poses=None, joints=None):
        """Analytic dCoM/dq for the requested joints (3 x n)."""
        poses = poses or self.frames(angles)
        joints = joints or self.names

        subtree_mass = {}
        subtree_moment = {}
        for name in reversed(self.names):
            origin, rotation = poses[name]
            mass = self.mass[name]
            moment = mass * (origin + rotation @ self.com_local[name])
            for child, parent in CHAIN.items():
                if parent == name and child in subtree_mass:
                    mass += subtree_mass[child]
                    moment += subtree_moment[child]
            subtree_mass[name] = mass
            subtree_moment[name] = moment

        jacobian = np.zeros((3, len(joints)))
        for column, name in enumerate(joints):
            origin, rotation = poses[name]
            axis_world = rotation @ np.asarray(self.joints[name]["axis"], dtype=np.float64)
            axis_world /= max(np.linalg.norm(axis_world), 1e-12)
            mass = subtree_mass[name]
            if mass < 1e-12:
                continue
            centre = subtree_moment[name] / mass
            jacobian[:, column] = (mass / self.total_mass) * np.cross(
                axis_world, centre - origin)
        return jacobian

    def sole(self, angles, side, poses=None):
        poses = poses or self.frames(angles)
        origin, rotation = poses[f"{side}LegLax"]
        return origin + rotation @ SOLE_OFFSET

    def clamp(self, name, value):
        low, high = self.limits[name]
        return max(low, min(high, value))
