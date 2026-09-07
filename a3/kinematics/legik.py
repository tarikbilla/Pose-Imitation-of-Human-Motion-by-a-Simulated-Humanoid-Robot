import math

import numpy as np

from .atlaskin import SOLE_OFFSET

LEG_JOINTS = ("LegUhz", "LegMhx", "LegLhy", "LegKny", "LegUay", "LegLax")

MAX_ITERATIONS = 24
DAMPING = 0.01
STEP_LIMIT = 0.30
POSITION_TOLERANCE = 2e-4
ROTATION_TOLERANCE = 3e-3
RETRY_DAMPING = 0.05
DEFAULT_SEED = (0.0, 0.0, -0.40, 0.80, -0.40, 0.0)
RESTART_KNEES = (0.80, 1.30, 0.35)
RESTART_THRESHOLD = 5e-3
WARM_ACCEPT = 0.02


def _rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _elementary(axis):
    if abs(axis[0]) > 0.5:
        return _rot_x
    if abs(axis[1]) > 0.5:
        return _rot_y
    return _rot_z


def rotation_error(current, target):
    delta = target @ current.T
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(trace)
    if angle < 1e-9:
        return np.zeros(3), 0.0
    axis = np.array([delta[2, 1] - delta[1, 2],
                     delta[0, 2] - delta[2, 0],
                     delta[1, 0] - delta[0, 1]])
    return axis * (angle / (2.0 * math.sin(angle))), angle


class LegIK:
    def __init__(self, model, side):
        self.model = model
        self.side = side
        self.names = tuple(side + suffix for suffix in LEG_JOINTS)
        self.limits = np.array([model.limits[name] for name in self.names])
        self.axes = [np.asarray(model.joints[name]["axis"], dtype=np.float64)
                     for name in self.names]
        self.anchors = [np.asarray(model.joints[name]["anchor"], dtype=np.float64)
                        for name in self.names]
        self.rotators = [_elementary(axis) for axis in self.axes]
        self.identity = np.eye(3)

    def chain(self, values):
        origin = np.zeros(3)
        rotation = self.identity
        origins = []
        rotations = []
        for index in range(6):
            origin = origin + rotation @ self.anchors[index]
            rotation = rotation @ self.rotators[index](values[index])
            origins.append(origin)
            rotations.append(rotation)
        return origins, rotations

    def pose(self, values):
        origins, rotations = self.chain(values)
        return origins[-1] + rotations[-1] @ SOLE_OFFSET, rotations[-1]

    def jacobian(self, origins, rotations, sole):
        columns = np.empty((6, 6))
        for index in range(6):
            axis = rotations[index] @ self.axes[index]
            columns[:3, index] = np.cross(axis, sole - origins[index])
            columns[3:, index] = axis
        return columns

    def clamp(self, values):
        return np.clip(values, self.limits[:, 0], self.limits[:, 1])

    def _descend(self, target_position, target_rotation, start, damping,
                 weight=1.0):
        current = self.clamp(np.asarray(start, dtype=np.float64))
        regular = np.eye(6) * (damping ** 2)
        position_error = float("inf")
        angle_error = float("inf")
        for _ in range(MAX_ITERATIONS):
            origins, rotations = self.chain(current)
            sole = origins[-1] + rotations[-1] @ SOLE_OFFSET
            offset = target_position - sole
            twist, angle_error = rotation_error(rotations[-1], target_rotation)
            position_error = float(np.linalg.norm(offset))
            if (position_error < POSITION_TOLERANCE
                    and angle_error * weight < ROTATION_TOLERANCE):
                break
            error = np.concatenate([offset, twist * weight])
            jac = self.jacobian(origins, rotations, sole)
            if weight != 1.0:
                jac = jac.copy()
                jac[3:, :] *= weight
            step = jac.T @ np.linalg.solve(jac @ jac.T + regular, error)
            norm = float(np.linalg.norm(step))
            if norm < 1e-6:
                break
            if norm > STEP_LIMIT:
                step *= STEP_LIMIT / norm
            current = self.clamp(current + step)
        return current, position_error, angle_error

    def solve(self, target_position, target_rotation, seed=None,
              weight=1.0):
        target_position = np.asarray(target_position, dtype=np.float64)
        starts = []
        if seed is not None:
            starts.append(np.asarray(seed, dtype=np.float64))
        starts.append(np.array(DEFAULT_SEED))

        best = None
        warm = seed is not None
        for index, start in enumerate(starts):
            values, position_error, angle_error = self._descend(
                target_position, target_rotation, start, DAMPING, weight)
            if best is None or position_error < best[1]:
                best = (values, position_error, angle_error)
            if position_error < RESTART_THRESHOLD:
                return best
            if warm and index == 0 and position_error < WARM_ACCEPT:
                return best

        for knee in RESTART_KNEES:
            start = np.array(DEFAULT_SEED)
            start[3] = knee
            start[2] = -0.5 * knee
            start[4] = -0.5 * knee
            values, position_error, angle_error = self._descend(
                target_position, target_rotation, start, RETRY_DAMPING,
                weight)
            if position_error < best[1]:
                best = (values, position_error, angle_error)
            if position_error < RESTART_THRESHOLD:
                break
        return best

    def as_dict(self, values):
        return {name: float(value) for name, value in zip(self.names, values)}
