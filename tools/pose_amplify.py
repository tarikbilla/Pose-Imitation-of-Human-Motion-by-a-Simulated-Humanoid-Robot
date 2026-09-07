import math

import numpy as np

REST = {
    "left_upper_arm": (0.039, 0.318, -0.945),
    "left_fore_arm": (0.039, 0.318, -0.945),
    "right_upper_arm": (0.114, -0.374, -0.918),
    "right_fore_arm": (0.114, -0.374, -0.918),
}
SCALARS = ("torso_yaw", "torso_pitch", "torso_roll", "head_yaw", "head_pitch")


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / norm if norm > 1e-9 else None


def _rotate(rest, axis, angle):
    axis = _unit(axis)
    if axis is None:
        return rest
    cos = math.cos(angle)
    sin = math.sin(angle)
    return (rest * cos + np.cross(axis, rest) * sin
            + axis * float(np.dot(axis, rest)) * (1.0 - cos))


def amplify(directions, scalars, factor):
    if factor == 1.0:
        return directions, scalars

    out_directions = {}
    for name, vector in directions.items():
        rest = REST.get(name)
        unit = _unit(vector) if vector is not None else None
        if rest is None or unit is None:
            out_directions[name] = vector
            continue
        rest = _unit(rest)
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(rest, unit)))))
        target = min(math.pi, factor * angle)
        out_directions[name] = _rotate(rest, np.cross(rest, unit), target).tolist()

    out_scalars = dict(scalars)
    for name in SCALARS:
        if name in out_scalars:
            out_scalars[name] = float(out_scalars[name]) * factor
    return out_directions, out_scalars
