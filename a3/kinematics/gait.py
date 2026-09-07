import numpy as np

from .lipm import Phase

DEFAULT_SINGLE = 0.62
DEFAULT_DOUBLE = 0.20
DEFAULT_CLEARANCE = 0.045


def other(side):
    return "R" if side == "L" else "L"


def plan_walk(feet, step_length, step_width, steps,
              single=DEFAULT_SINGLE, double=DEFAULT_DOUBLE, first="R",
              settle=0.6, taper_first=True):
    positions = {"L": np.array(feet["L"][:2], dtype=np.float64),
                 "R": np.array(feet["R"][:2], dtype=np.float64)}
    centre = 0.5 * (positions["L"][1] + positions["R"][1])
    phases = [Phase(0.5 * (positions["L"] + positions["R"]), settle,
                    None, "LR", False)]

    swing = first
    landings = []
    for index in range(steps):
        support = other(swing)
        advance = step_length
        if index == 0 and taper_first:
            advance = 0.5 * step_length
        if index == steps - 1:
            advance = 0.5 * step_length
        target = positions[support].copy()
        target[0] += advance
        target[1] = centre + (step_width * 0.5) * (1.0 if swing == "L" else -1.0)

        phases.append(Phase(positions[support], single, swing, support, True))
        landings.append((swing, target.copy(), phases[-1]))
        positions[swing] = target

        midpoint = 0.5 * (positions["L"] + positions["R"])
        phases.append(Phase(midpoint, double, None, "LR", False))
        swing = other(swing)

    phases.append(Phase(0.5 * (positions["L"] + positions["R"]), settle,
                        None, "LR", False))
    return phases, landings


def swing_profile(fraction):
    fraction = min(1.0, max(0.0, fraction))
    horizontal = fraction * fraction * (3.0 - 2.0 * fraction)
    vertical = np.sin(np.pi * fraction) ** 2
    return horizontal, vertical


def swing_pose(origin, target, fraction, clearance=DEFAULT_CLEARANCE):
    horizontal, vertical = swing_profile(fraction)
    position = np.array([
        origin[0] + (target[0] - origin[0]) * horizontal,
        origin[1] + (target[1] - origin[1]) * horizontal,
        origin[2] + (target[2] - origin[2]) * horizontal + clearance * vertical,
    ])
    return position


class GaitCommand:
    def __init__(self):
        self.cadence = 0.0
        self.step_length = 0.0
        self.step_width = 0.178
        self.active = False

    def update(self, cadence, step_length, step_width=None):
        self.cadence = float(cadence)
        self.step_length = float(step_length)
        if step_width is not None:
            self.step_width = float(step_width)
        self.active = self.cadence > 0.05

    def timings(self):
        if self.cadence <= 0.05:
            return DEFAULT_SINGLE, DEFAULT_DOUBLE
        period = 1.0 / max(self.cadence, 0.2)
        single = period * (DEFAULT_SINGLE / (DEFAULT_SINGLE + DEFAULT_DOUBLE))
        double = period - single
        return single, double
