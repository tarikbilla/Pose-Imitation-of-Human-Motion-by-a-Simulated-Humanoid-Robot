import os

DOUBLE_SUPPORT = "DS"
SHIFT = "SHIFT"
LIFT = "LIFT"
PLACE = "PLACE"

SHIFT_SECONDS = float(os.environ.get("A3_SHIFT_S", "0.55"))
LIFT_SECONDS = float(os.environ.get("A3_LIFT_S", "0.40"))
PLACE_SECONDS = float(os.environ.get("A3_PLACE_S", "0.35"))
SETTLE_SECONDS = float(os.environ.get("A3_STEP_SETTLE_S", "0.30"))

SHIFT_TARGET = float(os.environ.get("A3_SHIFT_Y", "0.055"))
SHIFT_TOLERANCE = float(os.environ.get("A3_SHIFT_TOL", "0.020"))
LIFT_KNEE = float(os.environ.get("A3_LIFT_KNEE", "0.55"))
LIFT_HIP = float(os.environ.get("A3_LIFT_HIP", "0.30"))
LIFT_ANKLE = float(os.environ.get("A3_LIFT_ANKLE", "0.18"))

CADENCE_MIN = 0.08


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class Stepper:
    """Marching state machine.

    The camera says *whether* and *how fast* to step; it never says how. The
    weight shift always completes before a foot is allowed to leave the ground,
    which is the ordering a direct joint copy destroys.
    """

    def __init__(self):
        self.state = DOUBLE_SUPPORT
        self.swing = "R"
        self.clock = 0.0
        self.active = False
        self.steps = 0
        self.blocked = 0.0

    def reset(self):
        self.__init__()

    @property
    def stance(self):
        return "L" if self.swing == "R" else "R"

    def _shift_sign(self):
        # positive body-y is the robot's left
        return 1.0 if self.stance == "L" else -1.0

    def update(self, dt, request, com_error_y):
        """request: 0..1 marching intensity from the human. Returns
        (com_target_y, swing_side, lift_fraction)."""
        self.clock += dt
        want = request > CADENCE_MIN

        if self.state == DOUBLE_SUPPORT:
            self.active = want
            if want and self.clock >= SETTLE_SECONDS:
                self.state = SHIFT
                self.clock = 0.0
            return 0.0, None, 0.0

        if self.state == SHIFT:
            target = SHIFT_TARGET * self._shift_sign()
            progress = smoothstep(self.clock / SHIFT_SECONDS)
            com_target = target * progress
            reached = abs(com_error_y - com_target) < SHIFT_TOLERANCE
            if self.clock >= SHIFT_SECONDS:
                if reached:
                    self.state = LIFT
                    self.clock = 0.0
                else:
                    self.blocked += dt
                    if self.blocked > 1.5:
                        self.state = DOUBLE_SUPPORT
                        self.clock = 0.0
                        self.blocked = 0.0
            return com_target, None, 0.0

        if self.state == LIFT:
            target = SHIFT_TARGET * self._shift_sign()
            fraction = smoothstep(self.clock / LIFT_SECONDS)
            if self.clock >= LIFT_SECONDS:
                self.state = PLACE
                self.clock = 0.0
            return target, self.swing, fraction

        if self.state == PLACE:
            target = SHIFT_TARGET * self._shift_sign()
            fraction = 1.0 - smoothstep(self.clock / PLACE_SECONDS)
            if self.clock >= PLACE_SECONDS:
                self.steps += 1
                self.swing = "L" if self.swing == "R" else "R"
                self.state = DOUBLE_SUPPORT
                self.clock = 0.0
                self.blocked = 0.0
            return target * max(0.0, fraction), self.swing, fraction

        return 0.0, None, 0.0

    def leg_offsets(self, swing, fraction):
        """Joint offsets that lift the swing leg clear of the ground."""
        if swing is None or fraction <= 0.0:
            return {}
        return {
            f"{swing}LegKny": LIFT_KNEE * fraction,
            f"{swing}LegLhy": -LIFT_HIP * fraction,
            f"{swing}LegUay": LIFT_ANKLE * fraction,
        }

    def describe(self):
        return {"state": self.state, "swing": self.swing,
                "steps": self.steps, "active": self.active}
