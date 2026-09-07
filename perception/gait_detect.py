import numpy as np

from .skeleton import A3_INDEX

SCALE_TAU = 0.35
LIFT_TAU = 0.60
LIFT_ON = 0.040
LIFT_OFF = 0.020
MIN_STEP_SECONDS = 0.25
MAX_STEP_SECONDS = 2.5
DISTANCE_TAU = 0.25
STEP_GAIN = float(1.0)
MIN_CONFIDENCE = 0.30


class GaitDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scale = None
        self.baseline = {"L": None, "R": None}
        self.airborne = {"L": False, "R": False}
        self.distance = None
        self.last_touchdown = None
        self.last_distance = None
        self.cadence = 0.0
        self.step_length = 0.0
        self.support = None
        self.events = []
        self.time = 0.0

    def _blend(self, previous, value, tau, dt):
        if previous is None:
            return value
        alpha = dt / max(tau, dt)
        return previous + alpha * (value - previous)

    def update(self, keypoints, scores, timestamp):
        dt = max(1e-3, timestamp - self.time)
        self.time = timestamp

        shoulder = 0.5 * (keypoints[A3_INDEX["left_shoulder"]]
                          + keypoints[A3_INDEX["right_shoulder"]])
        hip = 0.5 * (keypoints[A3_INDEX["left_hip"]]
                     + keypoints[A3_INDEX["right_hip"]])
        torso = float(abs(hip[1] - shoulder[1]))
        if torso < 1e-3:
            return self.state()
        self.scale = self._blend(self.scale, torso, SCALE_TAU, dt)

        radial = 1.0 / max(self.scale, 1e-6)
        self.distance = self._blend(self.distance, radial, DISTANCE_TAU, dt)

        ankles = {"L": keypoints[A3_INDEX["left_ankle"]],
                  "R": keypoints[A3_INDEX["right_ankle"]]}
        confidence = min(float(scores[A3_INDEX["left_ankle"]]),
                         float(scores[A3_INDEX["right_ankle"]]))
        if confidence < MIN_CONFIDENCE:
            return self.state()

        lowest = max(ankles["L"][1], ankles["R"][1])
        for side, point in ankles.items():
            self.baseline[side] = self._blend(self.baseline[side], lowest,
                                              LIFT_TAU, dt)
            height = (self.baseline[side] - point[1]) / self.scale
            if not self.airborne[side] and height > LIFT_ON:
                self.airborne[side] = True
            elif self.airborne[side] and height < LIFT_OFF:
                self.airborne[side] = False
                self._touchdown(side, timestamp)

        if self.last_touchdown is not None:
            since = timestamp - self.last_touchdown
            if since > MAX_STEP_SECONDS:
                self.cadence = 0.0
                self.step_length = 0.0
        return self.state()

    def _touchdown(self, side, timestamp):
        if self.last_touchdown is not None:
            interval = timestamp - self.last_touchdown
            if interval < MIN_STEP_SECONDS:
                return
            if interval <= MAX_STEP_SECONDS:
                self.cadence = 1.0 / interval
                if self.last_distance is not None:
                    travel = self.distance - self.last_distance
                    self.step_length = -STEP_GAIN * travel / max(self.distance, 1e-6)
        self.last_touchdown = timestamp
        self.last_distance = self.distance
        self.support = side
        self.events.append({"t": round(timestamp, 3), "side": side,
                            "cadence": round(self.cadence, 3),
                            "step_length": round(self.step_length, 4)})

    def state(self):
        return {"cadence": self.cadence,
                "step_length": self.step_length,
                "airborne": dict(self.airborne),
                "scale": self.scale,
                "distance": self.distance,
                "stepping": self.cadence > 0.05}
