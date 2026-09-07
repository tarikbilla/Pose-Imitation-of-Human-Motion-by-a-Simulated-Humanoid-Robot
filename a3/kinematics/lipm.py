import math

import numpy as np

GRAVITY = 9.81


def omega_for(height):
    return math.sqrt(GRAVITY / max(height, 1e-3))


def capture_point(com, com_velocity, omega):
    return np.asarray(com[:2]) + np.asarray(com_velocity[:2]) / omega


class Phase:
    def __init__(self, zmp, duration, swing, support, lift):
        self.zmp = np.asarray(zmp, dtype=np.float64)
        self.duration = float(duration)
        self.swing = swing
        self.support = support
        self.lift = bool(lift)
        self.start = 0.0
        self.dcm_start = None
        self.dcm_end = None

    @property
    def end(self):
        return self.start + self.duration


class WalkPattern:
    def __init__(self, com_height, omega=None):
        self.com_height = float(com_height)
        self.omega = float(omega) if omega else omega_for(com_height)
        self.phases = []
        self.horizon = 0.0

    def plan(self, phases):
        self.phases = list(phases)
        clock = 0.0
        for phase in self.phases:
            phase.start = clock
            clock += phase.duration
        self.horizon = clock

        terminal = self.phases[-1].zmp.copy()
        following = terminal
        for phase in reversed(self.phases):
            phase.dcm_end = following
            decay = math.exp(-self.omega * phase.duration)
            phase.dcm_start = phase.zmp + (phase.dcm_end - phase.zmp) * decay
            following = phase.dcm_start
        return self

    def phase_at(self, time):
        if not self.phases:
            return None, 0.0
        for phase in self.phases:
            if time < phase.end:
                return phase, time - phase.start
        last = self.phases[-1]
        return last, last.duration

    def dcm(self, time):
        phase, local = self.phase_at(time)
        if phase is None:
            return np.zeros(2)
        growth = math.exp(self.omega * local)
        return phase.zmp + (phase.dcm_start - phase.zmp) * growth

    def dcm_velocity(self, time):
        phase, _ = self.phase_at(time)
        if phase is None:
            return np.zeros(2)
        return self.omega * (self.dcm(time) - phase.zmp)

    def zmp(self, time):
        phase, _ = self.phase_at(time)
        return np.zeros(2) if phase is None else phase.zmp.copy()

    def integrate_com(self, dt, start=None):
        com = np.array(self.phases[0].zmp if start is None else start[:2],
                       dtype=np.float64)
        steps = int(round(self.horizon / dt))
        times = np.arange(steps + 1) * dt
        trace = np.zeros((steps + 1, 2))
        velocity = np.zeros((steps + 1, 2))
        for index, time in enumerate(times):
            trace[index] = com
            reference = self.dcm(time)
            rate = self.omega * (reference - com)
            velocity[index] = rate
            com = com + rate * dt
        return times, trace, velocity


def track_zmp(measured_dcm, reference_dcm, reference_zmp, omega, gain):
    return (reference_zmp
            + (1.0 + gain / omega) * (np.asarray(measured_dcm)
                                      - np.asarray(reference_dcm)))


def capture_step(measured_dcm, omega, remaining, support_zmp):
    growth = math.exp(omega * max(remaining, 0.0))
    return (np.asarray(support_zmp, dtype=np.float64)
            + (np.asarray(measured_dcm, dtype=np.float64)
               - np.asarray(support_zmp, dtype=np.float64)) * growth)
