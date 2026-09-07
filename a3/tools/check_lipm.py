import json
import math
import os
import sys

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)

from kinematics.gait import plan_walk
from kinematics.lipm import WalkPattern, capture_step, omega_for

DT = 0.002
COM_HEIGHT = 0.86
FOOT_HALF = 0.0624
TOE = 0.178
HEEL = 0.082


def implied_zmp(times, com, omega):
    acceleration = np.zeros_like(com)
    acceleration[1:-1] = (com[2:] - 2.0 * com[1:-1] + com[:-2]) / (DT ** 2)
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]
    return com - acceleration / (omega ** 2)


def support_bounds(pattern, times):
    lower = np.zeros((len(times), 2))
    upper = np.zeros((len(times), 2))
    for index, time in enumerate(times):
        phase, _ = pattern.phase_at(time)
        lower[index] = phase.zmp - np.array([HEEL, FOOT_HALF])
        upper[index] = phase.zmp + np.array([TOE, FOOT_HALF])
        if phase.support == "LR":
            lower[index, 1] -= 0.089
            upper[index, 1] += 0.089
    return lower, upper


def main():
    feet = {"L": (0.0, 0.089), "R": (0.0, -0.089)}
    report = {}

    for label, step_length, steps in (("in place", 0.0, 8),
                                      ("forward 0.20 m", 0.20, 10),
                                      ("forward 0.30 m", 0.30, 10)):
        phases, landings = plan_walk(feet, step_length, 0.178, steps)
        pattern = WalkPattern(COM_HEIGHT).plan(phases)
        times, com, velocity = pattern.integrate_com(DT)

        jumps = []
        for previous, following in zip(pattern.phases, pattern.phases[1:]):
            jumps.append(float(np.linalg.norm(previous.dcm_end - following.dcm_start)))

        dcm = np.array([pattern.dcm(t) for t in times])
        planned = np.array([pattern.zmp(t) for t in times])
        recovered = implied_zmp(times, com, pattern.omega)

        interior = slice(20, -20)
        zmp_error = np.linalg.norm(recovered[interior] - planned[interior], axis=1)

        lower, upper = support_bounds(pattern, times)
        inside = np.all((com >= lower - 1e-9) & (com <= upper + 1e-9), axis=1)

        travelled = float(com[-1, 0] - com[0, 0])
        expected = float(pattern.phases[-1].zmp[0] - pattern.phases[0].zmp[0])

        report[label] = {
            "steps": steps,
            "horizon_s": round(pattern.horizon, 3),
            "omega": round(pattern.omega, 4),
            "dcm_discontinuity_max_m": round(max(jumps), 12),
            "zmp_error_median_mm": round(float(np.median(zmp_error)) * 1000, 4),
            "zmp_error_p99_mm": round(float(np.percentile(zmp_error, 99)) * 1000, 4),
            "com_inside_support_percent": round(float(inside.mean()) * 100, 2),
            "com_bounded": bool(np.all(np.abs(com) < 10.0)),
            "com_travel_m": round(travelled, 4),
            "zmp_travel_m": round(expected, 4),
            "final_dcm_error_m": round(float(np.linalg.norm(
                dcm[-1] - pattern.phases[-1].zmp)), 6),
        }

    omega = omega_for(COM_HEIGHT)
    support = np.array([0.0, -0.089])
    dcm_now = np.array([0.12, 0.03])
    remaining = 0.35
    landing = capture_step(dcm_now, omega, remaining, support)
    dcm_touchdown = support + (dcm_now - support) * math.exp(omega * remaining)
    report["capture_step"] = {
        "landing": [round(float(v), 5) for v in landing],
        "dcm_at_touchdown": [round(float(v), 5) for v in dcm_touchdown],
        "captured_error_m": round(float(np.linalg.norm(
            dcm_touchdown - landing)), 12),
    }

    print(f"{'gait':18s}{'duration':>8s}{'DCM jump':>13s}{'ZMP error':>13s}"
          f"{'CoM drin':>10s}{'Weg':>9s}")
    for label in ("in place", "forward 0.20 m", "forward 0.30 m"):
        entry = report[label]
        print(f"{label:18s}{entry['horizon_s']:7.2f}s"
              f"{entry['dcm_discontinuity_max_m']:13.2e}"
              f"{entry['zmp_error_p99_mm']:10.4f}mm"
              f"{entry['com_inside_support_percent']:9.1f}%"
              f"{entry['com_travel_m']:8.3f}m")
    print()
    print(f"capture step: landing {report['capture_step']['landing']}, "
          f"residual {report['capture_step']['captured_error_m']:.2e} m")

    with open(os.path.join(A3_ROOT, "results", "lipm_check.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    walks = [report[k] for k in ("in place", "forward 0.20 m",
                                 "forward 0.30 m")]
    ok = (all(e["dcm_discontinuity_max_m"] < 1e-9 for e in walks)
          and all(e["zmp_error_p99_mm"] < 1.0 for e in walks)
          and all(e["com_bounded"] for e in walks)
          and report["capture_step"]["captured_error_m"] < 1e-9)
    print("criterion: DCM continuous, ZMP back-solve < 1 mm, CoM bounded  ->",
          "bestanden" if ok else "NICHT bestanden")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
