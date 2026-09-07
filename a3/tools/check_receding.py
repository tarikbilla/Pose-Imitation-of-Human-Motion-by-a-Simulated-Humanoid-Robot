import json
import os
import sys

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)

from kinematics.gait import plan_walk
from kinematics.lipm import WalkPattern

COM_HEIGHT = 0.865
SINGLE = 0.62
DOUBLE = 0.20


def preview(feet, length, horizon, first):
    phases, landings = plan_walk(feet, length, 0.178, horizon,
                                 single=SINGLE, double=DOUBLE, first=first,
                                 settle=DOUBLE, taper_first=False)
    return WalkPattern(COM_HEIGHT).plan(phases), landings


def cycle(feet, lengths, horizon, cycles, first="R"):
    swing = first
    jumps = []
    dcm_track = []
    positions = {"L": np.array(feet["L"], dtype=np.float64),
                 "R": np.array(feet["R"], dtype=np.float64)}
    for index in range(cycles):
        length = lengths[min(index, len(lengths) - 1)]
        pattern, landings = preview(positions, length, horizon, swing)
        boundary = pattern.phases[0].duration + pattern.phases[1].duration
        dcm_before = pattern.dcm(boundary)
        landing_swing, landing_target, _ = landings[0]
        positions[landing_swing] = landing_target.copy()
        swing = "L" if landing_swing == "R" else "R"
        following, _ = preview(positions, length, horizon, swing)
        dcm_after = following.dcm(0.0)
        jumps.append(float(np.linalg.norm(dcm_after - dcm_before)))
        dcm_track.append((round(float(dcm_before[0]), 4),
                          round(float(dcm_after[0]), 4)))
    return jumps, dcm_track, positions


def main():
    feet = {"L": np.array([0.0, 0.089]), "R": np.array([0.0, -0.089])}
    report = {}

    print(f"{'preview':>9s}{'step length':>15s}{'DCM jump med':>16s}"
          f"{'max':>10s}")
    for horizon in (1, 2, 3, 4, 6):
        jumps, _, _ = cycle(feet, [0.20], horizon, 8)
        report[f"horizon_{horizon}"] = {
            "median_mm": round(float(np.median(jumps)) * 1000, 4),
            "max_mm": round(float(max(jumps)) * 1000, 4),
        }
        print(f"{horizon:9d}{0.20:15.2f}{np.median(jumps) * 1000:14.4f}mm"
              f"{max(jumps) * 1000:8.4f}mm")

    print()
    print("alternating step length at preview 4 "
          "(forward, stop, backward, forward):")
    lengths = [0.20, 0.20, 0.20, 0.0, 0.0, -0.18, -0.18, -0.18, 0.0, 0.20, 0.20]
    jumps, track, final = cycle(feet, lengths, 4, len(lengths))
    for index, (length, jump, pair) in enumerate(zip(lengths, jumps, track)):
        print(f"  Zyklus {index:2d}  L={length:+.2f} m  "
              f"DCM jump {jump * 1000:7.4f} mm  "
              f"x: {pair[0]:+.4f} -> {pair[1]:+.4f}")
    report["varying"] = {
        "lengths": lengths,
        "median_mm": round(float(np.median(jumps)) * 1000, 4),
        "max_mm": round(float(max(jumps)) * 1000, 4),
        "final_feet": {k: [round(float(x), 4) for x in v] for k, v in final.items()},
    }
    print(f"\nEndposition Fuesse: L{report['varying']['final_feet']['L']} "
          f"R{report['varying']['final_feet']['R']}")

    with open(os.path.join(A3_ROOT, "results", "receding_check.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    stable = report["horizon_4"]["max_mm"] < 1.0
    varying = report["varying"]["max_mm"] < 5.0
    print(f"\ncriterion: preview 4 below 1 mm jump, alternating length "
          f"unter 5 mm  -> {'bestanden' if stable and varying else 'NICHT bestanden'}")
    return 0 if (stable and varying) else 1


if __name__ == "__main__":
    sys.exit(main())
