import argparse
import itertools
import json
import os
import sys

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)

from kinematics.atlaskin import AtlasModel
from kinematics.legik import LegIK

FOCAL_PX = 1000.0
TORSO_M = 0.50
STANCE_WIDTH = 0.178
CLEARANCE = 0.05
DT = 0.008


def load(path):
    frames = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if "angles" in entry:
                    frames.append(entry)
    return frames


def pelvis_track(frames):
    torso0 = frames[0]["torso_px"]
    hip0 = frames[0]["hip_px"]
    times = np.array([f["t"] for f in frames])
    forward = np.array([FOCAL_PX * TORSO_M * (1.0 / torso0 - 1.0 / max(f["torso_px"], 1.0))
                        for f in frames])
    lateral = np.array([-(f["hip_px"] - hip0) * TORSO_M / max(f["torso_px"], 1.0)
                        for f in frames])
    grid = np.arange(times[0], times[-1], DT)
    return grid, np.interp(grid, times, forward), np.interp(grid, times, lateral)


def smoothstep(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def simulate(grid, forward, lateral, trigger, swing_time, height):
    feet = {"L": np.array([0.0, +0.5 * STANCE_WIDTH, 0.0]),
            "R": np.array([0.0, -0.5 * STANCE_WIDTH, 0.0])}
    support = "L"
    swing = None
    swing_from = swing_to = None
    phase = 0.0
    relative = []
    steps = 0
    for index in range(len(grid)):
        planar = np.array([forward[index], lateral[index]])
        if swing is None:
            along = float(planar[0] - feet[support][0])
            if abs(along) > trigger:
                swing = "R" if support == "L" else "L"
                swing_from = feet[swing].copy()
                direction = 1.0 if along > 0.0 else -1.0
                swing_to = np.array([planar[0] + direction * trigger,
                                     planar[1] + 0.5 * STANCE_WIDTH
                                     * (1.0 if swing == "L" else -1.0), 0.0])
                phase = 0.0
        else:
            phase += DT / swing_time
            blend = smoothstep(min(1.0, phase))
            position = swing_from + (swing_to - swing_from) * blend
            position[2] = CLEARANCE * np.sin(np.pi * min(1.0, phase))
            feet[swing] = position
            if phase >= 1.0:
                feet[swing] = swing_to.copy()
                support = swing
                swing = None
                steps += 1
        pelvis = np.array([planar[0], planar[1], height])
        for side in ("L", "R"):
            relative.append(feet[side] - pelvis)
    return np.array(relative), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", default=os.path.join(
        A3_ROOT, "recordings", "testvideo_angles.jsonl"))
    args = parser.parse_args()

    frames = load(args.angles)
    grid, forward, lateral = pelvis_track(frames)
    print(f"pelvis path: {len(grid)} ticks, forward "
          f"{forward.min():+.3f}..{forward.max():+.3f} m, "
          f"lateral {lateral.min():+.3f}..{lateral.max():+.3f} m")

    model = AtlasModel()
    ik = LegIK(model, "L")

    print(f"\n{'height':>6s}{'trig':>6s}{'Swing':>7s}{'steps':>9s}"
          f"{'vor max':>9s}{'zurueck max':>13s}{'seit max':>10s}"
          f"{'IK p95':>9s}{'IK max':>9s}")
    best = None
    for height, trigger, swing_time in itertools.product(
            (0.81, 0.83, 0.85), (0.06, 0.07, 0.08, 0.09, 0.10), (0.24, 0.30)):
        relative, steps = simulate(grid, forward, lateral, trigger,
                                   swing_time, height)
        sample = relative[::13]
        errors = []
        seed = None
        for target in sample:
            values, error, _ = ik.solve(target, np.eye(3), seed=seed)
            seed = values
            errors.append(error * 1000.0)
        errors = np.array(errors)
        p95 = float(np.percentile(errors, 95))
        worst = float(errors.max())
        print(f"{height:6.2f}{trigger:6.2f}{swing_time:7.2f}{steps:9d}"
              f"{relative[:, 0].max():9.3f}{relative[:, 0].min():13.3f}"
              f"{np.abs(relative[:, 1]).max():10.3f}{p95:9.3f}{worst:9.3f}")
        score = (p95, worst)
        if best is None or score < best[0]:
            best = (score, (height, trigger, swing_time, steps))
    print(f"\nbest result: height {best[1][0]}, Trigger {best[1][1]}, "
          f"swing {best[1][2]} -> {best[1][3]} steps, "
          f"IK p95 {best[0][0]:.3f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
