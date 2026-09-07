import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.recording import Replay
from perception.retarget import UpperBodyRetargeter

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

DIRECTIONS = ("left_upper_arm", "left_fore_arm", "left_hand_normal",
              "right_upper_arm", "right_fore_arm", "right_hand_normal")
SCALARS = ("torso_yaw", "torso_pitch", "torso_roll", "head_yaw", "head_pitch")


def main():
    parser = argparse.ArgumentParser(description="Run a recording through the retargeter.")
    parser.add_argument("recording")
    args = parser.parse_args()

    replay = Replay(args.recording).load()
    retargeter = UpperBodyRetargeter()

    frames = 0
    valid = 0
    collected = {name: [] for name in DIRECTIONS}
    scalars = {name: [] for name in SCALARS}
    confidences = []
    unit_errors = []

    for entry in replay.entries:
        frames += 1
        if entry.get("kp") is None:
            continue
        keypoints = np.asarray(entry["kp"], dtype=np.float32)
        scores = np.asarray(entry["sc"], dtype=np.float32)
        out = retargeter(keypoints, scores, entry["t"])
        if out is None or not out["valid"]:
            continue
        valid += 1
        confidences.append(out["confidence"])
        for name, vector in out["directions"].items():
            collected[name].append(vector)
            unit_errors.append(abs(float(np.linalg.norm(vector)) - 1.0))
        for name in SCALARS:
            if name in out["scalars"]:
                scalars[name].append(out["scalars"][name])

    print("=" * 74)
    print("A3 / retargeting check")
    print("=" * 74)
    print(f"recording      {os.path.basename(args.recording)}")
    print(f"frames         {frames}")
    print(f"valid output   {valid}  ({valid / max(1, frames) * 100:.1f}%)")
    print(f"confidence     mean {np.mean(confidences):.3f}" if confidences else "")
    if unit_errors:
        print(f"unit-norm err  max {max(unit_errors):.2e}   (must be ~0)")
    print("-" * 74)

    print("direction vectors  (body frame: x forward, y left, z up)")
    print(f"{'field':<20}{'n':>7}{'x range':>18}{'y range':>18}{'z range':>18}")
    report = {}
    for name in DIRECTIONS:
        data = np.asarray(collected[name]) if collected[name] else None
        if data is None or not len(data):
            print(f"{name:<20}{0:>7}{'-':>18}{'-':>18}{'-':>18}")
            continue
        ranges = []
        for axis in range(3):
            ranges.append(f"{data[:, axis].min():+.2f}..{data[:, axis].max():+.2f}")
        print(f"{name:<20}{len(data):>7}{ranges[0]:>18}{ranges[1]:>18}{ranges[2]:>18}")
        report[name] = {
            "count": len(data),
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
        }

    print("-" * 74)
    print("scalar angles")
    print(f"{'field':<20}{'n':>7}{'min deg':>12}{'max deg':>12}{'span deg':>12}")
    for name in SCALARS:
        data = np.asarray(scalars[name]) if scalars[name] else None
        if data is None or not len(data):
            print(f"{name:<20}{0:>7}")
            continue
        lo = math.degrees(data.min())
        hi = math.degrees(data.max())
        print(f"{name:<20}{len(data):>7}{lo:>12.1f}{hi:>12.1f}{hi - lo:>12.1f}")
        report[name] = {"count": len(data), "min_deg": lo, "max_deg": hi}

    print("-" * 74)
    issues = []
    if valid / max(1, frames) < 0.9:
        issues.append(f"only {valid / max(1, frames) * 100:.0f}% of frames produced output")
    if unit_errors and max(unit_errors) > 1e-5:
        issues.append(f"direction vectors are not unit length (max err {max(unit_errors):.2e})")
    for name in ("left_upper_arm", "right_upper_arm"):
        if name in report:
            span = np.asarray(report[name]["max"]) - np.asarray(report[name]["min"])
            if span[0] < 0.2:
                issues.append(f"{name}: depth axis barely moves (span {span[0]:.2f}) - "
                              "foreshortening may not be resolving")
    if issues:
        for issue in issues:
            print(f"  ISSUE: {issue}")
    else:
        print("  retargeting output looks consistent")
    print("=" * 74)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "retarget_check.json"), "w", encoding="utf-8") as handle:
        json.dump({"frames": frames, "valid": valid, "fields": report,
                   "issues": issues}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
