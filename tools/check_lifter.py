import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.lift3d import H36M, Lifter3D
from perception.recording import Replay

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

LIMBS = {
    "upper arm L": ("left_shoulder", "left_elbow"),
    "fore arm L": ("left_elbow", "left_wrist"),
    "upper arm R": ("right_shoulder", "right_elbow"),
    "fore arm R": ("right_elbow", "right_wrist"),
    "thigh L": ("left_hip", "left_knee"),
    "shank L": ("left_knee", "left_ankle"),
    "thigh R": ("right_hip", "right_knee"),
    "shank R": ("right_knee", "right_ankle"),
}
LEGS = ("thigh L", "shank L", "thigh R", "shank R")


def main():
    parser = argparse.ArgumentParser(description="Verify the 3-D lifter on a recording.")
    parser.add_argument("--recording",
                        default=os.path.join(A3_ROOT, "recordings", "testvideo_wb.jsonl"))
    parser.add_argument("--every-nth", type=int, default=1)
    args = parser.parse_args()

    lifter = Lifter3D(every_nth=args.every_nth)
    print("=" * 68)
    print("A3 / 3-D lifter check")
    print("=" * 68)
    print(f"model      {lifter.describe()}")
    if lifter.warning:
        print(f"WARNING    {lifter.warning}")
    if not lifter.available:
        return 1

    replay = Replay(args.recording).load()
    poses = []
    raw2d = []
    timings = []
    for entry in replay.entries:
        if entry.get("kp") is None:
            continue
        keypoints = np.asarray(entry["kp"], dtype=np.float32)
        scores = np.asarray(entry["sc"], dtype=np.float32)
        pose = lifter(keypoints, scores)
        if pose is not None and lifter.ready:
            poses.append(pose)
            raw2d.append(keypoints)
            if lifter.inference_ms:
                timings.append(lifter.inference_ms)

    print(f"frames     {len(replay)} in, {len(poses)} lifted")
    if timings:
        print(f"inference  median {statistics.median(timings):.1f} ms  "
              f"({1000 / statistics.median(timings):.0f} Hz)")
    print("-" * 68)

    from perception.skeleton import A3_INDEX
    lateral, vertical = [], []
    d3 = {0: [], 1: [], 2: []}
    for pose, kp in zip(poses, raw2d):
        d2 = kp[A3_INDEX["left_elbow"]] - kp[A3_INDEX["left_shoulder"]]
        lateral.append(float(d2[0]))
        vertical.append(float(d2[1]))
        delta = pose[H36M["left_elbow"]] - pose[H36M["left_shoulder"]]
        for axis in range(3):
            d3[axis].append(float(delta[axis]))

    print("axis semantics (correlation of the 3-D axis with the image axes)")
    print(f"{'axis':<8}{'vs image x':>13}{'vs image y':>13}{'std':>10}")
    for axis in range(3):
        values = np.asarray(d3[axis])
        cx = float(np.corrcoef(values, lateral)[0, 1])
        cy = float(np.corrcoef(values, vertical)[0, 1])
        print(f"{'xyz'[axis]:<8}{cx:>13.3f}{cy:>13.3f}{values.std():>10.3f}")
    print("-" * 68)

    print("limb length stability   std/mean, lower is better")
    print(f"{'limb':<14}{'2D':>9}{'3D':>9}{'gain':>9}")
    report = {}
    gains_leg, gains_arm = [], []
    for label, (a, b) in LIMBS.items():
        two, three = [], []
        for pose, kp in zip(poses, raw2d):
            scale2 = float(np.linalg.norm(
                kp[A3_INDEX["neck"]] - kp[A3_INDEX["hip_center"]]))
            scale3 = float(np.linalg.norm(pose[H36M["neck"]] - pose[H36M["hip"]]))
            if scale2 < 1e-6 or scale3 < 1e-9:
                continue
            two.append(float(np.linalg.norm(kp[A3_INDEX[a]] - kp[A3_INDEX[b]])) / scale2)
            three.append(float(np.linalg.norm(pose[H36M[a]] - pose[H36M[b]])) / scale3)
        two, three = np.asarray(two), np.asarray(three)
        v2 = two.std() / two.mean()
        v3 = three.std() / three.mean()
        gain = (1 - v3 / v2) * 100
        (gains_leg if label in LEGS else gains_arm).append(gain)
        report[label] = {"var2d": v2, "var3d": v3, "gain_percent": gain}
        print(f"{label:<14}{v2:>9.3f}{v3:>9.3f}{gain:>8.0f}%")

    print("-" * 68)
    print(f"legs  mean gain {np.mean(gains_leg):+5.0f}%      "
          f"arms mean gain {np.mean(gains_arm):+5.0f}%")
    print("=" * 68)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "lifter_check.json"), "w", encoding="utf-8") as handle:
        json.dump({"limbs": report,
                   "leg_gain": float(np.mean(gains_leg)),
                   "arm_gain": float(np.mean(gains_arm)),
                   "median_ms": statistics.median(timings) if timings else None},
                  handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
