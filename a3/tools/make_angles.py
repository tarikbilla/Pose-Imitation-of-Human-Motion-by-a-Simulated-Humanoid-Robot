import argparse
import json
import os
import sys

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)
sys.path.insert(0, os.path.join(A3_ROOT, "webots", "controllers", "a3_upper_body"))

import armkin
from kinematics.atlaskin import AtlasModel
from kinematics.legik import LegIK
from perception.fullbody import FullBodyRetargeter
from perception.gait_detect import GaitDetector
from perception.lift3d import Lifter3D
from perception.locomotion import LocomotionTracker
from perception.recording import Replay
from perception.skeleton import A3_INDEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default=os.path.join(
        A3_ROOT, "recordings", "testvideo_wb.jsonl"))
    parser.add_argument("--out", default=os.path.join(
        A3_ROOT, "recordings", "testvideo_angles.jsonl"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=1e9)
    args = parser.parse_args()

    model = AtlasModel()
    ik = {"L": LegIK(model, "L"), "R": LegIK(model, "R")}
    lifter = Lifter3D()
    print("Lifter:", lifter.describe())
    retarget = FullBodyRetargeter(ik, arm_solver=armkin.solve_arm, lifter=lifter)
    gait = GaitDetector()
    locomotion = LocomotionTracker()

    replay = Replay(args.recording, loop=False, realtime=False).load()
    rows = []
    dropped = 0
    while True:
        item = replay.read()
        if item is None:
            break
        entry, keypoints, scores = item
        if keypoints is None:
            dropped += 1
            continue
        if not (args.start <= entry["t"] <= args.end):
            continue
        state = gait.update(keypoints, scores, entry["t"])
        move = locomotion.update(keypoints, scores, entry["t"])
        torso_px = state.get("scale") or 0.0
        hip_px = float(0.5 * (keypoints[A3_INDEX["left_hip"]][0]
                              + keypoints[A3_INDEX["right_hip"]][0]))
        out = retarget(keypoints, scores, entry["t"])
        if out is None:
            dropped += 1
            continue
        rows.append({
            "t": round(float(entry["t"]), 4),
            "seq": out["seq"],
            "angles": {name: round(float(value), 5)
                       for name, value in out["angles"].items()},
            "cadence": round(float(state["cadence"]), 4),
            "stepping": bool(state["stepping"]),
            "forward": round(float(move["forward"]), 5),
            "lateral": round(float(move["lateral"]), 5),
            "moving": bool(move["moving"]),
            "torso_px": round(float(torso_px), 3),
            "hip_px": round(hip_px, 2),
            "leg_error_mm": round(float(max(
                (info["position_mm"] for info in out["quality"].values()),
                default=0.0)), 4),
        })

    with open(args.out, "w", encoding="utf-8", newline=chr(10)) as handle:
        handle.write(json.dumps({"format": "a3-angles", "version": 1,
                                 "source": os.path.basename(args.recording),
                                 "frames": len(rows)}) + chr(10))
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + chr(10))

    errors = np.array([r["leg_error_mm"] for r in rows])
    joints = sorted({name for r in rows for name in r["angles"]})
    print(f"frames        {len(rows)}   dropped {dropped}")
    print(f"time span     {rows[0]['t']:.2f} .. {rows[-1]['t']:.2f} s")
    print(f"joints        {len(joints)} of 28")
    print(f"leg IK resid  median {np.median(errors):.4f} mm, "
          f"p99 {np.percentile(errors, 99):.3f} mm, max {errors.max():.3f} mm")
    print(f"written       {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
