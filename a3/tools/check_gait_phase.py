import argparse
import os
import sys

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)

from perception.fullbody import to_body, THIGH, SHANK, HIP_OFFSET
from perception.lift3d import Lifter3D, H36M
from perception.recording import Replay


def leg_offset(points, side):
    prefix = "left" if side == "L" else "right"
    hip = points[H36M[prefix + "_hip"]]
    knee = points[H36M[prefix + "_knee"]]
    ankle = points[H36M[prefix + "_ankle"]]
    thigh = to_body(knee - hip)
    shank = to_body(ankle - knee)
    if thigh is None or shank is None:
        return None
    origin = HIP_OFFSET.copy()
    if side == "R":
        origin[1] = -origin[1]
    return origin + thigh * THIGH + shank * SHANK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default=os.path.join(
        A3_ROOT, "recordings", "testvideo_wb.jsonl"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=1e9)
    args = parser.parse_args()

    lifter = Lifter3D()
    replay = Replay(args.recording, loop=False, realtime=False).load()
    times, feet, sizes = [], {"L": [], "R": []}, []
    from perception.skeleton import A3_INDEX
    while True:
        item = replay.read()
        if item is None:
            break
        entry, keypoints, scores = item
        if keypoints is None or not (args.start <= entry["t"] <= args.end):
            continue
        points = lifter(keypoints, scores)
        if points is None:
            continue
        offsets = {side: leg_offset(np.asarray(points), side)
                   for side in ("L", "R")}
        if any(value is None for value in offsets.values()):
            continue
        times.append(float(entry["t"]))
        for side in ("L", "R"):
            feet[side].append(offsets[side])
        sizes.append(abs(float(0.5 * (keypoints[A3_INDEX["left_ankle"]][1]
                                      + keypoints[A3_INDEX["right_ankle"]][1])
                             - keypoints[A3_INDEX["nose"]][1])))

    times = np.array(times)
    sizes = np.array(sizes)
    print(f"frames {len(times)}  time {times[0]:.2f} .. {times[-1]:.2f} s")
    print()
    for side in ("L", "R"):
        data = np.array(feet[side])
        x = data[:, 0]
        z = data[:, 2]
        height = z - z.min()
        dx = np.gradient(x, times)
        swing = height > np.percentile(height, 70)
        stance = height < np.percentile(height, 30)
        corr = float(np.corrcoef(height, dx)[0, 1])
        print(f"foot {side}:  sagittal {x.min():+.3f} .. {x.max():+.3f} m   "
              f"lift {height.max()*1000:.0f} mm")
        print(f"          dx/dt swing {dx[swing].mean():+.4f} m/s   "
              f"stance {dx[stance].mean():+.4f} m/s")
        print(f"          correlation height vs dx/dt  {corr:+.4f}"
              f"   -> {'FORWARD (correct)' if corr > 0 else 'BACKWARD (inverted)'}")
        print()
    scale_rate = np.gradient(sizes, times)
    print(f"apparent size: {sizes.min():.0f} .. {sizes.max():.0f} px  "
          f"(larger = closer to the camera)")
    lo, hi = int(0.05 * len(sizes)), int(0.95 * len(sizes))
    print(f"  start {sizes[:20].mean():.0f} px -> end {sizes[-20:].mean():.0f} px  "
          f"=> subject net {'CLOSER (forward)' if sizes[-20:].mean() > sizes[:20].mean() else 'FURTHER AWAY (backward)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
