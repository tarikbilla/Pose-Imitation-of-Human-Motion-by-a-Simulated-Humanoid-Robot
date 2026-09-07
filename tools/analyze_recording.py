import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.pose2d import HALPE26_INDEX, HALPE26_NAMES
from perception.recording import Replay

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

GOOD = 0.5
FOOT_KEYS = (
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe",
)
CORE_KEYS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def load(path):
    replay = Replay(path).load()
    keypoints = []
    scores = []
    times = []
    for entry in replay.entries:
        times.append(entry["t"])
        if entry.get("kp") is None:
            keypoints.append(np.full((len(HALPE26_NAMES), 2), np.nan))
            scores.append(np.zeros(len(HALPE26_NAMES)))
        else:
            keypoints.append(np.asarray(entry["kp"], dtype=np.float32))
            scores.append(np.asarray(entry["sc"], dtype=np.float32))
    return replay, np.asarray(times), np.asarray(keypoints), np.asarray(scores)


def longest_gap(detected):
    best = current = 0
    for value in detected:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def jitter(keypoints, scores, name):
    index = HALPE26_INDEX[name]
    track = keypoints[:, index, :]
    valid = scores[:, index] >= GOOD
    deltas = []
    for i in range(1, len(track)):
        if valid[i] and valid[i - 1]:
            deltas.append(np.linalg.norm(track[i] - track[i - 1]))
    return float(np.median(deltas)) if deltas else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Assess an A3 keypoint recording.")
    parser.add_argument("path")
    parser.add_argument("--window", type=float, default=5.0,
                        help="seconds per timeline bucket")
    args = parser.parse_args()

    replay, times, keypoints, scores = load(args.path)
    frames = len(times)
    if not frames:
        print("empty recording")
        return 1

    detected = scores.max(axis=1) > 0
    duration = times[-1] if frames else 0.0

    print("=" * 70)
    print("A3 / recording analysis")
    print("=" * 70)
    print(f"file            {os.path.basename(args.path)}")
    print(f"frames          {frames}   duration {duration:.1f} s")
    print(f"source          {replay.header.get('source', {}).get('path', 'n/a')}")
    print(f"detection rate  {detected.mean() * 100:.1f}%")
    gap = longest_gap(detected)
    print(f"longest gap     {gap} frames ({gap / 30:.2f} s at 30 fps)")
    print("-" * 70)

    print("per-keypoint confidence (mean over detected frames)")
    print(f"{'keypoint':<18}{'mean':>8}{'>=0.5':>9}{'jitter px':>12}")
    order = []
    for name in HALPE26_NAMES:
        index = HALPE26_INDEX[name]
        column = scores[detected, index]
        order.append((float(column.mean()), name, float((column >= GOOD).mean())))
    for mean, name, share in sorted(order):
        flag = "  <-- weak" if mean < 0.5 else ""
        print(f"{name:<18}{mean:>8.3f}{share * 100:>8.0f}%{jitter(keypoints, scores, name):>12.2f}{flag}")

    print("-" * 70)
    foot_means = [np.mean(scores[detected, HALPE26_INDEX[n]]) for n in FOOT_KEYS]
    core_means = [np.mean(scores[detected, HALPE26_INDEX[n]]) for n in CORE_KEYS]
    print(f"core joints     mean confidence {np.mean(core_means):.3f}")
    print(f"foot points     mean confidence {np.mean(foot_means):.3f}")

    print("-" * 70)
    print("timeline")
    print(f"{'window':<14}{'detect':>9}{'core':>9}{'feet':>9}")
    buckets = max(1, int(np.ceil(duration / args.window)))
    timeline = []
    for bucket in range(buckets):
        start = bucket * args.window
        end = start + args.window
        mask = (times >= start) & (times < end)
        if not mask.any():
            continue
        sub_detected = detected & mask
        rate = detected[mask].mean()
        if sub_detected.any():
            core = np.mean([scores[sub_detected, HALPE26_INDEX[n]].mean() for n in CORE_KEYS])
            feet = np.mean([scores[sub_detected, HALPE26_INDEX[n]].mean() for n in FOOT_KEYS])
        else:
            core = feet = 0.0
        timeline.append({"start": start, "detection": float(rate),
                         "core": float(core), "feet": float(feet)})
        print(f"{start:>5.0f}-{end:<7.0f}{rate * 100:>8.0f}%{core:>9.3f}{feet:>9.3f}")

    print("=" * 70)
    verdict = []
    if detected.mean() < 0.95:
        verdict.append(f"detection rate {detected.mean() * 100:.0f}% is below 95%")
    if np.mean(foot_means) < 0.5:
        verdict.append(f"foot confidence {np.mean(foot_means):.2f} is weak - "
                       "foot-lift events will be unreliable")
    if gap > 15:
        verdict.append(f"a {gap}-frame dropout will show as a jump in the robot")
    if verdict:
        for line in verdict:
            print(f"  ISSUE: {line}")
    else:
        print("  recording is usable for retargeting")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "file": args.path,
        "frames": frames,
        "duration_s": float(duration),
        "detection_rate": float(detected.mean()),
        "longest_gap_frames": int(gap),
        "core_confidence": float(np.mean(core_means)),
        "foot_confidence": float(np.mean(foot_means)),
        "per_keypoint": {name: {"mean": m, "above_threshold": s}
                         for m, name, s in order},
        "timeline": timeline,
        "issues": verdict,
    }
    with open(os.path.join(OUT_DIR, "recording_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
