import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.filters import KeypointFilter
from perception.recording import Replay

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

CANDIDATES = (
    (0.0, 0.000),
    (0.5, 0.003),
    (1.0, 0.007),
    (1.5, 0.007),
    (2.0, 0.010),
    (3.0, 0.020),
    (5.0, 0.050),
)


def load(path):
    replay = Replay(path).load()
    times, keypoints, scores = [], [], []
    for entry in replay.entries:
        if entry.get("kp") is None:
            continue
        times.append(entry["t"])
        keypoints.append(entry["kp"])
        scores.append(entry["sc"])
    return (np.asarray(times), np.asarray(keypoints, dtype=np.float64),
            np.asarray(scores, dtype=np.float64))


def jitter_of(track, valid):
    deltas = []
    for i in range(1, len(track)):
        if valid[i] and valid[i - 1]:
            deltas.append(np.linalg.norm(track[i] - track[i - 1]))
    return float(np.median(deltas)) if deltas else float("nan")


def speed_of(track, valid):
    speeds = []
    for i in range(1, len(track)):
        if valid[i] and valid[i - 1]:
            speeds.append(np.linalg.norm(track[i] - track[i - 1]))
    if not speeds:
        return 0.0
    return float(np.percentile(speeds, 90))


def main():
    parser = argparse.ArgumentParser(description="Tune the One-Euro filter on a recording.")
    parser.add_argument("recording")
    args = parser.parse_args()

    times, raw, scores = load(args.recording)
    frames, count, _ = raw.shape
    valid = scores >= 0.3

    print("=" * 78)
    print("A3 / One-Euro filter tuning")
    print("=" * 78)
    print(f"recording   {os.path.basename(args.recording)}   {frames} frames, {count} keypoints")

    baseline_jitter = np.nanmedian([jitter_of(raw[:, k], valid[:, k]) for k in range(count)])
    fast = np.nanmedian([speed_of(raw[:, k], valid[:, k]) for k in range(count)])
    print(f"raw jitter (median over keypoints)   {baseline_jitter:.2f} px")
    print(f"raw 90th-pct motion per frame        {fast:.2f} px")
    print("-" * 78)
    print(f"{'min_cutoff':>11}{'beta':>8}{'jitter':>10}{'reduction':>11}{'lag proxy':>12}{'max dev':>10}")

    results = []
    for min_cutoff, beta in CANDIDATES:
        if min_cutoff <= 0.0:
            results.append({"min_cutoff": 0.0, "beta": 0.0,
                            "jitter": float(baseline_jitter), "reduction": 0.0,
                            "lag_proxy": 0.0, "max_deviation": 0.0})
            print(f"{'off':>11}{'-':>8}{baseline_jitter:>10.2f}{'0%':>11}{0.0:>12.2f}{0.0:>10.2f}")
            continue

        filt = KeypointFilter(count, min_cutoff=min_cutoff, beta=beta)
        smoothed = np.zeros_like(raw)
        for i in range(frames):
            out, _ = filt(raw[i], scores[i], times[i])
            smoothed[i] = out

        jitter = np.nanmedian([jitter_of(smoothed[:, k], valid[:, k]) for k in range(count)])
        deviation = np.linalg.norm(smoothed - raw, axis=2)
        lag = float(np.median(deviation[valid]))
        worst = float(np.percentile(deviation[valid], 99))
        reduction = (1.0 - jitter / baseline_jitter) * 100.0

        results.append({"min_cutoff": min_cutoff, "beta": beta,
                        "jitter": float(jitter), "reduction": float(reduction),
                        "lag_proxy": lag, "max_deviation": worst})
        print(f"{min_cutoff:>11.1f}{beta:>8.3f}{jitter:>10.2f}{reduction:>10.0f}%"
              f"{lag:>12.2f}{worst:>10.2f}")

    print("-" * 78)
    usable = [r for r in results if r["min_cutoff"] > 0 and r["lag_proxy"] <= 2.0]
    if usable:
        best = max(usable, key=lambda r: r["reduction"])
        print(f"RECOMMENDED  min_cutoff={best['min_cutoff']}  beta={best['beta']}")
        print(f"             jitter {baseline_jitter:.2f} -> {best['jitter']:.2f} px "
              f"({best['reduction']:.0f}% less), median deviation {best['lag_proxy']:.2f} px")
    print("=" * 78)
    print("lag proxy = median distance between filtered and raw position.")
    print("Keep it well below the 90th-pct motion, or fast moves get dragged.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "filter_tuning.json"), "w", encoding="utf-8") as handle:
        json.dump({"raw_jitter": float(baseline_jitter),
                   "raw_motion_p90": float(fast),
                   "candidates": results}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
