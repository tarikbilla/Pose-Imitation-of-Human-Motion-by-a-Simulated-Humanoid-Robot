import json
import os
import sys
import time

import numpy as np

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)

from kinematics.atlaskin import SOLE_OFFSET, AtlasModel
from kinematics.legik import LegIK

SAMPLES = 2000
WALK_RANGE = {"LegUhz": 0.25, "LegMhx": 0.30, "LegLhy": 0.80,
              "LegKny": 1.20, "LegUay": 0.50, "LegLax": 0.30}


def sample(rng, ik):
    values = []
    for name in ik.names:
        span = WALK_RANGE[name[1:]]
        low, high = ik.model.limits[name]
        if name.endswith("Kny"):
            values.append(float(rng.uniform(0.0, span)))
        else:
            values.append(float(np.clip(rng.uniform(-span, span), low, high)))
    return np.array(values)


def cross_check(model, ik, rng, count=200):
    worst = 0.0
    neutral = {name: 0.0 for name in model.names}
    for _ in range(count):
        values = sample(rng, ik)
        angles = dict(neutral)
        for name, value in zip(ik.names, values):
            angles[name] = float(value)
        poses = model.frames(angles)
        origin, rotation = poses[ik.names[-1]]
        reference = origin + rotation @ SOLE_OFFSET
        fast, _ = ik.pose(values)
        worst = max(worst, float(np.linalg.norm(fast - reference)))
    return worst


def main():
    model = AtlasModel()
    rng = np.random.default_rng(20260905)
    report = {}

    for side in ("L", "R"):
        ik = LegIK(model, side)
        drift = cross_check(model, ik, rng)

        position_errors = []
        angle_errors = []
        cold = []
        warm = []
        seed = None

        for _ in range(SAMPLES):
            truth = sample(rng, ik)
            target_position, target_rotation = ik.pose(truth)

            started = time.perf_counter()
            solved, position_error, angle_error = ik.solve(
                target_position, target_rotation, seed=None)
            cold.append(time.perf_counter() - started)

            started = time.perf_counter()
            ik.solve(target_position, target_rotation, seed=seed)
            warm.append(time.perf_counter() - started)
            seed = solved

            position_errors.append(position_error)
            angle_errors.append(angle_error)

        track_position = []
        track_time = []
        base = np.array([0.0, 0.0, -0.45, 0.90, -0.45, 0.0])
        target_position, target_rotation = ik.pose(base)
        track_seed = base.copy()
        for _ in range(1500):
            target_position = target_position + rng.normal(0.0, 0.0015, 3)
            started = time.perf_counter()
            track_seed, error, _ = ik.solve(target_position, target_rotation,
                                            seed=track_seed)
            track_time.append(time.perf_counter() - started)
            track_position.append(error)

        position_errors = np.array(position_errors) * 1000.0
        angle_errors = np.degrees(np.array(angle_errors))
        report[side] = {
            "samples": SAMPLES,
            "fk_cross_check_m": round(drift, 12),
            "position_median_mm": round(float(np.median(position_errors)), 5),
            "position_p99_mm": round(float(np.percentile(position_errors, 99)), 5),
            "position_max_mm": round(float(position_errors.max()), 5),
            "angle_p99_deg": round(float(np.percentile(angle_errors, 99)), 5),
            "under_0p1mm_percent": round(float((position_errors < 0.1).mean()) * 100, 2),
            "cold_median_ms": round(float(np.median(cold)) * 1000, 3),
            "warm_median_ms": round(float(np.median(warm)) * 1000, 3),
            "tracking_median_ms": round(float(np.median(track_time)) * 1000, 3),
            "tracking_p99_ms": round(float(np.percentile(track_time, 99)) * 1000, 3),
            "tracking_p99_mm": round(float(np.percentile(track_position, 99)) * 1000, 5),
        }

    print(f"{'Seite':6s}{'FK-Abgleich':>14s}{'Pos med':>10s}{'Pos p99':>10s}"
          f"{'ang p99':>10s}{'<0.1mm':>9s}{'cold':>8s}{'tracking':>12s}")
    for side, entry in report.items():
        print(f"{side:6s}{entry['fk_cross_check_m']:14.2e}"
              f"{entry['position_median_mm']:9.4f}m{entry['position_p99_mm']:9.4f}m"
              f"{entry['angle_p99_deg']:9.4f}d{entry['under_0p1mm_percent']:8.1f}%"
              f"{entry['cold_median_ms']:7.2f}m"
              f"{entry['tracking_median_ms']:9.3f}ms")

    os.makedirs(os.path.join(A3_ROOT, "results"), exist_ok=True)
    with open(os.path.join(A3_ROOT, "results", "legik_check.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    ok = all(entry["position_p99_mm"] < 0.5
             and entry["fk_cross_check_m"] < 1e-9
             and entry["tracking_median_ms"] < 1.0
             and entry["tracking_p99_mm"] < 0.5
             for entry in report.values())
    print()
    for side, entry in report.items():
        print(f"  {side}: tracking median {entry['tracking_median_ms']:.3f} ms, "
              f"p99 {entry['tracking_p99_ms']:.3f} ms, "
              f"residual p99 {entry['tracking_p99_mm']:.4f} mm")
    print("criterion: p99 < 0.5 mm, FK cross-check < 1e-9 m, tracking < 1 ms  ->",
          "bestanden" if ok else "NICHT bestanden")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
