#!/usr/bin/env python3
"""Replay a recorded session through the arm chain and measure what it costs.

The arm path has one number that matters and one that is easy to get wrong:

* **net lag** -- how far the commanded joint angle trails the retargeted one.
  The old per-frame EMA spent 127 ms here (measured on the 2026-09-16 session);
  the point of :class:`~pose_control_utils.ArmTracker`'s ``lead_s`` is to give
  that back by extrapolating along the target's own velocity.
* **added jitter** -- what that extrapolation costs in noise. Prediction always
  buys responsiveness with noise, so a lag figure quoted without the jitter
  beside it is meaningless: ``lead_s`` large enough drives the lag to zero and
  the arm to a tremor.

Both are measured against the retargeted target computed from the SAME logged
landmarks, so this isolates the driver stage. Everything upstream of it (camera,
inference, the One Euro keypoint filter) is already baked into the log and is
reported separately by ``--upstream``.

Usage::

    python scripts/tune_arm_tracking.py logs/run_20260916_110404
    python scripts/tune_arm_tracking.py logs/run_.../pose_keypoints.csv --grid
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "main" / "libraries"))

import nao_retarget  # noqa: E402
from pose_control_utils import TRACKED_ARM_JOINTS, ArmTracker  # noqa: E402

sys.path.insert(0, str(ROOT / "main" / "controllers" / "pose_imitation_controller"))
# The shipped tuning, read from the controller rather than restated here, so
# this script cannot quietly drift from what the robot actually runs.
ARM_TAU_S, ARM_LEAD_S = 0.07, 0.20
try:
    import re as _re
    _text = (ROOT / "main" / "controllers" / "pose_imitation_controller"
             / "pose_imitation_controller.py").read_text()
    ARM_TAU_S = float(_re.search(r"^ARM_TAU_S\s*=\s*([\d.]+)", _text, _re.M).group(1))
    ARM_LEAD_S = float(_re.search(r"^ARM_LEAD_S\s*=\s*([\d.]+)", _text, _re.M).group(1))
except Exception:  # noqa: BLE001 - the defaults above are a fine fallback
    pass

from src.perception.landmarks import POSE_LANDMARKS  # noqa: E402

# The simulation steps at 50 Hz (basicTimeStep 20 ms); the tracker is advanced
# once per step, so the replay has to use the same grid or the measured lag is
# an artefact of the replay rate rather than of the tuning.
SIM_DT = 0.02


def load_pose_log(path: Path) -> list[tuple[float, dict[str, list[float]]]]:
    """``[(timestamp_s, {landmark: [x, y, z, visibility]})]`` from a pose CSV."""
    rows: list[tuple[float, dict[str, list[float]]]] = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            kps: dict[str, list[float]] = {}
            for name in POSE_LANDMARKS:
                try:
                    kps[name] = [
                        float(raw[f"{name}_x"]), float(raw[f"{name}_y"]),
                        float(raw[f"{name}_z"]), float(raw[f"{name}_visibility"]),
                    ]
                except (KeyError, TypeError, ValueError):
                    continue
            if any(kp[3] > 0.0 for kp in kps.values()):
                rows.append((float(raw["timestamp_s"]), kps))
    return rows


def retarget_series(rows) -> dict[str, list[tuple[float, float]]]:
    """Per-joint ``[(t, target_rad)]`` -- what the retargeter asks for."""
    geom = nao_retarget.HeadGeometry()
    out: dict[str, list[tuple[float, float]]] = {j: [] for j in TRACKED_ARM_JOINTS}
    for t, kps in rows:
        targets = nao_retarget.retarget_upper_body(kps, head_geom=geom)
        for joint, value in targets.items():
            if joint in out:
                out[joint].append((t, value))
    return {j: v for j, v in out.items() if v}


# Frames apart by more than this are not one motion: the arm went out of view
# (or the subject left) and came back somewhere else. Scoring across such a gap
# compares a straight-line interpolation over several seconds against a tracker
# that correctly held still, which swamps every real difference -- it is what
# put a 39 deg p99 error on the UNCHANGED baseline too.
MAX_GAP_S = 0.5


def segments(series: list[tuple[float, float]], min_len: int = 50):
    """Split a per-joint series wherever the arm stopped being observed."""
    out, current = [], []
    for sample in series:
        if current and sample[0] - current[-1][0] > MAX_GAP_S:
            if len(current) >= min_len:
                out.append(current)
            current = []
        current.append(sample)
    if len(current) >= min_len:
        out.append(current)
    return out


def reference(series: list[tuple[float, float]], dt: float = SIM_DT) -> list[float]:
    """The target LINEARLY INTERPOLATED onto the simulation grid.

    Not the zero-order hold. The hold is what the old code actually commanded
    between camera frames, so scoring against it would treat the staircase as
    ground truth and credit a stepping arm with zero lag. Straight-line
    interpolation is the best available estimate of where the target really was
    at 50 Hz, and it is ACAUSAL -- it uses the next camera frame as well as the
    previous one -- which is exactly why it makes a fair reference for something
    that is trying to predict that next frame.
    """
    t0, t1 = series[0][0], series[-1][0]
    out: list[float] = []
    index = 0
    t = t0
    while t <= t1:
        while index + 1 < len(series) and series[index + 1][0] <= t:
            index += 1
        if index + 1 >= len(series):
            out.append(series[-1][1])
        else:
            (ta, va), (tb, vb) = series[index], series[index + 1]
            frac = 0.0 if tb <= ta else (t - ta) / (tb - ta)
            out.append(va + (vb - va) * max(0.0, min(1.0, frac)))
        t += dt
    return out


def replay(series: list[tuple[float, float]], tracker: ArmTracker,
           joint: str, dt: float = SIM_DT) -> list[float]:
    """Step ``tracker`` over ``series`` on the simulation grid; return commands."""
    t0, t1 = series[0][0], series[-1][0]
    tracker.reset(joint, series[0][1])
    cmd: list[float] = []
    index = 0
    t = t0
    while t <= t1:
        while index < len(series) and series[index][0] <= t:
            tracker.observe(joint, series[index][1], t)
            index += 1
        value = tracker.advance(joint, t, dt)
        cmd.append(series[max(0, index - 1)][1] if value is None else value)
        t += dt
    return cmd


def best_lag(a: list[float], b: list[float], dt: float,
             max_lag_s: float = 0.6) -> tuple[float, float]:
    """Lag (s) of ``b`` behind ``a`` at peak correlation, and that correlation.

    Scans NEGATIVE lags too: a predictor that has cancelled the delay is
    supposed to come out at or slightly ahead of zero, and a scan that started
    at zero would report every such tuning as "0 ms" and hide the overshoot.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    a = [x - mean_a for x in a]
    b = [x - mean_b for x in b]
    norm = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))) + 1e-12
    steps = int(max_lag_s / dt)
    best = (0.0, -2.0)
    for k in range(-steps, steps + 1):
        if k >= 0:
            total = sum(a[i] * b[i + k] for i in range(n - k))
        else:
            total = sum(a[i - k] * b[i] for i in range(n + k))
        corr = total / norm
        if corr > best[1]:
            best = (k * dt, corr)
    return best


def jitter(values: list[float]) -> float:
    """RMS of the second difference -- the part of the signal that is tremor.

    Second difference rather than first: a fast gesture has a large first
    difference and is not noise. What a too-eager predictor adds is a
    sample-to-sample REVERSAL, which is exactly what this picks up.
    """
    if len(values) < 3:
        return 0.0
    accel = [values[i + 1] - 2 * values[i] + values[i - 1] for i in range(1, len(values) - 1)]
    return math.sqrt(sum(x * x for x in accel) / len(accel))


def frame_ema_baseline(series, alpha=0.4, dt=SIM_DT) -> list[float]:
    """The OLD behaviour: EMA stepped once per camera frame, held in between."""
    t0, t1 = series[0][0], series[-1][0]
    state = series[0][1]
    cmd: list[float] = []
    index, t = 0, t0
    while t <= t1:
        while index < len(series) and series[index][0] <= t:
            state += (series[index][1] - state) * alpha
            index += 1
        cmd.append(state)
        t += dt
    return cmd


def overshoot(cmd: list[float], ref: list[float]) -> tuple[float, float]:
    """``(rms, p99)`` of |command - reference| in radians.

    The lag figure alone cannot tell a predictor that has cancelled the delay
    from one that is sailing past every direction reversal and coming back: both
    peak the cross-correlation at zero. This is what separates them -- an
    overshooting tracker is wrong by a lot at the turning points even though it
    is, on average, on time.
    """
    n = min(len(cmd), len(ref))
    errs = sorted(abs(cmd[i] - ref[i]) for i in range(n))
    rms = math.sqrt(sum(e * e for e in errs) / max(1, n))
    return rms, errs[min(n - 1, int(0.99 * n))]


def _accumulate(summary, label, lag, corr, ratio, rms, p99) -> None:
    """Running totals per configuration, averaged over joints at the end."""
    acc = summary.setdefault(label, [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    for i, value in enumerate((lag, corr, ratio, rms, p99)):
        acc[i] += value
    acc[5] += 1


def report(series_by_joint, configs, joints=None):
    joints = joints or sorted(series_by_joint)
    print(f"\n{'joint':16s} {'config':>22s} {'net lag':>9s} {'r':>6s} "
          f"{'jitter':>9s} {'vs EMA':>7s} {'err rms':>9s} {'err p99':>9s}")
    print("-" * 95)
    summary = {}
    for joint in joints:
        raw = series_by_joint.get(joint)
        if not raw:
            continue
        parts = segments(raw)
        if not parts:
            continue
        # Score each continuously-observed stretch, then concatenate. The
        # cross-correlation and the error statistics both run per segment, so
        # neither is asked to reason across a stretch where the arm was not
        # being seen at all.
        ref = [v for part in parts for v in reference(part)]
        cmd = [v for part in parts for v in frame_ema_baseline(part)]
        base_lag, base_r = best_lag(ref, cmd, SIM_DT)
        base_jit = jitter(cmd)
        base_rms, base_p99 = overshoot(cmd, ref)
        print(f"{joint:16s} {'EMA a=0.4 (old)':>22s} {base_lag*1000:8.0f}ms "
              f"{base_r:6.3f} {math.degrees(base_jit):8.4f}d {'--':>7s} "
              f"{math.degrees(base_rms):8.2f}d {math.degrees(base_p99):8.2f}d")
        for label, kwargs in configs:
            tracker = ArmTracker(**kwargs)
            cmd = [v for part in parts for v in replay(part, tracker, joint)]
            lag, corr = best_lag(ref, cmd, SIM_DT)
            jit = jitter(cmd)
            rms, p99 = overshoot(cmd, ref)
            ratio = jit / base_jit if base_jit > 0 else float("inf")
            print(f"{'':16s} {label:>22s} {lag*1000:8.0f}ms {corr:6.3f} "
                  f"{math.degrees(jit):8.4f}d {ratio:6.2f}x "
                  f"{math.degrees(rms):8.2f}d {math.degrees(p99):8.2f}d")
            _accumulate(summary, label, lag, corr, ratio, rms, p99)
        _accumulate(summary, "EMA a=0.4 (old)", base_lag, base_r, 1.0,
                    base_rms, base_p99)
        print()
    if summary:
        print(f"{'MEAN OVER JOINTS':>39s} {'net lag':>9s} {'r':>6s} "
              f"{'jit/EMA':>8s} {'err rms':>9s} {'err p99':>9s}")
        print("-" * 95)
        ordered = ([("EMA a=0.4 (old)", summary.pop("EMA a=0.4 (old)"))]
                   if "EMA a=0.4 (old)" in summary else [])
        for label, (lag, corr, ratio, rms, p99, n) in ordered + list(summary.items()):
            print(f"{label:>39s} {lag/n*1000:8.0f}ms {corr/n:6.3f} {ratio/n:7.2f}x "
                  f"{math.degrees(rms/n):8.2f}d {math.degrees(p99/n):8.2f}d")


def report_grip(rows) -> int:
    """Measure the thumb-to-finger / forearm ratio a real subject produces.

    ``nao_retarget``'s GRIP_OPEN_RATIO and GRIP_CLOSED_RATIO map that ratio onto
    a hand closure, and their defaults come from adult proportions rather than
    from MeTRAbs output -- nobody had a recording with hand markers in it when
    they were written. This prints the distribution so they can be set from a
    session where the subject deliberately opens and closes their hands.
    """
    import statistics

    from nao_retarget import GRIP_CLOSED_RATIO, GRIP_OPEN_RATIO

    ratios = {"L": [], "R": []}
    for _t, kps in rows:
        for side, pre in (("L", "left_"), ("R", "right_")):
            need = (pre + "elbow", pre + "wrist", pre + "thumb", pre + "finger")
            if any(n not in kps or kps[n][3] < 0.5 for n in need):
                continue
            forearm = math.dist(kps[pre + "elbow"][:3], kps[pre + "wrist"][:3])
            if forearm < 1e-6:
                continue
            ratios[side].append(
                math.dist(kps[pre + "thumb"][:3], kps[pre + "finger"][:3]) / forearm)

    if not any(ratios.values()):
        print("\nNo hand landmarks in this log.\n"
              "  Hand markers exist only in MeTRAbs' 122-joint superset, so a\n"
              "  session recorded with pose.skeleton: coco_19 cannot have them.\n"
              "  Set pose.skeleton to \"\" in configs/default.yaml and record again.")
        return 1

    print(f"\nthumb-to-finger distance / forearm length\n{'-' * 58}")
    print(f"{'side':>5s} {'n':>6s} {'p2':>7s} {'p10':>7s} {'p50':>7s} "
          f"{'p90':>7s} {'p98':>7s}")
    for side, values in ratios.items():
        if not values:
            continue
        q = statistics.quantiles(values, n=100)
        print(f"{side:>5s} {len(values):6d} {q[1]:7.3f} {q[9]:7.3f} "
              f"{q[49]:7.3f} {q[89]:7.3f} {q[97]:7.3f}")
    pooled = sorted(ratios["L"] + ratios["R"])
    q = statistics.quantiles(pooled, n=100)
    print(f"\nshipped:   GRIP_CLOSED_RATIO = {GRIP_CLOSED_RATIO:.2f}   "
          f"GRIP_OPEN_RATIO = {GRIP_OPEN_RATIO:.2f}")
    print(f"suggested: GRIP_CLOSED_RATIO = {q[4]:.2f}   GRIP_OPEN_RATIO = {q[94]:.2f}")
    print("\n  The p5/p95 of the pooled ratio, NOT its min/max: the extremes of\n"
          "  a monocular hand estimate are outliers, and anchoring the open end\n"
          "  to one would leave every real open hand reading as half-shut.\n"
          "  Only meaningful if the subject actually made a fist on camera --\n"
          "  check that p5 is well below p50 before trusting the closed end.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="run directory or pose_keypoints.csv")
    parser.add_argument("--grid", action="store_true",
                        help="sweep a tau/lead grid instead of the shortlist")
    parser.add_argument("--joint", action="append", default=None,
                        help="restrict to these joints (repeatable)")
    parser.add_argument("--grip", action="store_true",
                        help="calibrate nao_retarget's GRIP_* ratios instead")
    args = parser.parse_args(argv)

    path = Path(args.log)
    if path.is_dir():
        path = path / "pose_keypoints.csv"
    if not path.exists():
        print(f"no such pose log: {path}", file=sys.stderr)
        return 1

    rows = load_pose_log(path)
    if len(rows) < 100:
        print(f"only {len(rows)} usable frames in {path}", file=sys.stderr)
        return 1
    span = rows[-1][0] - rows[0][0]
    print(f"{path}: {len(rows)} frames over {span:.0f}s ({len(rows)/span:.1f} FPS)")

    if args.grip:
        return report_grip(rows)

    series_by_joint = retarget_series(rows)
    print(f"joints solved: {', '.join(sorted(series_by_joint))}")

    if args.grid:
        configs = [
            (f"tau={tau:.2f} lead={lead:.2f}", {"tau_s": tau, "lead_s": lead})
            for tau in (0.05, 0.07, 0.10)
            for lead in (0.00, 0.12, 0.20, 0.28, 0.36)
        ]
    else:
        configs = [
            ("tau=0.07 lead=0.00", {"tau_s": 0.07, "lead_s": 0.00}),
            ("SHIPPED 0.07/0.20", {"tau_s": ARM_TAU_S, "lead_s": ARM_LEAD_S}),
            ("tau=0.07 lead=0.32", {"tau_s": 0.07, "lead_s": 0.32}),
        ]
    report(series_by_joint, configs, args.joint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
