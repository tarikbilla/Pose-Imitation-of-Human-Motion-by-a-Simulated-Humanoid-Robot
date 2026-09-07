import argparse
import json
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception import runtime
from perception.pose2d import HALPE26_NAMES, Pose2D

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IMAGE = os.path.join(A3_ROOT, "assets", "testperson.jpg")
OUT_DIR = os.path.join(A3_ROOT, "results")

RUNS = 15
WARMUP = 3


def benchmark(image, mode, device):
    try:
        estimator = Pose2D(mode=mode, device=device)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if estimator.device != device:
        return {"ok": False, "error": estimator.warning or "device fallback"}

    for _ in range(WARMUP):
        estimator(image)

    timings = []
    result = None
    for _ in range(RUNS):
        start = time.perf_counter()
        result = estimator(image)
        timings.append((time.perf_counter() - start) * 1000.0)

    keypoints, scores = result.person()
    return {
        "ok": True,
        "device": estimator.device,
        "mode": mode,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "fps": 1000.0 / statistics.median(timings),
        "people": 0 if not result.found else len(result.keypoints),
        "mean_score": float(scores.mean()) if scores is not None else 0.0,
        "keypoints": 0 if keypoints is None else len(keypoints),
    }


def main():
    parser = argparse.ArgumentParser(description="A3 perception smoke test.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--mode", default="balanced", choices=("lightweight", "balanced", "performance"))
    parser.add_argument("--cpu-compare", action="store_true", help="also benchmark the CPU provider")
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="letterbox the test image to 1080x1920 to match the rotated Brio frame",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("M2 / perception smoke test")
    print("=" * 66)
    info = runtime.describe()
    print(f"onnxruntime            {info['onnxruntime_version']}")
    print(f"providers              {info['available_providers']}")
    if not info["directml"]:
        print("FATAL: DmlExecutionProvider missing")
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"FATAL: cannot read image {args.image}")
        return 1
    if args.portrait:
        import numpy as np

        target_w, target_h = 1080, 1920
        scale = min(target_w / image.shape[1], target_h / image.shape[0])
        resized = cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
        )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - resized.shape[0]) // 2
        x0 = (target_w - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        image = canvas

    print(f"image                  {args.image}  {image.shape[1]}x{image.shape[0]}")
    print(f"mode                   {args.mode}")
    print("-" * 66)
    print("loading models (first run downloads ~150 MB)...")

    report = {"image": args.image, "mode": args.mode, "runtime": info, "results": {}}

    dml = benchmark(image, args.mode, runtime.DEVICE_DML)
    report["results"]["dml"] = dml
    if dml["ok"]:
        print(
            f"DirectML   median {dml['median_ms']:7.2f} ms  ({dml['fps']:5.1f} fps)"
            f"  people={dml['people']} kpts={dml['keypoints']} score={dml['mean_score']:.3f}"
        )
    else:
        print(f"DirectML   FAILED: {dml['error']}")

    if args.cpu_compare:
        cpu = benchmark(image, args.mode, runtime.DEVICE_CPU)
        report["results"]["cpu"] = cpu
        if cpu["ok"]:
            print(
                f"CPU        median {cpu['median_ms']:7.2f} ms  ({cpu['fps']:5.1f} fps)"
                f"  people={cpu['people']} kpts={cpu['keypoints']} score={cpu['mean_score']:.3f}"
            )
            if dml["ok"]:
                speedup = cpu["median_ms"] / dml["median_ms"]
                report["dml_speedup"] = speedup
                print(f"speedup    {speedup:.2f}x")
        else:
            print(f"CPU        FAILED: {cpu['error']}")

    print("-" * 66)
    ok = dml["ok"] and dml["people"] > 0 and dml["keypoints"] == len(HALPE26_NAMES)
    print("VERDICT:", "PASS" if ok else "FAIL")
    print("=" * 66)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m2_perception.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
