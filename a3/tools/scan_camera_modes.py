import argparse
import json
import os
import sys
import time

import cv2

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

MODES = (
    (1920, 1080, "MJPG"),
    (1920, 1080, "YUY2"),
    (1280, 720, "MJPG"),
    (1280, 720, "YUY2"),
    (960, 540, "MJPG"),
    (640, 480, "MJPG"),
)

ORDERS = ("fourcc_first", "size_first")
WARMUP = 8
MEASURE = 45


def fourcc_to_text(value):
    value = int(value)
    if value <= 0:
        return "none"
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def configure(capture, width, height, fourcc, order):
    code = cv2.VideoWriter_fourcc(*fourcc)
    if order == "fourcc_first":
        capture.set(cv2.CAP_PROP_FOURCC, code)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FOURCC, code)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def measure(source, backend, width, height, fourcc, order):
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        return None
    configure(capture, width, height, fourcc, order)

    for _ in range(WARMUP):
        capture.read()

    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))

    start = time.perf_counter()
    received = 0
    for _ in range(MEASURE):
        ok, frame = capture.read()
        if ok and frame is not None:
            received += 1
    elapsed = time.perf_counter() - start
    capture.release()

    return {
        "requested": {"width": width, "height": height, "fourcc": fourcc, "order": order},
        "actual": {"width": actual_w, "height": actual_h, "fourcc": actual_fourcc},
        "fps": received / elapsed if elapsed else 0.0,
        "frames": received,
    }


def main():
    parser = argparse.ArgumentParser(description="Find a camera mode that sustains 30 fps.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--backend", default="dshow", choices=("dshow", "msmf"))
    args = parser.parse_args()

    backend = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_MSMF

    print("=" * 74)
    print(f"M2 / camera mode scan   source={args.source}  backend={args.backend}")
    print("=" * 74)
    print(f"{'requested':<22}{'order':<14}{'actual':<22}{'fps':>8}")
    print("-" * 74)

    results = []
    for width, height, fourcc in MODES:
        for order in ORDERS:
            entry = measure(args.source, backend, width, height, fourcc, order)
            if entry is None:
                print(f"{width}x{height} {fourcc:<8} {order:<14} cannot open")
                continue
            results.append(entry)
            actual = entry["actual"]
            print(
                f"{width}x{height} {fourcc:<8}"
                f"{order:<14}"
                f"{actual['width']}x{actual['height']} {actual['fourcc']:<8}"
                f"{entry['fps']:>8.1f}"
            )

    print("-" * 74)
    usable = [r for r in results if r["fps"] >= 24.0]
    if usable:
        usable.sort(
            key=lambda r: (r["actual"]["width"] * r["actual"]["height"], r["fps"]),
            reverse=True,
        )
        best = usable[0]
        print("BEST MODE")
        print(f"  {best['actual']['width']}x{best['actual']['height']} "
              f"{best['actual']['fourcc']} at {best['fps']:.1f} fps")
        print(f"  set fourcc {best['requested']['fourcc']} with order "
              f"{best['requested']['order']}")
    else:
        best = max(results, key=lambda r: r["fps"]) if results else None
        print("WARNING: no mode reached 24 fps.")
        if best:
            print(f"  fastest was {best['actual']['width']}x{best['actual']['height']} "
                  f"{best['actual']['fourcc']} at {best['fps']:.1f} fps")
        print("  Check that the camera is on a USB 3 port (blue connector).")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m2_camera_modes.json"), "w", encoding="utf-8") as handle:
        json.dump({"backend": args.backend, "results": results}, handle, indent=2)
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
