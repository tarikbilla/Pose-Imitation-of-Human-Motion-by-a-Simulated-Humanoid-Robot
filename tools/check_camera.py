import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.capture import Camera, CameraProfile, apply_rotation

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

BACKENDS = (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF))
MEASURE_FRAMES = 60


def fourcc_to_text(value):
    value = int(value)
    if value <= 0:
        return "none"
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def probe_indices(limit=3):
    found = []
    for index in range(limit):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = capture.isOpened()
        if opened:
            ok, frame = capture.read()
            if ok and frame is not None:
                found.append({"index": index, "shape": list(frame.shape)})
        capture.release()
    return found


def main():
    parser = argparse.ArgumentParser(description="Verify the camera before the pose run.")
    parser.add_argument("--profile", default="brio100")
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--scan", action="store_true", help="scan camera indices 0-3")
    args = parser.parse_args()

    print("=" * 66)
    print("M2 / camera check")
    print("=" * 66)

    if args.scan:
        print("scanning indices 0-3 ...")
        for entry in probe_indices():
            print(f"    index {entry['index']}  frame {entry['shape'][1]}x{entry['shape'][0]}")
        print("-" * 66)

    try:
        profile = CameraProfile.load(args.profile)
    except FileNotFoundError:
        print(f"profile '{args.profile}' not found, using defaults")
        profile = CameraProfile(name=args.profile)
    if args.source is not None:
        profile.source = args.source

    print(f"profile           {profile.name}")
    print(f"requested         {profile.capture_width}x{profile.capture_height} "
          f"@ {profile.fps} fps  fourcc={profile.fourcc}  rotation={profile.rotation}")
    print("-" * 66)

    report = {"profile": profile.name, "requested": {
        "width": profile.capture_width, "height": profile.capture_height,
        "fps": profile.fps, "fourcc": profile.fourcc}, "backends": {}}

    working = None
    for label, backend in BACKENDS:
        camera = Camera(profile, backend=backend)
        try:
            camera.open()
        except RuntimeError as exc:
            print(f"{label:<8} cannot open: {exc}")
            report["backends"][label] = {"ok": False, "error": str(exc)}
            continue

        for _ in range(10):
            camera.read()

        settings = camera.actual_settings()
        frame = camera.read()
        if frame is None:
            print(f"{label:<8} opened but no frame")
            report["backends"][label] = {"ok": False, "error": "no frame"}
            camera.release()
            continue

        start = time.perf_counter()
        received = 0
        for _ in range(MEASURE_FRAMES):
            if camera.read() is not None:
                received += 1
        elapsed = time.perf_counter() - start
        measured_fps = received / elapsed if elapsed else 0.0

        entry = {
            "ok": True,
            "reported_width": settings["width"],
            "reported_height": settings["height"],
            "reported_fps": settings["fps"],
            "fourcc": fourcc_to_text(settings["fourcc"]),
            "rotated_shape": [frame.shape[1], frame.shape[0]],
            "measured_fps": measured_fps,
        }
        report["backends"][label] = entry

        print(f"{label:<8} sensor {settings['width']}x{settings['height']} "
              f"@ {settings['fps']:.0f} fps  fourcc={entry['fourcc']}")
        print(f"{'':<8} after rotation {frame.shape[1]}x{frame.shape[0]}  "
              f"measured {measured_fps:.1f} fps")

        if working is None:
            working = (label, camera, frame)
        else:
            camera.release()

    print("-" * 66)
    if working is None:
        print("FATAL: no working camera backend.")
        print("Check that the Brio is connected and not used by another app.")
        return 1

    label, camera, frame = working
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = apply_rotation(frame, -profile.rotation % 360) if profile.rotation else frame
    cv2.imwrite(os.path.join(OUT_DIR, "camera_raw.png"), raw)
    cv2.imwrite(os.path.join(OUT_DIR, "camera_rotated.png"), frame)
    camera.release()

    entry = report["backends"][label]
    issues = []
    if entry["reported_width"] != profile.capture_width:
        issues.append(
            f"resolution mismatch: asked {profile.capture_width}x{profile.capture_height}, "
            f"got {entry['reported_width']}x{entry['reported_height']}"
        )
    if entry["measured_fps"] < profile.fps * 0.8:
        issues.append(f"measured {entry['measured_fps']:.1f} fps, expected ~{profile.fps}")
    if entry["fourcc"].upper() not in (profile.fourcc.upper(), "MJPG"):
        issues.append(f"fourcc is {entry['fourcc']}, MJPG gives the best bandwidth")

    print(f"using backend     {label}")
    print(f"images written    results/camera_raw.png, results/camera_rotated.png")
    if issues:
        print()
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("all requested settings honoured")

    report["issues"] = issues
    report["backend_used"] = label
    with open(os.path.join(OUT_DIR, "m2_camera.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 66)
    print("Next: stand fully in frame, then run tools\\calibrate_rotation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
