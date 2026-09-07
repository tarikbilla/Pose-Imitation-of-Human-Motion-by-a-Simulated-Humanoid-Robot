import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception import runtime
from perception.capture import Camera, CameraProfile, apply_rotation
from perception.overlay import draw_hud, draw_skeleton
from perception.pose2d import HALPE26_INDEX, Pose2D

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

CANDIDATES = (0, 90, 180, 270)
FRAMES_PER_CANDIDATE = 8
UPRIGHT_KEYS = ("head", "neck", "left_ankle", "right_ankle")


def uprightness(keypoints, scores):
    head = HALPE26_INDEX["head"]
    left = HALPE26_INDEX["left_ankle"]
    right = HALPE26_INDEX["right_ankle"]
    if min(scores[head], scores[left], scores[right]) < 0.3:
        return 0.0
    ankle_y = (keypoints[left][1] + keypoints[right][1]) / 2.0
    ankle_x = (keypoints[left][0] + keypoints[right][0]) / 2.0
    dy = ankle_y - keypoints[head][1]
    dx = abs(ankle_x - keypoints[head][0])
    span = abs(dy) + dx + 1e-6
    return max(0.0, dy) / span


def score_rotation(estimator, frames, degrees):
    detections = 0
    confidences = []
    uprights = []
    for frame in frames:
        rotated = apply_rotation(frame, degrees)
        result = estimator(rotated)
        keypoints, scores = result.person()
        if keypoints is None:
            continue
        detections += 1
        confidences.append(float(np.mean(scores)))
        uprights.append(uprightness(keypoints, scores))

    if not detections:
        return {"degrees": degrees, "detection_rate": 0.0, "confidence": 0.0,
                "uprightness": 0.0, "score": 0.0}

    rate = detections / len(frames)
    confidence = float(np.mean(confidences))
    upright = float(np.mean(uprights))
    return {
        "degrees": degrees,
        "detection_rate": rate,
        "confidence": confidence,
        "uprightness": upright,
        "score": rate * confidence * (0.35 + 0.65 * upright),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find the camera rotation that makes the subject upright."
    )
    parser.add_argument("--profile", default="brio100")
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--mode", default="lightweight")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--preview", action="store_true", help="show the winning frame")
    args = parser.parse_args()

    try:
        profile = CameraProfile.load(args.profile)
    except FileNotFoundError:
        print(f"profile '{args.profile}' not found, using defaults")
        profile = CameraProfile(name=args.profile)
    if args.source is not None:
        profile.source = args.source

    print("=" * 66)
    print("M2 / camera rotation calibration")
    print("=" * 66)
    print(f"profile           {profile.name}  source={profile.source}")
    print("Stand fully in frame, upright, arms slightly away from the body.")
    print("-" * 66)

    estimator = Pose2D(mode=args.mode, device=runtime.DEVICE_DML)
    if estimator.warning:
        print(f"WARNING: {estimator.warning}")
    estimator.warmup()

    probe = CameraProfile(**{**profile.__dict__, "rotation": 0})
    frames = []
    with Camera(probe) as camera:
        settings = camera.actual_settings()
        print(f"camera reports    {settings.get('width')}x{settings.get('height')} @ {settings.get('fps')}")
        for _ in range(10):
            camera.read()
        while len(frames) < FRAMES_PER_CANDIDATE:
            frame = camera.read()
            if frame is not None:
                frames.append(frame)

    if not frames:
        print("FATAL: no frames captured")
        return 1

    print("-" * 66)
    print(f"{'rotation':>10}{'detect':>10}{'confid':>10}{'upright':>10}{'score':>10}")
    results = []
    for degrees in CANDIDATES:
        entry = score_rotation(estimator, frames, degrees)
        results.append(entry)
        print(
            f"{degrees:>8}deg{entry['detection_rate']:>10.2f}"
            f"{entry['confidence']:>10.3f}{entry['uprightness']:>10.3f}{entry['score']:>10.3f}"
        )

    best = max(results, key=lambda entry: entry["score"])
    print("-" * 66)
    if best["score"] <= 0.0:
        print("FAILED: no person detected in any orientation.")
        print("Check lighting, framing and that you are fully visible.")
        return 1

    print(f"BEST ROTATION     {best['degrees']} degrees")
    height, width = apply_rotation(frames[0], best["degrees"]).shape[:2]
    print(f"frame after rot   {width}x{height}")
    if best["degrees"] in (90, 270):
        print("-> portrait mounting confirmed: the long sensor axis is vertical")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m2_rotation.json"), "w", encoding="utf-8") as handle:
        json.dump({"candidates": results, "best": best}, handle, indent=2)

    if not args.no_save:
        profile.rotation = best["degrees"]
        path = profile.save()
        print(f"saved             {path}")

    if args.preview:
        frame = apply_rotation(frames[-1], best["degrees"])
        result = estimator(frame)
        keypoints, scores = result.person()
        if keypoints is not None:
            draw_skeleton(frame, keypoints, scores)
        draw_hud(frame, [f"rotation {best['degrees']} deg", f"score {best['score']:.3f}"])
        cv2.imshow("A3 rotation calibration", frame)
        print("press any key to close the preview")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
