import argparse
import collections
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception import runtime
from perception.capture import CameraProfile, mirror_for_display
from perception.overlay import draw_hud, draw_skeleton
from perception.pose2d import Pose2D
from perception.recording import Recorder
from perception.source import VideoSource, open_source

WINDOW = "A3 pose"
HELP = "q quit   m mirror   space pause   s snapshot"


def main():
    parser = argparse.ArgumentParser(
        description="A3 pose viewer for a live camera or a recorded video."
    )
    parser.add_argument("--source", default=None,
                        help="camera index (default from profile) or path to a video file")
    parser.add_argument("--profile", default="brio100")
    parser.add_argument("--model", default="wholebody", choices=("wholebody", "body"),
                        help="wholebody adds 21 keypoints per hand")
    parser.add_argument("--mode", default="lightweight",
                        choices=("lightweight", "balanced", "performance"))
    parser.add_argument("--no-smooth", action="store_true",
                        help="disable the One-Euro filter")
    parser.add_argument("--rotation", type=int, default=None,
                        help="override rotation in degrees (0/90/180/270)")
    parser.add_argument("--loop", action="store_true", help="loop a video source")
    parser.add_argument("--realtime", action="store_true",
                        help="pace a video source at its own frame rate")
    parser.add_argument("--record", default=None,
                        help="write keypoints to this .jsonl recording")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--headless", action="store_true", help="no preview window")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=720)
    args = parser.parse_args()

    try:
        profile = CameraProfile.load(args.profile)
    except FileNotFoundError:
        print(f"profile '{args.profile}' not found, using defaults")
        profile = CameraProfile(name=args.profile)

    spec = args.source if args.source is not None else profile.source

    print("loading models ...")
    estimator = Pose2D(model=args.model, mode=args.mode,
                       device=runtime.DEVICE_DML, smooth=not args.no_smooth)
    if estimator.warning:
        print(f"WARNING: {estimator.warning}")
    estimator.warmup()
    print(f"device    {estimator.device}   model {args.model}/{args.mode}"
          f"   hands {'yes' if estimator.supports_hands else 'no'}"
          f"   smoothing {'off' if args.no_smooth else 'on'}")

    source = open_source(
        spec, profile=profile, loop=args.loop,
        realtime=args.realtime, rotation=args.rotation,
    )
    info = source.describe()
    print(f"source    {info}")

    is_video = isinstance(source, VideoSource)
    mirror = (not args.no_mirror) and not is_video

    recorder = None
    if args.record:
        metadata = {"source": info, "model": args.model, "mode": args.mode,
                    "smoothed": not args.no_smooth, "mirrored": False}
        recorder = Recorder(args.record, metadata).open()
        print(f"recording {args.record}")

    print(HELP)

    latencies = collections.deque(maxlen=30)
    frame_times = collections.deque(maxlen=30)
    snapshots = 0
    processed = 0
    detected = 0
    paused = False
    last = time.perf_counter()
    started = time.perf_counter()

    try:
        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    print("source exhausted")
                    break

                result = estimator(frame, timestamp=source.timestamp())
                latencies.append(result.inference_ms)
                keypoints, scores = result.person()
                processed += 1
                if keypoints is not None:
                    detected += 1

                if recorder is not None:
                    recorder.write(keypoints, scores, timestamp=source.timestamp())

                now = time.perf_counter()
                frame_times.append(now - last)
                last = now

                if recorder is not None and processed % 60 == 0:
                    rate = detected / processed * 100.0
                    print(f"  {processed} frames, {rate:.0f}% detected")

            if not args.headless:
                display = frame
                shown = keypoints
                if mirror:
                    display, shown = mirror_for_display(frame, keypoints)
                display = display.copy()
                if shown is not None:
                    draw_skeleton(display, shown, scores)

                fps = len(frame_times) / sum(frame_times) if sum(frame_times) else 0.0
                mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
                lines = [
                    f"{fps:5.1f} fps   infer {mean_latency:5.1f} ms",
                    f"{display.shape[1]}x{display.shape[0]}  mirror {'on' if mirror else 'off'}",
                    f"person {'yes' if keypoints is not None else 'NO'}"
                    + (f"  score {scores.mean():.2f}" if scores is not None else ""),
                ]
                if is_video:
                    lines.append(f"frame {source.frame_index}/{source.frame_count}"
                                 f"  {source.progress * 100:.0f}%")
                if paused:
                    lines.append("PAUSED")
                draw_hud(display, lines)

                if display.shape[1] > args.max_width:
                    scale = args.max_width / display.shape[1]
                    display = cv2.resize(display,
                                         (args.max_width, int(display.shape[0] * scale)))
                cv2.imshow(WINDOW, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    mirror = not mirror
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    out = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), "..", "results", f"snapshot_{snapshots}.png"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    cv2.imwrite(out, display)
                    snapshots += 1
                    print(f"saved {out}")

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        source.release()
        if recorder is not None:
            recorder.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print("-" * 60)
    print(f"frames processed  {processed}")
    if processed:
        print(f"detection rate    {detected / processed * 100:.1f}%")
        print(f"throughput        {processed / elapsed:.1f} fps")
        print(f"mean inference    {sum(latencies) / len(latencies):.1f} ms")
    if recorder is not None:
        print(f"recording written {args.record}  ({recorder.frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
