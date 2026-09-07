import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.overlay import draw_hud, draw_skeleton
from perception.pose2d import HALPE26_INDEX
from perception.recording import Replay

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

FOOT_KEYS = ("left_heel", "right_heel", "left_big_toe", "right_big_toe",
             "left_small_toe", "right_small_toe")


def ffmpeg_executable():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def open_writer(path, width, height, fps, crf):
    command = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", path,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(command, stdin=subprocess.PIPE, creationflags=creationflags)


def main():
    parser = argparse.ArgumentParser(
        description="Render the recorded skeleton on top of the source video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "overlay.mp4"))
    parser.add_argument("--start", type=float, default=0.0, help="start second")
    parser.add_argument("--duration", type=float, default=0.0, help="0 = all")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--stills", type=int, default=6,
                        help="also write this many evenly spaced PNG frames")
    args = parser.parse_args()

    replay = Replay(args.recording).load()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"FATAL: cannot open {args.video}")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

    print("=" * 66)
    print("A3 / overlay render")
    print("=" * 66)
    print(f"video       {os.path.basename(args.video)}  {total} frames @ {fps:.2f} fps")
    print(f"recording   {os.path.basename(args.recording)}  {len(replay)} entries")
    if len(replay) != total:
        print(f"  note: {len(replay)} keypoint frames vs {total} video frames")

    first_frame = int(args.start * fps)
    last_frame = total if args.duration <= 0 else min(
        total, first_frame + int(args.duration * fps)
    )
    if first_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = open_writer(args.out, width, height, fps, args.crf)

    still_at = set()
    if args.stills > 0:
        span = last_frame - first_frame
        still_at = {first_frame + int(span * i / max(1, args.stills - 1))
                    for i in range(args.stills)}

    index = first_frame
    written = 0
    stills_written = 0
    missing = 0

    while index < last_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        keypoints = scores = None
        if index < len(replay.entries):
            entry = replay.entries[index]
            if entry.get("kp") is not None:
                keypoints = np.asarray(entry["kp"], dtype=np.float32)
                scores = np.asarray(entry["sc"], dtype=np.float32)
        if keypoints is None:
            missing += 1

        if args.scale != 1.0:
            frame = cv2.resize(frame, (width, height))
            if keypoints is not None:
                keypoints = keypoints * args.scale

        if keypoints is not None:
            draw_skeleton(frame, keypoints, scores)
            feet = np.mean([scores[HALPE26_INDEX[n]] for n in FOOT_KEYS])
            lines = [
                f"frame {index}   t {index / fps:5.2f}s",
                f"mean score {scores.mean():.2f}   feet {feet:.2f}",
            ]
        else:
            lines = [f"frame {index}   t {index / fps:5.2f}s", "NO DETECTION"]
        draw_hud(frame, lines, width=int(300 * args.scale))

        writer.stdin.write(frame.tobytes())
        written += 1

        if index in still_at:
            still_path = os.path.join(OUT_DIR, f"overlay_still_{stills_written:02d}.png")
            cv2.imwrite(still_path, frame)
            stills_written += 1

        index += 1

    capture.release()
    writer.stdin.close()
    writer.wait()

    size_mb = os.path.getsize(args.out) / 1e6 if os.path.isfile(args.out) else 0.0
    print("-" * 66)
    print(f"frames written   {written}")
    print(f"without pose     {missing}")
    print(f"stills written   {stills_written}  (results/overlay_still_*.png)")
    print(f"output           {args.out}  ({size_mb:.1f} MB)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
