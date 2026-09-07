import argparse
import json
import os
import re
import subprocess
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(A3_ROOT, "results")

WANT_WIDTH = 1080
WANT_HEIGHT = 1920
WANT_FPS = 30.0


def ffmpeg_executable():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def ffprobe_json(path):
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        return None
    probe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if not os.path.isfile(probe):
        return None
    command = [
        probe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def ffmpeg_stream_info(path):
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        return {}
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True
    )
    text = (result.stderr or "") + (result.stdout or "")

    info = {}
    rotation = re.search(r"rotation of ([-\d.]+) degrees", text)
    if rotation:
        info["rotation"] = float(rotation.group(1))
    else:
        tag = re.search(r"^\s*rotate\s*:\s*(\d+)", text, re.MULTILINE)
        if tag:
            info["rotation"] = float(tag.group(1))
        elif "displaymatrix" in text.lower():
            info["rotation"] = "displaymatrix present"

    video = re.search(r"Stream #\d+:\d+.*?: Video: ([^\s,]+)[^,]*, ([^\s,]+)", text)
    if video:
        info["codec"] = video.group(1)
        info["pix_fmt"] = video.group(2)
    return info


def main():
    parser = argparse.ArgumentParser(description="Validate an exported video for the A3 pipeline.")
    parser.add_argument("path")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"FATAL: file not found: {args.path}")
        return 1

    capture = cv2.VideoCapture(args.path)
    if not capture.isOpened():
        print("FATAL: OpenCV cannot open this file.")
        return 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = capture.read()
    capture.release()

    size_mb = os.path.getsize(args.path) / 1e6
    duration = frames / fps if fps else 0.0

    print("=" * 66)
    print("A3 / video check")
    print("=" * 66)
    print(f"file           {os.path.basename(args.path)}  ({size_mb:.1f} MB)")
    print(f"decoded size   {width}x{height}")
    print(f"frame rate     {fps:.3f}")
    print(f"frames         {frames}   ({duration:.1f} s)")

    codec = None
    rotation = None
    pix_fmt = None
    probe = ffprobe_json(args.path)
    if probe:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            codec = stream.get("codec_name")
            pix_fmt = stream.get("pix_fmt")
            tags = stream.get("tags", {}) or {}
            rotation = tags.get("rotate")
            for entry in stream.get("side_data_list", []) or []:
                if "rotation" in entry:
                    rotation = entry["rotation"]
            print(f"codec          {codec}  {pix_fmt}")
            print(f"colour         {stream.get('color_space') or 'unspecified'} / "
                  f"{stream.get('color_transfer') or 'unspecified'}")
            break
    else:
        info = ffmpeg_stream_info(args.path)
        codec = info.get("codec")
        pix_fmt = info.get("pix_fmt")
        rotation = info.get("rotation")
        if codec:
            print(f"codec          {codec}  {pix_fmt or ''}")
        print(f"rotation tag   {rotation if rotation is not None else 'none'}")

    print("-" * 66)
    problems = []
    warnings = []

    if width > height:
        problems.append(
            f"landscape ({width}x{height}). Export portrait {WANT_WIDTH}x{WANT_HEIGHT} "
            "- set the Resolve timeline resolution, not just the render size."
        )
    elif (width, height) != (WANT_WIDTH, WANT_HEIGHT):
        warnings.append(f"portrait but {width}x{height}, expected {WANT_WIDTH}x{WANT_HEIGHT}")

    if abs(fps - WANT_FPS) > 0.5:
        warnings.append(f"frame rate {fps:.2f}, expected {WANT_FPS:.0f}")

    if rotation not in (None, 0, "0"):
        problems.append(
            f"rotation metadata is {rotation}. The pixels are not physically rotated; "
            "different tools will disagree. Re-export from a portrait timeline."
        )

    if codec and codec not in ("h264", "hevc", "mpeg4", "vp9", "av1"):
        warnings.append(f"unusual codec {codec}")
    if pix_fmt and "10le" in pix_fmt:
        warnings.append(f"{pix_fmt} is 10-bit; 8-bit yuv420p is safer")

    if not ok or first is None:
        problems.append("cannot decode the first frame")
    else:
        mean = float(first.mean())
        if mean < 25:
            warnings.append(f"very dark first frame (mean {mean:.0f}) - check exposure")
        elif mean > 230:
            warnings.append(f"very bright first frame (mean {mean:.0f})")
        preview = os.path.join(OUT_DIR, "video_first_frame.png")
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(preview, first)
        print(f"first frame    written to {os.path.relpath(preview, A3_ROOT)}")

    for problem in problems:
        print(f"  PROBLEM: {problem}")
    for warning in warnings:
        print(f"  warning: {warning}")
    if not problems and not warnings:
        print("  everything looks right")

    print("-" * 66)
    print("VERDICT:", "USABLE" if not problems else "RE-EXPORT NEEDED")
    if not problems:
        print()
        print("Next:")
        print(f'  tools\\live_pose.py --source "{args.path}" --realtime')
        print(f'  tools\\live_pose.py --source "{args.path}" --headless '
              f'--record results\\session.jsonl')
    print("=" * 66)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
