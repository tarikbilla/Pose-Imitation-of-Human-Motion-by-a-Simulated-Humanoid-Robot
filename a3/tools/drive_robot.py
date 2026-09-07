import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.recording import Replay
from perception.retarget import UpperBodyRetargeter
from pose_amplify import amplify
from transport import schema, udp


RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
READY_MARKERS = (os.path.join(RESULTS, "m4_ready"),
                 os.path.join(RESULTS, "m3_ready"),
                 os.path.join(RESULTS, "puppet_ready"))
PREVIEW_WINDOW = "A3 reference stream"


def wait_for_controller(timeout):
    if any(os.path.isfile(m) for m in READY_MARKERS):
        return True
    print(f"waiting for the Webots controller (max {timeout:.0f} s) ...")
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if any(os.path.isfile(m) for m in READY_MARKERS):
            print("controller ready")
            return True
        time.sleep(0.25)
    print("controller did not report ready; sending anyway")
    return False


def open_preview(replay, explicit_path, height):
    import cv2

    from perception.overlay import draw_hud, draw_skeleton

    path = explicit_path
    if path is None:
        source = replay.header.get("source") or {}
        path = source.get("path")
    if not path or not os.path.isfile(path):
        print(f"preview disabled: source video not found ({path})")
        return None

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(f"preview disabled: cannot open {path}")
        return None

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    probe_scale = height / 1920.0
    cv2.resizeWindow(PREVIEW_WINDOW, int(1080 * probe_scale), height)
    cv2.moveWindow(PREVIEW_WINDOW, max(0, 1920 - int(1080 * probe_scale) - 12), 8)
    print(f"preview   {os.path.basename(path)}")
    return {"capture": capture, "cv2": cv2, "height": height,
            "draw_skeleton": draw_skeleton, "draw_hud": draw_hud, "index": -1}


def show_preview(preview, entry, keypoints, scores, out, sent):
    cv2 = preview["cv2"]
    capture = preview["capture"]

    target = entry.get("i", preview["index"] + 1)
    if target != preview["index"] + 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    preview["index"] = target

    ok, frame = capture.read()
    if not ok or frame is None:
        return True

    if keypoints is not None:
        preview["draw_skeleton"](frame, keypoints, scores)

    scalars = out["scalars"]
    directions = out["directions"]

    def arrow(name):
        vector = directions.get(name)
        if vector is None:
            return "   --"
        return f"({vector[0]:+.2f},{vector[1]:+.2f},{vector[2]:+.2f})"

    seconds = entry["t"]
    lines = [
        f"packet {sent}   conf {out['confidence']:.2f}",
        f"torso yaw {math.degrees(scalars.get('torso_yaw', 0)):+5.0f}"
        f"  pitch {math.degrees(scalars.get('torso_pitch', 0)):+4.0f}"
        f"  roll {math.degrees(scalars.get('torso_roll', 0)):+4.0f}",
        f"L upper {arrow('left_upper_arm')}",
        f"L fore  {arrow('left_fore_arm')}",
        f"R upper {arrow('right_upper_arm')}",
        f"R fore  {arrow('right_fore_arm')}",
        f"hands   {'yes' if 'left_hand_normal' in directions else 'no':>3}",
    ]
    scale = preview["height"] / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), preview["height"]))

    width = frame.shape[1]
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (width, 86), (16, 16, 20), -1)
    cv2.addWeighted(banner, 0.85, frame, 0.15, 0, frame)

    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    cv2.putText(frame, f"FRAME {target}", (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (120, 240, 140), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{minutes:d}:{remainder:05.2f}   {seconds:6.2f}s", (14, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 235, 235), 2, cv2.LINE_AA)

    preview["draw_hud"](frame, lines, origin=(10, 96), width=min(340, width - 20))
    cv2.imshow(PREVIEW_WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def run_replay(args):
    replay = Replay(args.recording, loop=args.loop, realtime=not args.fast).load()
    retargeter = UpperBodyRetargeter()

    print(f"replay    {os.path.basename(args.recording)}  "
          f"{len(replay)} frames, {replay.duration:.1f} s")

    preview = None
    if args.preview:
        preview = open_preview(replay, args.video, args.preview_height)

    if args.wait_ready:
        wait_for_controller(args.wait_timeout)

    with open(os.path.join(RESULTS, "driver_started"), "w",
              encoding="utf-8") as handle:
        handle.write("1")
    sender = udp.Sender(args.host, args.port)
    print(f"sending   {args.host}:{args.port}  "
          f"{'realtime' if not args.fast else 'as fast as possible'}"
          f"{'  looping' if args.loop else ''}")

    sent = 0
    skipped = 0
    started = time.perf_counter()
    try:
        while True:
            item = replay.read()
            if item is None:
                break
            entry, keypoints, scores = item
            if keypoints is None:
                skipped += 1
                continue
            out = retargeter(keypoints, scores, entry["t"])
            if out is None or not out["valid"]:
                skipped += 1
                continue
            directions, scalars = amplify(
                out["directions"], out["scalars"], args.amplify)
            packet = schema.build(
                out["seq"], out["t"], directions,
                scalars, out["confidence"], lower=out.get("lower"),
            )
            sender.send(packet)
            sent += 1
            if preview is not None:
                if not show_preview(preview, entry, keypoints, scores, out, sent):
                    print("preview closed")
                    break
            if sent % 150 == 0:
                print(f"  sent {sent}  t={entry['t']:5.1f}s")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        sender.close()
        if preview is not None:
            preview["capture"].release()
            preview["cv2"].destroyAllWindows()

    elapsed = time.perf_counter() - started
    print("-" * 56)
    print(f"packets sent   {sent}")
    print(f"frames skipped {skipped}")
    print(f"wall time      {elapsed:.1f} s  ({sent / max(elapsed, 1e-6):.1f} pkt/s)")
    return 0


def run_live(args):
    import cv2

    from perception import runtime
    from perception.capture import CameraProfile
    from perception.pose2d import Pose2D
    from perception.source import open_source

    profile = CameraProfile.load(args.profile)
    estimator = Pose2D(model="wholebody", mode=args.mode, device=runtime.DEVICE_DML)
    estimator.warmup()
    retargeter = UpperBodyRetargeter()
    sender = udp.Sender(args.host, args.port)
    source = open_source(args.source or profile.source, profile=profile)

    print(f"live      {source.describe()}")
    print(f"sending   {args.host}:{args.port}   press Ctrl+C to stop")

    sent = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            timestamp = source.timestamp()
            result = estimator(frame, timestamp=timestamp)
            keypoints, scores = result.person()
            if keypoints is None:
                continue
            out = retargeter(keypoints, scores, timestamp)
            if out is None or not out["valid"]:
                continue
            sender.send(schema.build(out["seq"], out["t"], out["directions"],
                                     out["scalars"], out["confidence"],
                                     lower=out.get("lower")))
            sent += 1
            if sent % 90 == 0:
                print(f"  sent {sent}  infer {result.inference_ms:.0f} ms")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        source.release()
        sender.close()
    print(f"packets sent {sent}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Stream upper-body references to the Webots controller."
    )
    parser.add_argument("--recording", help="replay a .jsonl recording")
    parser.add_argument("--source", help="camera index or video path for live mode")
    parser.add_argument("--profile", default="brio100")
    parser.add_argument("--mode", default="lightweight")
    parser.add_argument("--host", default=udp.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=udp.DEFAULT_PORT)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--amplify", type=float, default=1.0,
                        help="scale pose deviation from rest for stress tests")
    parser.add_argument("--fast", action="store_true",
                        help="send without real-time pacing")
    parser.add_argument("--preview", action="store_true",
                        help="show the source video with the streamed skeleton")
    parser.add_argument("--video", default=None,
                        help="explicit video path for --preview")
    parser.add_argument("--preview-height", type=int, default=1010,
                        help="preview window height in pixels")
    parser.add_argument("--wait-ready", action="store_true",
                        help="wait until the Webots controller reports ready")
    parser.add_argument("--wait-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.recording:
        return run_replay(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
