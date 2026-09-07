import argparse
import collections
import math
import os
import sys
import time

import cv2

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, A3_ROOT)
sys.path.insert(0, os.path.join(A3_ROOT, "webots", "controllers",
                                "a3_upper_body"))

import armkin
from kinematics.atlaskin import AtlasModel
from kinematics.legik import LegIK
from perception import runtime
from perception.capture import CameraProfile, mirror_for_display
from perception.fullbody import FullBodyRetargeter
from perception.lift3d import Lifter3D
from perception.locomotion import LocomotionTracker
from perception.overlay import draw_hud, draw_skeleton
from perception.pose2d import Pose2D
from perception.site import Site
from perception.recording import Recorder
from perception.source import FALLBACK_REASON as source_fallback
from perception.source import VideoSource, open_source
from transport import angles as angle_codec
from transport import udp

WINDOW = "A3 puppet - live"
HELP = "q quit   m mirror   n reset datum   s snapshot"
ANGLE_PORT = 8768

WATCH = (("torso", "BackMby"), ("turn", "BackLbz"), ("neck", "NeckAy"),
         ("knee L", "LLegKny"), ("knee R", "RLegKny"))


def main():
    parser = argparse.ArgumentParser(
        description="Live camera -> full-body retargeting -> Webots puppet")
    parser.add_argument("--source", default=None,
                        help="Camera index or video path")
    parser.add_argument("--site", default=None,
                        help="Site profile: home, lab (default: A3_SITE)")
    parser.add_argument("--profile", default=None,
                        help="Camera profile, overrides the site")
    parser.add_argument("--mode", default=None,
                        choices=("lightweight", "balanced", "performance"))
    parser.add_argument("--host", default=udp.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=ANGLE_PORT)
    parser.add_argument("--rotation", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", default=None,
                        help="Also record the raw keypoints")
    parser.add_argument("--max-width", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 = no limit")
    args = parser.parse_args()

    site = Site.load(args.site)
    site.apply_environment()
    camera_name = args.profile or site.camera
    pose_mode = args.mode or site.pose_mode
    try:
        profile = CameraProfile.load(camera_name)
    except FileNotFoundError:
        print(f"camera profile '{camera_name}' not found, using defaults")
        profile = CameraProfile(name=camera_name)

    device, warning = runtime.resolve_device(site.device, site.device_fallback)
    if warning:
        print(f"WARNING: {warning}")
    print(f"site      {site.name}: camera {camera_name}, device {device}, "
          f"capture backend {site.backend}")
    print("loading models ...")
    estimator = Pose2D(model="wholebody", mode=pose_mode, device=device)
    estimator.warmup()
    lifter = Lifter3D(device=device)
    model = AtlasModel()
    retarget = FullBodyRetargeter({"L": LegIK(model, "L"),
                                   "R": LegIK(model, "R")},
                                  arm_solver=armkin.solve_arm, lifter=lifter)

    source = open_source(args.source if args.source is not None else profile.source,
                         profile=profile, loop=args.loop,
                         realtime=args.realtime, rotation=args.rotation,
                         backend=site.backend,
                         device_hint=site.capture_device_hint or None)
    for reason in source_fallback:
        print(f"WARNING: ffmpeg capture unavailable, using OpenCV -> {reason}")
    is_video = isinstance(source, VideoSource)
    mirror = (not args.no_mirror) and not is_video

    sender = udp.Sender(args.host, args.port, codec=angle_codec)
    recorder = None
    if args.record:
        recorder = Recorder(args.record, {"source": source.describe(),
                                          "model": "wholebody",
                                          "mode": args.mode,
                                          "mirrored": False}).open()

    print(f"device    {estimator.device}   lifter {lifter.describe()['device']}")
    print(f"source    {source.describe()}")
    print(f"sending   {args.host}:{args.port}")
    print(HELP)

    tracker = None
    latencies = collections.deque(maxlen=30)
    periods = collections.deque(maxlen=30)
    sent = 0
    seen = 0
    detected = 0
    snapshots = 0
    last = time.perf_counter()
    started = last

    try:
        while True:
            if args.seconds and time.perf_counter() - started > args.seconds:
                print("time limit reached")
                break
            frame = source.read()
            if frame is None:
                print("source exhausted")
                break
            if tracker is None:
                height, width = frame.shape[:2]
                focal, origin = profile.focal_for(width, height)
                tracker = LocomotionTracker(focal=focal,
                                            body=site.body_height_m,
                                            frame_long=max(width, height))
                print(f"frame     {width}x{height}   focal "
                      f"{tracker.focal:.0f} px ({origin})")
                if "UNCALIBRATED" in origin:
                    print("WARNING: no camera calibration for this profile; "
                          "absolute travel distance will be wrong. See "
                          "docs/PORTING_LAB.md.")
            stamp = source.timestamp()
            result = estimator(frame, timestamp=stamp)
            latencies.append(result.inference_ms)
            keypoints, scores = result.person()
            seen += 1

            out = None
            move = tracker.state()
            if keypoints is not None:
                if recorder is not None:
                    recorder.write(keypoints, scores, timestamp=stamp)
                out = retarget(keypoints, scores, stamp)
                if out is not None:
                    detected += 1
                    move = tracker.update(keypoints, scores, stamp)
                    sender.send(angle_codec.build(
                        out["seq"], out["t"], out["angles"],
                        forward=move["forward"], lateral=move["lateral"],
                        confidence=float(scores.mean())))
                    sent += 1

            now = time.perf_counter()
            periods.append(now - last)
            last = now

            span = sum(periods)
            fps = len(periods) / span if span else 0.0
            infer = sum(latencies) / len(latencies) if latencies else 0.0

            if args.headless:
                if seen % 60 == 0:
                    print(f"  {seen} frames, {sent} packets, {fps:4.1f} fps, "
                          f"inference {infer:4.0f} ms, "
                          f"forward {move['forward']:+.2f} m")
                continue

            display, shown = (mirror_for_display(frame, keypoints)
                              if mirror else (frame, keypoints))
            display = display.copy()
            if shown is not None:
                draw_skeleton(display, shown, scores)

            lines = [
                f"{fps:5.1f} fps   inference {infer:5.1f} ms",
                f"person {'yes' if out is not None else 'waiting ...'}"
                + (f"   score {scores.mean():.2f}" if scores is not None else ""),
                f"packets {sent}   detected {100.0 * detected / max(seen, 1):.0f} %",
                "",
                f"distance  {move['distance_m']:5.2f} m"
                + ("" if move["ready"] else "   (calibrating ...)"),
                f"forward {move['forward']:+5.2f} m   lateral "
                f"{move['lateral']:+5.2f} m",
            ]
            if out is not None:
                lines.append("")
                for label, joint in WATCH:
                    value = out["angles"].get(joint)
                    if value is not None:
                        lines.append(f"{label:<9}{math.degrees(value):+6.1f} deg")
            draw_hud(display, lines)

            if display.shape[1] > args.max_width:
                scale = args.max_width / display.shape[1]
                display = cv2.resize(display, (args.max_width,
                                               int(display.shape[0] * scale)))
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("n"):
                tracker.reset()
                retarget.reset()
                print("datum reset")
            if key == ord("s"):
                path = os.path.join(A3_ROOT, "results",
                                    f"live_{snapshots}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, display)
                print(f"saved {path}")
                snapshots += 1
    except KeyboardInterrupt:
        print("abgebrochen")
    finally:
        source.release()
        sender.close()
        if recorder is not None:
            recorder.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"frames {seen}, detected {detected}, packets {sent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
