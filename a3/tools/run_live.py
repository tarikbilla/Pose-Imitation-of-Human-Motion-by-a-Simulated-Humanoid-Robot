import argparse
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
A3_ROOT = os.path.abspath(os.path.join(TOOLS, ".."))
WEBOTS = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                      "Webots", "msys64", "mingw64", "bin", "webots.exe")
WORLD = os.path.join(A3_ROOT, "webots", "worlds", "m8_puppet.wbt")
RESULTS = os.path.join(A3_ROOT, "results")
RUN_JSON = os.path.join(RESULTS, "puppet_run.json")
READY = os.path.join(RESULTS, "puppet_ready")


def main():
    parser = argparse.ArgumentParser(
        description="Start the Webots puppet and the live camera driver")
    parser.add_argument("--source", default=None,
                        help="Camera index or video path (default: from the camera profile)")
    parser.add_argument("--site", default=None,
                        help="Site profile: home, lab (default: A3_SITE)")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--seconds", type=float, default=3600.0,
                        help="Upper bound for the session")
    parser.add_argument("--boot", type=float, default=12.0,
                        help="Time to wait until Webots has opened the port")
    parser.add_argument("--record", default=None,
                        help="Also record the raw keypoints of the session")
    parser.add_argument("--headless", action="store_true",
                        help="no monitoring window")
    parser.add_argument("--driver-seconds", type=float, default=0.0,
                        help="Stop the driver after n seconds (0 = manual)")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for path in (RUN_JSON, READY):
        try:
            os.remove(path)
        except OSError:
            pass

    env = dict(os.environ)
    env.update({"A3_LIVE": "1", "A3_WALL_CLOCK": "1", "A3_WAIT_DRIVER": "0",
                "A3_RUN_SECONDS": str(args.seconds)})

    log_path = os.path.join(RESULTS, "_live.log")
    log = open(log_path, "w", encoding="utf-8")
    webots = subprocess.Popen(
        [WEBOTS, "--batch", "--mode=realtime", "--stdout", "--stderr", WORLD],
        env=env, cwd=A3_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"Webots started (PID {webots.pid}), waiting {args.boot:.0f} s "
          f"for the receiver ...")
    time.sleep(args.boot)
    if webots.poll() is not None:
        log.close()
        print("Webots exited early, see", log_path)
        return 1

    command = [sys.executable, "-u", os.path.join(TOOLS, "live_puppet.py")]
    if args.site:
        command += ["--site", args.site]
    if args.profile:
        command += ["--profile", args.profile]
    if args.mode:
        command += ["--mode", args.mode]
    if args.source is not None:
        command += ["--source", args.source]
    if args.record:
        command += ["--record", args.record]
    if args.headless:
        command.append("--headless")
    if args.driver_seconds:
        command += ["--seconds", str(args.driver_seconds)]

    print("starting driver - close the window or press q to end the "
          "session\n")
    try:
        subprocess.run(command, cwd=A3_ROOT, env=env)
    except KeyboardInterrupt:
        print("interrupted")

    print("\nwaiting for the simulation to finish ...")
    deadline = time.time() + 30.0
    while time.time() < deadline and not os.path.exists(RUN_JSON):
        if webots.poll() is not None:
            break
        time.sleep(0.5)
    if webots.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(webots.pid)],
                       capture_output=True)
    log.close()

    if not os.path.exists(RUN_JSON):
        print("no result written, see", log_path)
        return 1
    with open(RUN_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print("--- session ---")
    for key in ("live_packets", "duration_s", "travel_forward",
                "travel_range_m", "plant_events", "real_ground_min_mm",
                "real_airborne_percent", "upper_error_median_deg"):
        if key in data:
            print(f"  {key:<24}{data[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
