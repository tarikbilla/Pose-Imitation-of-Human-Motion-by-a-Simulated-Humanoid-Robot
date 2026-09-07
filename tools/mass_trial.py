import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
A3_ROOT = os.path.abspath(os.path.join(TOOLS, ".."))
sys.path.insert(0, TOOLS)

import mass_tables as mt

WEBOTS = os.path.join(os.environ.get("WEBOTS_HOME", os.path.join(
    os.path.expanduser("~"), "AppData", "Local", "Programs", "Webots")),
    "msys64", "mingw64", "bin", "webots.exe")
WORLD = os.path.join(A3_ROOT, "webots", "worlds", "m3_upper_body.wbt")
RESULTS = os.path.join(A3_ROOT, "results")
READY = os.path.join(RESULTS, "m3_ready")
RUN_JSON = os.path.join(RESULTS, "m3_run.json")
PYTHON = r"C:\venvs\a3-pose\Scripts\python.exe"
RECORDING = os.path.join(A3_ROOT, "recordings", "testvideo_wb.jsonl")


def clear():
    for path in (READY, RUN_JSON):
        try:
            os.remove(path)
        except OSError:
            pass


def one_run(variant, arm_scale, safety, timeout, amplify):
    subprocess.run([PYTHON, os.path.join(TOOLS, "build_atlas_proto.py"),
                    "--masses", variant],
                   check=True, capture_output=True, cwd=A3_ROOT)
    clear()
    env = dict(os.environ)
    env["A3_ARM_SCALE"] = str(arm_scale)
    env["A3_SAFETY"] = "1" if safety else "0"
    env["A3_FINISH_IDLE"] = "3"
    started = time.time()
    webots = subprocess.Popen(
        [WEBOTS, "--batch", "--mode=fast", "--minimize", "--no-rendering",
         "--stdout", "--stderr", WORLD],
        env=env, cwd=A3_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver = subprocess.Popen(
        [PYTHON, os.path.join(TOOLS, "drive_robot.py"),
         "--recording", RECORDING, "--fast", "--wait-ready",
         "--amplify", str(amplify)],
        env=env, cwd=A3_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(RUN_JSON) or webots.poll() is not None:
            break
        time.sleep(0.5)
    time.sleep(1.5)
    for process in (driver, webots):
        if process.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True)
    wall = time.time() - started
    if not os.path.exists(RUN_JSON):
        return {"variant": variant, "ok": False, "wall_s": round(wall, 1)}
    with open(RUN_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    trace = data.get("trace") or []
    imit = [s["imit"] for s in trace if "imit" in s]
    scales = [s.get("scale", 1.0) for s in trace]
    return {
        "variant": variant,
        "ok": True,
        "fallen": bool(data.get("fallen")),
        "packets": data.get("packets"),
        "dropped": data.get("dropped"),
        "duration_s": round(float(data.get("duration_s") or 0.0), 1),
        "stand_height": data.get("stand_height"),
        "imit_median_deg": round(math.degrees(statistics.median(imit)), 2) if imit else None,
        "imit_p95_deg": round(math.degrees(
            sorted(imit)[int(0.95 * (len(imit) - 1))]), 2) if imit else None,
        "authority_median": round(statistics.median(scales), 3) if scales else None,
        "wall_s": round(wall, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(mt.VARIANTS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--arm-scale", type=float, default=1.0)
    parser.add_argument("--safety", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--amplify", type=float, default=1.0)
    parser.add_argument("--out", default=os.path.join(RESULTS, "mass_trial.json"))
    args = parser.parse_args()

    rows = []
    for variant in args.variants:
        for index in range(args.runs):
            row = one_run(variant, args.arm_scale, args.safety,
                          args.timeout, args.amplify)
            row["run"] = index
            row["amplify"] = args.amplify
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump({"arm_scale": args.arm_scale, "safety": args.safety,
                           "amplify": args.amplify, "rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
