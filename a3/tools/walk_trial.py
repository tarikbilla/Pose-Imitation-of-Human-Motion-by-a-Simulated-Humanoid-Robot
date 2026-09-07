import argparse
import itertools
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
A3_ROOT = os.path.abspath(os.path.join(TOOLS, ".."))
sys.path.insert(0, TOOLS)

WEBOTS = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                      "Webots", "msys64", "mingw64", "bin", "webots.exe")
WORLD = os.path.join(A3_ROOT, "webots", "worlds", "m7_walk.wbt")
RUN_JSON = os.path.join(A3_ROOT, "results", "walk_run.json")
READY = os.path.join(A3_ROOT, "results", "walk_ready")
PYTHON = r"C:\venvs\a3-pose\Scripts\python.exe"

KEEP = ("fallen", "steps_done", "steps_planned", "travel_forward",
        "travel_lateral", "planned_forward", "com_error_median",
        "dcm_error_median", "dcm_error_max", "ik_error_max_mm",
        "duration_s", "horizon_s", "profile", "com_height", "omega")


def build_proto(profile):
    subprocess.run([PYTHON, os.path.join(TOOLS, "build_atlas_proto.py"),
                    "--masses", profile],
                   check=True, capture_output=True, cwd=A3_ROOT)


def one_run(settings, timeout):
    for path in (RUN_JSON, READY):
        try:
            os.remove(path)
        except OSError:
            pass
    env = dict(os.environ)
    env.update({key: str(value) for key, value in settings.items()})
    started = time.time()
    process = subprocess.Popen(
        [WEBOTS, "--batch", "--mode=fast", "--minimize", "--no-rendering",
         WORLD], env=env, cwd=A3_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = started + timeout
    while time.time() < deadline:
        if os.path.exists(RUN_JSON):
            time.sleep(1.0)
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)
    if process.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True)
    wall = time.time() - started
    if not os.path.exists(RUN_JSON):
        return {"ok": False, "wall_s": round(wall, 1)}
    with open(RUN_JSON, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    row = {key: data.get(key) for key in KEEP}
    row["ok"] = True
    row["wall_s"] = round(wall, 1)
    if row["planned_forward"]:
        row["travel_ratio"] = round(row["travel_forward"] / row["planned_forward"], 3)
    else:
        row["travel_ratio"] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=["core"])
    parser.add_argument("--step-length", nargs="+", type=float, default=[0.15])
    parser.add_argument("--gain", nargs="+", type=float, default=[0.0])
    parser.add_argument("--sign", nargs="+", type=float, default=[1.0])
    parser.add_argument("--adapt", nargs="+", default=["0"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=200.0)
    parser.add_argument("--out", default=os.path.join(A3_ROOT, "results",
                                                      "walk_trial.json"))
    args = parser.parse_args()

    rows = []
    current_profile = None
    combos = itertools.product(args.profiles, args.step_length, args.gain,
                               args.sign, args.adapt)
    for profile, length, gain, sign, adapt in combos:
        if profile != current_profile:
            build_proto(profile)
            current_profile = profile
        for index in range(args.runs):
            settings = {
                "A3_MASS_PROFILE": profile,
                "A3_STEP_LENGTH": length,
                "A3_STEP_COUNT": args.steps,
                "A3_DCM_GAIN": gain,
                "A3_DCM_SIGN": sign,
                "A3_ADAPT_STEPS": adapt,
                "A3_LOG_EVERY": 25,
            }
            row = one_run(settings, args.timeout)
            row.update({"variant_profile": profile, "step_length": length,
                        "gain": gain, "sign": sign, "adapt": adapt,
                        "run": index})
            rows.append(row)
            flag = "FEHLER" if not row.get("ok") else (
                "gefallen" if row["fallen"] else "ok      ")
            print(f"{profile:6s} L={length:.2f} g={gain:4.1f} s={sign:+.0f} "
                  f"a={adapt} r{index}  {flag}  "
                  f"steps={row.get('steps_done')}/{row.get('steps_planned')} "
                  f"Weg={row.get('travel_forward')} "
                  f"quer={row.get('travel_lateral')} "
                  f"dcm={row.get('dcm_error_median')}/{row.get('dcm_error_max')} "
                  f"ik={row.get('ik_error_max_mm')}mm", flush=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump({"rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
