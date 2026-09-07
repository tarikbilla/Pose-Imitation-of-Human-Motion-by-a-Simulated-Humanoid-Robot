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
RUN_JSON = os.path.join(A3_ROOT, "results", "puppet_run.json")
READY = os.path.join(A3_ROOT, "results", "puppet_ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--mode", default="fast",
                        choices=("fast", "realtime", "run"))
    args = parser.parse_args()

    for path in (RUN_JSON, READY):
        try:
            os.remove(path)
        except OSError:
            pass
    env = dict(os.environ)
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value

    started = time.time()
    log = open(os.path.join(A3_ROOT, "results", "_puppet.log"), "w",
               encoding="utf-8")
    flags = ["--batch", f"--mode={args.mode}", "--stdout", "--stderr"]
    if not args.visible:
        flags[2:2] = ["--minimize", "--no-rendering"]
    process = subprocess.Popen(
        [WEBOTS] + flags + [WORLD],
        env=env, cwd=A3_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = started + args.timeout
    while time.time() < deadline:
        if os.path.exists(RUN_JSON):
            time.sleep(0.8)
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)
    if process.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True)
    log.close()
    with open(os.path.join(A3_ROOT, "results", "_puppet.log"), "r",
              encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "WARNING" not in line and line.strip():
                print(line.rstrip())
    print(f"--- Wanduhr {time.time() - started:.1f} s ---")
    if os.path.exists(RUN_JSON):
        with open(RUN_JSON, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        print(json.dumps({k: v for k, v in data.items()
                          if k not in ("trace", "steps")}, indent=2))
        return 0
    print("kein Ergebnis", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
