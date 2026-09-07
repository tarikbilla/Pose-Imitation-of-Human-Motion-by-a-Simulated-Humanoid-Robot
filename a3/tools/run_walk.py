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
WORLD = os.path.join(A3_ROOT, "webots", "worlds", "m7_walk.wbt")
RUN_JSON = os.path.join(A3_ROOT, "results", "walk_run.json")
READY = os.path.join(A3_ROOT, "results", "walk_ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--env", nargs="*", default=[])
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
    process = subprocess.Popen(
        [WEBOTS, "--batch", "--mode=fast", "--minimize", "--no-rendering",
         "--stdout", "--stderr", WORLD],
        env=env, cwd=A3_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)

    lines = []
    deadline = started + args.timeout
    while time.time() < deadline:
        if os.path.exists(RUN_JSON):
            time.sleep(1.0)
            break
        if process.poll() is not None:
            break
        line = process.stdout.readline()
        if line:
            lines.append(line.rstrip())
            print(line.rstrip(), flush=True)

    if process.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True)

    print(f"--- Wanduhr {time.time() - started:.1f} s ---")
    if os.path.exists(RUN_JSON):
        with open(RUN_JSON, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        print(json.dumps({k: v for k, v in data.items() if k != "trace"},
                         indent=2))
        return 0
    print("no result written", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
