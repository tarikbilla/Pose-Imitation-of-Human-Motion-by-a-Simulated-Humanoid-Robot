import argparse
import os
import subprocess
import sys

DEFAULT_VENV = r"C:\venvs\a3-pose"

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTROLLERS_DIR = os.path.join(A3_ROOT, "webots", "controllers")


def venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_venv(venv_dir, base_python):
    if os.path.isfile(venv_python(venv_dir)):
        print(f"venv exists            {venv_dir}")
        return
    print(f"creating venv          {venv_dir}")
    subprocess.check_call([base_python, "-m", "venv", venv_dir])


NO_DEPS_PACKAGES = ("rtmlib",)


def install_requirements(venv_dir):
    python = venv_python(venv_dir)
    requirements = os.path.join(A3_ROOT, "requirements.txt")
    print("installing requirements")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    subprocess.check_call([python, "-m", "pip", "install", "-r", requirements, "--quiet"])
    for package in NO_DEPS_PACKAGES:
        print(f"installing {package} (--no-deps)")
        subprocess.check_call(
            [python, "-m", "pip", "install", package, "--no-deps", "--quiet"]
        )


def verify_directml(venv_dir):
    python = venv_python(venv_dir)
    probe = (
        "import onnxruntime as ort;"
        "providers = ort.get_available_providers();"
        "print('providers:', providers);"
        "raise SystemExit(0 if 'DmlExecutionProvider' in providers else 3)"
    )
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode == 3:
        print()
        print("ERROR: DmlExecutionProvider is missing.")
        print("       Something installed plain `onnxruntime` and displaced")
        print("       `onnxruntime-directml`. Repair with:")
        print(f"       {python} -m pip uninstall onnxruntime onnxruntime-directml -y")
        print(f"       {python} -m pip install onnxruntime-directml")
        return False
    return result.returncode == 0


def write_runtime_ini(venv_dir):
    python = venv_python(venv_dir)
    if not os.path.isdir(CONTROLLERS_DIR):
        return
    written = 0
    for entry in sorted(os.listdir(CONTROLLERS_DIR)):
        controller_dir = os.path.join(CONTROLLERS_DIR, entry)
        if not os.path.isdir(controller_dir):
            continue
        target = os.path.join(controller_dir, "runtime.ini")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("[python]\n")
            handle.write(f"COMMAND = {python}\n")
        written += 1
    print(f"runtime.ini written    {written} controller(s)")


def check_path_length(venv_dir):
    if os.name != "nt":
        return
    if len(A3_ROOT) > 120:
        print()
        print("WARNING: repository path is long and Windows long paths are off.")
        print(f"         {len(A3_ROOT)} chars: {A3_ROOT}")
        print("         Keep the venv outside the repo (this script does).")
    if venv_dir.lower().startswith(A3_ROOT.lower()):
        print()
        print("WARNING: venv lives inside the repo. Package installs with deep")
        print("         directory trees (onnx) will fail with WinError 206.")


def main():
    parser = argparse.ArgumentParser(description="Set up the A3 Python environment.")
    parser.add_argument("--venv", default=DEFAULT_VENV, help="virtualenv location")
    parser.add_argument("--python", default=sys.executable, help="base interpreter")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    venv_dir = os.path.abspath(args.venv)

    print("=" * 62)
    print("A3 environment setup")
    print("=" * 62)
    print(f"repo root              {A3_ROOT}")

    create_venv(venv_dir, args.python)
    if not args.skip_install:
        install_requirements(venv_dir)
    write_runtime_ini(venv_dir)
    check_path_length(venv_dir)

    print("-" * 62)
    healthy = verify_directml(venv_dir)

    print("=" * 62)
    if healthy:
        print("done. verify the full GPU path with:")
        print(f"  {venv_python(venv_dir)} tools\\check_perception.py --cpu-compare")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
