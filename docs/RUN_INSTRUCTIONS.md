# Run Instructions — Target PC (Ubuntu)

Complete deployment guide for running the project on a fresh Ubuntu machine after cloning the repository.
This document assumes you will `git pull` on the **target PC** and run from there.

> Tested on: **Ubuntu 22.04 LTS** and **Ubuntu 24.04 LTS** with **Python 3.11**.

---

## 0. TL;DR (Copy–Paste)

```bash
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
python run.py --no-webots
```

A window titled **"Pose Imitation - Camera Feed"** opens showing your camera with the 33-landmark skeleton drawn on top. Press **`q`** or **`ESC`** to quit.

---

## 1. Hardware & OS Requirements

| Item | Minimum | Recommended |
|---|---|---|
| OS    | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU   | 4 cores          | 8+ cores |
| RAM   | 8 GB             | 16 GB+ |
| GPU   | optional         | NVIDIA w/ CUDA |
| Camera| any UVC webcam   | Sony A7 III via Elgato Cam Link 4K |
| Disk  | 5 GB free        | 20 GB free |

Verify the camera is detected:
```bash
ls /dev/video*
v4l2-ctl --list-devices   # apt install v4l-utils if missing
```

---

## 2. System Packages

Run once on the target PC:

```bash
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git ffmpeg v4l-utils \
    libgl1 libglib2.0-0
```

> ⚠️ MediaPipe is **not** compatible with Python 3.13+. Stay on 3.10 / 3.11 / 3.12.

---

## 3. Clone the Repository

```bash
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
```

If you already cloned previously, just pull the latest changes:

```bash
git pull
```

---

## 4. Python Environment

### 4.1 Automatic (recommended)

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

This script:
- installs system packages if missing,
- creates `.venv/`,
- installs all Python deps from `requirements.txt`,
- lists detected `/dev/video*` cameras.

### 4.2 Manual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

---

## 5. Camera Permissions

Add your user to the `video` group (once):

```bash
sudo usermod -aG video "$USER"
# log out and back in for it to take effect
```

Test the camera directly with OpenCV:

```bash
python -c "import cv2; c=cv2.VideoCapture(0); ok,_=c.read(); print('camera ok:', ok); c.release()"
```

---

## 6. Run the Demo (No Webots Needed)

Just see the live camera feed with detected joints:

```bash
python run.py --no-webots
```

Or with the Makefile:

```bash
make demo
```

You should see:
- a window titled **"Pose Imitation - Camera Feed"**,
- the 33-landmark skeleton drawn on top of your body,
- a HUD with FPS, latency, frame count, and landmark count,
- **`Status: HUMAN DETECTED`** when at least ~30% of landmarks are visible.

Keyboard:
- **`q`** or **`ESC`** — quit.

---

## 7. Run Full Pipeline with Webots

### 7.1 Install Webots
Download the latest `.deb` from <https://cyberbotics.com/> and install:

```bash
wget https://github.com/cyberbotics/webots/releases/download/R2024a/webots_2024a_amd64.deb
sudo apt install ./webots_2024a_amd64.deb
```

### 7.2 Open the world

1. Launch Webots: `webots &`
2. `File → Open World…` → select `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`.
3. In the scene tree, set the humanoid robot's `controller` field to `pose_imitation_controller`.
4. `Tools → Preferences → Python command` → set to:
   ```text
   /<absolute-repo-path>/.venv/bin/python
   ```
5. Save the world.

### 7.3 Start the simulation
Press **▶ Play** in Webots. The controller binds UDP port `8765`.

### 7.4 Start the pipeline (separate terminal)

```bash
source .venv/bin/activate
python run.py
```

---

## 8. Configuration

Edit `configs/default.yaml` to change resolution, FPS targets, source, etc.

CLI flags override config values:

```text
--source 0                  # webcam index
--source data/sample.mp4    # video file
--no-webots                 # skip UDP send
--no-display                # run headless (servers/SSH)
--max-frames 300            # auto-stop
--log-level DEBUG           # verbose logs
```

---

## 9. Output Artifacts

Each run writes CSVs to `logs/run_YYYYMMDD_HHMMSS/`:

- `pose_keypoints.csv` — all 33 landmarks with `(x, y, z, visibility)` per frame.
- `joint_targets.csv` — retargeted robot joint angles (radians).

Use these for offline metrics (per-joint MAE, latency histograms).

---

## 10. Verify Build

```bash
pytest -q          # unit tests
ruff check src     # lint
make headless      # 100-frame smoke run (no display, no webots)
```

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open video source 0` | `ls /dev/video*` — confirm camera; `sudo usermod -aG video $USER` and re-login. |
| Black window / no skeleton | Increase lighting; ensure full body is visible; check `pose.use_mediapipe: true` in config. |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0`. |
| `qt.qpa.plugin: could not load` | `sudo apt install -y libxcb-xinerama0`. |
| Low FPS | Drop `input.width/height` to `640×480`; in `pose_estimator.py` set `model_complexity=0`. |
| Webots Python errors | `Tools → Preferences → Python command` must point to `.venv/bin/python` of this repo. |
| UDP packets not received | Both processes must run on same host; firewall must allow `127.0.0.1:8765/udp`. |

---

## 12. Git Workflow (push from dev → pull on target)

On your **development PC**:

```bash
git add .
git commit -m "feat: ready for target deployment"
git push origin main
```

On the **target Ubuntu PC**:

```bash
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt        # if requirements changed
python run.py --no-webots              # quick smoke test
python run.py                           # full pipeline with Webots
```

That's it — you're live.
