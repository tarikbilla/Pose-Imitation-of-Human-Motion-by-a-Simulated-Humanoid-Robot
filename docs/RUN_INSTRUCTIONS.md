# Run Instructions — Target PC (Ubuntu + Conda)

Complete deployment guide for running the project on a fresh Ubuntu machine after cloning the repository.
This document assumes you will `git pull` on the **target PC** and run from there, using a **Conda environment**.

> Tested on: **Ubuntu 22.04 LTS** and **Ubuntu 24.04 LTS** with **Python 3.11** in a Conda env named `y313`.

---

## 0. TL;DR (Copy–Paste)

```bash
# 1. Activate the conda environment
conda activate y313

# 2. Clone or pull the repository
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot

# 3. Install dependencies into the conda env
pip install -r requirements.txt

# 4. Run the demo (no Webots needed)
python run.py --no-webots
```

A window titled **"Pose Imitation - Camera Feed"** opens showing your camera with the 33-landmark skeleton drawn on top, **following your movement in real time**. Press **`q`** or **`ESC`** to quit.

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
    git ffmpeg v4l-utils \
    libgl1 libglib2.0-0
```

> ⚠️ MediaPipe is **not** compatible with Python 3.13+. Use Python **3.10**, **3.11**, or **3.12** inside your Conda env.

---

## 3. Conda Environment (`y313`)

### 3.1 Create the env (first time only)

If the env does not exist yet on the target PC:

```bash
conda create -n y313 python=3.11 -y
```

### 3.2 Activate it (every session)

```bash
conda activate y313
```

You should see your prompt change to `(y313) user@host:~$`.

Verify Python version:

```bash
python --version
# Python 3.11.x
```

---

## 4. Clone / Pull the Repository

First-time clone:

```bash
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
```

Subsequent updates:

```bash
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
git pull
```

---

## 5. Install Python Dependencies

With the conda env active:

```bash
conda activate py313
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

Verify MediaPipe imports successfully:

```bash
python -c "import mediapipe as mp; print('mediapipe', mp.__version__)"
```

> If this fails, the pipeline will raise `PoseEstimatorError` at startup. Fix MediaPipe before running.

---

## 6. Camera Permissions

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

## 7. Run the Demo (No Webots Needed)

```bash
conda activate y313
python run.py --no-webots
```

You should see:
- a window titled **"Pose Imitation - Camera Feed"**,
- the **33-landmark skeleton drawn on your body**, following your motion in real time,
- a HUD showing **FPS**, **Latency**, **Frame**, **Landmarks visible**, **Status: HUMAN DETECTED**, and **Source: MediaPipe**.

Keyboard shortcuts:
- **`q`** or **`ESC`** — quit.

If the HUD shows **"Source: SYNTHETIC"**, MediaPipe is **not** installed correctly — re-run Step 5.

---

## 8. Run Full Pipeline with Webots

### 8.1 Install Webots
Download the latest `.deb` from <https://cyberbotics.com/> and install:

```bash
wget https://github.com/cyberbotics/webots/releases/download/R2024a/webots_2024a_amd64.deb
sudo apt install ./webots_2024a_amd64.deb
```

### 8.2 Open the world

1. Launch Webots: `webots &`
2. `File → Open World…` → select `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`.
3. In the scene tree, set the humanoid robot's `controller` field to `pose_imitation_controller`.
4. `Tools → Preferences → Python command` → set to the conda env's Python:
   ```bash
   # find your conda env path:
   conda activate y313 && which python
   # e.g.  /home/<user>/miniconda3/envs/y313/bin/python
   ```
5. Paste that path into the Webots Python command field. Save the world.

### 8.3 Start the simulation
Press **▶ Play** in Webots. The controller binds UDP port `8765`.

### 8.4 Start the pipeline (separate terminal)

```bash
conda activate y313
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
python run.py
```

---

## 9. Configuration

Edit `configs/default.yaml` to tune behavior. Important keys:

```yaml
input:
  source: 0              # webcam index, or "data/sample.mp4"
  width: 1280
  height: 720
  flip_horizontal: true  # selfie-mirror: skeleton tracks your left/right naturally

pose:
  use_mediapipe: true
  model_complexity: 1    # 0=lite (fastest), 1=full, 2=heavy
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  allow_synthetic_fallback: false   # keep false in production
```

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

## 10. Output Artifacts

Each run writes CSVs to `logs/run_YYYYMMDD_HHMMSS/`:

- `pose_keypoints.csv` — all 33 landmarks with `(x, y, z, visibility)` per frame.
- `joint_targets.csv` — retargeted robot joint angles (radians).

---

## 11. Verify Build

```bash
conda activate y313
pytest -q          # unit tests
ruff check src     # lint
```

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Window opens but skeleton does not follow you | HUD shows "Source: SYNTHETIC" → MediaPipe missing. Run `pip install -r requirements.txt` inside the active conda env. |
| `PoseEstimatorError: MediaPipe is not installed` | Same as above. Confirm with `python -c "import mediapipe"`. |
| `Cannot open video source 0` | `ls /dev/video*` — confirm camera; `sudo usermod -aG video $USER` and re-login. |
| Skeleton tracks but mirrored | Set `input.flip_horizontal: false` in config. |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0`. |
| `qt.qpa.plugin: could not load` | `sudo apt install -y libxcb-xinerama0`. |
| Low FPS | Drop `input.width/height` to `640×480`; set `pose.model_complexity: 0`. |
| Webots Python errors | Set Webots `Python command` to `which python` from the active `y313` conda env (Step 8.2.4). |
| UDP packets not received | Same host; firewall must allow `127.0.0.1:8765/udp`. |

---

## 13. Git Workflow (push from dev → pull on target)

On your **development PC**:

```bash
git add .
git commit -m "feat: ready for target deployment"
git push origin main
```

On the **target Ubuntu PC**:

```bash
conda activate y313
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
git pull origin main
pip install -r requirements.txt        # if requirements changed
python run.py --no-webots              # quick smoke test (window + live skeleton)
python run.py                          # full pipeline with Webots
```

That's it — you're live.
