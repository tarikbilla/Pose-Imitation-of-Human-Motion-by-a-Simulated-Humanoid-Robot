# Run Instructions — VS Code + Webots (Conda-Only)

This guide shows how to run the project from both sides: the Python pipeline in VS Code, and the Webots robot controller.
**All dependencies are managed via Conda.** No pip or external package managers needed.

> **Environment**: Ubuntu 22.04 / 24.04, Python 3.12 in Conda env `py312`, MediaPipe 0.10.13 (installed via pip within conda), Webots R2024a

---

## 0. Quick Start

### Option A: Camera-only preview in VS Code
```bash
conda activate py312
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
python run.py --no-webots
```

### Option B: Full Webots live imitation
1. Launch Webots and open `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
2. Set the robot controller to `pose_imitation_controller`
3. Start the simulation in Webots
4. In VS Code terminal:
```bash
conda activate py312
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
python run.py
```

---

## 1. Requirements

### Software
- Ubuntu 22.04 or 24.04
- Conda with Python 3.12
- Webots installed
- Working webcam available on the machine

### Project files
- `run.py` — Python pipeline
- `configs/default.yaml` — pipeline configuration
- `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt` — Webots world
- `main/controllers/pose_imitation_controller/pose_imitation_controller.py` — Webots robot controller

---

## 2. Create and activate the Conda environment

### 2.1 Create env (first time only)
```bash
conda create -n py312 python=3.12 -y
```

### 2.2 Activate env
```bash
conda activate py312
```

### 2.3 Install all dependencies from environment.yml
```bash
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
conda env create -f environment.yml -y
```

### 2.4 Install MediaPipe via conda's pip (one-time)
```bash
conda activate py312
conda run -n py312 pip install mediapipe==0.10.13
```

### 2.5 Verify installation
```bash
conda activate py312
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); print('Has solutions:', hasattr(mp, 'solutions'))"
```

You should see: `MediaPipe version: 0.10.13` and `Has solutions: True`

---

## 3. VS Code setup

### 3.1 Open the project in VS Code
Open this folder in VS Code:
```
/home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
```

### 3.2 Select the Conda interpreter in VS Code
- Press `Ctrl+Shift+P`
- Select `Python: Select Interpreter`
- Choose: `./miniconda3/envs/py312/bin/python` (or similar path showing `py312`)

> If `py312` doesn't appear, run `conda activate py312` in a terminal first to ensure the env exists.

### 3.3 Open an integrated terminal
- `Terminal → New Terminal`
- Confirm active env:
```bash
python --version
```

---

## 4. Run the Python pipeline (all via conda)

### 4.1 Activate the conda environment in terminal
```bash
conda activate py312
```

### 4.2 Camera-only demo
```bash
python run.py --no-webots
```

### 4.3 Headless mode (no camera window)
```bash
python run.py --no-display --no-webots
```

### 4.4 Full Webots integration
```bash
python run.py
```

> Note: the current Webots controller focuses on safe upper-body imitation.
> Lower-body joints (`LHipPitch`, `RHipPitch`, `TorsoPitch`) are ignored to keep the robot upright until full balance and ankle control are implemented.

### 4.5 Optional flags
- `--source 0` — use webcam index 0  
- `--max-frames 300` — stop after 300 frames
- `--log-level DEBUG` — verbose debug logs

Example:
```bash
python run.py --log-level DEBUG
```

---

## 5. Webots setup

### 5.1 Install Webots
Download and install Webots from <https://cyberbotics.com/>.

### 5.2 Open the world
1. Launch Webots: `webots &`
2. Open:
```
main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt
```

### 5.3 Set the controller
In the Webots scene tree, set the humanoid robot's `controller` field to:
```
pose_imitation_controller
```

### 5.4 Set the Python command in Webots to use conda env
1. Open Webots and go to `Tools → Preferences`
2. Find the `Python command` field
3. Set it to the **absolute path** of the conda env's Python:
```bash
/home/CSPM26/miniconda3/envs/py312/bin/python
```
4. Click `OK` to save and close

---

## 6. Run the full system (Conda + VS Code + Webots)

### Step-by-step execution
1. **In Webots**:
   - Open the world: `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
   - Set the humanoid robot's controller to: `pose_imitation_controller`
   - Verify Python command is set to: `/home/CSPM26/miniconda3/envs/py312/bin/python`
   - Press **▶ Play** to start simulation

2. **In VS Code terminal**:
   ```bash
   conda activate py312
   cd /home/CSPM26/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
   python run.py
   ```

3. **Move in front of the camera** — the robot will follow your movements

### What happens behind the scenes
- VS Code Python pipeline (conda `py312`) detects your pose using MediaPipe
- Camera feed with landmarks appears on screen
- Joint commands are sent over UDP port `8765` to Webots
- The Webots controller (using the same `py312` env) receives commands and moves the robot
- Robot limbs follow your motion in real time

---

## 7. How to verify it works

### In VS Code
- The camera window appears
- HUD shows `Source: MediaPipe`
- `Landmarks: XX/33` updates
- `Status: ✓ HUMAN DETECTED`

### In Webots
- The humanoid robot moves in response to the pipeline
- The controller logs show UDP activity
- If the robot does not move, verify the simulation is playing and the controller is active

---

## 8. Important configuration points

Edit `configs/default.yaml` for tuning:
```yaml
input:
  source: 0
  width: 1280
  height: 720
  flip_horizontal: true

pose:
  use_mediapipe: true
  model_complexity: 1
  min_detection_confidence: 0.35
  min_tracking_confidence: 0.35
  allow_synthetic_fallback: false
```

Important: keep `allow_synthetic_fallback: false` so the system uses real MediaPipe pose tracking.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `conda env create -f environment.yml -y` to install all conda dependencies. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Ensure correct MediaPipe version: `conda run -n py312 pip install mediapipe==0.10.13` |
| HUD shows `Source: SYNTHETIC` | MediaPipe not loaded. Verify: `python -c "import mediapipe; print(hasattr(mediapipe, 'solutions'))"` |
| Skeleton does not follow movement | Ensure `--no-webots` is NOT used. If using it, camera-only mode is expected (no Webots). |
| Robot does not move in Webots | (1) Webots is playing (▶), (2) controller is `pose_imitation_controller`, (3) Python command set correctly. |
| Webots controller fails to start | Verify Webots Python command: `Tools → Preferences → Python command = /home/CSPM26/miniconda3/envs/py312/bin/python` |
| Camera fails to open | Check: `ls /dev/video*` exists, add user to video group: `sudo usermod -aG video $USER`, then reboot. |

---

## 10. Optional checks

Run tests:
```bash
pytest -q
```

Check Python syntax:
```bash
python -m py_compile src/perception/pose_estimator.py
```

---

## 11. Quick VS Code run sequence

1. Open repository in VS Code
2. Select `py312` conda interpreter (Ctrl+Shift+P → Python: Select Interpreter)
3. Open integrated terminal
4. Activate conda env:
   ```bash
   conda activate py312
   ```
5. **Camera-only preview**:
   ```bash
   python run.py --no-webots
   ```
6. **With Webots live robot control**:
   ```bash
   python run.py
   ```

---

## 12. Quick Webots setup sequence

1. Launch Webots
2. File → Open World: `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
3. Scene tree: Set humanoid robot `controller` = `pose_imitation_controller`
4. Tools → Preferences → Python command = `/home/CSPM26/miniconda3/envs/py312/bin/python`
5. Click OK to save
6. Press **▶ Play** button to start simulation
7. In VS Code: `conda activate py312` → `python run.py`

---

## 13. Important notes

- **All commands use conda**: `conda activate py312` before running any Python code
- **Environment file**: `environment.yml` is the single source of truth for all dependencies
- **MediaPipe version**: Fixed at `0.10.13` for Python 3.12 compatibility  
- **Webots UDP port**: Controller listens on `8765` (do not change)
- **Pipeline sends automatically**: Joint commands to Webots start immediately when camera detects motion
- **Camera-only testing**: Use `--no-webots` flag to test perception without starting Webots
- **No pip in workspace**: All deps are managed via conda (MediaPipe installed via conda's pip for compatibility)

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
