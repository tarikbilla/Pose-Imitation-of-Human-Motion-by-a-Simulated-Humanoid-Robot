# Run Instructions — VS Code + Webots (Conda-Only)

This guide shows how to run the project from both sides: the Python pipeline in VS Code, and the Webots robot controller.
**All dependencies are managed via Conda.** No pip or external package managers needed.

> **Environment**: Ubuntu 22.04 / 24.04, Python 3.12 in Conda env `py312`, TensorFlow + TensorFlow-Hub running [MeTRAbs](https://github.com/isarandi/metrabs) (installed via pip within conda), Webots R2024a
>
> **First run downloads the MeTRAbs model** (371 MB on disk) into `~/.cache/metrabs`;
> set `$METRABS_CACHE_DIR` to put it elsewhere. It is deliberately *not* kept in
> a temp directory, so a reboot does not throw it away.
>
> **GPU required.** MeTRAbs needs a CUDA-enabled TensorFlow build for real-time
> inference -- see step 2.4. Without a GPU, either run on a different machine or
> set `pose.allow_synthetic_fallback: true` in `configs/default.yaml` (the robot
> will NOT follow you in that mode; it's only useful to smoke-test the rest of
> the pipeline). The pretrained model weights are non-commercial-use only
> (training-data license) -- see MeTRAbs'
> [MODELS_6_DATASETS.md](https://github.com/isarandi/metrabs/blob/master/docs/MODELS_6_DATASETS.md).

---

## How long this takes

Measured 2026-09-18 on the target PC (RTX 3090 Ti + RTX 3070, Python 3.12).
Per-frame latency is in [`WORKFLOW.md`](WORKFLOW.md#latency-budget); this table
is the wall-clock you actually wait through.

### First-time setup — **~20–40 min**, mostly network

| Step | Duration | Bounded by |
|---|---|---|
| 2.1 `conda create` | ~1–2 min | local |
| 2.3 `conda env create -f environment.yml` | **~5–15 min** | download |
| 2.4 `pip install tensorflow tensorflow-hub` | **~5–10 min** | download (~600 MB) |
| 2.5 Verify TF sees the GPU | seconds | — |
| 2.6 First model download (371 MB) | **~2–5 min** | download |
| 5.1 Install Webots | ~5–10 min | download |

### Every run — **~45–70 s before the robot tracks you**

| Step | Duration | Note |
|---|---|---|
| MeTRAbs model load | **31 s** | cold load of the cached SavedModel |
| TensorFlow graph warm-up | **15 s** | the first `estimate()` call |
| Webots world load | ~10–20 s | in parallel if you use `make run` |
| IMU auto-zero | 1.0 s standing | needs feet loaded, or the tilt gates stay idle |

> **The first ~45 s is not a hang.** The OpenCV window opens *before* TensorFlow
> finishes warming up, so the earliest frames are untracked and the robot will
> not move. Wait for `Pose estimator: MeTRAbs (real human tracking active)`.

### While running

| Thing | Duration |
|---|---|
| Camera loop | **63 ms/frame p50 → 16 FPS** |
| You move → motors move | **~121 ms** |
| You start walking → robot starts | **~1.35 s** |
| You stop walking → robot stops | **0.8–3.2 s** |
| You turn 90° → robot has turned | **4.6 s** |
| Fall → reset → ready again | ~2–3 s |
| `pytest -q` (575 tests) | **3.6 min** |

---

## 0. Quick Start

### Option A: Camera-only preview in VS Code
```bash
conda activate py312
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
python run.py --no-webots
```

### Option B: Full Webots live imitation — one command
```bash
conda activate py312
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
make run                       # == python run.py --launch-webots
```
This opens the project world in Webots and starts the pipeline against it. Press
▶ in Webots if the simulation does not start playing on its own.

### Option C: drive a Webots you opened yourself
1. Launch Webots and open `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
2. Set the robot controller to `pose_imitation_controller`
3. Start the simulation in Webots
4. In VS Code terminal:
```bash
conda activate py312
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
make pipeline                  # == python run.py --no-launch-webots
```
(Plain `python run.py --launch-webots` is also safe here: it detects the running
Webots and attaches to it rather than opening a second one.)

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
*(~5–15 min, network-bound)*
```bash
cd /home/<user>/CS_Group_C_2026/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
conda env create -f environment.yml -y
```

### 2.4 Install TensorFlow + TensorFlow-Hub via conda's pip (one-time)
*(~5–10 min, ~600 MB download)*
Install a TensorFlow build matching this machine's CUDA/cuDNN driver version
(check with `nvidia-smi`, then see the
[TensorFlow GPU install guide](https://www.tensorflow.org/install/pip) for the
matching version pin -- this project was written without GPU access to verify
an exact version, see `requirements.txt`):
```bash
conda activate py312
conda run -n py312 pip install "tensorflow>=2.12,<2.16" "tensorflow-hub>=0.15,<0.17"
```

### 2.5 Verify installation
```bash
conda activate py312
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

You should see a TensorFlow version and a non-empty GPU list, e.g.
`GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. If
the list is empty, the pipeline will refuse to start (see the GPU note above)
-- fix the CUDA/cuDNN install before continuing.

### 2.6 First model download + skeleton check (one-time)
*(~2–5 min: 371 MB download, then a ~31 s model load and a ~15 s warm-up)*
The first run downloads and caches the MeTRAbs model (this can take a few
minutes). Also confirms `src/perception/landmarks.py`'s joint-name mapping
actually matches this model (see that file's docstring for why this matters):
```bash
python scripts/inspect_metrabs_skeleton.py
```
It should print `OK: every canonical landmark matched, no leftover raw names.`
at the end. If it instead lists `MISSING canonical landmarks`, fix
`CANONICAL_TO_RAW_ALIASES` in `src/perception/landmarks.py` using the raw
names the script printed before doing anything else -- every downstream joint
lookup depends on that mapping being correct.

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

> Note: the controller drives the **whole body** — arms, head and legs. In front
> of the camera you can:
>
> * **squat** → the robot squats (symmetric, statically balanced);
> * **raise one leg** → the robot shifts its weight onto the other foot and then
>   raises the matching leg, as far as its own centre-of-mass model says is safe;
> * **walk / march** → the robot walks across the floor for real, using Webots'
>   pre-balanced NAO walk clips (it marches in place if no clips are installed);
> * **turn your body** → the robot steps round to face the same way, closing the
>   loop on its InertialUnit heading.
>
> One knob controls all of it: `LEG_CONTROL` at the top of
> `main/controllers/pose_imitation_controller/pose_imitation_controller.py`
> (`"auto"` = everything, `"pose"` = stay on the spot, `"engine"` = march in
> place, `"off"` = upper body only). Set `walk.enabled: false` in
> `configs/default.yaml` to stop streaming gait/yaw cues altogether.
> Full explanation: `main/controllers/pose_imitation_controller/README.md`.

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
- VS Code Python pipeline (conda `py312`) detects your pose using MeTRAbs (GPU)
- Camera feed with landmarks appears on screen
- Joint commands are sent over UDP port `8765` to Webots
- The Webots controller (using the same `py312` env) receives commands and moves the robot
- Robot limbs follow your motion in real time

---

## 7. How to verify it works

### In VS Code
- The camera window appears
- HUD shows `Source: MeTRAbs`
- `Landmarks: XX/19` updates
- `Status: ✓ HUMAN DETECTED`

### In Webots
- The humanoid robot moves in response to the pipeline
- The controller console prints a status line every 100 frames, e.g.
  `Frame 400 | sim 49.8 Hz | tracking | legs=pose | 8 joints applied`.
  `legs=` tells you which layer is driving the legs (`pose`, `march:march`,
  `motion:forward`, `stand`) — the quickest way to see what the robot thinks you
  are doing.
- At startup it prints `Locomotion clips found: forward, turn_left, turn_right, …`.
  If it instead warns that **no NAO .motion files were found**, the robot will
  march in place rather than walking across the floor: set `$WEBOTS_HOME`, or copy
  the clips from `<webots>/projects/robots/softbank/nao/motions/` into
  `main/controllers/pose_imitation_controller/motions/`.

### If nothing moves at all
Read the startup block the controller prints in the Webots console — it lists
every layer as `ON` or `OFF`, and flags anything that is off with what it costs
you. If that block never appeared, the controller died before its first step;
look for a Python traceback (usually Webots' Python interpreter is missing NumPy:
`Tools → Preferences → Python command`).

The per-100-frame status line then says in plain language what the legs are
doing, e.g. `legs: holding: centre of mass not yet over the stance foot` or
`legs: knees and feet both out of frame; leg lift cannot be seen`.

### If the robot never walks forward
Forward walking needs the gait detector to see you *marching*, and that needs
your **knees** in frame. Watch the `Gait: … conf` field in the camera window: it
has to stay above 0.6. Recorded runs had the knees visible on only 53–65% of
frames with a detected torso — step back from the camera until your knees show.

### If a raised leg does nothing
Check `lift-cue` in that status line. `none` means neither your feet nor your
knees are in frame, so the lift is literally invisible — step back from the
camera until at least your knees show.

### If the legs do not move at all
The most common cause is a world file missing the NAO foot/floor contact pair —
the feet then slide on the floor and absorb every leg command. The committed
world already has it; see section 7 of
`main/controllers/pose_imitation_controller/README.md`.
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
  use_metrabs: true
  model_url: "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip"
  skeleton: coco_19
  default_fov_degrees: 55.0
  detector_threshold: 0.3
  num_aug: 1
  max_detections: 1
  require_gpu: true
  allow_synthetic_fallback: false
```

Important: keep `allow_synthetic_fallback: false` so the system uses real MeTRAbs pose tracking.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `conda env create -f environment.yml -y` to install all conda dependencies. |
| `MetrabsUnavailableError: No GPU visible to TensorFlow` | Fix the CUDA/cuDNN install (see step 2.4-2.5), or run on a machine with a GPU. |
| `ModuleNotFoundError: No module named 'tensorflow_hub'` | `conda run -n py312 pip install "tensorflow>=2.12,<2.16" "tensorflow-hub>=0.15,<0.17"` |
| `PoseEstimatorError: ... joint names did not match ...` | Run `python scripts/inspect_metrabs_skeleton.py` and fix `CANONICAL_TO_RAW_ALIASES` in `src/perception/landmarks.py` using the raw names it prints. |
| HUD shows `Source: SYNTHETIC` | MeTRAbs not loaded (see the log line above the HUD for why -- usually the GPU check or a skeleton mismatch). |
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
- **TensorFlow/GPU**: MeTRAbs needs a CUDA-enabled TensorFlow build matching this machine's driver -- see step 2.4
- **Webots UDP port**: Controller listens on `8765` (do not change)
- **Pipeline sends automatically**: Joint commands to Webots start immediately when camera detects motion
- **Camera-only testing**: Use `--no-webots` flag to test perception without starting Webots
- **No pip in workspace**: All deps are managed via conda (TensorFlow/TensorFlow-Hub installed via conda's pip for compatibility)

| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0`. |
| `qt.qpa.plugin: could not load` | `sudo apt install -y libxcb-xinerama0`. |
| Low FPS | Drop `input.width/height` to `640×480`; switch `pose.model_url` to a smaller/faster backbone (e.g. `metrabs_rn18_y4` or `metrabs_mob3s_y4`); lower `pose.num_aug` (already 1 by default). |
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
