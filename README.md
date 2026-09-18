# Pose Imitation of Human Motion by a Simulated Humanoid Robot 

A Linux-first, real-time pipeline that:

1. Captures live video from a webcam (or replays a video file).
2. Runs **[MeTRAbs](https://github.com/isarandi/metrabs)** (GPU-accelerated) to
   detect a human and estimate **absolute 3D body landmarks** (millimeters,
   camera-frame), not just a 2D projection.
3. Draws the live skeleton on top of the camera feed in an OpenCV window.
4. Streams the landmarks (plus gait and body-yaw cues) to a **Webots** simulated
   NAO H25 over UDP.
5. The Webots controller retargets them to the full NAO pose and drives the robot
   while keeping it on its feet.

> **GPU required.** MeTRAbs needs a CUDA-enabled TensorFlow install for
> real-time inference -- see [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md).
> The pretrained model weights are non-commercial-use only (training-data
> license; the MeTRAbs code itself is MIT) -- see
> [MODELS_6_DATASETS.md](https://github.com/isarandi/metrabs/blob/master/docs/MODELS_6_DATASETS.md).

## What the robot does

| You | The robot |
|---|---|
| Move your arms | Follows both arms (shoulder pitch **and** roll, elbow) |
| Turn / nod your head | Follows head yaw and pitch |
| Squat | Squats 1:1 with you, to 40° hip / 80° knee |
| **Spread your legs** | Widens its stance 1:1 with you, to ~30° per leg (limited by how far the ankle can keep the sole flat) |
| **Raise one leg** | Shifts its weight onto the other foot, *then* raises the matching leg — as far as its own centre-of-mass model says is safe |
| Lean, or split your stance | Follows partly — these move the centre of mass, so they stay limited |
| **Walk / march** | Walks across the floor for real (its world coordinates change), using Webots' pre-balanced NAO walk clips; marches in place if no clips are installed |
| **Turn your body** | Steps round to face the same way — through the **full circle**, including turning to face behind you — closing the loop on the InertialUnit heading |

The lower body is the interesting part: a single camera cannot see whether NAO's
centre of mass is over a foot, so the camera only ever supplies the *desired* leg
pose and the robot's own forward-kinematics CoM model decides how much of it is
safe to execute.

How much gets through depends on the *symmetry* of the pose, not its size. A
wider stance or a deeper squat is mirror-symmetric: it moves the centre of mass
not at all, and a wider stance actually enlarges the support polygon — so those
pass at full authority, 1:1 with you. A lean or a split stance does move the
centre of mass, so it stays limited. Details and the maths:
[`main/controllers/pose_imitation_controller/README.md`](main/controllers/pose_imitation_controller/README.md).

> Full requirements specification: [`docs/PRD.md`](docs/PRD.md)
> Complete install & run guide (for the target PC): [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

---

## Quickstart (Ubuntu + Conda)

```bash
# 1. Activate the conda environment (create it first if needed):
#    conda create -n y313 python=3.11 -y
conda activate py313

# 2. Clone or pull
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot

# 3. Install dependencies into the conda env
pip install -r requirements.txt

# 4. Camera-only demo (no Webots required) — shows live skeleton overlay:
python run.py --no-webots

# ...or the whole thing, Webots included:
make run
```

### How long each step takes

| Step | Duration | How often |
|---|---|---|
| 3. `pip install` / `conda env create` | **~5–15 min** | once per machine (network-bound) |
| First run: MeTRAbs model download (371 MB) | **~2–5 min** | once per machine |
| **Every** run: model load | **31 s** | each pipeline start |
| **Every** run: TensorFlow graph warm-up | **15 s** | the first inference call |
| **→ total start-up before tracking begins** | **~45–50 s** | budget this before a demo |
| Webots world load | **~10–20 s** | each run |
| `pytest -q` (575 tests) | **3.6 min** | per change |

> **The first ~45 s is not a hang.** The OpenCV window opens before TensorFlow
> has finished warming up, so early frames are untracked. Wait for
> `Pose estimator: MeTRAbs (real human tracking active)` in the console.

> **First run downloads the MeTRAbs model** (371 MB) into `~/.cache/metrabs`
> (override with `$METRABS_CACHE_DIR`). It is kept there, not in a temp
> directory, so a reboot does not throw it away.

Press **`q`** or **`ESC`** in the window to quit.

> Full setup guide: [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

---

## Performance & Latency

Measured 2026-09-18 on the target PC (RTX 3090 Ti + RTX 3070, Webots
`basicTimeStep` 20 ms, `metrabs_eff2s_y4` at 1920×1080). Full per-stage
breakdown and method: [`docs/WORKFLOW.md`](docs/WORKFLOW.md#latency-budget).

**Two budgets, different causes — don't confuse them.**

### 1. Motion latency — you move, the motors move: **~121 ms**

| Stage | Library | p50 |
|---|---|---|
| **MeTRAbs inference + capture + overlay** | TensorFlow + TF-Hub, OpenCV | **~63 ms** (whole loop) |
| Keypoint smoothing (3 axes) | `OneEuroFilter` | 0.07 ms |
| Gait cue + action cue | in-repo | 0.04 ms |
| UDP send → receive | `socket` + `json` | 0.005 ms |
| Controller step quantisation | Webots | ≤20 ms |
| Arm/head filter residual | `ArmTracker` | 38 ms |

Whole-loop camera period, measured over **31 243 recorded frames with a real
subject**: **63 ms p50 / 95 ms p90 → 16.0 FPS** effective, against a
`runtime.latency_budget_ms` of 150 ms.

Inference is essentially all of that 63 ms. Per-mode it costs **75.9 ms** when
the YOLOv4 person detector runs and **36.2 ms** on a tracked box; at
`pose.detect_interval: 2` the mix averages ~56 ms, and capture, overlay, cues,
smoothing and UDP together add under 7 ms. *(A no-subject frame returns in
~42 ms — detector only, pose network skipped — so don't benchmark on blank
input.)*

Everything that is not the GPU costs **0.1 ms combined**. If you want this
faster, change `pose.model_url` to a smaller backbone; tuning anything else is
tuning noise.

### 2. Decision latency — you act, the robot commits to a clip: **seconds**

A motion clip is a *commitment*: while it plays it owns the 12 leg joints, open
loop. This is what people actually perceive as lag.

| Event | Measured | Bounded by |
|---|---|---|
| Walk starts | **~1.35 s** | 790 ms cue + 560 ms prepare ramp |
| Walk stops | **0.8 – 3.2 s** | 165 ms cue + 600 ms latch + ≤2.46 s clip |
| Walk speed | **0.073 m/s** | motor-limited — the clip already peaks at 84% of rated speed |
| Turn 90° | **4.6 s**, 1 clip, ±1.1° | the turn clip, at 20.7 °/s |
| Turn 180° | **8.5 s**, 1 clip, ±3.9° | " |
| Squat / one-leg / stance width | **~121 ms** | no clip — pose imitation is continuous |

Webots realtime factor: **0.984** (20.3 ms wall per 20 ms step, over 5 725 s).

---

## Repository Layout

```text
.
├── configs/default.yaml              # runtime configuration
├── docs/
│   ├── PRD.md                        # product requirements (incl. MeTRAbs 3D landmarks)
│   └── RUN_INSTRUCTIONS.md           # full setup guide for target PC
├── main/                             # Webots project root
│   ├── worlds/…​.wbt                  # world + REQUIRED NAO foot/floor contact
│   ├── controllers/pose_imitation_controller/
│   └── libraries/                    # Webots-free (unit-tested) robot maths
│       ├── nao_retarget.py           # landmarks → NAO angles (incl. per-leg solve)
│       ├── lower_body.py             # weight-shift / single-leg-lift sequencer
│       ├── balance.py                # model-based CoM balance
│       ├── gait.py                   # in-place march engine
│       ├── walk_motion.py            # motion clips + heading (yaw) servo
│       └── pose_control_utils.py     # NaoPoseDriver: limits, smoothing, logging
├── scripts/setup_ubuntu.sh           # one-shot Ubuntu setup
├── src/
│   ├── perception/                   # video input + pose estimation + visualizer
│   ├── retargeting/                  # keypoints → joint angles
│   ├── utils/                        # config, fps controller, smoother, csv logger
│   ├── pipeline.py                   # end-to-end orchestrator
│   ├── run.py                        # CLI entrypoint
│   ├── types.py                      # dataclasses
│   └── webots_bridge.py              # UDP bridge to Webots controller
├── tests/                            # pytest unit tests
├── Makefile                          # make setup | run | demo | test | lint
├── requirements.txt
└── run.py                            # `python run.py`
```

## CLI Flags

```text
python run.py [--config configs/default.yaml]
              [--source 0|path/to/video.mp4]
              [--no-webots]            # skip UDP send (pure perception demo)
              [--launch-webots]        # start Webots too (whole demo, one command)
              [--no-launch-webots]     # never start it; Webots is already open
              [--no-display]           # headless, no OpenCV window
              [--max-frames N]
              [--log-level INFO|DEBUG|WARNING|ERROR]
```

`--launch-webots` **attaches to an already-running Webots** rather than opening a
second instance — two of them would each load the world and then fight over
UDP 8765.

## Make Targets

| Target | Description |
|---|---|
| `make setup`    | Provision venv + apt deps (Ubuntu). |
| `make run`      | The whole demo: launches Webots **and** the pipeline. |
| `make pipeline` | Pipeline only, against a Webots you opened yourself. |
| `make demo`     | Camera + window only (no Webots needed). |
| `make headless` | 100 frames, no window — CI smoke test. |
| `make test`     | Run pytest. |
| `make lint`     | Run ruff. |
| `make format`   | Black + ruff --fix. |

## Tests

```bash
pytest -q        # 575 tests, ~3.6 min
```

`tests/test_controller_integration.py` dominates the runtime — it drives the
real control loop against a fake Webots. For a fast inner loop while working on
the locomotion maths, `pytest tests/test_walk_motion.py -q` runs in seconds.

All the robot-side maths lives in `main/libraries/` and imports **no** Webots
module, so it is fully unit-tested on a machine without Webots installed.

## Webots setup in one line

```bash
make run          # or: python run.py --launch-webots
```

That opens `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
and starts the pipeline against it. Press ▶ in Webots if the simulation is
paused. To drive a Webots you opened yourself, use `make pipeline` instead.

Two world settings are **required**, and both are already in the committed world
file — if you build your own world, copy them or the legs will not work:

```
WorldInfo {
  basicTimeStep 20                     # Webots' default 32 ms is too coarse for NAO
  contactProperties [
    ContactProperties {                # Nao.proto tags its soles "NAO foot material";
      material2 "NAO foot material"    # without this pair the feet slide and the
      coulombFriction [ 8 ]            # robot cannot load a foot or take a step
      bounce 0
      bounceVelocity 0.003
    }
  ]
}
```

## If something does not move

The controller prints a startup block in the Webots console listing every layer
as `ON`/`OFF`, then a status line every 100 frames saying in plain language what
the legs are doing and why (`legs: holding: centre of mass not yet over the
stance foot`, `lift-cue=none` when your legs are out of frame, and so on). Start
there — it names the cause in one line.

See [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md) for Webots configuration and troubleshooting.
