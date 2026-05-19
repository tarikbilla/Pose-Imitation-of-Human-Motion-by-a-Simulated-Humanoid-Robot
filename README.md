# Pose Imitation of Human Motion by a Simulated Humanoid Robot

A Linux-first, real-time pipeline that:

1. Captures live video from a webcam (or replays a video file).
2. Runs **MediaPipe Pose** to detect a human and all **33 body landmarks**.
3. Draws the live skeleton on top of the camera feed in an OpenCV window.
4. Retargets the human pose to humanoid joint commands.
5. Streams those commands to a **Webots** simulated humanoid via UDP.

> Full requirements specification: [`docs/PRD.md`](docs/PRD.md)
> Complete install & run guide (for the target PC): [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

---

## Quickstart (Ubuntu)

```bash
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
bash scripts/setup_ubuntu.sh
source .venv/bin/activate

# Camera-only demo (no Webots required) — shows live skeleton overlay:
python run.py --no-webots
```

Press **`q`** or **`ESC`** in the window to quit.

---

## Repository Layout

```text
.
├── configs/default.yaml              # runtime configuration
├── docs/
│   ├── PRD.md                        # product requirements (incl. 33 landmarks)
│   └── RUN_INSTRUCTIONS.md           # full setup guide for target PC
├── main/                             # Webots project root
│   ├── worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt
│   └── controllers/pose_imitation_controller/
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
              [--no-display]           # headless, no OpenCV window
              [--max-frames N]
              [--log-level INFO|DEBUG|WARNING|ERROR]
```

## Make Targets

| Target | Description |
|---|---|
| `make setup`    | Provision venv + apt deps (Ubuntu). |
| `make run`      | Full pipeline (camera + Webots bridge + window). |
| `make demo`     | Camera + window only (no Webots needed). |
| `make headless` | 100 frames, no window — CI smoke test. |
| `make test`     | Run pytest. |
| `make lint`     | Run ruff. |
| `make format`   | Black + ruff --fix. |

## Tests

```bash
pytest -q
```

See [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md) for Webots configuration and troubleshooting.
