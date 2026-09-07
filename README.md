# A3 — Pose Imitation with Atlas

Rebuild of the pipeline camera → 3D pose → retargeting → Webots Atlas.
Self-contained; shares no code with `src/`, `main/` or `scripts/`.

## Start here

| Document | Contents |
|---|---|
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | **Consolidated findings** — every decision with its rationale, all measurements, all failures |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | **How to run it** — setup, live operation, parameters, troubleshooting |
| [`docs/PORTING_LAB.md`](docs/PORTING_LAB.md) | **Moving to the lab rig** — Sony A7 III, NVIDIA/CUDA, Linux/macOS |

The per-milestone records `docs/M0_RESULTS.md` … `docs/M8_RESULTS.md`,
`docs/PLAN.md` and `docs/SUMMARY.md` remain as the raw history;
`FINDINGS.md` consolidates all of them.

## Run it live

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\run_live.py
```

Starts Webots and the camera driver, and opens a monitoring window showing the
camera image with the detected 2D skeleton drawn on it. `q` quits, `m` toggles
mirroring, `n` resets the distance datum, `s` saves a snapshot.

## Run it against a recording

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\make_angles.py
C:\venvs\a3-pose\Scripts\python.exe tools\run_puppet.py --env A3_ANGLES=<absolute path>
```

Deterministic and reproducible; this is how every reported measurement was
taken. Results land in `results/puppet_run.json`.

The whole-body QP from M4 (`webots\worlds\m4_wholebody.wbt`) is **not**
recommended: measured, its imitation error is 15.6° against 5.0° for direct
control. See [`docs/FINDINGS.md`](docs/FINDINGS.md) §8.

## Setup

Requirements: Python 3.11, Webots R2025a, Windows.

```bat
python tools\setup_env.py
```

Creates the virtual environment at `C:\venvs\a3-pose`, installs dependencies and
writes each Webots controller's `runtime.ini`.

> The venv lives **outside** the repository on purpose. The repository path is
> long and Windows long paths are disabled on this machine; a venv inside the
> repository makes installs fail with `WinError 206`.

> `rtmlib` is installed with `--no-deps` on purpose: installed normally it pulls
> in `onnxruntime` (CPU), which replaces `onnxruntime-directml` and silently
> removes the GPU provider. `setup_env.py` checks this after every run.

One-time, for the 3D lifter:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
C:\venvs\a3-pose\Scripts\python.exe tools\build_lifter.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_lifter.py
```

Downloads MotionBERT, patches it for DirectML and exports to `models/` (62 MB,
not in the repository).

Verify the GPU path:

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\check_directml.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_perception.py --cpu-compare
```

Expected: `DmlExecutionProvider` available, speedup above 2× over CPU.

## Site profiles

The repository supports two rigs side by side:

```bat
rem home office: Brio 100, AMD RX 6800 XT, DirectML  (default)
tools\run_live.py --site home

rem laboratory: Sony A7 III, NVIDIA, CUDA
tools\run_live.py --site lab
```

Defined in `configs/sites/`. See [`docs/PORTING_LAB.md`](docs/PORTING_LAB.md).

## Camera

The Brio 100 is mounted **portrait** — this nearly doubles the usable vertical
angle (30.4° → 51.6°), halves the required distance to about 2 m and puts 1.8×
more pixels on the body. The frame is rotated inside `Camera.read()`, that is
**before** any inference; the whole chain then works in the rotated image.

```bat
rem determine and store the rotation automatically
C:\venvs\a3-pose\Scripts\python.exe tools\calibrate_rotation.py --preview

rem live view with skeleton
C:\venvs\a3-pose\Scripts\python.exe tools\live_pose.py
```

Keys: `q` quit · `m` mirror · `r` advance rotation · `s` save.

## Robot model

`webots/protos/AtlasA3.proto` is **generated**, not maintained by hand:

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --with-unfixed
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --kinematic
```

Against the Webots stock model: 28 `PositionSensor`s added, the **segment masses
redistributed**, the placeholder `DEFAULT_PHYSICS` replaced per link by its own
negligible `Physics` node, supervisor enabled. `--kinematic` additionally strips
all physics for the puppet.

Atlas carries its real Boston Dynamics distribution: 89.000 kg, of which
**27.8 % sits in the arms** — in humans it is 10 %. One arm weighs 12.35 kg.
Pose imitation moves exactly those arms, and as shipped this makes about a
quarter of all reachable upper-body poses statically unstable. `--masses` writes
a different distribution into local copies of the sub-PROTOs
(`webots/protos/vendor/`), preserving `centerOfMass` and scaling inertia with it.

This also removes the inertia fault from M0: Webots merges the shared
`DEFAULT_PHYSICS` with `inertiaMatrix [1 1 1 0 0 0]` into every link, which made
the ankles 83.7× too sluggish.

Profiles: `spec` (as shipped), `human`, `legs`, `core` (default), `combo`.
`tools/check_proto.py` verifies the product against the profile table.

## Worlds

```bat
set WB=%LOCALAPPDATA%\Programs\Webots\msys64\mingw64\bin\webots.exe
"%WB%" --batch --mode=fast --minimize webots\worlds\<world>.wbt
```

| World | Purpose |
|---|---|
| `m0_inertia_probe` | Webots merging semantics on a minimal model |
| `m0_atlas_probe` | inventory of the stock Atlas |
| `m1_atlas_a3` | verification of `AtlasA3` |
| `m1_inertia_AtlasA3[Unfixed]` | A/B test of the physics fix |
| `m1_identify` | plant identification |
| `m1_stance` | standing pose, posture control, disturbance test |
| `m3_upper_body` | upper body follows, legs rigid |
| `m4_wholebody` | whole-body QP (rejected) |
| `m7_walk` | DCM walking with physics |
| `m8_puppet` | the puppet, recorded or live |

Controllers write their results as JSON to `results/` — Webots does not forward
controller `stdout` to the console on Windows.

## Status

| Milestone | State |
|---|---|
| M0 foundations and verification | **complete** |
| M1 model, sensing, posture control | **complete** |
| M2 GPU perception | **complete** |
| M3 upper-body imitation | **complete** |
| M4 whole-body QP | **rejected** — 15.6° against 5.0° for M3 |
| M5 stepping | **superseded by M7** |
| M6 mass distribution | **complete** — profile `core` |
| M7 walking (DCM, with physics) | **complete** — 2.6 m, 14 steps, no fall |
| M8 puppet (no physics) | **complete** — live camera, foot contact, travel |

Current live state: 476 frames → 476 packets → 450 received, forward travel
tracked to 19.3 mm median error, airborne 1.29 %, ground penetration beyond
10 mm in 0.12 % of frames, upper-body error 0.937°.
