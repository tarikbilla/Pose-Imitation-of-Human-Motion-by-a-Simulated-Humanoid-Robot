# Operations Guide

How to set up, run and read the A3 pose-imitation pipeline.

For the reasoning behind any of it, see [`FINDINGS.md`](FINDINGS.md). For moving
to different hardware, see [`PORTING_LAB.md`](PORTING_LAB.md).

---

## 1. Setup

Requirements: Python 3.11, Webots R2025a, Windows (see `PORTING_LAB.md` for
Linux and macOS).

```bat
python tools\setup_env.py
```

This creates the virtual environment at `C:\venvs\a3-pose`, installs the
dependencies and writes a `runtime.ini` for every Webots controller.

Two things this script does on purpose, both of which break the pipeline
silently if changed:

> **The virtual environment lives outside the repository.** The repository path
> is long and Windows long paths are disabled on this machine; a venv inside the
> repository makes installs fail with `WinError 206`.

> **`rtmlib` is installed with `--no-deps`.** Installed normally it pulls in
> `onnxruntime` (CPU), which replaces `onnxruntime-directml` and silently
> removes the GPU provider. The pipeline then runs 8× slower with no error.
> `setup_env.py` verifies this after every run.

> **`runtime.ini` needs absolute interpreter paths.** With a relative path
> Webots crashes on load with no error message (SIGSEGV, exit 139). These files
> are generated and git-ignored because they are machine-specific.

### One-time: build the 3D lifter

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
C:\venvs\a3-pose\Scripts\python.exe tools\build_lifter.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_lifter.py
```

Downloads MotionBERT, patches it for DirectML and exports to `models/` (62 MB,
not in the repository).

### Verify the GPU path

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\check_directml.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_perception.py --cpu-compare
```

Expected: `DmlExecutionProvider` available, speedup above 2× over CPU. On the
RX 6800 XT it is about 8×.

---

## 2. Running live

One command starts everything:

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\run_live.py
```

It starts Webots in real-time mode with `A3_LIVE=1`, waits for the receiver to
open its port, then starts the driver with the monitoring window.

The monitoring window shows the camera image with the detected 2D skeleton drawn
on it, plus frame rate, inference time, distance, forward and lateral travel,
and the most relevant joint angles.

| Key | Action |
|---|---|
| `q` | quit the session |
| `m` | toggle mirroring (display only, never the data) |
| `n` | reset the distance datum |
| `s` | save a snapshot to `results/` |

If the sender goes quiet for longer than `A3_LIVE_TIMEOUT` (10 s), the
simulation ends by itself.

**Stand in your starting position before the datum is set.** The distance datum
is taken during the first 2 seconds of a valid pose. Either be in position when
you start, or press `n` once you are standing where you want zero to be.

### Useful options

```bat
rem different site profile (see PORTING_LAB.md)
run_live.py --site lab

rem replay a video file instead of the camera
run_live.py --source path\to\video.mp4

rem no window, for automated runs
run_live.py --headless --driver-seconds 30

rem also record the raw keypoints of the session
run_live.py --record recordings\session.jsonl
```

Running the two halves separately is sometimes easier for debugging:

```bat
rem terminal 1
C:\venvs\a3-pose\Scripts\python.exe tools\run_puppet.py --visible --mode realtime --env A3_LIVE=1

rem terminal 2
C:\venvs\a3-pose\Scripts\python.exe tools\live_puppet.py
```

---

## 3. Running against a recording

Deterministic and reproducible; this is how all reported measurements were
taken.

```bat
rem 1. record keypoints from a video or the camera
C:\venvs\a3-pose\Scripts\python.exe tools\live_pose.py --source video.mp4 --record recordings\take.jsonl

rem 2. turn keypoints into joint angles
C:\venvs\a3-pose\Scripts\python.exe tools\make_angles.py --recording recordings\take.jsonl

rem 3. drive the puppet
C:\venvs\a3-pose\Scripts\python.exe tools\run_puppet.py --env A3_ANGLES=<absolute path to angles file>
```

Results are written to `results/puppet_run.json`. Webots does not forward
controller stdout on Windows, so JSON files are the only reliable output
channel.

---

## 4. Reading the results

`results/puppet_run.json` contains the summary and a trace. The values worth
looking at:

### Ground contact — measured on the simulation's own foot nodes

| Field | Meaning | Good value |
|---|---|---|
| `real_ground_min_mm` | deepest penetration of any sole corner | > −40 mm |
| `real_airborne_percent` | fraction of frames with both feet above 5 mm | < 2 % |
| `deep_count` | frames penetrating more than 10 mm | < 10 of ~3400 |
| `real_slide_p95_L/R_mm` | per-frame slide of a planted foot | < 0.2 mm |

These come from `getFromProtoDef` on the ankle joints, not from the controller's
own set-points. The `slide_*` and `penetration_*` fields are the controller's
own view and are kept only for comparison — they cannot detect a fault in the
controller's own model.

### Travel

| Field | Meaning |
|---|---|
| `travel_forward` | net distance travelled, body frame |
| `travel_target_m` | what the camera measured |
| `travel_error_median_mm` | tracking error against the camera signal |
| `travel_range_m` | total excursion |

### Imitation quality

| Field | Meaning | Good value |
|---|---|---|
| `upper_error_median_deg` | commanded vs achieved, upper body | < 1° |
| `leg_error_p95_deg` | joint tracking lag | < 2° |
| `leg_correction_median_deg` | how far the stance leg deviates from the 1:1 pose | the price of travelling |
| `pelvis_height_min/max` | posture; below ~0.80 m starts to look like a squat | |

---

## 5. Environment variables

The controller is tuned through the environment. Defaults are the values all
reported measurements were taken with.

### Live and source

| Variable | Default | Meaning |
|---|---|---|
| `A3_SITE` | `home` | site profile: `home`, `lab` |
| `A3_LIVE` | `0` | take angles from UDP instead of a file |
| `A3_LIVE_PORT` | `8768` | UDP port for the angle stream |
| `A3_LIVE_TIMEOUT` | `10.0` | seconds of silence before stopping |
| `A3_ANGLES` | — | absolute path to a precomputed angle file |
| `A3_RUN_SECONDS` | `60` | session length cap |
| `A3_WALL_CLOCK` | `1` | pace against wall clock (live) or simulation time |

### Foot contact and stepping

| Variable | Default | Meaning |
|---|---|---|
| `A3_PLANT_ON` | `0.008` | height below which a foot is considered planted |
| `A3_PLANT_OFF` | `0.020` | height above which a planted foot is released |
| `A3_MIN_STANCE` | `0.25` | minimum stance time before a forced step |
| `A3_STRAIN_HOLD` | `0.10` | how long the lock must be unreachable before stepping |
| `A3_RELEASE_ERROR` | `0.015` | IK residual that counts as unreachable |

### Travel and posture

| Variable | Default | Meaning |
|---|---|---|
| `A3_TRAVEL_TAU` | `0.35` | how fast the pelvis follows the camera signal |
| `A3_DRIFT_LIMIT` | `0.20` | maximum pelvis lead over the stance anchor |
| `A3_HEIGHT_FLOOR` | `0.84` | pelvis never goes below this |
| `A3_BASE_TAU` | `0.15` | horizontal pelvis smoothing |
| `A3_COMMAND_RATE` | `3.0` | rad/s cap on commanded joint motion |
| `A3_STEP_GAIN` | `1.0` | how strongly the swing foot corrects travel error |

Raising `A3_DRIFT_LIMIT` buys travel at the cost of leg distortion and ground
contact. Measured: 0.20 gives +0.407 m with 4 deep frames and 21.7° p95 leg
correction; 0.25 gives +0.442 m with 23 deep frames and 30.9°.

### Retargeting

| Variable | Default | Meaning |
|---|---|---|
| `A3_STRIDE_GAIN` | `1.0` | sagittal leg amplitude; 1.0 is true 1:1 |
| `A3_HEAD_GAIN` | `1.0` | neck pitch scale |
| `A3_HEAD_SCALE` | `60.0` | degrees per unit of the nose-to-ear measure |
| `A3_TORSO_GAIN` | `1.0` | torso amplitude |
| `A3_BODY_M` | `1.75` | subject height, scales absolute distance |
| `A3_FOCAL_RATIO` | `0.72` | fallback focal estimate when no calibration exists |

---

## 6. Rebuilding the robot model

`webots/protos/AtlasA3.proto` and `AtlasA3Kin.proto` are **generated**, not
edited by hand:

```bat
rem physics model, mass profile core
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --masses core

rem kinematic model for the puppet
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --kinematic

rem plus the unmodified control variant for A/B tests
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --with-unfixed
```

Mass profiles: `spec` (as shipped), `human`, `legs`, `core` (default), `combo`.
`tools/check_proto.py` verifies the product against the profile table.

---

## 7. Verification commands

| Check | Command |
|---|---|
| GPU path | `tools\check_directml.py` |
| PROTO integrity and mass sum | `tools\check_proto.py` |
| Leg inverse kinematics | `tools\check_legik.py` |
| DCM trajectory generator | `tools\check_lipm.py` |
| Receding horizon | `tools\check_receding.py` |
| 3D lifter axes and accuracy | `tools\check_lifter.py` |
| Gait phase in a recording | `tools\check_gait_phase.py` |
| Perception, with CPU comparison | `tools\check_perception.py --cpu-compare` |

---

## 8. Worlds

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
| `m4_wholebody` | whole-body QP (rejected, see FINDINGS §8) |
| `m7_walk` | DCM walking with physics |
| `m8_puppet` | the puppet, with and without a live stream |

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 8× slower than expected, no error | plain `onnxruntime` shadowing the DirectML build | re-run `setup_env.py`; check `check_directml.py` |
| Webots exits immediately, no message | relative interpreter path in `runtime.ini` | re-run `setup_env.py` |
| Camera stream stops after ~200 frames | pipe backpressure | already fixed by the draining reader; verify `drain: True` in the source description |
| `no live packet received` | driver started before Webots bound the port, or a firewall prompt | start Webots first, or raise `--boot` |
| Robot stands still, packets counted but low | pose valid but nobody in frame | check `detected` in the driver output |
| Travel is far too large or too small | wrong focal length or subject height | see `PORTING_LAB.md` §3 |
| Distance datum obviously wrong | calibrated while nobody was in frame | press `n` while standing in position |
| `WinError 206` during install | venv inside the repository | use `C:\venvs\a3-pose` |
