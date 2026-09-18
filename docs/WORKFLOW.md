# System Workflow Documentation

## Overview

This document describes the complete workflow and architecture of the **Pose Imitation of Human Motion by a Simulated Humanoid Robot** system. The pipeline captures human motion from video input, processes it through computer vision algorithms, and drives a simulated humanoid robot in Webots to imitate the detected poses in real-time.

Each phase below carries a **Libraries** block (what it is built on) and a
**Latency** block (what it costs, measured on this project). The consolidated
budget is in [Latency Budget](#latency-budget); one-off costs that are measured
in minutes rather than milliseconds are in [One-Off Costs](#one-off-costs-minutes-not-milliseconds).

> **Measurement conditions for every timing in this document.** Measured
> 2026-09-18 on the project's own target PC: NVIDIA RTX 3090 Ti (24 GB) + RTX
> 3070 (8 GB), Python 3.12, TensorFlow GPU build, Webots `basicTimeStep` 20 ms.
> Perception stages were timed by replaying 4 000 recorded frames from
> `logs/run_20260918_125223/pose_keypoints.csv`; MeTRAbs was timed on 30 warm
> 1920x1080 frames with the shipped `metrabs_eff2s_y4` backbone, `num_aug=1`,
> `max_detections=1`; the controller loop was read from
> `logs/webots_joint_trajectory_1789730415.csv` (8 153 s of simulation).
> Inference must be timed **with a subject in shot**: a blank frame short-circuits
> the pose network and reads ~25% fast.
> `p50` is the median, `p90`/`p99` the tail. Re-measure on other hardware --
> the GPU term dominates and nothing else here is close to it.

---

## System Architecture

```
┌─────────────────┐
│  Video Input    │
│  (Camera/File)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Video Capture   │
│  & Processing   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pose Detection │
│  (MeTRAbs, GPU) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retargeting &  │
│  Joint Mapping  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Smoothing &   │
│   Filtering     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  UDP Bridge to  │   landmarks + gait cues + body yaw
│     Webots      │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ Webots controller  (main/controllers/…)                  │
│                                                          │
│   arms + head ── nao_retarget.retarget_upper_body        │
│                                                          │
│   legs ── ARBITER: exactly ONE layer per step            │
│     1. locomotion    walk_motion + Webots .motion clips  │
│     2. march engine  gait.GaitEngine (in place)          │
│     3. pose imitation lower_body.LowerBodyController     │
│     4. stand         balance.BalanceController           │
│                          │                               │
│                     NaoPoseDriver                        │
│                     clamp → smooth → setPosition         │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Webots NAO H25  │
│    Simulation   │
└─────────────────┘
```

> The robot-side maths lives in `main/libraries/` and imports **no** Webots
> module, so every layer above is unit-tested off-simulation (`pytest -q`).

---

## Phase 1: Video Input & Capture

### Technology Stack
- **OpenCV (cv2)** - Video capture and image processing
- **Platform-specific backends**:
  - Linux: V4L2 (Video4Linux2)
  - macOS: AVFoundation
  - Windows: DirectShow

### Components
- `src/perception/video_input.py` - `VideoSource` class

### Process Flow

1. **Source Selection**
   - Accepts webcam index (e.g., `0` for default camera)
   - Accepts video file path (e.g., `data/sample.mp4`)
   - Primary hardware: Sony A7 III camera via HDMI-to-USB capture

2. **Backend Initialization**
   - Automatically selects optimal backend for the OS
   - Falls back to `CAP_ANY` if preferred backend fails
   - Configures buffer size to minimize latency

3. **Frame Capture Configuration**
   - Resolution: 1280×720 (configurable, up to 1920×1080)
   - Frame rate: Adaptive 25-100 FPS based on pipeline performance
   - Initial FPS: 30 (configured in `configs/default.yaml`)
   - Optional horizontal flip for selfie-style tracking

4. **Frame Generation**
   - Yields `VideoFrame` objects containing:
     - Frame index (sequential counter)
     - Timestamp (seconds since start)
     - BGR image array (numpy ndarray)

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `opencv-python` | >=4.9,<5.0 | `VideoCapture`, backend selection, colour conversion, buffer sizing |
| V4L2 / AVFoundation / DirectShow | OS | the capture backend OpenCV drives |

### Latency
| Item | Measured | Note |
|---|---|---|
| Capture + decode + colour convert | **~7 ms** p50 | Derived: the 63 ms whole-loop period minus the ~56 ms inference mix. 1920x1080 costs capture bandwidth, not inference time -- MeTRAbs rescales internally, so dropping to 1280x720 helps only if capture is the bottleneck. |
| `cv2.CAP_PROP_BUFFERSIZE` | 1 frame | Anything larger adds a whole frame period of stale video before the pipeline ever sees it. |

### Key Features
- Robust error handling with consecutive failure tracking
- Platform-aware backend selection
- Adaptive frame rate to match pipeline throughput
- Low-latency configuration (buffer size = 1)

---

## Phase 2: Pose Detection & Estimation

### Technology Stack
- **[MeTRAbs](https://github.com/isarandi/metrabs)** - GPU-accelerated absolute-3D human pose estimation (TensorFlow / TensorFlow-Hub, with a built-in YOLOv4 person detector)
- **NumPy** - Numerical computations
- **OpenCV** - Image preprocessing

### Components
- `src/perception/pose_estimator.py` - `PoseEstimator` class
- `src/perception/metrabs_model.py` - model loading, GPU check, skeleton introspection, camera intrinsics
- `src/perception/landmarks.py` - Landmark definitions (19 keypoints, `coco_19` skeleton)

### Process Flow

1. **Initialization**
   - Checks a GPU is visible to TensorFlow (fails loudly otherwise -- see
     `pose.require_gpu` in configs/default.yaml)
   - Loads the MeTRAbs SavedModel via TensorFlow-Hub (`pose.model_url`,
     default: EfficientNetV2-S backbone, cached locally after first download)
   - Reads the model's actual joint names for `pose.skeleton` (`coco_19`) and
     matches them against this project's canonical landmark names

2. **Frame Processing**
   - Converts BGR frame to RGB
   - Feeds the image to `model.detect_poses(...)` (detector + 3D pose network
     in one call), with `num_aug=1` and `max_detections=1` for latency
   - Receives 3D poses (mm, camera frame), 2D pixel poses, and a detection box

3. **Landmark Extraction**
   - **19 Keypoints** including:
     - Face: nose, eyes, ears
     - Upper body: shoulders, elbows, wrists
     - Torso: neck, pelvis, hips
     - Lower body: knees, ankles
   - Each landmark contains:
     - (x, y, z) absolute METRIC coordinates in millimeters, camera frame (x right, y down, z forward/away)
     - Visibility PROXY (0-1): in-frame/in-detection-box confidence, NOT a true per-joint occlusion estimate (MeTRAbs has no per-joint confidence output)

4. **Output Generation**
   - Creates `PoseFrame` object with:
     - Timestamp
     - Frame index
     - Dictionary of named keypoints (e.g., "left_shoulder", "right_elbow")

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `tensorflow` (GPU build) | >=2.12,<2.16 | runs the MeTRAbs SavedModel |
| `tensorflow-hub` | >=0.15,<0.17 | loads the model from the TF-Hub zip / local cache |
| `numpy` | >=1.26,<2.1 | landmark arrays, the joint-jump filter |
| `opencv-python` | >=4.9,<5.0 | image handover to the model |

### Latency
**This phase is the pipeline's cost.** Every other perception stage together is
under 0.1 ms; this one is ~56 ms.

| Item | Measured | Note |
|---|---|---|
| Whole perception loop, real subject | **p50 63 ms / p90 95 ms → 16.0 FPS** | 31 243 recorded frames. Inference is essentially all of it. |
| Inference, detector frame | **75.9 ms** | the YOLOv4 person detector runs |
| Inference, tracked-box frame | **36.2 ms** | box carried from the previous detection |
| Mix at `detect_interval: 2` | **~56 ms** | what the loop actually pays per frame |
| `estimate()` on a **blank** frame | 42 ms | detector only, pose network skipped — **not** a valid benchmark; always measure with a subject in shot |
| First call after load | **15.1 s** | one-off TensorFlow graph warm-up; the pipeline is not usable until it has passed |
| Model load, cold | **31 s** | one-off per process, from the 371 MB `~/.cache/metrabs` |
| `pose.skeleton` choice | **no cost** | every named skeleton is an index gather out of the same 122-joint superset: 152.1 ms/call for the superset against 152.4 ms for `coco_19` on the earlier reference machine |

The backbone is the one knob that moves this number (`pose.model_url`):
`metrabs_mob3s_y4` / `metrabs_rn18_y4` are faster and less accurate,
`metrabs_eff2l_y4` slower and more accurate. `pose.detect_interval` (default 2)
reruns the YOLOv4 person detector only every N frames and tracks in between.

### Key Features
- Real GPU-accelerated 3D pose tracking
- Explicit failure handling (no silent CPU fallback unless configured)
- Optional synthetic fallback mode for testing (disabled by default)

### Detected Landmarks
```
Head: nose, left/right eye, left/right ear
Upper Body: left/right shoulder, left/right elbow, left/right wrist
Torso: neck, pelvis, left/right hip
Lower Body: left/right knee, left/right ankle
```

---

## Phase 3: Pose Retargeting & Joint Mapping (fallback channel)

> **Where retargeting really happens.** This Python-side mapper is now the
> *fallback* path only. The primary channel streams the raw landmarks and the
> **controller** retargets them (`main/libraries/nao_retarget.py`), because
> driving the robot's full pose needs the actual limb geometry, NAO's real joint
> axes/signs/limits, and the robot's own balance state — none of which the Python
> side has. The controller uses these pre-computed angles only when a frame
> arrives with no `keypoints`. See Phase 8.

### Technology Stack
- **NumPy** - Vector mathematics
- **Math** - Trigonometric calculations
- **Python dataclasses** - Type-safe data structures

### Components
- `src/retargeting/mapper.py` - `RetargetingMapper` class
- `src/types.py` - `JointCommand` dataclass

### Process Flow

1. **Keypoint Vector Extraction**
   - Computes limb vectors from consecutive landmarks:
     - Left arm: shoulder → elbow → wrist
     - Right arm: shoulder → elbow → wrist
     - Left leg: hip → knee
     - Right leg: hip → knee
     - Torso: hip → shoulder

2. **Joint Angle Calculation**
   - **Shoulder Pitch** (left/right):
     - Computed from upper arm vector angle
     - Uses `atan2(-y, |x|)` for pitch angle
   
   - **Elbow Roll** (left/right):
     - Angle between upper and lower arm vectors
     - Left: `π - angle_between(upper, lower)`
     - Right: `-(π - angle_between(upper, lower))`
   
   - **Hip Pitch** (left/right):
     - Computed from hip-to-knee vector
     - Uses `atan2(y, |x|)` for pitch angle
   
   - **Torso Pitch**:
     - Average of left and right hip-to-shoulder angles

3. **Joint Limiting**
   - Clips angles to safe robot joint limits:
     - Shoulder Pitch: -119° to +119°
     - Elbow Roll: 0° to 135° (left), -135° to 0° (right)
     - Hip Pitch: -88° to +27°
     - Torso Pitch: -30° to +30°

4. **Command Generation**
   - Creates `JointCommand` with:
     - Timestamp (from input frame)
     - Frame index
     - Dictionary of joint angles in radians

### Mapped Joints
- `LShoulderPitch` / `RShoulderPitch`
- `LElbowRoll` / `RElbowRoll`
- `LHipPitch` / `RHipPitch`
- `TorsoPitch`

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `numpy` | >=1.26,<2.1 | vector maths for the geometric IK |
| `math` | stdlib | angle solving (`atan2`, `acos`) |

### Latency
| Item | Measured | Note |
|---|---|---|
| `RetargetingMapper.map()` | **<0.01 ms** | below timer resolution over 4 000 replayed frames |

This is the *fallback* channel only. The primary path streams raw landmarks and
solves on the robot side in `main/libraries/nao_retarget.py` — see Phase 8.

### Key Features
- Geometric inverse kinematics approach
- Hardware-safe joint limits
- Handles both upper and lower body
- Frame-accurate synchronization

---

## Phase 4: Smoothing & Filtering

### Technology Stack
- **Exponential Smoothing Algorithm**
- **Python collections** - State management

### Components
- `src/utils/filtering.py` - `ExponentialSmoother` class

### Process Flow

1. **Exponential Smoothing**
   - Formula: `smoothed = α × current + (1 - α) × previous`
   - Default α = 0.35 (configurable in `configs/default.yaml`)
   - Lower α = more smoothing (slower response)
   - Higher α = less smoothing (faster response)

2. **State Management**
   - Maintains previous values for all joints
   - Per-joint smoothing (independent filtering)
   - First frame uses current value as previous

3. **Temporal Filtering**
   - Reduces jitter from pose estimation noise
   - Maintains motion continuity
   - Prevents abrupt joint changes

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `src/utils/filtering.py` | in-repo | `OneEuroFilter` (keypoints), `ExponentialSmoother` (angles) |
| `numpy` | >=1.26,<2.1 | array maths |

### Latency
| Item | Measured | Note |
|---|---|---|
| `OneEuroFilter.update()`, one axis | **0.022 ms** p50 | x, y and z filter separately, so ~0.07 ms per frame |
| Delay it *adds* | **speed-dependent, by design** | A fixed-alpha EMA charges a flat `(1-alpha)/alpha` samples whether the subject moves or not — 70 ms at `alpha=0.5`, nearly half the 150 ms budget. One Euro spends that delay only while the subject is still and gets out of the way when they move. |

### Key Features
- Real-time filtering (minimal latency)
- Per-joint independent smoothing
- Configurable smoothing strength
- Zero-phase lag (causal filter)

---

## Phase 5: Adaptive FPS Control

### Technology Stack
- **Python time module** - Performance monitoring
- **Collections.deque** - Rolling window statistics

### Components
- `src/utils/fps.py` - `AdaptiveFPSController` class

### Process Flow

1. **Latency Monitoring**
   - Measures per-frame processing time
   - Maintains rolling average over recent frames
   - Compares against target latency budget (default: 150ms)

2. **Dynamic FPS Adjustment**
   - **If latency > budget**: Decrease FPS (reduce load)
   - **If latency < budget**: Increase FPS (improve responsiveness)
   - Adjustment step: ±5 FPS (configurable)
   - Constraints: 25-100 FPS range

3. **Throttling**
   - Sleeps between frames to match target FPS
   - Prevents resource over-utilization
   - Balances responsiveness vs. CPU usage

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `src/utils/fps.py` | in-repo | `AdaptiveFPSController` |
| standard `time` | stdlib | monotonic clock, inter-frame sleep |

### Latency
| Item | Measured | Note |
|---|---|---|
| Controller overhead | **<0.01 ms** | arithmetic only |
| `latency_budget_ms` | **150 ms** (config) | the target it steers toward |
| Achieved loop period | **63 ms p50 / 95 ms p90** | 16.0 FPS effective over 31 243 recorded frames. The ~56 ms GPU term means the 100 FPS ceiling is unreachable on this backbone; the controller settles where the GPU allows. |

### Configuration
- `initial_fps`: 30
- `min_fps`: 25
- `max_fps`: 100
- `fps_step`: 5
- `latency_budget_ms`: 150

### Key Features
- Automatic performance tuning
- Hardware-adaptive operation
- Maintains real-time performance
- Prevents system overload

---

## Phase 6: Visualization & Feedback

### Technology Stack
- **OpenCV** - GUI window and drawing
- Custom skeleton renderer (`SkeletonOverlay`), projecting MeTRAbs' 3D mm
  landmarks back to pixels with a pinhole intrinsic matrix

### Components
- `src/perception/visualizer.py` - `SkeletonOverlay` class

### Process Flow

1. **Skeleton Drawing**
   - Renders 33 detected landmarks as circles
   - Draws connections between landmarks (bones)
   - Color-coded by body part:
     - Face landmarks
     - Upper body connections
     - Lower body connections

2. **Window Management**
   - Creates OpenCV window "Pose Imitation"
   - Real-time display of annotated video
   - Keyboard controls:
     - `q` or `ESC` to quit
     - Window close button to exit

3. **Optional Headless Mode**
   - Flag: `--no-display`
   - Disables visualization for server/SSH environments
   - Pipeline continues without GUI

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `opencv-python` | >=4.9,<5.0 | `imshow`, drawing primitives, `waitKey` |

### Latency
| Item | Measured | Note |
|---|---|---|
| Overlay draw + `imshow` | part of the ~21 ms non-GPU budget | `--no-display` removes it; use for headless runs and benchmarking |

### Key Features
- Real-time visual feedback
- Low-overhead rendering
- Optional headless operation
- User-friendly controls

---

## Phase 7: Webots Bridge & Communication

### Technology Stack
- **UDP Sockets** - Low-latency network communication
- **JSON** - Data serialization
- **Python socket module**

### Components
- `src/webots_bridge.py` - `WebotsBridge` class
- Webots controller: `main/controllers/pose_imitation_controller/`

### Process Flow

1. **UDP Socket Initialization**
   - Creates UDP socket (connectionless, low latency)
   - Target: `127.0.0.1:8765` (localhost, configurable)
   - No handshake required (fire-and-forget)

2. **Command Serialization**
   - Builds the frame as JSON:
     ```json
     {
       "timestamp_s": 1.234,
       "frame_index": 42,
       "keypoints": {
         "left_shoulder": [-160.2, -580.4, 1980.1, 0.99],
         "left_knee":     [-114.8, 428.7, 2010.5, 0.97],
         "left_ankle":    [-119.3, 826.9, 2005.1, 0.95]
       },
       "gait": {
         "state": "march", "cadence_hz": 0.95, "phase": 1.83,
         "swing_side": 1, "intensity": 0.7, "turn": 0.4, "conf": 0.98,
         "body_yaw_rad": 0.42, "yaw_conf": 0.99
       },
       "joint_angles_rad": { "LShoulderPitch": 0.52 }
     }
     ```
   - `keypoints` **(primary)** — 19 landmarks as `[x, y, z, visibility]` in
     absolute METRIC camera-frame coordinates (millimeters; x right, y down, z
     forward/away). Head, shoulders, elbows, wrists, hips, knees, ankles, neck,
     pelvis; the curated subset keeps the packet small (NFR-1). `visibility` is
     a PROXY (in-frame/in-box confidence), not true per-joint occlusion --
     MeTRAbs has no per-joint confidence output. MeTRAbs' `coco_19` skeleton has
     no separate heel/toe landmarks (unlike the old MediaPipe 33-point set), so
     the controller's ground-line/lift detection falls back to ankle-only.
   - `gait` — cadence/phase/stop for the march engine, plus `body_yaw_rad`
     (**an angle**, so the controller can close a heading loop on it) and
     `yaw_conf`, which is independent of `conf` because the yaw needs only the
     shoulders and hips and stays usable when the legs leave the frame.
   - `joint_angles_rad` **(fallback)** — used only when a frame carries no
     `keypoints`.
   - Additive and backward compatible: a controller ignores fields it does not
     know, so the protocol can grow without breaking older builds.
   - Encodes as UTF-8 bytes

3. **UDP Transmission**
   - Sends datagram to Webots controller
   - No acknowledgment required
   - Minimal overhead (~1ms per frame)

4. **Optional Disabling**
   - Flag: `--no-webots`
   - Enables pure perception demo mode
   - Useful for testing without Webots

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `socket` (stdlib) | stdlib | connectionless UDP, no handshake and no retransmit |
| `json` (stdlib) | stdlib | payload encoding |

### Latency
| Item | Measured | Note |
|---|---|---|
| `sendto` + `recv` on localhost | **0.005 ms p50 / 0.008 ms p99** | 5 000 round trips |
| Payload size | **2 472 bytes** | one datagram; no fragmentation on loopback |
| Queue drain policy | keeps only the **freshest** datagram | The controller runs at 50 Hz and the camera at 16 Hz, so a queue could only ever add age. Draining to the newest frame is what keeps the stale-frame term at zero. |

UDP is chosen over TCP precisely here: a retransmitted pose frame is worse than
a dropped one, because by the time it arrives the human has moved.

### Key Features
- Ultra-low latency (<5ms network overhead)
- Fire-and-forget messaging
- No connection management overhead
- Easy to disable for testing

---

## Phase 8: Webots Robot Simulation & Whole-Body Control

### Technology Stack
- **Webots R2025a** - Robot simulation environment (NAO H25 proto)
- **Python Webots API** - `Robot`, `Motion`, motors, position sensors, IMU, gyro, FSRs
- **UDP Socket** - Command reception (port 8765)
- **NumPy** - forward kinematics and the centre-of-mass model

### Components
| File | Role |
|---|---|
| `main/worlds/…​.wbt` | world; **must** define the NAO foot/floor contact pair (see below) |
| `main/controllers/pose_imitation_controller/pose_imitation_controller.py` | Webots glue + the lower-body **arbiter** |
| `main/libraries/nao_retarget.py` | landmarks → NAO angles (arms, head, closed-form per-leg solve) |
| `main/libraries/lower_body.py` | weight-shift / single-leg-lift sequencer and its safety gates |
| `main/libraries/gait.py` | in-place march engine |
| `main/libraries/balance.py` | model-based CoM balance (FK + link masses + Fibonacci search) |
| `main/libraries/walk_motion.py` | `.motion` clip discovery, heading (yaw) servo, locomotion planning |
| `main/libraries/pose_control_utils.py` | `NaoPoseDriver`: limits, smoothing, velocity caps, trajectory log |

### Process Flow

1. **Initialization**
   - Look up all 24 motors and their `<name>S` position sensors
   - Enable the InertialUnit (gravity **and** heading), gyro, accelerometer and
     the 3-axis foot force sensors (`LFsr` / `RFsr`)
   - Discover Webots' pre-balanced NAO `.motion` clips on disk
   - Bind the non-blocking UDP listener on port 8765

2. **Command reception** (per simulation step)
   - Drain the UDP backlog and keep only the **freshest** frame (latency, NFR-1)
   - Arms and head are commanded directly from the landmarks
   - The legs' observation is only *latched*, for whichever layer runs this step

3. **Lower-body arbitration — exactly one commander per step**
   Two layers commanding the 12 leg joints at once means they fight and the
   robot falls, so the arbiter picks one, in priority order:

   | Priority | Layer | When |
   |---|---|---|
   | 1 | **Locomotion** — play a `.motion` clip | heading needs correcting, or the human is walking and clips exist |
   | 2 | **March engine** — `gait.GaitEngine` | human is walking but no clips are installed (marches in place) |
   | 3 | **Pose imitation** — `lower_body.LowerBodyController` | default: squat, leg abduction, single-leg lift |
   | 4 | **Stand** — `balance.BalanceController` only | legs disabled |

4. **Single-leg lift: LOAD → SINGLE → UNLOAD**
   - **LOAD** — lean so the CoM moves over the stance foot. The lean *sign* is
     probed against the CoM model rather than hard-coded.
   - **SINGLE** — only once `stance_margin > 0` (and the FSRs agree) does the
     swing leg follow the human, its authority scaled continuously by that margin
   - **UNLOAD** — human lowers the foot, or margin/tilt safety closes the gate
   - Both blends are rate-limited, so there is always a smooth path back to the
     symmetric crouch

5. **Turning is a closed loop**
   `error = wrap_pi(desired_heading − InertialUnit yaw)`, where the desired
   heading tracks the human's measured torso yaw. Turn clips fire until the error
   closes, so the rotation converges despite coarse clips and noisy tracking —
   and it works while standing still.

6. **Motion playback hand-off**
   While a clip plays it owns the **whole** body: per-joint commanding is
   suspended and the motor velocity caps are lifted (a velocity-capped motor
   cannot reach a clip's keyframes, and the pre-balanced gait then arrives late at
   every foot placement and topples). Clips play to completion — a clip boundary
   is a balanced double-support pose, the only safe place to hand control back —
   and the smoothers are reseeded from the position sensors on reclaim.

7. **Simulation step**
   `basicTimeStep` is 20 ms. Webots' default of 32 ms is too coarse for NAO leg
   control and destabilises the walk clips; the controller warns if it sees more.

### Driven joints
All 24: head yaw/pitch, both arms (shoulder pitch **and** roll, elbow yaw/roll,
wrist), and the full 12-joint leg chain (hip yaw-pitch/roll/pitch, knee pitch,
ankle pitch/roll). NAO has no torso joint, so torso pitch is dropped and torso
*yaw* is realised by stepping round.

### World requirement (not optional)
```
WorldInfo {
  basicTimeStep 20
  contactProperties [
    ContactProperties {
      material2 "NAO foot material"
      coulombFriction [ 8 ]
      bounce 0
      bounceVelocity 0.003
    }
  ]
}
```
`Nao.proto` tags its soles `"NAO foot material"`. Without a matching
`ContactProperties` the pair falls back to Webots' low-friction bouncy default:
the feet slide, the robot cannot load one foot or take a step, and the leg
controller *looks frozen* because every leg command is absorbed by foot slip.

### Libraries used
| Library | Version | Used for |
|---|---|---|
| Webots `controller` module | R2025a | `Robot`, `Motion`, `InertialUnit`, `Supervisor`, `TouchSensor` |
| `numpy` | >=1.26,<2.1 | forward kinematics and the centre-of-mass model in `balance.py` |
| `main/libraries/*` | in-repo | `nao_retarget`, `lower_body`, `balance`, `gait`, `walk_motion`, `clip_forge`, `clip_safety`, `pose_control_utils` |

### Latency

**Control loop**

| Item | Measured | Note |
|---|---|---|
| `basicTimeStep` | **20 ms** | Webots' 32 ms default is too coarse for NAO's legs |
| Wall time per step | **20.3 ms** | realtime factor **0.983** over 8 153 s of simulation |
| `plan_action()`, the leg arbiter | **0.002 ms** p50 | the decision is free; what it commits to is not |
| Arm/head chain residual lag | **38 ms** | `ARM_TAU_S` 0.07 s + `ARM_LEAD_S` 0.20 s, measured over 5 473 frames. The previous frame-rate-clocked EMA cost 108 ms. |
| `STALE_AFTER_S` | **0.5 s** | no command for this long → hold pose, then stand down |

**Decision latency — human does a thing, robot does it**

A clip is a *commitment*: while one plays it owns the 12 leg joints, open loop.
These are the numbers that decide how responsive the robot feels, and they are
seconds, not milliseconds.

| Event | Measured | Where the time goes |
|---|---|---|
| Walk **starts** | **790 ms** (cue) + **560 ms** (prepare ramp) | The action cue wants 80 mm of pelvis travel across its 0.8 s window before committing; the ramp puts the legs in the clip's opening crouch at `LEG_POSE_RATE` 1.5 rad/s. |
| Walk **stops** | **165 ms** (cue) + **600 ms** (`WALK_LATCH_RELEASE_S`) + up to **2.46 s** (clip) | The clip term is one stride period (1.28 s) to reach the free exit phase, then a 1.18 s settle. Its closing stand-up (another 1.28 s) is skipped — see `GaitCycle.rest_s`. |
| Walk speed | **0.073 m/s** sustained | `Forwards50.motion` cycled. Motor-limited: the clip already peaks at **84%** of the motors' rated speed, so it cannot be retimed faster. |
| Turn 90° | **4.6 s**, one clip, residual **1.1°** | `TurnLeft180`/`TurnRight180`, entered after their 1.20 s opening crouch and stopped at the certified keyframe nearest the heading error. |
| Turn 180° | **8.5 s**, one clip, residual **3.9°** | |
| Turn rate | **20.7 °/s** | against 13.8 °/s for the 40° clips, ~60% of whose runtime is their own start/stop transient |
| Smallest turn served | **21°** | below this the heading error is left alone |
| Squat / one-leg / stance width | **continuous, no clip** | pose imitation, so no commitment and no clip latency |

**Watchdogs**

| Constant | Value | Purpose |
|---|---|---|
| `CLIP_PREPARE_TIMEOUT_S` | 2.5 s | per-action ramp ceiling |
| `CLIP_PREPARE_RUN_TIMEOUT_S` | 3.5 s | total ramp ceiling across a dithering planner |
| `MOTION_WATCHDOG_S` | 8.0 s | fallback when a clip never reports itself finished |
| `CLIP_EXIT_TOLERANCE_S` | 0.03 s | how near a certified keyframe counts as being at one |

### Key Features
- Realistic physics with correct foot friction
- One commander per step — no layer fighting
- Model-verified weight transfer before any foot leaves the ground
- Gyro-lead fall detection that aborts a motion clip before the tilt threshold
- Graceful degradation: no clips → march in place; no NumPy → hard-capped lift;
  tracking lost → ramp back to the balanced crouch

---

## Phase 9: Logging & Telemetry

### Technology Stack
- **Python logging module** - Structured logging
- **CSV files** - Time-series data storage
- **Python pathlib** - File management

### Components
- `src/utils/logger.py` - `CsvRunLogger` class
- `src/utils/config.py` - Configuration management

### Process Flow

1. **Run Initialization**
   - Creates timestamped run directory: `logs/run_YYYYMMDD_HHMMSS/`
   - Initializes CSV files for:
     - Joint commands
     - Pose keypoints
     - Performance metrics

2. **Per-Frame Logging**
   - Records joint angles with timestamps
   - Logs detected keypoint coordinates
   - Captures processing latency metrics

3. **Performance Metrics**
   - FPS (frames per second)
   - Frame processing time
   - Detection confidence scores
   - UDP transmission success

4. **Post-Run Analysis**
   - CSV files can be analyzed with:
     - Python (pandas, matplotlib)
     - MATLAB
     - Excel
   - Enables quantitative evaluation

### Log Files
- `joint_commands.csv` - Robot control signals
- `pose_keypoints.csv` - Detected human pose
- `performance.csv` - System metrics

### Libraries used
| Library | Version | Used for |
|---|---|---|
| `csv` | stdlib | trajectory and keypoint logs |
| `logging` | stdlib | console startup block and periodic status line |
| `matplotlib` | >=3.8,<4.0 | offline plots in `scripts/` |

### Latency
| Item | Measured | Note |
|---|---|---|
| One trajectory row | negligible against the 20 ms step | 131 columns, buffered CSV write |
| Status line | every **100 frames** (`STATUS_EVERY`) ≈ 2 s of simulation | plain-language summary of what the legs are doing and why |
| Log growth | ~**220 MB** per 5 700 s episode | `logs/` is git-ignored; prune between sessions |

### Key Features
- Automatic timestamped organization
- CSV format for universal compatibility
- Frame-accurate synchronization
- Minimal performance overhead

---

## Complete Pipeline Summary

### End-to-End Flow

```
1. Video Frame (30-100 FPS)
   ↓
2. Pose Detection (MeTRAbs, GPU) → 19 landmarks (absolute 3D, mm)
   ↓
3. Joint Mapping (Geometric IK) → 7 joint angles (Python-side fallback path)
   ↓
4. Smoothing (Exponential filter, keypoints + angles) → Filtered
   ↓
5. UDP Send (JSON over UDP) → Webots controller
   ↓
6. Robot Actuation (Webots physics) → Humanoid imitation
```

<a name="latency-budget"></a>
### Latency Budget

Every row below was measured on this project (see the conditions note at the top
of this document). Two things are worth reading off it before anything else:

1. **One stage costs ~56 ms and the rest of perception costs 0.07 ms.** Tuning
   anything but the GPU stage is tuning noise.
2. **The robot's *motion* latency is ~120 ms; its *decision* latency is seconds.**
   These are different budgets with different causes, and conflating them is how
   "the robot is laggy" got misdiagnosed for four sessions — the imitation loop
   was never the problem, the clip commitment was.

#### A. Motion latency — you move, the robot's motors move

| # | Stage | Library | p50 | p90 |
|---|---|---|---|---|
| 1 | Capture + decode + overlay | OpenCV / V4L2 | ~7 ms | — |
| 2 | **MeTRAbs inference** (mix at `detect_interval: 2`) | TensorFlow + TF-Hub | **~56 ms** | — |
| 3 | Keypoint smoothing (3 axes) | `OneEuroFilter` | 0.07 ms | — |
| 4 | Gait cue | `gait_cues.py` | 0.030 ms | 0.034 ms |
| 5 | Action cue | `action_cues.py` | 0.011 ms | 0.012 ms |
| 6 | UDP send → receive | `socket` + `json` | 0.005 ms | 0.008 ms |
| 7 | Controller step quantisation | Webots | ≤20 ms | — |
| 8 | Arm/head filter residual | `ArmTracker` | 38 ms | — |
| | **Sum (arms/head)** | | **~121 ms** | **~153 ms** |

Measured whole-loop camera period: **63 ms p50 / 95 ms p90 → 16.0 FPS**
effective, over 31 243 recorded frames. The `runtime.latency_budget_ms` target
is 150 ms, so the median loop sits inside budget and the p90 sits just outside
it; the adaptive FPS controller cannot recover that because the ~56 ms GPU term
is a floor, not a load.

> The sum is a sum of independently measured stages, not a single end-to-end
> stopwatch reading. An end-to-end measurement would need the controller to log
> the pipeline's `frame_index`, which it currently does not (its `frame_index`
> column is its own control-step counter). That is the one number in this
> document worth adding instrumentation for.

#### B. Decision latency — you do a thing, the robot decides to do it

| Event | Measured | Dominated by |
|---|---|---|
| Walk starts | **~1.35 s** | 790 ms cue + 560 ms prepare ramp |
| Walk stops | **0.8 – 3.2 s** | 165 ms cue + 600 ms latch + ≤2.46 s clip |
| Turn 90° | **4.6 s** | the clip, at 20.7 °/s |
| Turn 180° | **8.5 s** | the clip |
| Squat / one-leg / stance | **motion latency only (~121 ms)** | no clip involved |

#### C. What each budget is bounded by

| Budget | Bound | Can it be improved? |
|---|---|---|
| Motion latency | MeTRAbs inference, ~56 ms | Yes — a smaller backbone (`metrabs_mob3s_y4`), or a larger `pose.detect_interval`, at a stated accuracy cost |
| Walk speed | 0.073 m/s | **No** — the clip peaks at 84% of rated motor speed; retiming it faster is not available |
| Turn rate | 20.7 °/s | Turn clips run at 52% of rated speed, so there *is* headroom — but retiming a dynamically balanced clip needs live validation, not the certifier (which already fails the shipped clips) |
| Walk stop | 2.46 s | Already cut from 3.73 s by trimming the clip's dead tail |
| Walk start | 1.35 s | The 790 ms is a deliberate trade: the 80 mm travel witness cut false walking 18.8% → 1.7% |

<a name="one-off-costs-minutes-not-milliseconds"></a>
### One-Off Costs (minutes, not milliseconds)

These are the things that make you wait. Everything above is per-frame; this is
per-session or per-install.

| Step | Duration | Frequency | Note |
|---|---|---|---|
| `conda env create -f environment.yml` | **~5–15 min** | once per machine | network-bound; TensorFlow + CUDA runtime is the bulk |
| MeTRAbs model download | **~2–5 min** (371 MB) | once per machine | cached in `~/.cache/metrabs`, survives reboots (`$METRABS_CACHE_DIR` to relocate) |
| MeTRAbs model load | **31 s** | **every pipeline start** | cold load of the SavedModel |
| TensorFlow graph warm-up | **15.1 s** | **every pipeline start** | the first `estimate()` call; the window opens before this finishes, so the first ~15 s of video is not tracked |
| **Total pipeline start-up** | **~45–50 s** | every run | budget this before a demo — it is not a hang |
| Webots world load | **~10–20 s** | every run | NAO + floor + contact properties |
| IMU auto-zero calibration | **1.0 s** standing (`IMU_CALIBRATION_S`), 20 samples min | every episode | the robot must be standing with its feet loaded or tilt gates stay idle |
| `pytest -q` (full suite) | **215 s (3.6 min)**, 575 tests | per change | `tests/test_controller_integration.py` dominates |
| Fall → `simulationReset` → ready | **~2–3 s** | per fall | `FALL_CONFIRM_S` 1.0 s + reset + re-calibration |

---

## Technology Stack Summary

### Core Technologies

| Phase | Technology | Purpose |
|-------|-----------|---------|
| Video Input | OpenCV + V4L2/AVFoundation | Cross-platform video capture |
| Pose Detection | MeTRAbs (GPU) | Absolute-3D human pose estimation |
| Retargeting | NumPy + Math | Geometric inverse kinematics |
| Smoothing | Exponential Filter | Temporal noise reduction |
| Communication | UDP Sockets + JSON | Low-latency data transmission |
| Simulation | Webots | Physics-based robot simulation |
| Visualization | OpenCV GUI | Real-time feedback |
| Logging | CSV + Python logging | Data recording & analysis |

### Programming Languages
- **Python 3.10-3.12** - Primary implementation language
- **YAML** - Configuration files

### Key Libraries
- `opencv-python` (≥4.9) - Computer vision
- `tensorflow` (≥2.12, GPU build) + `tensorflow-hub` - MeTRAbs pose estimation
- `numpy` (≥1.26) - Numerical computing
- `pyyaml` (≥6.0) - Configuration parsing
- `scipy` (≥1.11) - Signal processing

### Development Tools
- `pytest` - Unit testing
- `ruff` - Fast Python linter
- `black` - Code formatting
- `make` - Build automation

---

## Configuration

All pipeline parameters are configurable via `configs/default.yaml`:

### Input Configuration
```yaml
input:
  source: 0                    # Camera index or file path
  width: 1280                  # Frame width
  height: 720                  # Frame height
  flip_horizontal: true        # Mirror mode
```

### Pose Detection Configuration
```yaml
pose:
  use_metrabs: true             # Enable MeTRAbs
  model_url: "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip"
  skeleton: coco_19
  default_fov_degrees: 55.0
  detector_threshold: 0.3
  num_aug: 1
  max_detections: 1
  require_gpu: true             # refuse to start without a GPU
  allow_synthetic_fallback: false
```

### Retargeting Configuration
```yaml
retargeting:
  smoothing_alpha: 0.35        # Smoothing strength (0-1)
```

### Runtime Configuration
```yaml
runtime:
  initial_fps: 30
  min_fps: 25
  max_fps: 100
  fps_step: 5
  latency_budget_ms: 150
```

### Webots Configuration
```yaml
webots_bridge:
  enabled: true
  host: 127.0.0.1
  port: 8765
```

---

## Usage Modes

### 1. Full Pipeline (Camera + Webots)
```bash
python run.py
```
- Captures from camera
- Displays skeleton overlay
- Sends commands to Webots robot

### 2. Camera-Only Demo (No Webots)
```bash
python run.py --no-webots
```
- Displays skeleton overlay only
- No robot simulation required
- Useful for testing pose detection

### 3. Headless Mode (No Display)
```bash
python run.py --no-display
```
- Runs without GUI window
- For server/SSH environments
- Still sends to Webots if enabled

### 4. Video File Replay
```bash
python run.py --source path/to/video.mp4
```
- Replays recorded video
- Deterministic execution
- Useful for evaluation

### 5. Limited Frame Count
```bash
python run.py --max-frames 100
```
- Stops after N frames
- For testing and benchmarking

---

## System Requirements

### Hardware
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 4GB minimum (8GB recommended)
- **Camera**: USB webcam or Sony A7 III via HDMI capture
- **GPU**: REQUIRED (CUDA-enabled, matching TensorFlow build) for real-time MeTRAbs inference

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.10, 3.11, or 3.12
- **Webots**: R2023b or later
- **Conda**: Recommended for environment management

### Network
- Localhost UDP port 8765 available
- No internet required (except for initial package installation)

---

## Error Handling & Robustness

### Video Input Failures
- Automatic backend fallback (V4L2 → CAP_ANY)
- Consecutive failure tracking (max 30 failures)
- Graceful degradation with logging

### Pose Detection Failures
- Frame-by-frame retry on detection failure
- Confidence threshold filtering
- Optional synthetic fallback for testing

### Network Failures
- UDP is fire-and-forget (no blocking on failure)
- Dropped packets don't crash pipeline
- Webots controller handles missing frames

### Graceful Shutdown
- Signal handlers for Ctrl+C (SIGINT)
- Proper resource cleanup:
  - Camera release
  - Socket closure
  - Window destruction
  - Log file finalization

---

## Future Extensions

### Potential Enhancements
1. **Deep Learning Retargeting**: Replace geometric IK with learned mapping
2. **Multi-Person Tracking**: Support multiple humans simultaneously
3. **Hand Pose Estimation**: Add finger-level control
4. **Balance Controller**: Advanced stability using reinforcement learning
5. **Hardware Deployment**: Port to physical humanoid robot
6. **Remote Operation**: Network-based teleoperation over internet
7. **Gesture Recognition**: Command robot via hand gestures
8. **Motion Recording**: Record and replay motion sequences

---

## References

- **MeTRAbs**: https://github.com/isarandi/metrabs
- **Webots Documentation**: https://cyberbotics.com/doc/guide/index
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Project Repository**: https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot

---

## Appendix: File Structure

```
src/
├── run.py                    # CLI entrypoint
├── pipeline.py               # Main orchestrator
├── types.py                  # Data structures
├── webots_bridge.py          # UDP communication
├── perception/
│   ├── video_input.py        # Video capture
│   ├── pose_estimator.py     # MeTRAbs wrapper
│   ├── landmarks.py          # Landmark definitions
│   └── visualizer.py         # Skeleton overlay
├── retargeting/
│   └── mapper.py             # Joint angle computation
└── utils/
    ├── config.py             # YAML configuration
    ├── filtering.py          # Smoothing algorithms
    ├── fps.py                # Adaptive FPS control
    └── logger.py             # CSV logging

main/
├── worlds/
│   └── Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt
└── controllers/
    └── pose_imitation_controller/
        ├── pose_imitation_controller.py
        └── pose_imitation_controller_advanced.py

configs/
└── default.yaml              # Runtime configuration

docs/
├── PRD.md                    # Product requirements
├── RUN_INSTRUCTIONS.md       # Setup guide
└── WORKFLOW.md               # This document

tests/
├── test_landmarks.py
├── test_retargeting.py
└── test_utils.py
```

---

**Document Version**: 1.0  
**Last Updated**: May 29, 2026  
**Maintainer**: Tarik Billa
