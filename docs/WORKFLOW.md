# System Workflow Documentation

## Overview

This document describes the complete workflow and architecture of the **Pose Imitation of Human Motion by a Simulated Humanoid Robot** system. The pipeline captures human motion from video input, processes it through computer vision algorithms, and drives a simulated humanoid robot in Webots to imitate the detected poses in real-time.

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
│  (MediaPipe)    │
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
│  UDP Bridge to  │
│     Webots      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Webots Humanoid │
│    Simulation   │
└─────────────────┘
```

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

### Key Features
- Robust error handling with consecutive failure tracking
- Platform-aware backend selection
- Adaptive frame rate to match pipeline throughput
- Low-latency configuration (buffer size = 1)

---

## Phase 2: Pose Detection & Estimation

### Technology Stack
- **MediaPipe Pose** - Google's ML-based human pose estimation
- **NumPy** - Numerical computations
- **OpenCV** - Image preprocessing

### Components
- `src/perception/pose_estimator.py` - `PoseEstimator` class
- `src/perception/landmarks.py` - Landmark definitions (33 keypoints)

### Process Flow

1. **Initialization**
   - Loads MediaPipe Pose model with configurable complexity:
     - 0 = Lite (fastest)
     - 1 = Full (balanced, default)
     - 2 = Heavy (most accurate, slowest)
   - Sets detection and tracking confidence thresholds (default: 0.35)
   - Enables landmark smoothing for temporal stability

2. **Frame Processing**
   - Converts BGR frame to RGB (MediaPipe requirement)
   - Feeds image to MediaPipe Pose detector
   - Receives 33 body landmarks per frame

3. **Landmark Extraction**
   - **33 Keypoints** including:
     - Face: nose, eyes, ears, mouth
     - Upper body: shoulders, elbows, wrists, hands
     - Torso: hips
     - Lower body: knees, ankles, feet, toes
   - Each landmark contains:
     - (x, y, z) coordinates (normalized 0-1 for x,y; depth for z)
     - Visibility score (0-1)

4. **Output Generation**
   - Creates `PoseFrame` object with:
     - Timestamp
     - Frame index
     - Dictionary of named keypoints (e.g., "left_shoulder", "right_elbow")

### Key Features
- Real-time human pose tracking (25-100 FPS)
- Robust cross-platform support
- Explicit failure handling (no silent fallbacks unless configured)
- Optional synthetic fallback mode for testing (disabled by default)

### Detected Landmarks
```
Head: nose, left/right eye (inner/outer), left/right ear
Upper Body: left/right shoulder, left/right elbow, left/right wrist
Hands: left/right pinky, left/right index, left/right thumb
Torso: left/right hip
Lower Body: left/right knee, left/right ankle
Feet: left/right heel, left/right foot_index
```

---

## Phase 3: Pose Retargeting & Joint Mapping

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
- **MediaPipe Drawing Utils** - Skeleton rendering

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
   - Converts `JointCommand` to JSON:
     ```json
     {
       "timestamp_s": 1.234,
       "frame_index": 42,
       "joint_angles_rad": {
         "LShoulderPitch": 0.52,
         "RShoulderPitch": -0.31,
         ...
       }
     }
     ```
   - Encodes as UTF-8 bytes

3. **UDP Transmission**
   - Sends datagram to Webots controller
   - No acknowledgment required
   - Minimal overhead (~1ms per frame)

4. **Optional Disabling**
   - Flag: `--no-webots`
   - Enables pure perception demo mode
   - Useful for testing without Webots

### Key Features
- Ultra-low latency (<5ms network overhead)
- Fire-and-forget messaging
- No connection management overhead
- Easy to disable for testing

---

## Phase 8: Webots Robot Simulation

### Technology Stack
- **Webots R2023b+** - Robot simulation environment
- **Python Webots API** - Controller interface
- **UDP Socket** - Command reception

### Components
- World file: `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`
- Controller: `main/controllers/pose_imitation_controller/pose_imitation_controller.py` or `pose_imitation_controller_advanced.py`

### Process Flow

1. **Simulation Initialization**
   - Loads humanoid robot model
   - Initializes robot joints (motors)
   - Creates UDP socket listener on port 8765

2. **Command Reception**
   - Listens for UDP datagrams (non-blocking)
   - Deserializes JSON payload
   - Extracts joint angle commands

3. **Joint Actuation**
   - Sets target positions for robot motors
   - Applies joint angles in radians
   - Webots physics engine handles:
     - Motor dynamics
     - Collision detection
     - Balance physics
     - Gravity simulation

4. **Simulation Step**
   - Webots updates at simulation timestep (typically 32ms)
   - Robot moves to target pose
   - Physics constraints enforced

### Supported Joints
- Shoulders (pitch/roll)
- Elbows (roll)
- Hips (pitch)
- Torso (pitch)
- Additional joints can be added as needed

### Key Features
- Realistic physics simulation
- Real-time robot visualization
- Balance and collision handling
- Extendable robot model

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
2. Pose Detection (MediaPipe) → 33 landmarks
   ↓
3. Joint Mapping (Geometric IK) → 7 joint angles
   ↓
4. Smoothing (Exponential filter) → Filtered angles
   ↓
5. UDP Send (JSON over UDP) → Webots controller
   ↓
6. Robot Actuation (Webots physics) → Humanoid imitation
```

### Key Performance Characteristics
- **End-to-end latency**: 50-150ms (adaptive)
- **Frame rate**: 25-100 FPS (adaptive)
- **Pose detection**: 25-35ms per frame
- **Retargeting**: <1ms per frame
- **UDP transmission**: <1ms per frame
- **Smoothing**: <1ms per frame

---

## Technology Stack Summary

### Core Technologies

| Phase | Technology | Purpose |
|-------|-----------|---------|
| Video Input | OpenCV + V4L2/AVFoundation | Cross-platform video capture |
| Pose Detection | MediaPipe Pose | ML-based human pose estimation |
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
- `mediapipe` (≥0.10.13) - Pose estimation
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
  use_mediapipe: true          # Enable MediaPipe
  model_complexity: 1          # 0=lite, 1=full, 2=heavy
  min_detection_confidence: 0.35
  min_tracking_confidence: 0.35
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
- **GPU**: Optional (MediaPipe uses CPU by default)

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

- **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose.html
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
│   ├── pose_estimator.py     # MediaPipe wrapper
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
