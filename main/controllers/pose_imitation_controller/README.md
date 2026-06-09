# Webots NAO Pose Imitation Controller

Real-time control system for the simulated **NAO (H25)** humanoid robot based on
human pose tracking.

## Overview

This system enables the Webots NAO robot to imitate human movements in real-time by:
1. Capturing human pose with MediaPipe (up to 33 landmarks per frame) — Python side
2. Mapping human pose to generic joint angles — Python side (`src/retargeting/`)
3. Sending joint commands via UDP (port 8765) to the Webots controller
4. The controller **re-maps those angles to NAO's joint conventions**, smooths,
   clamps to mechanical limits, and drives the motors while holding balance

### Architecture

The Python pipeline streams, per frame, **both** raw MediaPipe landmarks and a
small set of pre-computed joint angles. The controller prefers the landmarks and
does **full-body retargeting** itself; the joint angles are a fallback.

- [`main/libraries/nao_retarget.py`](../../libraries/nao_retarget.py) — landmarks → NAO joints (shoulder pitch+roll, elbow, head, optional legs)
- [`main/libraries/pose_control_utils.py`](../../libraries/pose_control_utils.py) — `NaoPoseDriver` (limits, smoothing, standing balance) + logging
- the two controller files are thin Webots/UDP wrappers

```
UDP frame ─► controller (socket loop)
                 │  keypoints (preferred)         joint_angles_rad (fallback)
                 ▼                                       │
            retarget_full_body()  ◄──────────────────────┘ map_pipeline_angles()
                 │  NAO joint targets
                 ▼
            NaoPoseDriver
                 │  1. clamp to NAO limits     (FR-5)
                 │  2. exponential smoothing   (FR-6)
                 │  3. setPosition + capped velocity
                 ▼
            NAO motors  (+ standing posture held for balance — NFR-4)
```

### What follows the human (full body)

| Human motion | NAO joints | Notes |
|---|---|---|
| Arm up / down | `ShoulderPitch` | vertical component of the upper-arm |
| Arm left / right (abduction) | `ShoulderRoll` | lateral component — **now working** |
| Elbow bend | `ElbowRoll` | angle between upper-arm and forearm |
| Head turn / nod | `HeadYaw`, `HeadPitch` | from nose vs. shoulder midline (`DRIVE_HEAD`) |
| Legs (hip/knee/ankle) | `HipPitch`, `KneePitch`, `AnklePitch` | `DRIVE_LEGS` — **off by default**, no balance controller yet |

The shoulder is the key part: NAO's 2-DOF shoulder is recovered from the single
observed arm direction by splitting it into a vertical part (`ShoulderPitch`) and
a lateral part (`ShoulderRoll`) — depth-free, so it is robust for a frontal
camera. Every joint is gated on landmark **visibility**, so anything out of frame
(usually the legs) is simply not commanded and the driver holds its last pose.

**Tunables** (top of each controller): `DRIVE_HEAD`, `DRIVE_LEGS`,
`SWAP_SIDES` (mirror-image mapping), `SMOOTHING_ALPHA`, `VELOCITY_SCALE`.
If left/right feels mirrored the wrong way, set `SWAP_SIDES = True`.

### Why a mapping layer is required

The Python pipeline emits angles in a neutral convention that does **not** match
NAO's joints. The driver corrects this:

| Issue | Pipeline sends | NAO expects | Correction |
|---|---|---|---|
| Left elbow | `LElbowRoll` **positive** when bent | **negative** (-1.54 … -0.03) | negate |
| Right elbow | `RElbowRoll` **negative** when bent | **positive** (0.03 … 1.54) | negate |
| Shoulder pitch | arm-down ≈ -1.57 | arm-down ≈ +1.57 | negate |
| Torso | `TorsoPitch` | *no such motor on NAO* | dropped |

Without this layer `Motor.setPosition()` silently clamps the elbows straight and
they never bend — which is the bug the previous controller hit.

## Controllers

### 1. `pose_imitation_controller.py` (Standard) — **recommended**

- Re-maps pipeline angles to NAO conventions and clamps to limits
- Exponential smoothing + per-joint velocity caps for smooth motion
- Holds a stable standing posture (legs straight & stiff) so NAO does not fall
- Drains the UDP backlog each step and applies only the freshest frame (low latency)
- Holds the last pose if commands stop arriving (stale detection)
- Position-sensor feedback + stuck-motor warnings
- This is the controller wired into the world file's `Nao { controller ... }`

### 2. `pose_imitation_controller_advanced.py` (Advanced)

Same `NaoPoseDriver` core, tuned for experimentation:
- Smoother (laggier) motion settings
- Optional **leg driving** (`DRIVE_LEGS = True`) for lower-body experiments —
  off by default because NAO has no balance controller yet and will fall
- Verbose per-joint diagnostics (commanded vs. measured, average error, stuck flags)

**Both controllers require** the shared `pose_control_utils.py` library (auto-added
to `sys.path`).

### Output: joint-trajectory log (FR-7 / US-3)

With `ENABLE_TRAJECTORY_LOG = True` (default), each run writes a CSV to
`<project>/logs/webots_joint_trajectory_<epoch>.csv` containing, per simulation
frame: `wall_time_s, sim_time_s, frame_index`, and for every driven joint a
`<joint>_cmd_rad` (commanded) and `<joint>_meas_rad` (achieved, from the position
sensor) column. This is the data the evaluation step uses to compute per-joint
MAE (target vs. achieved) and timing. Logging is fully defensive — any I/O error
disables it without disturbing the real-time loop. The `logs/` directory is
git-ignored.

## Communication Protocol

The controllers receive JSON-formatted UDP packets on **port 8765**:

```json
{
  "timestamp_s": 1234567890.123,
  "frame_index": 45,
  "joint_angles_rad": { "LShoulderPitch": 0.5, "RElbowRoll": -1.1 },
  "keypoints": {
    "left_shoulder":  [0.40, 0.40, -0.1, 0.99],
    "left_elbow":     [0.22, 0.40, -0.1, 0.98],
    "left_wrist":     [0.06, 0.40, -0.1, 0.97],
    "right_shoulder": [0.60, 0.40, -0.1, 0.99],
    "nose":           [0.50, 0.22, -0.2, 0.99]
  }
}
```

### Fields:
- **timestamp_s**: Frame timestamp in seconds
- **frame_index**: Frame number from pipeline
- **keypoints** *(preferred)*: MediaPipe landmarks `name → [x, y, z, visibility]`
  in normalized image coordinates (`x` right, `y` down, both 0–1). The controller
  retargets these to the full NAO pose. A curated subset (~17 landmarks: head,
  shoulders, elbows, wrists, hips, knees, ankles) is streamed to keep packets small.
- **joint_angles_rad** *(fallback)*: pre-computed joint angles, used only when no
  `keypoints` are present.

## Supported Joints (NAO H25 hardware ranges)

These are the **actual NAO motor ranges** the driver clamps to. Note the elbow
sign convention — it is the opposite of what the pipeline sends, which is why the
driver negates those channels.

| Pipeline key | NAO motor | NAO range | Driven by default |
|---|---|---|---|
| LShoulderPitch | LShoulderPitch | -119.5° to +119.5° | ✅ (negated) |
| RShoulderPitch | RShoulderPitch | -119.5° to +119.5° | ✅ (negated) |
| LElbowRoll | LElbowRoll | **-88.5° to -2°** | ✅ (negated) |
| RElbowRoll | RElbowRoll | **+2° to +88.5°** | ✅ (negated) |
| LHipPitch | LHipPitch | -88° to +27.7° | ⛔ legs gated (balance) |
| RHipPitch | RHipPitch | -88° to +27.7° | ⛔ legs gated (balance) |
| TorsoPitch | — | *no NAO motor* | ❌ dropped |

Joints not driven by imitation (shoulder roll, elbow yaw, wrists, head, and the
full leg chain) are held at a neutral standing posture so the robot keeps a
natural pose and stays balanced. Leg driving can be enabled in the advanced
controller via `DRIVE_LEGS = True`.

## Configuration

### UDP Settings
- **Address**: `127.0.0.1` (localhost)
- **Port**: `8765`
- **Timeout**: `1.0 second`
- **Buffer**: `65536 bytes`

### Motion Settings
- **Max Velocity**: `2.0 rad/s` (smooth but responsive)
- **Position Tolerance**: `0.01 rad`
- **Update Rate**: Same as Webots timestep (typically 32-64ms)

### Logging
Set `log_level` in pipeline config to see:
- Frame-by-frame updates (DEBUG)
- Motor status (INFO)
- Performance metrics (INFO)
- Errors and warnings (WARNING/ERROR)

## Real-Time Performance

### Expected Performance
- **Latency**: <50ms end-to-end (camera → detection → command → robot)
- **FPS**: 30-50 fps typical, up to 100 fps possible
- **Smoothness**: Smooth continuous motion due to velocity control
- **Responsiveness**: Immediate response to detected human movements

### Optimization Tips
1. **Lower detection confidence** if humans are not detected consistently
2. **Reduce frame resolution** (e.g., 640x480) for higher FPS
3. **Enable smooth_landmarks** in MediaPipe config
4. **Use model_complexity: 1** for balanced performance

## Integration with Pipeline

The controllers integrate seamlessly with the Python pipeline:

```
Human in Camera
    ↓
MediaPipe Pose Detection (25-27/33 landmarks)
    ↓
Retargeting Mapper (human pose → robot joints)
    ↓
Smoothing Filter (exponential smoothing)
    ↓
UDP Command (JSON packet)
    ↓
Webots Controller (receives and applies)
    ↓
Robot Moves in Real-Time
```

## Running the System

### 1. Start Webots
```bash
# Open the world file with the pose imitation controller
# main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt
```

### 2. Run Python Pipeline
```bash
conda activate py312
cd /path/to/project
python run.py
```

The pipeline will:
- Detect human pose from webcam
- Send commands to Webots (port 8765)
- Show landmarks on camera feed
- Log joint tracking

### 3. Move in Front of Camera
The robot will follow your movements in real-time!

## Troubleshooting

### Robot Not Moving
1. Check if controller is running in Webots
2. Verify UDP port 8765 is not blocked
3. Check pipeline log for "Webots bridge sending to 127.0.0.1:8765"
4. Ensure human is detected ("Human detected" status)

### Jerky/Stuttering Motion
1. Lower `min_detection_confidence` (0.35 default)
2. Reduce camera resolution
3. Disable Webots graphics (run headless) for better performance

### Stuck Motors
1. Check motor names in controller match robot PROTO
2. Verify joint angle ranges are within limits
3. Check for motor max torque settings in Webots

### No UDP Messages Received
```bash
# On Linux, check if port is listening:
netstat -uln | grep 8765

# Or use UDP sniffer:
tcpdump -i lo udp port 8765
```

## Advanced Usage

### Using Motor Health Monitor
The advanced controller automatically:
- Tracks position errors per frame
- Detects stuck motors
- Logs diagnostics every 200 frames

Check logs for warnings like:
```
Motor 'RElbowRoll' may be stuck (error: 0.0850 rad)
```

### Custom Motor Configurations
Edit `pose_control_utils.py`:
```python
def get_default_motor_configs() -> Dict[str, MotorConfig]:
    return {
        "CustomJoint": MotorConfig(
            name="CustomJoint",
            min_angle=math.radians(-90),
            max_angle=math.radians(90),
            max_velocity=1.5,
            max_acceleration=5.0,
        ),
    }
```

## Performance Metrics

Monitor these in real-time via logs:

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| FPS | >30 | 20-30 | <20 |
| Latency | <50ms | 50-100ms | >100ms |
| Visible Landmarks | >20/33 | 15-20/33 | <15/33 |
| Position Error | <0.01 rad | 0.01-0.05 rad | >0.05 rad |

## Development Notes

### Code Structure
```
main/
├── controllers/
│   └── pose_imitation_controller/
│       ├── pose_imitation_controller.py (standard)
│       └── pose_imitation_controller_advanced.py (advanced)
└── libraries/
    └── pose_control_utils.py (utilities)
```

### Testing Controller Locally
For testing without full Webots:
```python
# Mock controller for development
import socket
import json
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 8765))

while True:
    data, addr = sock.recvfrom(65535)
    payload = json.loads(data)
    print(f"Frame {payload['frame_index']}: {len(payload['joint_angles_rad'])} joints")
```

## Future Enhancements

- [ ] Support for NAO V6 specific motors
- [ ] Inverse kinematics for more natural movement
- [ ] Multi-sensor feedback (accelerometers, gyros)
- [ ] Collision avoidance
- [ ] Motion recording and playback
- [ ] Network communication over internet

## License

See main project LICENSE file.

## Questions?

For issues or questions:
1. Check logs: `logs/run_*/pose_keypoints.csv`
2. Enable DEBUG logging: `python run.py --log-level DEBUG`
3. Check Webots console for controller errors
4. Verify UDP connectivity: `netstat -uln | grep 8765`
