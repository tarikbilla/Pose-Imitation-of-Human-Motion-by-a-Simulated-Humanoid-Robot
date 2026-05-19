# Webots Pose Imitation Controller

Real-time control system for the simulated humanoid robot based on human pose tracking.

## Overview

This system enables the Webots humanoid robot to imitate human movements in real-time by:
1. Capturing human pose with MediaPipe (21-27/33 landmarks per frame)
2. Mapping human pose to robot joint angles
3. Sending joint commands via UDP to the Webots controller
4. The controller applies joint angles and controls motor movements

## Controllers

### 1. `pose_imitation_controller.py` (Standard)
**Recommended for most use cases.**

Features:
- Real-time joint angle application
- Smooth velocity-based motion
- Position feedback tracking
- Comprehensive logging
- UDP command reception on port 8765

**Usage:**
```bash
# Set as the main controller in Webots world file
# The controller will start listening for commands on UDP 8765
```

### 2. `pose_imitation_controller_advanced.py` (Advanced)
**For advanced motion control and diagnostics.**

Additional features:
- Joint limit enforcement
- Motor health monitoring
- Stuck motor detection
- Adaptive velocity control
- Enhanced diagnostics

**Requires:** `pose_control_utils.py` library

## Communication Protocol

The controllers receive JSON-formatted UDP packets on **port 8765**:

```json
{
  "timestamp_s": 1234567890.123,
  "frame_index": 45,
  "joint_angles_rad": {
    "LShoulderPitch": 0.5,
    "RShoulderPitch": -0.3,
    "LElbowRoll": 1.2,
    "RElbowRoll": -1.1,
    "LHipPitch": -0.1,
    "RHipPitch": 0.2,
    "TorsoPitch": 0.05
  }
}
```

### Fields:
- **timestamp_s**: Frame timestamp in seconds
- **frame_index**: Frame number from pipeline
- **joint_angles_rad**: Dictionary of joint names to angles (radians)

## Supported Joints

| Joint Name | Range | Purpose |
|---|---|---|
| LShoulderPitch | -119° to +119° | Left shoulder pitch |
| RShoulderPitch | -119° to +119° | Right shoulder pitch |
| LElbowRoll | 0° to +135° | Left elbow roll |
| RElbowRoll | -135° to 0° | Right elbow roll |
| LHipPitch | -88° to +27° | Left hip pitch |
| RHipPitch | -88° to +27° | Right hip pitch |
| TorsoPitch | -30° to +30° | Torso pitch (new) |

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
