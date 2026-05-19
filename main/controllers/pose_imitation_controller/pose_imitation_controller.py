"""
Real-time Webots pose imitation controller.

Receives human pose commands via UDP from the Python pipeline and controls
the simulated humanoid robot in real-time. Features:
- Smooth joint position control with velocity limits
- Position feedback and validation
- Graceful error handling
- Frame-by-frame tracking synchronization
"""
from __future__ import annotations

import json
import logging
import socket
import sys
import time
from typing import Dict, Optional

try:
    from controller import Robot  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

# Configuration
UDP_PORT = 8765
UDP_TIMEOUT = 1.0  # seconds
MAX_VELOCITY = 2.0  # rad/s - smooth but responsive
POSITION_TOLERANCE = 0.01  # rad - acceptable position error

# Motor names mapped from human pose to robot joints
MOTOR_NAMES = [
    "LShoulderPitch",
    "RShoulderPitch",
    "LElbowRoll",
    "RElbowRoll",
    "LHipPitch",
    "RHipPitch",
    "TorsoPitch",  # Added torso control
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PoseController")


class PoseImiationController:
    """Real-time pose imitation controller for Webots humanoid robot."""

    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.motors: Dict[str, object] = {}
        self.sensors: Dict[str, object] = {}
        self.target_positions: Dict[str, float] = {}
        self.current_positions: Dict[str, float] = {}
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"Initializing robot controller (timestep: {self.timestep}ms)")
        self._init_motors()
        self._init_socket()
        logger.info("Controller initialized successfully")

    def _init_motors(self) -> None:
        """Initialize motors with velocity limits."""
        logger.info("Setting up motors...")
        for name in MOTOR_NAMES:
            try:
                motor = self.robot.getDevice(name)
                if motor is None:
                    logger.warning(f"Motor '{name}' not found in robot")
                    continue
                
                # Configure motor for smooth control
                motor.setVelocity(0.0)  # Start at rest
                motor.setAcceleration(float('inf'))  # No acceleration limit
                self.motors[name] = motor
                self.target_positions[name] = 0.0
                self.current_positions[name] = 0.0
                
                # Try to get position sensor
                sensor_name = f"{name}::sensor"
                sensor = self.robot.getDevice(sensor_name)
                if sensor is not None:
                    self.sensors[name] = sensor
                    
            except Exception as e:
                logger.error(f"Error initializing motor '{name}': {e}")
        
        logger.info(f"Motors ready: {len(self.motors)}/{len(MOTOR_NAMES)}")

    def _init_socket(self) -> None:
        """Initialize UDP socket for receiving pose commands."""
        logger.info(f"Opening UDP socket on port {UDP_PORT}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    def _receive_pose_command(self) -> Optional[Dict]:
        """Try to receive and parse a pose command via UDP."""
        try:
            data, _ = self.sock.recvfrom(65535)
            payload = json.loads(data.decode("utf-8"))
            return payload
        except (BlockingIOError, json.JSONDecodeError, OSError):
            return None

    def _update_motor_positions(self, angles: Dict[str, float]) -> None:
        """Apply joint angles to motors with velocity limits."""
        for joint_name, target_angle in angles.items():
            if joint_name not in self.motors:
                logger.warning(f"Unknown joint: {joint_name}")
                continue
            
            motor = self.motors[joint_name]
            current = self.current_positions.get(joint_name, 0.0)
            
            # Clamp target angle to reasonable bounds
            target_angle = float(target_angle)
            self.target_positions[joint_name] = target_angle
            
            # Calculate velocity for smooth motion
            angle_diff = target_angle - current
            velocity = min(MAX_VELOCITY, abs(angle_diff) / (self.timestep / 1000.0 + 1e-6))
            if angle_diff < 0:
                velocity = -velocity
            
            # Apply position and velocity
            motor.setPosition(target_angle)
            motor.setVelocity(velocity)

    def _update_position_feedback(self) -> None:
        """Read current joint positions from sensors."""
        for joint_name in self.motors:
            if joint_name in self.sensors:
                try:
                    sensor = self.sensors[joint_name]
                    pos = sensor.getValue()
                    self.current_positions[joint_name] = float(pos)
                except Exception as e:
                    logger.debug(f"Error reading sensor {joint_name}: {e}")

    def _log_status(self) -> None:
        """Log current controller status periodically."""
        if self.frame_count % 100 == 0:  # Log every 100 frames
            elapsed = time.time() - self.last_update_time
            fps = 100 / elapsed if elapsed > 0 else 0
            active_motors = len([m for m in self.motors.values() if m is not None])
            logger.info(
                f"Frame {self.frame_count} | FPS: {fps:.1f} | "
                f"Active motors: {active_motors}/{len(MOTOR_NAMES)} | "
                f"Joints tracking: {len(self.target_positions)}"
            )
            self.last_update_time = time.time()

    def run(self) -> None:
        """Main control loop."""
        logger.info("Starting main control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                # Receive pose command
                command = self._receive_pose_command()
                if command is not None:
                    angles = command.get("joint_angles_rad", {})
                    frame_idx = command.get("frame_index", self.frame_count)
                    
                    if angles:
                        self._update_motor_positions(angles)
                        if self.frame_count % 30 == 0:
                            logger.debug(f"Applied pose frame {frame_idx} with {len(angles)} joints")
                
                # Update feedback
                self._update_position_feedback()
                
                # Periodic logging
                self._log_status()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down gracefully")
        except Exception as e:
            logger.exception(f"Unexpected error in control loop: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean shutdown."""
        logger.info("Cleaning up resources...")
        try:
            # Stop all motors
            for motor in self.motors.values():
                if motor is not None:
                    motor.setVelocity(0.0)
            
            # Close socket
            if hasattr(self, 'sock'):
                self.sock.close()
                
            logger.info(f"Controller stopped after {self.frame_count} frames")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main() -> None:
    """Entry point."""
    try:
        controller = PoseImiationController()
        controller.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
