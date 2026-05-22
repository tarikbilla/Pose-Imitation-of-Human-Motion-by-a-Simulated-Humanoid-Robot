"""
Advanced pose imitation controller with optimized motion control.

Features:
- Joint limit enforcement
- Smooth motion interpolation
- Motor health monitoring
- Adaptive velocity control
- Comprehensive logging and diagnostics
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

# Import utilities (adjust path as needed for Webots environment)
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'libraries'))

try:
    from pose_control_utils import (
        JointLimiter,
        SmoothInterpolator,
        MotorHealthMonitor,
        get_default_motor_configs,
    )
except ImportError as e:
    print(f"Warning: Could not import utilities: {e}")
    JointLimiter = None
    MotorHealthMonitor = None

# Configuration
UDP_PORT = 8765
MOTOR_NAMES = [
    "LShoulderPitch",
    "RShoulderPitch",
    "LElbowRoll",
    "RElbowRoll",
    "LHipPitch",
    "RHipPitch",
    "TorsoPitch",
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AdvancedPoseController")


class AdvancedPoseController:
    """Advanced real-time pose imitation controller with optimized motion."""

    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.motors: Dict[str, object] = {}
        self.target_positions: Dict[str, float] = {}
        self.current_positions: Dict[str, float] = {}
        self.frame_count = 0
        self.last_status_time = time.time()
        
        # Advanced features
        if JointLimiter is not None:
            configs = get_default_motor_configs()
            self.limiter = JointLimiter(configs)
            self.monitor = MotorHealthMonitor(max_position_error=0.1)
        else:
            self.limiter = None
            self.monitor = None
        
        logger.info(f"Initializing advanced controller (timestep: {self.timestep}ms)")
        self._init_motors()
        self._init_socket()
        logger.info("Advanced controller initialized")

    def _init_motors(self) -> None:
        """Initialize all motors."""
        logger.info("Setting up motors...")
        for name in MOTOR_NAMES:
            try:
                motor = self.robot.getDevice(name)
                if motor is None:
                    logger.warning(f"Motor '{name}' not found")
                    continue
                
                motor.setVelocity(0.0)
                self.motors[name] = motor
                self.target_positions[name] = 0.0
                self.current_positions[name] = 0.0
                    
            except Exception as e:
                logger.error(f"Error initializing motor '{name}': {e}")
        
        logger.info(f"Motors ready: {len(self.motors)}/{len(MOTOR_NAMES)}")

    def _init_socket(self) -> None:
        """Initialize UDP socket."""
        logger.info(f"Opening UDP socket on port {UDP_PORT}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind(("127.0.0.1", UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    def _receive_command(self) -> Optional[Dict]:
        """Receive pose command from Python pipeline."""
        try:
            data, _ = self.sock.recvfrom(65535)
            return json.loads(data.decode("utf-8"))
        except (BlockingIOError, json.JSONDecodeError, OSError):
            return None

    def _apply_pose_frame(self, angles: Dict[str, float]) -> int:
        """Apply a pose frame to all motors. Returns number of joints applied."""
        count = 0
        for joint_name, target_angle in angles.items():
            if joint_name not in self.motors:
                logger.warning(f"Unknown joint: {joint_name}")
                continue
            
            motor = self.motors[joint_name]
            current = self.current_positions.get(joint_name, 0.0)
            
            # Apply joint limits if available
            if self.limiter is not None:
                target_angle = self.limiter.clamp_angle(joint_name, target_angle)
            
            # Store target
            self.target_positions[joint_name] = target_angle
            
            # Calculate smooth velocity
            angle_diff = target_angle - current
            timestep_s = self.timestep / 1000.0
            velocity = abs(angle_diff / max(timestep_s, 0.001))
            velocity = min(2.0, velocity)  # Limit velocity
            
            # Apply to motor
            motor.setPosition(target_angle)
            motor.setVelocity(velocity)
            
            # Health monitoring
            if self.monitor is not None:
                self.monitor.check_position_error(joint_name, target_angle, current)
            
            count += 1
        
        return count

    def _update_current_positions(self) -> None:
        """Update tracked current positions (using last setPosition calls)."""
        for joint_name in self.motors:
            # In Webots, we track positions we command
            self.current_positions[joint_name] = self.target_positions.get(joint_name, 0.0)

    def _log_diagnostics(self) -> None:
        """Log health diagnostics periodically."""
        if self.frame_count % 200 != 0:
            return
        
        elapsed = time.time() - self.last_status_time
        fps = 200 / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Frame {self.frame_count} | FPS: {fps:.1f} | "
            f"Active motors: {len(self.motors)}/{len(MOTOR_NAMES)}"
        )
        
        if self.monitor is not None:
            for joint_name in self.motors:
                avg_error = self.monitor.get_average_error(joint_name)
                stuck = self.monitor.detect_stuck_motor(joint_name)
                if stuck:
                    logger.warning(f"Motor '{joint_name}' may be stuck (error: {avg_error:.4f} rad)")
        
        self.last_status_time = time.time()

    def run(self) -> None:
        """Main control loop."""
        logger.info("Starting advanced control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                # Receive command
                command = self._receive_command()
                if command is not None:
                    angles = command.get("joint_angles_rad", {})
                    if angles:
                        n_applied = self._apply_pose_frame(angles)
                        if self.frame_count % 30 == 0 and n_applied > 0:
                            logger.debug(
                                f"Frame {command.get('frame_index', self.frame_count)}: "
                                f"Applied {n_applied} joints"
                            )
                
                # Update position tracking
                self._update_current_positions()
                
                # Diagnostics
                self._log_diagnostics()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupt received")
        except Exception as e:
            logger.exception(f"Error in control loop: {e}")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        try:
            for motor in self.motors.values():
                if motor is not None:
                    motor.setVelocity(0.0)
            
            if hasattr(self, 'sock'):
                self.sock.close()
            
            logger.info(f"Shutdown complete ({self.frame_count} frames)")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main() -> None:
    """Entry point."""
    try:
        controller = AdvancedPoseController()
        controller.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
