"""
Utilities for pose imitation control in Webots.

Provides helper functions for motor control, smooth interpolation,
and safety checks for the pose imitation system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MotorConfig:
    """Configuration for a motor joint."""
    name: str
    min_angle: float  # radians
    max_angle: float  # radians
    max_velocity: float  # rad/s
    max_acceleration: float  # rad/s^2


class JointLimiter:
    """Enforces joint angle limits and smooth motion constraints."""

    def __init__(self, configs: Dict[str, MotorConfig]) -> None:
        self.configs = configs

    def clamp_angle(self, joint_name: str, angle: float) -> float:
        """Clamp angle to joint limits."""
        if joint_name not in self.configs:
            return angle
        
        config = self.configs[joint_name]
        return max(config.min_angle, min(config.max_angle, angle))

    def calculate_velocity(
        self,
        joint_name: str,
        current_angle: float,
        target_angle: float,
        timestep_ms: float,
    ) -> float:
        """Calculate appropriate velocity for smooth motion."""
        if joint_name not in self.configs:
            return 0.0
        
        config = self.configs[joint_name]
        angle_diff = target_angle - current_angle
        
        # Time to reach target at max velocity
        time_s = timestep_ms / 1000.0
        required_velocity = angle_diff / (time_s + 1e-6)
        
        # Limit velocity
        limited = min(config.max_velocity, abs(required_velocity))
        return limited if angle_diff >= 0 else -limited

    def validate_trajectory(
        self,
        joint_name: str,
        positions: list[float],
    ) -> bool:
        """Check if a trajectory respects joint limits."""
        if joint_name not in self.configs:
            return True
        
        config = self.configs[joint_name]
        for pos in positions:
            if pos < config.min_angle or pos > config.max_angle:
                return False
        return True


class SmoothInterpolator:
    """Performs smooth interpolation between poses."""

    @staticmethod
    def linear_interpolate(
        current: float,
        target: float,
        alpha: float,  # 0.0 = current, 1.0 = target
    ) -> float:
        """Linear interpolation between current and target."""
        return current * (1.0 - alpha) + target * alpha

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out function for smooth acceleration/deceleration."""
        if t < 0.5:
            return 4.0 * t ** 3
        else:
            return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0

    @staticmethod
    def smooth_step(
        current: float,
        target: float,
        smoothing_factor: float,
    ) -> float:
        """Smooth step function for exponential smoothing."""
        return current + (target - current) * smoothing_factor


class MotorHealthMonitor:
    """Monitors motor health and detects anomalies."""

    def __init__(self, max_position_error: float = 0.05) -> None:
        self.max_position_error = max_position_error
        self.position_errors: Dict[str, list[float]] = {}
        self.velocity_spikes: Dict[str, int] = {}

    def check_position_error(
        self,
        joint_name: str,
        target: float,
        current: float,
    ) -> bool:
        """Check if position tracking error is within bounds."""
        error = abs(target - current)
        if joint_name not in self.position_errors:
            self.position_errors[joint_name] = []
        
        self.position_errors[joint_name].append(error)
        # Keep last 100 errors
        if len(self.position_errors[joint_name]) > 100:
            self.position_errors[joint_name].pop(0)
        
        return error <= self.max_position_error

    def get_average_error(self, joint_name: str) -> float:
        """Get average position error for a joint."""
        if joint_name not in self.position_errors:
            return 0.0
        
        errors = self.position_errors[joint_name]
        return sum(errors) / len(errors) if errors else 0.0

    def detect_stuck_motor(self, joint_name: str) -> bool:
        """Detect if a motor is stuck (not moving when commanded)."""
        if joint_name not in self.position_errors:
            return False
        
        recent_errors = self.position_errors[joint_name][-10:]
        if not recent_errors:
            return False
        
        # Motor is stuck if error is consistently high
        avg = sum(recent_errors) / len(recent_errors)
        return avg > self.max_position_error * 2


def get_default_motor_configs() -> Dict[str, MotorConfig]:
    """Return default motor configurations for NAO-like humanoid."""
    deg2rad = math.radians
    
    return {
        "LShoulderPitch": MotorConfig(
            name="LShoulderPitch",
            min_angle=deg2rad(-119),
            max_angle=deg2rad(119),
            max_velocity=2.0,
            max_acceleration=10.0,
        ),
        "RShoulderPitch": MotorConfig(
            name="RShoulderPitch",
            min_angle=deg2rad(-119),
            max_angle=deg2rad(119),
            max_velocity=2.0,
            max_acceleration=10.0,
        ),
        "LElbowRoll": MotorConfig(
            name="LElbowRoll",
            min_angle=deg2rad(0),
            max_angle=deg2rad(135),
            max_velocity=2.0,
            max_acceleration=10.0,
        ),
        "RElbowRoll": MotorConfig(
            name="RElbowRoll",
            min_angle=deg2rad(-135),
            max_angle=deg2rad(0),
            max_velocity=2.0,
            max_acceleration=10.0,
        ),
        "LHipPitch": MotorConfig(
            name="LHipPitch",
            min_angle=deg2rad(-88),
            max_angle=deg2rad(27),
            max_velocity=1.5,
            max_acceleration=8.0,
        ),
        "RHipPitch": MotorConfig(
            name="RHipPitch",
            min_angle=deg2rad(-88),
            max_angle=deg2rad(27),
            max_velocity=1.5,
            max_acceleration=8.0,
        ),
        "TorsoPitch": MotorConfig(
            name="TorsoPitch",
            min_angle=deg2rad(-30),
            max_angle=deg2rad(30),
            max_velocity=1.0,
            max_acceleration=5.0,
        ),
    }
