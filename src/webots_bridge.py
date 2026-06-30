from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Dict, Optional

from src.type_defs import JointCommand, Keypoint

# Landmarks streamed to the Webots controller for full-body retargeting.
# A curated subset keeps the UDP packet small (low latency, NFR-1) while still
# covering every joint the controller maps: arms, head and legs.
KEYPOINTS_TO_STREAM = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass
class WebotsBridge:
    host: str
    port: int

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _encode(
        self,
        command: JointCommand,
        keypoints: Optional[Dict[str, Keypoint]] = None,
        gait: Optional[Dict[str, object]] = None,
    ) -> bytes:
        payload: Dict[str, object] = {
            "timestamp_s": command.timestamp_s,
            "frame_index": command.frame_index,
            "joint_angles_rad": command.joint_angles_rad,
        }
        if keypoints:
            payload["keypoints"] = {
                name: [kp.x, kp.y, kp.z, kp.visibility]
                for name, kp in keypoints.items()
                if name in KEYPOINTS_TO_STREAM
            }
        if gait is not None:
            # Compact walk command (cadence/phase/swing/stop) for the on-robot
            # gait engine. Additive and optional: older controllers ignore it.
            payload["gait"] = gait
        return json.dumps(payload).encode("utf-8")

    def send_joint_command(self, command: JointCommand) -> None:
        """Backward-compatible: send joint angles only."""
        self._sock.sendto(self._encode(command), (self.host, self.port))

    def send_pose_frame(
        self,
        command: JointCommand,
        keypoints: Optional[Dict[str, Keypoint]] = None,
        gait: Optional[Dict[str, object]] = None,
    ) -> None:
        """Send joint angles plus raw landmarks (full-body retargeting) and an
        optional gait command (real-time walking).

        The controller prefers ``keypoints`` (full-body) and falls back to
        ``joint_angles_rad`` when no landmarks are present; ``gait`` drives the
        on-robot walk engine and is ignored by builds that don't support it.
        """
        self._sock.sendto(self._encode(command, keypoints, gait), (self.host, self.port))

    def close(self) -> None:
        self._sock.close()
