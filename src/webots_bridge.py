from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from src.type_defs import JointCommand, Keypoint

# Landmarks streamed to the Webots controller for full-body retargeting.
# A curated subset keeps the UDP packet small (low latency, NFR-1) while still
# covering every joint the controller maps: arms, head and legs.
#
# MeTRAbs' coco_19 skeleton has no separate heel/toe points (unlike MediaPipe's
# 33-landmark set) -- the controller's ground-line/lift detection falls back to
# ankle-only (see main/libraries/nao_retarget.py's LowerBodyRetargeter, which
# already degrades gracefully when heel landmarks are absent). ``neck`` and
# ``pelvis`` are included because the Webots-side retargeter uses them to build
# a per-frame torso-local reference frame from real 3D geometry.
KEYPOINTS_TO_STREAM = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "neck", "pelvis",
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
        keypoints: dict[str, Keypoint] | None = None,
        gait: dict[str, object] | None = None,
        action: dict[str, object] | None = None,
    ) -> bytes:
        payload: dict[str, object] = {
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
        if action is not None:
            # Which lower-body CLIP the human is asking for, plus the evidence
            # behind it (see src/perception/action_cues.py). Separate from
            # "gait" on purpose: gait answers "is there a marching rhythm" from
            # a periodic proxy, this answers "what is this person doing" from
            # measured travel. Additive and optional, like gait.
            payload["action"] = action
        return json.dumps(payload).encode("utf-8")

    def send_joint_command(self, command: JointCommand) -> None:
        """Backward-compatible: send joint angles only."""
        self._sock.sendto(self._encode(command), (self.host, self.port))

    def send_pose_frame(
        self,
        command: JointCommand,
        keypoints: dict[str, Keypoint] | None = None,
        gait: dict[str, object] | None = None,
        action: dict[str, object] | None = None,
    ) -> None:
        """Send joint angles plus raw landmarks (full-body retargeting) and an
        optional gait command (real-time walking).

        The controller prefers ``keypoints`` (full-body) and falls back to
        ``joint_angles_rad`` when no landmarks are present; ``gait`` drives the
        on-robot walk engine and is ignored by builds that don't support it.
        """
        self._sock.sendto(self._encode(command, keypoints, gait, action),
                          (self.host, self.port))

    def close(self) -> None:
        self._sock.close()
