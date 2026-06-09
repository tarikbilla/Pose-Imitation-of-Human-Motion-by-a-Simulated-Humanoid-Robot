from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from src.type_defs import JointCommand


@dataclass
class WebotsBridge:
    host: str
    port: int

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_joint_command(self, command: JointCommand) -> None:
        payload = {
            "timestamp_s": command.timestamp_s,
            "frame_index": command.frame_index,
            "joint_angles_rad": command.joint_angles_rad,
        }
        message = json.dumps(payload).encode("utf-8")
        self._sock.sendto(message, (self.host, self.port))

    def close(self) -> None:
        self._sock.close()
