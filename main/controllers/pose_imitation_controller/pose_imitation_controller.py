from __future__ import annotations

import json
import socket
from typing import Dict

from controller import Robot  # type: ignore

UDP_PORT = 8765
MOTOR_NAMES = [
    "LShoulderPitch",
    "RShoulderPitch",
    "LElbowRoll",
    "RElbowRoll",
    "LHipPitch",
    "RHipPitch",
]


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    motors: Dict[str, object] = {}
    for name in MOTOR_NAMES:
        motor = robot.getDevice(name)
        motors[name] = motor

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", UDP_PORT))
    sock.setblocking(False)

    latest = {}

    while robot.step(timestep) != -1:
        try:
            data, _ = sock.recvfrom(65535)
            payload = json.loads(data.decode("utf-8"))
            latest = payload.get("joint_angles_rad", {})
        except BlockingIOError:
            pass

        for joint_name, angle in latest.items():
            motor = motors.get(joint_name)
            if motor is not None:
                motor.setPosition(float(angle))


if __name__ == "__main__":
    main()
