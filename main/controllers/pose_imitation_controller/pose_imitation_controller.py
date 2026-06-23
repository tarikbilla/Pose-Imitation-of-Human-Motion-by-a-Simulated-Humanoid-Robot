"""
Real-time Webots NAO pose imitation controller.

Receives generic human-pose joint commands via UDP from the Python pipeline
(`src/`) and drives the simulated NAO humanoid in real time.

Responsibilities (the NAO-specific mapping, smoothing, joint limiting and
standing stabilization) live in ``pose_control_utils.NaoPoseDriver``; this file
is just the Webots glue: open the socket, step the simulation, hand each frame
to the driver, and shut down cleanly.

Protocol (UDP, port 8765, JSON):
    {
      "timestamp_s": 1234567890.123,
      "frame_index": 45,
      "joint_angles_rad": {"LShoulderPitch": 0.5, "RElbowRoll": -1.1, ...}
    }
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from typing import Dict, Optional

try:
    from controller import Supervisor  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

# Make the shared library importable regardless of Webots' working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402

# --- Configuration ---------------------------------------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
SOCKET_RCVBUF = 1 << 16

# Full-body imitation. Legs follow the user as a statically-balanced crouch +
# lateral sway (symmetric, ankle-compensated) so the robot tracks your body
# without a dynamic balance controller and without falling (see PRD NFR-4 and
# nao_retarget._lower_body). Set False to fall back to upper-body-only.
DRIVE_LEGS = True
DRIVE_HEAD = True         # head yaw/pitch follow the human head
SWAP_SIDES = False        # True = mirror-image mapping (robot's left <-> your right)
SMOOTHING_ALPHA = 0.4     # EMA factor for joint targets (0..1, higher = snappier)
VELOCITY_SCALE = 0.5      # fraction of each joint's hardware max velocity
LEG_VELOCITY_FACTOR = 0.5 # extra slow-down on leg joints (eases crouch/sway in)
STALE_AFTER_S = 0.5       # hold pose if no command for this long

# Base stabilization (the "don't fall over" fix). NAO is a free-standing biped;
# moving the legs/body shifts the centre of mass off its tiny feet and gravity
# topples it. With no dynamic balance controller available, we instead remove
# the toppling degrees of freedom: the Supervisor pins the pelvis upright and in
# place horizontally while leaving its *height* free, so the robot can still
# crouch (body lowers under gravity, feet stay on the floor) and animate every
# joint, but it cannot tip over. Requires `supervisor TRUE` on the Nao node.
# Set False to run as a plain free-standing robot (will fall when legs move).
KEEP_UPRIGHT = True

# Log commanded vs. achieved joint angles to <project>/logs/ for offline
# imitation-fidelity metrics (PRD FR-7 / US-3).
ENABLE_TRAJECTORY_LOG = True
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PoseController")


class BaseStabilizer:
    """Keeps the robot from toppling by constraining its floating base.

    NAO has no dynamic balance controller here, so instead of *recovering* from
    a fall we *prevent* one by removing the unstable degrees of freedom of the
    pelvis (the controller's own node): each step we re-pin its horizontal
    position and upright orientation and zero the horizontal/angular velocity.
    The vertical axis is left free, so the body still drops when the legs crouch
    and rests on the feet — the robot can imitate the user's body and legs but
    physically cannot tip over.

    Needs ``supervisor TRUE`` on the Nao node. If the controller is not a
    supervisor (``getSelf()`` is None) the stabilizer disables itself and the
    robot runs as a plain — fall-prone — free-standing biped.
    """

    def __init__(self, robot, logger) -> None:
        self.log = logger
        self.enabled = False
        node = robot.getSelf() if hasattr(robot, "getSelf") else None
        if node is None:
            self.log("Base stabilizer OFF (controller is not a Supervisor; "
                     "set 'supervisor TRUE' on the Nao node to prevent falling)")
            return
        try:
            self.node = node
            self.translation = node.getField("translation")
            self.rotation = node.getField("rotation")
            self._xy = self.translation.getSFVec3f()[:2]   # lock-in spot on the floor
            self._upright = list(self.rotation.getSFRotation())  # initial upright pose
            self.enabled = True
            self.log("Base stabilizer ON (pelvis pinned upright; height free for crouch)")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Base stabilizer OFF ({exc})")
            self.enabled = False

    def step(self) -> None:
        if not self.enabled:
            return
        try:
            pos = self.translation.getSFVec3f()
            # Lock horizontal position + upright orientation; keep current height.
            self.translation.setSFVec3f([self._xy[0], self._xy[1], pos[2]])
            self.rotation.setSFRotation(self._upright)
            # Bleed off any toppling/sliding momentum, preserve vertical motion.
            vx, vy, vz, wx, wy, wz = self.node.getVelocity()
            self.node.setVelocity([0.0, 0.0, vz, 0.0, 0.0, 0.0])
        except Exception as exc:  # noqa: BLE001
            self.log(f"Base stabilizer disabled mid-run ({exc})")
            self.enabled = False


class PoseImitationController:
    """Webots glue around :class:`NaoPoseDriver`."""

    def __init__(self) -> None:
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        logger.info("Initializing NAO pose controller (timestep: %dms)", self.timestep)

        self.stabilizer = BaseStabilizer(self.robot, logger.info) if KEEP_UPRIGHT else None

        self.driver = NaoPoseDriver(
            self.robot,
            drive_legs=DRIVE_LEGS,
            drive_head=DRIVE_HEAD,
            swap_sides=SWAP_SIDES,
            smoothing_alpha=SMOOTHING_ALPHA,
            velocity_scale=VELOCITY_SCALE,
            leg_velocity_factor=LEG_VELOCITY_FACTOR,
            stale_after_s=STALE_AFTER_S,
            logger=logger.info,
        )
        self._init_socket()

        self.trajectory_log = None
        if ENABLE_TRAJECTORY_LOG:
            self.trajectory_log = JointTrajectoryLogger(
                LOG_DIR, self.driver.logged_joints, logger=logger.info
            )

        self.frame_count = 0
        self._last_log_time = time.time()
        logger.info("Controller initialized; waiting for pose commands...")

    def _init_socket(self) -> None:
        logger.info("Opening UDP socket on %s:%d ...", UDP_HOST, UDP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    def _drain_latest_command(self) -> Optional[Dict]:
        """Return the most recent pose command, discarding any backlog.

        UDP can queue several frames between simulation steps. We only care
        about the freshest pose, so we drain the buffer and keep the last one
        (keeps end-to-end latency low — PRD NFR-1).
        """
        latest: Optional[Dict] = None
        while True:
            try:
                data, _ = self.sock.recvfrom(SOCKET_RCVBUF)
            except (BlockingIOError, OSError):
                break
            try:
                latest = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue
        return latest

    def _log_status(self) -> None:
        if self.frame_count % 100 != 0:
            return
        elapsed = time.time() - self._last_log_time
        fps = 100 / elapsed if elapsed > 0 else 0.0
        stats = self.driver.stats
        state = "STALE (holding)" if stats.stale else "tracking"
        logger.info(
            "Frame %d | sim %.1f Hz | %s | last frame applied %d joints",
            self.frame_count, fps, state, stats.joints_last_applied,
        )
        for name in self.driver.stuck_motors():
            logger.warning(
                "Motor '%s' may be stuck (avg err %.3f rad)",
                name, self.driver.health.average_error(name),
            )
        self._last_log_time = time.time()

    def run(self) -> None:
        logger.info("Starting control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                if self.stabilizer is not None:
                    self.stabilizer.step()
                now = self.robot.getTime()
                command = self._drain_latest_command()
                if command is not None:
                    # Prefer full-body retargeting from raw landmarks; fall back
                    # to pre-computed joint angles if only those were sent.
                    keypoints = command.get("keypoints")
                    if keypoints:
                        self.driver.update_from_keypoints(keypoints, now_s=now)
                    else:
                        angles = command.get("joint_angles_rad", {})
                        if angles:
                            self.driver.update(angles, now_s=now)
                else:
                    self.driver.check_stale(now)

                self.driver.read_feedback()
                if self.trajectory_log is not None:
                    self.trajectory_log.record(
                        now, self.frame_count, self.driver.commanded, self.driver.measured
                    )
                self._log_status()
                self.frame_count += 1
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in control loop: %s", exc)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        try:
            self.driver.stop()
        finally:
            if self.trajectory_log is not None:
                self.trajectory_log.close()
            if hasattr(self, "sock"):
                self.sock.close()
            logger.info("Controller stopped after %d frames", self.frame_count)


def main() -> None:
    try:
        PoseImitationController().run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
