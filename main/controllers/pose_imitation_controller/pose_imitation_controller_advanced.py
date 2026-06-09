"""
Advanced NAO pose imitation controller.

Same UDP protocol and :class:`NaoPoseDriver` core as the standard controller,
but tuned for experimentation:

* Lower smoothing alpha and velocity scale for extra-smooth motion.
* Optional leg driving (hip pitch) for lower-body experiments — OFF by default
  because NAO has no balance controller yet and will fall (PRD NFR-4 / risks).
* Verbose per-joint diagnostics: tracking error, stuck-motor detection, and the
  measured-vs-commanded position table.

Use the standard ``pose_imitation_controller`` for normal demos; use this one
when you need to inspect motor behaviour or trial lower-body imitation.
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
    from controller import Robot  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402

# --- Configuration ---------------------------------------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
SOCKET_RCVBUF = 1 << 16

DRIVE_LEGS = False        # set True to experiment with full leg imitation
DRIVE_HEAD = True
SWAP_SIDES = False        # True = mirror-image mapping
SMOOTHING_ALPHA = 0.3     # smoother (laggier) than the standard controller
VELOCITY_SCALE = 0.4
STALE_AFTER_S = 0.5
DIAGNOSTICS_EVERY = 200   # frames

ENABLE_TRAJECTORY_LOG = True
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AdvancedPoseController")


class AdvancedPoseController:
    """Diagnostics-focused wrapper around :class:`NaoPoseDriver`."""

    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        logger.info("Initializing advanced NAO controller (timestep: %dms)", self.timestep)

        self.driver = NaoPoseDriver(
            self.robot,
            drive_legs=DRIVE_LEGS,
            drive_head=DRIVE_HEAD,
            swap_sides=SWAP_SIDES,
            smoothing_alpha=SMOOTHING_ALPHA,
            velocity_scale=VELOCITY_SCALE,
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
        self._last_diag_time = time.time()
        logger.info("Advanced controller initialized")

    def _init_socket(self) -> None:
        logger.info("Opening UDP socket on %s:%d ...", UDP_HOST, UDP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    def _drain_latest_command(self) -> Optional[Dict]:
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

    def _log_diagnostics(self) -> None:
        if self.frame_count % DIAGNOSTICS_EVERY != 0:
            return
        elapsed = time.time() - self._last_diag_time
        fps = DIAGNOSTICS_EVERY / elapsed if elapsed > 0 else 0.0
        stats = self.driver.stats
        logger.info(
            "Frame %d | sim %.1f Hz | %s | applied %d joints",
            self.frame_count, fps,
            "STALE" if stats.stale else "tracking",
            stats.joints_last_applied,
        )
        for name in sorted(self.driver.motors):
            if name not in self.driver.commanded:
                continue
            cmd = self.driver.commanded[name]
            meas = self.driver.measured.get(name, float("nan"))
            err = self.driver.health.average_error(name)
            flag = "  <-- STUCK" if self.driver.health.is_stuck(name) else ""
            logger.debug(
                "  %-15s cmd %+.3f  meas %+.3f  avgerr %.3f%s",
                name, cmd, meas, err, flag,
            )
        for name in self.driver.stuck_motors():
            logger.warning("Motor '%s' may be stuck", name)
        self._last_diag_time = time.time()

    def run(self) -> None:
        logger.info("Starting advanced control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                now = self.robot.getTime()
                command = self._drain_latest_command()
                if command is not None:
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
                self._log_diagnostics()
                self.frame_count += 1
        except KeyboardInterrupt:
            logger.info("Interrupt received")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in control loop: %s", exc)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        logger.info("Shutting down...")
        try:
            self.driver.stop()
        finally:
            if self.trajectory_log is not None:
                self.trajectory_log.close()
            if hasattr(self, "sock"):
                self.sock.close()
            logger.info("Shutdown complete (%d frames)", self.frame_count)


def main() -> None:
    try:
        AdvancedPoseController().run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
