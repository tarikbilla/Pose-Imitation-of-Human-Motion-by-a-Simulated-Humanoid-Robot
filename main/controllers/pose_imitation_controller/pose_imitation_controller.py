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
import math
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

# Make the shared library importable regardless of Webots' working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402

# --- Configuration ---------------------------------------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
SOCKET_RCVBUF = 1 << 16

# Legs follow the user as a shallow, symmetric, statically-balanced crouch only
# (no lean/step — those topple a free-standing NAO). The robot stands under
# gravity and squats when you squat; arms + head track as usual. Set False for
# upper-body-only. See nao_retarget._lower_body and PRD NFR-4.
DRIVE_LEGS = True
DRIVE_HEAD = True         # head yaw/pitch follow the human head
SWAP_SIDES = False        # True = mirror-image mapping (robot's left <-> your right)
SMOOTHING_ALPHA = 0.4     # EMA factor for joint targets (0..1, higher = snappier)
VELOCITY_SCALE = 0.5      # fraction of each joint's hardware max velocity
LEG_VELOCITY_FACTOR = 0.5 # extra slow-down on leg joints (eases crouch/sway in)
STALE_AFTER_S = 0.5       # hold pose if no command for this long

# Model-based CoM balance feedback (recovers the depth/balance info the 2D camera
# loses). Forward kinematics + NAO link masses estimate the centre of mass each
# step; the InertialUnit gives the gravity direction; a Fibonacci-spiral search
# nudges the ankles/hips to keep the CoM over the feet. Runs in normal
# controller mode (no Supervisor). See main/libraries/balance.py.
ENABLE_BALANCE = True
INERTIAL_UNIT_NAME = "inertial unit"  # NAO IMU device (gravity/tilt sensing)

# Real-time walking. The human's detected gait (cadence/phase/swing/stop) drives
# an on-robot walk engine (main/libraries/gait.py) that produces balance-stable
# leg motion — the robot replicates the human's WALK, not their raw leg angles.
#   WALK_TIER = "march": Tier A double-support weight-shift march (default,
#       never fully unloads a foot -> stays in the regime the symmetric balance
#       loop keeps upright; reads as marching/walking-in-place, no falls).
#   WALK_TIER = "step":  Tier B single-support stepping (EXPERIMENTAL; only lifts
#       a foot when the CoM model + foot-force sensors confirm it is safe, else
#       it stays double-support). Unproven on a free-standing NAO — opt in only.
#   WALK_TIER = "stand": legs held in the static crouch (walk engine idle).
ENABLE_WALK = True
WALK_TIER = "march"
GAIT_SMOOTHING_ALPHA = 0.7      # snappier than the arm EMA so the gait waveform survives
GAIT_LEG_VELOCITY_FACTOR = 0.85  # raised leg velocity while walking (vs. gentle crouch)
GYRO_NAME = "gyro"
ACCELEROMETER_NAME = "accelerometer"
# Candidate NAO foot force-sensor device names (Tier B only; best-effort, the
# names vary by proto version — any that resolve are summed per foot, the rest
# are ignored and the engine falls back to its model-only stance gate).
FSR_DEVICES = {
    "L": ["LFsr", "LFootFSR", "LFoot/Fsr", "force_sensor_left"],
    "R": ["RFsr", "RFootFSR", "RFoot/Fsr", "force_sensor_right"],
}

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


class PoseImitationController:
    """Webots glue around :class:`NaoPoseDriver`."""

    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        logger.info("Initializing NAO pose controller (timestep: %dms)", self.timestep)

        self.driver = NaoPoseDriver(
            self.robot,
            drive_legs=DRIVE_LEGS,
            drive_head=DRIVE_HEAD,
            swap_sides=SWAP_SIDES,
            smoothing_alpha=SMOOTHING_ALPHA,
            velocity_scale=VELOCITY_SCALE,
            leg_velocity_factor=LEG_VELOCITY_FACTOR,
            stale_after_s=STALE_AFTER_S,
            enable_balance=ENABLE_BALANCE,
            enable_walk=ENABLE_WALK,
            walk_tier=WALK_TIER,
            gait_smoothing_alpha=GAIT_SMOOTHING_ALPHA,
            gait_leg_velocity_factor=GAIT_LEG_VELOCITY_FACTOR,
            logger=logger.info,
        )
        self._init_imu()
        self._init_walk_sensors()
        self._init_socket()

        self.trajectory_log = None
        if ENABLE_TRAJECTORY_LOG:
            self.trajectory_log = JointTrajectoryLogger(
                LOG_DIR, self.driver.logged_joints, logger=logger.info
            )

        self.frame_count = 0
        self._last_log_time = time.time()
        logger.info("Controller initialized; waiting for pose commands...")

    def _init_imu(self) -> None:
        """Enable the InertialUnit so balance can sense the torso's true tilt."""
        self.imu = None
        if not ENABLE_BALANCE:
            return
        imu = self.robot.getDevice(INERTIAL_UNIT_NAME)
        if imu is None:
            logger.warning("InertialUnit '%s' not found; balance runs CoM-only",
                           INERTIAL_UNIT_NAME)
            return
        try:
            imu.enable(self.timestep)
            self.imu = imu
            logger.info("InertialUnit enabled for balance feedback")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable InertialUnit: %s", exc)

    def _torso_tilt(self) -> tuple:
        """(roll, pitch) of the torso in rad from the IMU; (0, 0) if unavailable."""
        if self.imu is None:
            return (0.0, 0.0)
        try:
            roll, pitch, _yaw = self.imu.getRollPitchYaw()
            if math.isnan(roll) or math.isnan(pitch):
                return (0.0, 0.0)
            return (roll, pitch)
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)

    def _init_walk_sensors(self) -> None:
        """Enable the gyro/accelerometer and any foot force sensors used by the
        walk engine. All best-effort and NaN-guarded: Tier A needs none of them,
        and Tier B falls back to the model-only stance gate when they are absent.
        """
        self.gyro = None
        self.fsr: Dict[str, list] = {"L": [], "R": []}
        if not ENABLE_WALK:
            return
        for name in (GYRO_NAME, ACCELEROMETER_NAME):
            dev = self.robot.getDevice(name)
            if dev is not None:
                try:
                    dev.enable(self.timestep)
                except Exception:  # noqa: BLE001
                    continue
                if name == GYRO_NAME:
                    self.gyro = dev
        for side, names in FSR_DEVICES.items():
            for name in names:
                dev = self.robot.getDevice(name)
                if dev is None:
                    continue
                try:
                    dev.enable(self.timestep)
                except Exception:  # noqa: BLE001
                    continue
                self.fsr[side].append(dev)
        n_fsr = len(self.fsr["L"]) + len(self.fsr["R"])
        logger.info("Walk sensors: gyro=%s, foot-force sensors=%d",
                    self.gyro is not None, n_fsr)

    def _read_fsr(self):
        """Per-foot vertical load {"L": fz, "R": fz} from the FSRs, or None.

        Returns None when no force sensors resolved (Tier A doesn't need them;
        Tier B then uses the CoM-model stance gate alone)."""
        if not self.fsr["L"] and not self.fsr["R"]:
            return None
        out = {}
        for side in ("L", "R"):
            total = 0.0
            for dev in self.fsr[side]:
                try:
                    v = dev.getValue()
                except Exception:  # noqa: BLE001
                    continue
                if v is not None and not math.isnan(v):
                    total += abs(float(v))
            out[side] = total
        return out

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
        if self.driver.enable_walk:
            m = self.driver.gait_meta
            logger.info(
                "  walk[%s] amp=%.2f cadence=%.2fHz phase=%.2f single_support=%s",
                WALK_TIER, float(m.get("amp_gain", 0.0)), float(m.get("cadence", 0.0)),
                float(m.get("phase", 0.0)), m.get("single_support", False),
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
                    # Hand the latest gait command to the walk engine (if any).
                    self.driver.set_gait_command(command.get("gait"))
                else:
                    # No fresh frame: if we've gone stale, tell the walk engine to
                    # stop (it ramps amplitude to 0 -> back to the static crouch).
                    if self.driver.check_stale(now):
                        self.driver.set_gait_command(None)

                self.driver.read_feedback()
                torso_rp = self._torso_tilt()
                # Continuous lower-body control every step, regardless of how
                # often pose frames arrive. While walking, the gait engine is the
                # sole leg commander (and folds in balance during double support);
                # otherwise the standalone balance loop keeps the CoM over the feet.
                if self.driver.enable_walk:
                    self.driver.gait_tick(now, torso_rp, fsr=self._read_fsr())
                else:
                    self.driver.balance_tick(torso_rp)
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
