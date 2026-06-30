"""End-to-end pose imitation pipeline."""
from __future__ import annotations

import logging
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional

import cv2

from src.perception.gait_cues import GaitCueExtractor
from src.perception.pose_estimator import PoseEstimator
from src.perception.video_input import VideoSource
from src.perception.visualizer import SkeletonOverlay
from src.retargeting.mapper import RetargetingMapper, default_joint_limits
from src.type_defs import JointCommand
from src.utils.config import Config
from src.utils.filtering import ExponentialSmoother
from src.utils.fps import AdaptiveFPSController
from src.utils.logger import CsvRunLogger
from src.webots_bridge import WebotsBridge

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    config: Config
    show_window: bool = True
    enable_webots: bool = True
    max_frames: int = 0  # 0 = unlimited
    source_override: Optional[str] = None


@dataclass
class PoseImitationPipeline:
    options: PipelineOptions

    _stop_requested: bool = field(default=False, init=False)

    def request_stop(self, *_: object) -> None:
        logger.info("Stop requested; shutting down gracefully.")
        self._stop_requested = True

    def _resolve_source(self) -> int | str:
        if self.options.source_override is not None:
            value = self.options.source_override
        else:
            value = self.options.config.get("input.source", 0)
        return int(value) if str(value).isdigit() else value

    def run(self) -> int:
        cfg = self.options.config
        source = self._resolve_source()

        fps_controller = AdaptiveFPSController(
            min_fps=float(cfg.get("runtime.min_fps", 25)),
            max_fps=float(cfg.get("runtime.max_fps", 100)),
            latency_budget_ms=float(cfg.get("runtime.latency_budget_ms", 150)),
            step_fps=float(cfg.get("runtime.fps_step", 5)),
            _current_fps=float(cfg.get("runtime.initial_fps", 30)),
        )

        capture = VideoSource(
            source=source,
            width=int(cfg.get("input.width", 1280)),
            height=int(cfg.get("input.height", 720)),
            preferred_fps=fps_controller.current_fps,
        )
        estimator = PoseEstimator(
            use_mediapipe=bool(cfg.get("pose.use_mediapipe", True)),
            model_complexity=int(cfg.get("pose.model_complexity", 1)),
            min_detection_confidence=float(cfg.get("pose.min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("pose.min_tracking_confidence", 0.5)),
            allow_synthetic_fallback=bool(cfg.get("pose.allow_synthetic_fallback", False)),
        )
        if estimator.is_real:
            logger.info(
                "Pose estimator: MediaPipe (real human tracking active). "
                "Detection threshold: %.2f, Tracking threshold: %.2f",
                estimator.min_detection_confidence, estimator.min_tracking_confidence
            )
        else:
            logger.error("Pose estimator: SYNTHETIC fallback (will NOT follow the human). This is a fallback mode.")

        flip_horizontal = bool(cfg.get("input.flip_horizontal", True))
        mapper = RetargetingMapper(joint_limits=default_joint_limits())
        smoother = ExponentialSmoother(alpha=float(cfg.get("retargeting.smoothing_alpha", 0.35)))
        overlay = SkeletonOverlay(show=self.options.show_window)

        # Real-time walking: distil the human's gait into a compact command the
        # on-robot walk engine executes (the robot replicates the walk, not raw
        # leg angles — monocular depth is unreliable). Computed every frame so
        # cadence/phase stay warm; streamed only when walking is enabled.
        walk_enabled = bool(cfg.get("walk.enabled", True))
        gait_extractor = GaitCueExtractor(
            window_s=float(cfg.get("walk.cue_window_s", 1.3)),
            amp_start=float(cfg.get("walk.amp_start", 0.08)),
            amp_stop=float(cfg.get("walk.amp_stop", 0.05)),
            conf_min=float(cfg.get("walk.cue_conf_min", 0.6)),
        )

        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        log_dir = Path(cfg.get("logging.output_dir", "logs")) / run_name
        run_logger = CsvRunLogger(log_dir)
        logger.info("Logging run to %s", log_dir)

        bridge: Optional[WebotsBridge] = None
        if self.options.enable_webots and bool(cfg.get("webots_bridge.enabled", True)):
            bridge = WebotsBridge(
                host=str(cfg.get("webots_bridge.host", "127.0.0.1")),
                port=int(cfg.get("webots_bridge.port", 8765)),
            )
            logger.info("Webots bridge sending to %s:%d", bridge.host, bridge.port)
        else:
            logger.info("Webots bridge disabled.")

        # Register signal handlers for graceful shutdown.
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        latency_window: Deque[float] = deque(maxlen=30)
        max_frames = self.options.max_frames or int(cfg.get("runtime.max_frames", 0))
        exit_code = 0

        try:
            for frame in capture.read_loop(target_period_s=fps_controller.target_period_s):
                if self._stop_requested:
                    break

                start = time.perf_counter()
                image = cv2.flip(frame.image_bgr, 1) if flip_horizontal else frame.image_bgr
                pose = estimator.estimate(image, frame.timestamp_s, frame.frame_index)
                run_logger.log_pose(pose)

                gait_cmd = gait_extractor.update(pose)

                command = mapper.map_pose(pose)
                if command.joint_angles_rad:
                    smoothed = smoother.update(command.joint_angles_rad)
                    command = JointCommand(
                        timestamp_s=command.timestamp_s,
                        joint_angles_rad=smoothed,
                        frame_index=command.frame_index,
                    )
                    run_logger.log_joint_command(command)
                    if bridge is not None:
                        # Stream joint angles + raw landmarks (full-body
                        # retargeting) and the gait command (real-time walking).
                        bridge.send_pose_frame(
                            command, pose.keypoints,
                            gait=gait_cmd.as_dict() if walk_enabled else None,
                        )

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latency_window.append(elapsed_ms)
                avg_latency = sum(latency_window) / len(latency_window)
                fps_controller.update(measured_latency_ms=elapsed_ms)
                effective_fps = 1000.0 / max(avg_latency, 1e-3)

                if self.options.show_window:
                    n_joints = len(command.joint_angles_rad) if command else 0
                    visible_landmarks = sum(1 for kp in pose.keypoints.values() if kp.visibility > 0.3)
                    hud = [
                        f"Target FPS: {fps_controller.current_fps:5.1f}",
                        f"Joints: {n_joints}   Visible: {visible_landmarks}/33",
                        "Source: MediaPipe" if estimator.is_real else "Source: SYNTHETIC",
                        f"Gait: {gait_cmd.state:5s} {gait_cmd.cadence_hz:.2f}Hz "
                        f"conf {gait_cmd.conf:.2f}",
                    ]
                    canvas = overlay.draw(
                        image, pose,
                        fps=effective_fps,
                        latency_ms=avg_latency,
                        extra_hud=hud,
                    )
                    if not overlay.show_frame(canvas):
                        logger.info("User requested quit (ESC/q).")
                        break

                if max_frames > 0 and frame.frame_index + 1 >= max_frames:
                    logger.info("Reached max_frames=%d; stopping.", max_frames)
                    break
        except Exception:  # noqa: BLE001
            logger.exception("Pipeline failure")
            exit_code = 1
        finally:
            capture.release()
            estimator.close()
            run_logger.close()
            overlay.close()
            if bridge is not None:
                bridge.close()
            logger.info("Pipeline stopped.")

        return exit_code
