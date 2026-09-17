"""End-to-end pose imitation pipeline."""
from __future__ import annotations

import logging
import math
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.perception import metrabs_model
from src.perception.action_cues import ActionCue
from src.perception.gait_cues import GaitCueExtractor
from src.perception.pose_estimator import PoseEstimator
from src.perception.video_input import VideoSource
from src.perception.visualizer import SkeletonOverlay
from src.retargeting.mapper import RetargetingMapper, default_joint_limits
from src.type_defs import JointCommand, Keypoint, PoseFrame
from src.utils.config import Config
from src.utils.filtering import ExponentialSmoother, OneEuroFilter
from src.utils.fps import AdaptiveFPSController
from src.utils.logger import CsvRunLogger
from src.webots_bridge import WebotsBridge

logger = logging.getLogger(__name__)


class _KeypointSmoother:
    """Speed-adaptive smoothing of raw 3D keypoints.

    MeTRAbs, unlike MediaPipe (``smooth_landmarks=True``), does not smooth
    across frames itself -- each frame's pose is estimated independently, so
    something has to damp the jitter. A fixed-alpha EMA charged a flat
    ``(1 - alpha) / alpha`` samples of delay for that whether the subject was
    moving or not: at this pipeline's measured 14.3 FPS, ``alpha = 0.5`` cost
    70 ms, nearly half of ``runtime.latency_budget_ms``, and the robot visibly
    lagged the human. ``OneEuroFilter`` spends that delay only while the
    subject is still, and gets out of the way when they move -- see
    ``src/utils/filtering.py``.

    ``visibility`` passes through unsmoothed: it reflects the current frame's
    detection quality, not a lagging physical quantity.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ) -> None:
        def _f() -> OneEuroFilter:
            return OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

        self._x, self._y, self._z = _f(), _f(), _f()
        self._prev_t: float | None = None

    def update(self, keypoints: dict, timestamp_s: float) -> dict:
        if not keypoints:
            return keypoints
        dt = 0.0 if self._prev_t is None else timestamp_s - self._prev_t
        self._prev_t = timestamp_s
        xs = self._x.update({n: kp.x for n, kp in keypoints.items()}, dt)
        ys = self._y.update({n: kp.y for n, kp in keypoints.items()}, dt)
        zs = self._z.update({n: kp.z for n, kp in keypoints.items()}, dt)
        return {
            n: Keypoint(x=xs[n], y=ys[n], z=zs[n], visibility=kp.visibility)
            for n, kp in keypoints.items()
        }


@dataclass
class PipelineOptions:
    config: Config
    show_window: bool = True
    enable_webots: bool = True
    max_frames: int = 0  # 0 = unlimited
    source_override: str | None = None


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
            use_metrabs=bool(cfg.get("pose.use_metrabs", True)),
            model_url=str(cfg.get("pose.model_url", metrabs_model.DEFAULT_MODEL_URL)),
            skeleton=str(cfg.get("pose.skeleton", metrabs_model.DEFAULT_SKELETON)),
            default_fov_degrees=float(cfg.get("pose.default_fov_degrees", 55.0)),
            detector_threshold=float(cfg.get("pose.detector_threshold", 0.3)),
            num_aug=int(cfg.get("pose.num_aug", 1)),
            max_detections=int(cfg.get("pose.max_detections", 1)),
            detect_interval=int(cfg.get("pose.detect_interval", 5)),
            box_padding=float(cfg.get("pose.box_padding", 0.18)),
            max_joint_jump_mm=float(cfg.get("pose.max_joint_jump_mm", 300.0)),
            require_gpu=bool(cfg.get("pose.require_gpu", True)),
            allow_synthetic_fallback=bool(cfg.get("pose.allow_synthetic_fallback", False)),
        )
        keypoint_smoother = _KeypointSmoother(
            min_cutoff=float(cfg.get("pose.smoothing.min_cutoff", 1.0)),
            beta=float(cfg.get("pose.smoothing.beta", 0.005)),
            d_cutoff=float(cfg.get("pose.smoothing.d_cutoff", 1.0)),
        )
        if estimator.is_real:
            logger.info(
                "Pose estimator: MeTRAbs (real human tracking active). "
                "Skeleton: %s, detector threshold: %.2f",
                estimator.skeleton, estimator.detector_threshold,
            )
        else:
            logger.error(
                "Pose estimator: SYNTHETIC fallback (will NOT follow the human). "
                "This is a fallback mode."
            )

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
            cross_window_s=float(cfg.get("walk.cross_window_s", 3.0)),
            stop_window_s=float(cfg.get("walk.stop_window_s", 0.8)),
            amp_start=float(cfg.get("walk.amp_start", 0.08)),
            amp_stop=float(cfg.get("walk.amp_stop", 0.05)),
            conf_min=float(cfg.get("walk.cue_conf_min", 0.6)),
            conf_grace_frames=int(cfg.get("walk.cue_conf_grace_frames", 2)),
        )

        # Which lower-body CLIP the human is asking for. Separate from the gait
        # cue above, which answers a narrower question (is there a marching
        # rhythm) and answers it from a proxy signal. See action_cues.
        action_cue = ActionCue()

        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        log_dir = Path(cfg.get("logging.output_dir", "logs")) / run_name
        run_logger = CsvRunLogger(log_dir)
        logger.info("Logging run to %s", log_dir)

        bridge: WebotsBridge | None = None
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

        latency_window: deque[float] = deque(maxlen=30)
        max_frames = self.options.max_frames or int(cfg.get("runtime.max_frames", 0))
        exit_code = 0

        try:
            # Pass the CALLABLE, not the value: ``target_period_s`` is a
            # property, so handing it over directly freezes the period at
            # whatever it was when the generator was built and the adaptive
            # controller below can never actually change the capture rate
            # (FR-1b). See VideoSource.read_loop.
            for frame in capture.read_loop(
                target_period_s=lambda: fps_controller.target_period_s
            ):
                if self._stop_requested:
                    break

                start = time.perf_counter()
                image = cv2.flip(frame.image_bgr, 1) if flip_horizontal else frame.image_bgr
                pose = estimator.estimate(image, frame.timestamp_s, frame.frame_index)
                if pose.keypoints:
                    pose = PoseFrame(
                        timestamp_s=pose.timestamp_s,
                        keypoints=keypoint_smoother.update(
                            pose.keypoints, pose.timestamp_s
                        ),
                        frame_index=pose.frame_index,
                    )
                run_logger.log_pose(pose)

                gait_cmd = gait_extractor.update(pose)
                action_cmd = action_cue.update(pose)

                command = mapper.map_pose(pose)
                if command.joint_angles_rad:
                    smoothed = smoother.update(command.joint_angles_rad)
                    command = JointCommand(
                        timestamp_s=command.timestamp_s,
                        joint_angles_rad=smoothed,
                        frame_index=command.frame_index,
                    )
                    run_logger.log_joint_command(command)
                # The landmarks are the PRIMARY channel -- the controller does its
                # own full-body retargeting from them and only falls back to these
                # joint angles. So the frame goes out whenever a human was
                # detected: gating the send on the legacy mapper succeeding meant a
                # partial pose silently froze the robot mid-motion.
                if bridge is not None and (pose.keypoints or command.joint_angles_rad):
                    bridge.send_pose_frame(
                        command, pose.keypoints,
                        gait=gait_cmd.as_dict() if walk_enabled else None,
                        action=action_cmd.as_dict() if walk_enabled else None,
                    )

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latency_window.append(elapsed_ms)
                avg_latency = sum(latency_window) / len(latency_window)
                fps_controller.update(measured_latency_ms=elapsed_ms)
                effective_fps = 1000.0 / max(avg_latency, 1e-3)

                if self.options.show_window:
                    n_joints = len(command.joint_angles_rad) if command else 0
                    hud = [
                        f"Target FPS: {fps_controller.current_fps:5.1f}",
                        # NOT the landmark count -- _draw_hud already prints
                        # that, and two lines saying 25/25 is one line of noise.
                        # This is the count of RETARGETED joints, which is a
                        # different question: landmarks are what was seen, this
                        # is what the robot was actually told to do.
                        f"Joints commanded: {n_joints}",
                        "Source: MeTRAbs" if estimator.is_real else "Source: SYNTHETIC",
                        f"Gait: {gait_cmd.state:5s} {gait_cmd.cadence_hz:.2f}Hz "
                        f"conf {gait_cmd.conf:.2f}",
                        f"Body yaw: {math.degrees(gait_cmd.body_yaw_rad):+6.1f} deg "
                        f"conf {gait_cmd.yaw_conf:.2f}",
                        f"Action: {action_cmd.action:<14s} conf {action_cmd.confidence:.2f}",
                        f"  {action_cmd.reason[:46]}",
                    ]
                    h, w = image.shape[:2]
                    canvas = overlay.draw(
                        image, pose,
                        intrinsics=estimator.intrinsics_for(w, h),
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
