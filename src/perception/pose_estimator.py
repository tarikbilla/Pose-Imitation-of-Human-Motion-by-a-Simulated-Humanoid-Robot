"""MeTRAbs-based human pose estimator.

Design goals (unchanged from the MediaPipe-era version this replaces):
- Loud, explicit failures (no silent fallback) when MeTRAbs cannot load or no
  GPU is visible.
- Optional synthetic fallback only when explicitly enabled in config.
- Returns a canonical `PoseFrame` per frame, with a visibility PROXY per
  landmark (see below -- MeTRAbs has no true per-joint confidence).

What changed vs. MediaPipe
---------------------------
MediaPipe returned normalized [0,1] image coordinates plus a weak, unreliable
depth channel. MeTRAbs returns absolute METRIC 3D coordinates in MILLIMETERS,
in the camera's coordinate frame (x right, y down, z forward/away from the
camera) -- see ``metrabs_model.py`` and the upstream docs/API.md. Downstream
code (retargeting, gait cues) now works in real 3D instead of reconstructing
it from a 2D projection.

MeTRAbs also has no per-joint visibility/confidence output (only one
detection-box confidence per person). ``visibility`` here is therefore a
PROXY: 1.0 for a joint whose 2D projection lands well inside the detected
person's box and the image frame, decaying to 0.0 near the image edge or
outside the box, and 0.0 everywhere when no person was detected at all. It is
not a measure of true occlusion -- flagged explicitly so downstream
`visibility >= threshold` gating (nao_retarget.py, gait_cues.py, mapper.py) is
understood to be an approximation, not truth.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.perception import metrabs_model
from src.perception.landmarks import (
    OPTIONAL_LANDMARKS,
    REQUIRED_LANDMARKS,
    build_raw_to_canonical_map,
)
from src.type_defs import Keypoint, PoseFrame

logger = logging.getLogger(__name__)

# How close (in pixels) to the image/box edge a joint's 2D projection must be
# before its visibility proxy starts decaying to 0.
_EDGE_MARGIN_FRAC = 0.06


class PoseEstimatorError(RuntimeError):
    """Raised when MeTRAbs cannot be initialised and fallback is disabled."""


@dataclass
class PoseEstimator:
    """Wraps the MeTRAbs TF-Hub model with GPU checking and robust logging."""

    use_metrabs: bool = True
    model_url: str = metrabs_model.DEFAULT_MODEL_URL
    skeleton: str = metrabs_model.DEFAULT_SKELETON
    default_fov_degrees: float = 55.0
    detector_threshold: float = 0.3
    num_aug: int = 1
    max_detections: int = 1
    require_gpu: bool = True
    allow_synthetic_fallback: bool = False
    # Frames between full person-detections. MeTRAbs' detect_poses runs YOLOv4
    # AND the pose network; measured on this project's target machine that is
    # 75.9 ms/frame, of which the detector alone is 39.7 ms -- 52% of the budget,
    # spent re-finding a person who has moved a few pixels. estimate_poses with a
    # supplied box costs 36.2 ms and returns bit-identical joints (max difference
    # 0.00 mm on a held pose). Measured end-to-end on a live moving subject
    # (90 frames, this camera, RTX 3090 Ti):
    #
    #   interval  FPS   median drift  p95 drift   phantom poses
    #      1      12.1        -            -            -
    #      2      16.0      6.8 mm     120 mm          0
    #      3      17.7      6.8 mm     330 mm          0
    #      5      17.6      232 mm     536 mm          0
    #
    # 2 is the honest choice: +32% throughput for millimetres of error. Past 3
    # the drift grows fast and the speed does NOT -- the continuity gate starts
    # rejecting the tracked pose, so those frames pay for both inference passes.
    # 1 restores detect-every-frame.
    detect_interval: int = 2
    # How much to pad the tracked box, as a fraction of its size, so a person
    # moving between detections stays inside it.
    box_padding: float = 0.18
    # Largest median per-joint jump (mm) between consecutive frames that a pose
    # from the tracked box may show before it is disbelieved and re-detected.
    # 300 mm at 12-25 FPS is several metres per second -- far beyond a human, and
    # the signature of the box having drifted onto something that is not one.
    max_joint_jump_mm: float = 300.0

    _model: object = field(default=None, init=False, repr=False)
    _tf: object = field(default=None, init=False, repr=False)
    _raw_names: list[str] = field(default_factory=list, init=False, repr=False)
    _raw_to_canonical: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _intrinsics_cache: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    # Box carried between detections, as [left, top, width, height].
    _tracked_box: object = field(default=None, init=False, repr=False)
    # Joint-span (w, h) observed on the frame the tracked box was DETECTED on,
    # so later frames can tell how much the subject's apparent size has changed.
    _ref_extent: object = field(default=None, init=False, repr=False)
    _frames_since_detect: int = field(default=0, init=False, repr=False)
    _prev_poses3d: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = None

        if not self.use_metrabs:
            self._warn_about_fallback("pose.use_metrabs is False")
            return

        try:
            if self.require_gpu:
                metrabs_model.require_gpu()
            self._tf = metrabs_model.require_tensorflow()
            self._model = metrabs_model.load_model(self.model_url)
            info = metrabs_model.skeleton_info(self.model_url, self.skeleton)
        except metrabs_model.MetrabsUnavailableError as exc:
            if self.allow_synthetic_fallback:
                logger.error(str(exc))
                self._warn_about_fallback("MeTRAbs unavailable")
                self._model = None
                return
            raise PoseEstimatorError(str(exc)) from exc

        self._raw_names = list(info.names)
        self._raw_to_canonical = build_raw_to_canonical_map(self._raw_names)
        matched_canonical = set(self._raw_to_canonical.values())
        # The hand markers exist only in the 122-joint superset, so their
        # absence is a capability report, not a misconfiguration: coco_19 is
        # still a perfectly good body skeleton and the robot's hand joints
        # simply go untracked, exactly as they did before hands were added.
        absent_hands = [n for n in OPTIONAL_LANDMARKS if n not in matched_canonical]
        if absent_hands:
            logger.warning(
                "Skeleton '%s' has no hand landmarks (%s), so NAO's ElbowYaw, "
                "WristYaw and finger joints will not be driven. Set "
                "pose.skeleton to \"\" (the 122-joint superset) for hand "
                "tracking -- it costs no extra inference time.",
                self.skeleton, ", ".join(sorted(absent_hands)),
            )
        missing = [n for n in REQUIRED_LANDMARKS if n not in matched_canonical]
        if missing:
            msg = (
                f"MeTRAbs skeleton '{self.skeleton}' joint names did not match "
                f"{len(missing)} expected canonical landmark(s): {missing}. "
                f"Raw model joint names were: {self._raw_names}. "
                "src/perception/landmarks.py's CANONICAL_TO_RAW_ALIASES was "
                "written without access to a live model and needs correcting -- "
                "run scripts/inspect_metrabs_skeleton.py and update the alias "
                "table to match the raw names printed above."
            )
            if self.allow_synthetic_fallback:
                logger.error(msg)
                self._warn_about_fallback("landmark name mismatch")
                self._model = None
                return
            raise PoseEstimatorError(msg)

        logger.info(
            "MeTRAbs model initialised (skeleton=%s, %d joints, detector_threshold=%.2f).",
            self.skeleton, len(self._raw_names), self.detector_threshold,
        )

    # ------------------------------------------------------------------ public

    @property
    def is_real(self) -> bool:
        """True if the real MeTRAbs model is active (not the synthetic fallback)."""
        return self._model is not None

    def intrinsics_for(self, width: int, height: int) -> np.ndarray:
        """Pinhole intrinsic matrix used for this estimator's frames, cached
        per resolution. Exposed so the visualizer can project 3D points back
        to pixels for the overlay.
        """
        key = (width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = metrabs_model.intrinsic_matrix(
                width, height, self.default_fov_degrees
            )
        return self._intrinsics_cache[key]

    def estimate(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
        if self._model is None:
            return self._estimate_fallback(image_bgr, timestamp_s, frame_index)

        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self._tf.convert_to_tensor(rgb, dtype=self._tf.uint8)

        pred, box = self._infer(image_tensor, width, height, frame_index)
        if pred is None:
            return PoseFrame(timestamp_s=timestamp_s, keypoints={}, frame_index=frame_index)

        poses3d = pred["poses3d"].numpy()[0]  # [num_joints, 3] mm, camera frame
        poses2d = pred["poses2d"].numpy()[0]  # [num_joints, 2] pixels

        keypoints: dict[str, Keypoint] = {}
        for j, raw_name in enumerate(self._raw_names):
            canonical = self._raw_to_canonical.get(raw_name)
            if canonical is None:
                continue
            vis = self._visibility_proxy(poses2d[j], box, width, height)
            x, y, z = poses3d[j]
            keypoints[canonical] = Keypoint(
                x=float(x), y=float(y), z=float(z), visibility=vis
            )

        self._prev_poses3d = poses3d
        if self._frames_since_detect == 0:
            # This frame ran the detector: latch its box and the joint span that
            # goes with it as the reference for the frames that reuse them.
            pts = poses2d[np.isfinite(poses2d).all(axis=1)]
            if pts.shape[0] >= 4:
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                self._ref_extent = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
                self._tracked_box = np.asarray(box, dtype=np.float32)
            else:
                self._ref_extent = None
                self._tracked_box = None
        else:
            self._tracked_box = self._box_from_joints(poses2d, width, height)
        visible_count = sum(1 for kp in keypoints.values() if kp.visibility > 0.3)
        logger.debug(
            "Frame %d: Human detected (box conf %.2f) with %d/%d landmarks "
            "above the 0.3 visibility threshold",
            frame_index, float(box[4]), visible_count, len(keypoints),
        )
        return PoseFrame(timestamp_s=timestamp_s, keypoints=keypoints, frame_index=frame_index)

    def close(self) -> None:
        # The TF-Hub model has no explicit teardown; nothing to release.
        pass

    # ----------------------------------------------------------------- helpers

    def _infer(self, image_tensor, width: int, height: int, frame_index: int):
        """Run MeTRAbs, skipping the person-detector when a good box is known.

        Returns ``(prediction, box)`` where ``box`` is ``[left, top, w, h, conf]``,
        or ``(None, None)`` when nobody is in view.

        The detector is by far the most expensive part of ``detect_poses`` and
        re-runs from scratch every frame; between detections the previous frame's
        joints already bound the person more tightly than a fresh detection would.
        Any failure of the cheap path -- an empty result, a degenerate box -- drops
        straight back to a full detection on the same frame, so the worst case is
        one wasted pose-network pass, never a dropped frame.
        """
        use_tracked = (
            self.detect_interval > 1
            and self._tracked_box is not None
            and self._frames_since_detect < self.detect_interval - 1
        )
        if use_tracked:
            boxes4 = self._tf.constant([self._tracked_box[:4]], dtype=self._tf.float32)
            try:
                pred = self._model.estimate_poses(
                    image_tensor,
                    boxes4,
                    default_fov_degrees=self.default_fov_degrees,
                    num_aug=self.num_aug,
                    skeleton=self.skeleton,
                )
                if pred["poses3d"].shape[0] > 0 and self._pose_is_plausible(
                    pred["poses3d"].numpy()[0]
                ):
                    self._frames_since_detect += 1
                    return pred, self._tracked_box
            except Exception as exc:  # noqa: BLE001
                # A shape/signature mismatch on some model build: stop trying the
                # fast path rather than paying for a failure on every frame.
                logger.warning(
                    "estimate_poses unavailable (%s); detecting every frame.", exc
                )
                self.detect_interval = 1
            self._tracked_box = None

        pred = self._model.detect_poses(
            image_tensor,
            default_fov_degrees=self.default_fov_degrees,
            detector_threshold=self.detector_threshold,
            max_detections=self.max_detections,
            num_aug=self.num_aug,
            skeleton=self.skeleton,
        )
        boxes = pred["boxes"].numpy()
        self._frames_since_detect = 0
        if boxes.shape[0] == 0:
            logger.debug("Frame %d: No human detected by MeTRAbs", frame_index)
            self._tracked_box = None
            self._ref_extent = None
            self._prev_poses3d = None
            return None, None
        return pred, boxes[0]

    def _pose_is_plausible(self, poses3d: np.ndarray) -> bool:
        """Is this pose from the tracked box worth believing?

        ``estimate_poses`` has no detector and no notion of "nobody there": given
        a box it ALWAYS returns a skeleton, so once the subject walks away the
        cheap path keeps emitting a confident pose of an empty room. That is far
        worse than the frame it saves -- a dropped frame makes the robot hold
        still, an invented one makes it move to a phantom.

        The test is TEMPORAL, not spatial. An earlier version asked how many
        joints projected inside the picture, which sounds obvious and is useless
        here: MeTRAbs projects occluded and out-of-frame joints too, so a
        perfectly good pose of a nearby subject routinely puts a third of its
        joints outside a 1080-line frame (measured: a joint span of y = -44 to
        1438). That test rejected every good pose, and the fast path then paid
        for both inference passes -- 121 ms a frame instead of 82.

        A human body cannot teleport between two consecutive frames, so compare
        against the previous pose instead. Nothing to compare against (the first
        tracked frame after a detection) is accepted: the box is freshest there.
        """
        if self._prev_poses3d is None or self._prev_poses3d.shape != poses3d.shape:
            return True
        delta = np.linalg.norm(poses3d - self._prev_poses3d, axis=1)
        return bool(float(np.median(delta)) <= self.max_joint_jump_mm)

    def _box_from_joints(self, poses2d: np.ndarray, width: int, height: int):
        """Carry the DETECTOR's box forward, re-centred and re-scaled on this
        frame's joints.

        Not a box drawn around the joints: MeTRAbs' 2D joint projections span a
        far larger region than YOLOv4's box -- measured on this camera, 1.3x its
        width and 1.9x its height, because joints outside the frame are still
        projected. Cropping to that instead of to the detector's box feeds the
        pose network a picture it was never calibrated for, and the joints move
        by up to 250 mm. Keeping the detector's own width/height and only
        following the subject's centre and apparent size preserves the crop
        geometry, so the cheap path stays faithful to the expensive one.
        """
        if self._tracked_box is None or self._ref_extent is None:
            return None
        pts = poses2d[np.isfinite(poses2d).all(axis=1)]
        if pts.shape[0] < 4:
            return None
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cw, ch = float(x1 - x0), float(y1 - y0)
        if not (cw > 1.0 and ch > 1.0):
            return None
        ref_w, ref_h = self._ref_extent
        # Apparent size change since the last detection: the subject walking
        # toward or away from the camera. Averaged over both axes because a
        # raised arm widens the joint span without the person coming closer.
        scale = 0.5 * (cw / max(ref_w, 1e-6) + ch / max(ref_h, 1e-6))
        if not (0.4 < scale < 2.5):
            return None    # implausible: re-detect rather than guess
        _, _, bw, bh, conf = self._tracked_box
        bw, bh = float(bw) * scale, float(bh) * scale
        cx, cy = float(x0 + x1) * 0.5, float(y0 + y1) * 0.5
        if bw > 4.0 * width or bh > 4.0 * height:
            return None
        return np.array(
            [cx - bw * 0.5, cy - bh * 0.5, bw, bh, conf], dtype=np.float32
        )

    def _visibility_proxy(
        self, px_py: np.ndarray, box: np.ndarray, width: int, height: int
    ) -> float:
        """1.0 well inside the image and detection box, decaying to 0.0 near
        the edges. See module docstring -- this is a stand-in for MediaPipe's
        real per-joint visibility, which MeTRAbs does not provide.

        Two things this must NOT do, both of which silently emptied the
        retargeter (it drops any joint scoring under ``VIS_THRESHOLD`` = 0.5):

        * Measure distance to the RAW detection box. That box is fitted to the
          body, so the extremities the retargeter needs most -- ankles at its
          floor, wrists and shoulders at its widest -- sit on the boundary by
          construction and score ~0 however plainly visible they are. The box is
          padded by ``box_padding`` first, which is what that setting was always
          documented to do.
        * Scale by the detector's confidence. ``box[4]`` is a whole-PERSON score
          ("is someone there"), not a per-joint one, and folding it in multiplies
          every joint by the same sub-1.0 factor. Measured on this camera it sits
          at 0.45-0.50, so a fully-in-frame shoulder scored 0.489 and was thrown
          away for being 0.011 under the gate -- only 13.1% of joint-observations
          survived, against 31.4% on the in-frame test alone. Confidence is
          already enforced upstream by ``detector_threshold``, so applying it
          again here only conflates "is there a person" with "is this joint
          trustworthy".
        """
        px, py = float(px_py[0]), float(px_py[1])
        left, top, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        def edge_score(pos: float, lo: float, hi: float, margin: float) -> float:
            if pos < lo or pos > hi:
                return 0.0
            dist = min(pos - lo, hi - pos)
            return max(0.0, min(1.0, dist / max(margin, 1e-6)))

        # In-frame confidence: a joint outside the image was extrapolated by the
        # pose network rather than seen, so this term is the one doing real work.
        image_margin = _EDGE_MARGIN_FRAC * max(width, height)
        image_score = min(
            edge_score(px, 0.0, width, image_margin),
            edge_score(py, 0.0, height, image_margin),
        )

        # In-box confidence, against the padded box and a margin scaled to the
        # BOX. Scaling it to the image (as this once did) made the margin larger
        # than many boxes, so every joint in a distant subject scored 0.
        pad_x, pad_y = self.box_padding * w, self.box_padding * h
        box_margin = _EDGE_MARGIN_FRAC * max(w, h)
        box_score = min(
            edge_score(px, left - pad_x, left + w + pad_x, box_margin),
            edge_score(py, top - pad_y, top + h + pad_y, box_margin),
        )
        return min(image_score, box_score)

    def _warn_about_fallback(self, reason: str) -> None:
        logger.warning(
            "Using SYNTHETIC pose fallback (%s). The skeleton will NOT follow the "
            "human; fix MeTRAbs/GPU setup to enable real pose tracking.",
            reason,
        )

    def _estimate_fallback(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
        """Synthetic deterministic pose generator for environments without a
        working MeTRAbs/GPU setup. Coordinates are in the same mm/camera-frame
        convention real frames use (a person ~1.7m tall standing ~2m from the
        camera), so downstream retargeting math stays consistent."""
        t = frame_index / 20.0
        swing = 120.0 * math.sin(t)  # mm
        z0 = 2000.0  # mm from camera

        def kp(x: float, y: float, z: float = z0) -> Keypoint:
            return Keypoint(x=x, y=y, z=z, visibility=1.0)

        base: dict[str, Keypoint] = {
            "nose":            kp(0.0, -750.0),
            "left_eye":        kp(-20.0, -770.0),
            "right_eye":       kp(20.0, -770.0),
            "left_ear":        kp(-60.0, -750.0),
            "right_ear":       kp(60.0, -750.0),
            "neck":            kp(0.0, -600.0),
            "left_shoulder":   kp(-160.0, -580.0),
            "right_shoulder":  kp(160.0, -580.0),
            "left_elbow":      kp(-220.0, -300.0 + swing),
            "right_elbow":     kp(220.0, -300.0 - swing),
            "left_wrist":      kp(-260.0, -60.0 + swing * 1.3),
            "right_wrist":     kp(260.0, -60.0 - swing * 1.3),
            "pelvis":          kp(0.0, 0.0),
            "left_hip":        kp(-110.0, 20.0),
            "right_hip":       kp(110.0, 20.0),
            "left_knee":       kp(-115.0, 430.0),
            "right_knee":      kp(115.0, 430.0),
            "left_ankle":      kp(-120.0, 830.0),
            "right_ankle":     kp(120.0, 830.0),
        }
        return PoseFrame(timestamp_s=timestamp_s, keypoints=base, frame_index=frame_index)
