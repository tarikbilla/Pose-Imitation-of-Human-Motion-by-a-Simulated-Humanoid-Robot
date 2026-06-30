"""Monocular gait-cue extraction: human keypoints -> a compact gait command.

The robot cannot copy human *leg joint angles* to walk (monocular depth is
unreliable and direct leg-copy topples a free-standing NAO — see
``main/libraries/nao_retarget`` and the project history). Instead we distil the
human's lower-body motion into a small, depth-free **gait command** that an
on-robot walk engine (``main/libraries/gait.py``) turns into balance-stable
steps. The human supplies *intent* (am I marching? how fast? which leg is up?
stop), not joint targets.

Why these cues are monocular-robust
-----------------------------------
Every cue is a **sign or relative magnitude of normalized, hip-rooted 2D
landmark motion** — never an absolute angle and never depth ``z``:

* Primary signal ``s(t)`` = (left-knee height − right-knee height), normalized by
  shoulder width. It is *anti-phase between the two knees*, so it is large only
  when the legs alternate (marching) and is structurally immune to arm swing
  (arms are not in it). Its zero-crossings give the step cadence; its sign gives
  which knee is currently raised; its amplitude gives how vigorously the human
  marches.
* Normalization by a *smoothed* body width makes the cues scale-invariant, so
  they survive the subject walking toward/away from the camera.

The extractor is stateful (it keeps a short timestamped history) but is a pure
function of the landmark stream — no camera, no Webots, no RNG — so it is
unit-testable off-simulation and deterministic for recorded-video replay
(PRD Acceptance #3).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from src.type_defs import Keypoint, PoseFrame

# Lower-body landmarks the cue extractor needs visible to trust a gait reading.
_REQUIRED = ("left_hip", "right_hip", "left_knee", "right_knee")
_OPTIONAL = ("left_ankle", "right_ankle", "left_shoulder", "right_shoulder")

TWO_PI = 2.0 * math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class GaitCommand:
    """Compact, JSON-serializable gait command sent to the walk engine.

    ``state``       : "idle" or "march" (the engine treats anything it doesn't
                      recognize as "idle" -> decay to the stable crouch).
    ``cadence_hz``  : human step-cycle frequency (full L+R cycle), >= 0.
    ``phase``       : current gait phase in radians [0, 2*pi); 0 ~ left-knee up.
    ``swing_side``  : +1 left knee raised, -1 right knee raised, 0 neither.
    ``intensity``   : [0, 1] how vigorously the human marches (knee-lift amp).
    ``turn``        : [-1, 1] heavily-smoothed turn intent (experimental, low gain).
    ``conf``        : [0, 1] fraction of required lower-body landmarks visible.
    """

    state: str
    cadence_hz: float
    phase: float
    swing_side: int
    intensity: float
    turn: float
    conf: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "cadence_hz": round(self.cadence_hz, 4),
            "phase": round(self.phase, 4),
            "swing_side": self.swing_side,
            "intensity": round(self.intensity, 4),
            "turn": round(self.turn, 4),
            "conf": round(self.conf, 3),
        }


IDLE = GaitCommand("idle", 0.0, 0.0, 0, 0.0, 0.0, 0.0)


class GaitCueExtractor:
    """Turn a stream of :class:`PoseFrame`s into a smoothed :class:`GaitCommand`.

    Parameters
    ----------
    window_s:
        Length of the timestamped history used for cadence/amplitude (s).
    amp_start / amp_stop:
        Normalized peak-to-peak of the knee-differential signal needed to START
        marching and below which we STOP. ``amp_start > amp_stop`` gives
        hysteresis; the gap (plus the conf gate) rejects jitter and the
        "waving arms while standing still" aliasing case (bias to STOP).
    conf_min:
        Minimum visible-landmark fraction; below it we report idle so the robot
        holds its crouch when the legs leave frame.
    cadence_max_hz:
        Hard clamp on reported cadence (defensive against noisy crossings).
    start_cycles:
        Number of consistent alternating half-steps (zero-crossings) required
        before we declare "march" (slow to start), while we drop to idle
        immediately when the amplitude collapses (instant to stop).
    """

    def __init__(
        self,
        *,
        window_s: float = 1.3,
        amp_start: float = 0.08,
        amp_stop: float = 0.05,
        conf_min: float = 0.6,
        cadence_max_hz: float = 2.5,
        start_cycles: int = 2,
    ) -> None:
        self.window_s = float(window_s)
        self.amp_start = float(amp_start)
        self.amp_stop = float(amp_stop)
        self.conf_min = float(conf_min)
        self.cadence_max_hz = float(cadence_max_hz)
        self.start_cycles = int(start_cycles)

        # (timestamp_s, normalized knee-differential signal s)
        self._hist: Deque[Tuple[float, float]] = deque()
        self._cross_times: Deque[float] = deque(maxlen=6)
        self._last_sign: int = 0
        self._scale_ema: Optional[float] = None  # smoothed body width
        self._turn_ema: float = 0.0
        self._state: str = "idle"

    # -- public API ---------------------------------------------------------
    def update(self, pose: PoseFrame) -> GaitCommand:
        """Ingest one pose frame and return the current gait command."""
        kps = pose.keypoints
        conf = self._confidence(kps)
        if conf < self.conf_min:
            # Legs not reliably visible: reset cadence state, hold idle.
            self._decay_to_idle()
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, self._turn_ema, conf)

        scale = self._body_scale(kps)
        s = self._knee_diff_signal(kps, scale)
        t = pose.timestamp_s
        self._push(t, s)
        self._update_crossings(t, s)

        amp = self._amplitude()
        cadence = self._cadence_hz()
        # Hysteresis state machine (bias to stop).
        if self._state == "march":
            if amp < self.amp_stop or cadence <= 0.0:
                self._state = "idle"
        else:
            if amp >= self.amp_start and len(self._cross_times) >= self.start_cycles:
                self._state = "march"

        self._turn_ema += 0.05 * (self._turn_signal(kps, scale) - self._turn_ema)

        if self._state != "march":
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, self._turn_ema, conf)

        phase = self._phase(s, cadence)
        swing = 1 if s > 0.01 else (-1 if s < -0.01 else 0)
        # Map amplitude to a [0,1] intensity (amp_start..~3x amp_start -> 0..1).
        intensity = _clamp((amp - self.amp_stop) / (3.0 * self.amp_start), 0.0, 1.0)
        cadence = _clamp(cadence, 0.0, self.cadence_max_hz)
        return GaitCommand("march", cadence, phase, swing, intensity, self._turn_ema, conf)

    def reset(self) -> None:
        self._hist.clear()
        self._cross_times.clear()
        self._last_sign = 0
        self._scale_ema = None
        self._turn_ema = 0.0
        self._state = "idle"

    # -- internals ----------------------------------------------------------
    def _confidence(self, kps: Dict[str, Keypoint]) -> float:
        vis = [kps[n].visibility for n in _REQUIRED if n in kps]
        if len(vis) < len(_REQUIRED):
            return 0.0
        return sum(1.0 for v in vis if v >= 0.5) / len(_REQUIRED)

    def _body_scale(self, kps: Dict[str, Keypoint]) -> float:
        """Smoothed body width (shoulder span, hip span fallback) for normalization."""
        width = 0.0
        if "left_shoulder" in kps and "right_shoulder" in kps:
            width = abs(kps["left_shoulder"].x - kps["right_shoulder"].x)
        if width < 1e-3 and "left_hip" in kps and "right_hip" in kps:
            width = abs(kps["left_hip"].x - kps["right_hip"].x)
        width = max(width, 1e-3)
        # EMA so the normalizer doesn't wobble as the subject moves in depth.
        self._scale_ema = width if self._scale_ema is None else (
            self._scale_ema + 0.1 * (width - self._scale_ema)
        )
        return max(self._scale_ema, 1e-3)

    def _knee_diff_signal(self, kps: Dict[str, Keypoint], scale: float) -> float:
        """(left-knee height − right-knee height) / scale. Image y is DOWN, so a
        *raised* knee has a smaller y; height = hip_y − knee_y is larger when the
        knee is up. The differential is anti-phase between legs -> immune to arm
        swing, large only during alternating marching."""
        hip_y = 0.5 * (kps["left_hip"].y + kps["right_hip"].y)
        left_h = hip_y - kps["left_knee"].y
        right_h = hip_y - kps["right_knee"].y
        return (left_h - right_h) / scale

    def _turn_signal(self, kps: Dict[str, Keypoint], scale: float) -> float:
        """Experimental, low-gain turn proxy: lateral offset of shoulder midline
        from hip midline (a frontal-camera yaw/lean hint). Heavily smoothed by
        the caller; not part of the graded precision claim."""
        if not all(n in kps for n in ("left_shoulder", "right_shoulder")):
            return 0.0
        sh_x = 0.5 * (kps["left_shoulder"].x + kps["right_shoulder"].x)
        hip_x = 0.5 * (kps["left_hip"].x + kps["right_hip"].x)
        return _clamp((sh_x - hip_x) / scale, -1.0, 1.0)

    def _push(self, t: float, s: float) -> None:
        self._hist.append((t, s))
        cutoff = t - self.window_s
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.popleft()

    def _update_crossings(self, t: float, s: float) -> None:
        # Sign with a small deadband so noise near zero doesn't fake crossings.
        sign = 1 if s > 0.02 else (-1 if s < -0.02 else 0)
        if sign != 0 and self._last_sign != 0 and sign != self._last_sign:
            self._cross_times.append(t)
        if sign != 0:
            self._last_sign = sign
        # Drop crossings older than the window.
        cutoff = t - self.window_s
        while self._cross_times and self._cross_times[0] < cutoff:
            self._cross_times.popleft()

    def _amplitude(self) -> float:
        if len(self._hist) < 3:
            return 0.0
        vals = [s for _, s in self._hist]
        return max(vals) - min(vals)  # peak-to-peak

    def _cadence_hz(self) -> float:
        """Full-cycle (L+R) frequency from half-step zero-crossing intervals."""
        if len(self._cross_times) < 2:
            return 0.0
        times = list(self._cross_times)
        intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not intervals:
            return 0.0
        half_period = sum(intervals) / len(intervals)  # time between crossings
        if half_period <= 1e-3:
            return 0.0
        # One full gait cycle = two half-steps (two crossings).
        return _clamp(1.0 / (2.0 * half_period), 0.0, self.cadence_max_hz)

    def _phase(self, s: float, cadence: float) -> float:
        """Gait phase in [0, 2*pi) from the signal and its derivative.

        For s = A*sin(phi), phi = atan2(s, s_dot/omega). We estimate s_dot from
        the last two samples and omega from the cadence; the on-robot engine only
        uses this to phase-lock its own integrated clock, so an approximate value
        is fine.
        """
        if cadence <= 0.0 or len(self._hist) < 2:
            return 0.0
        (t0, s0), (t1, s1) = self._hist[-2], self._hist[-1]
        dt = max(t1 - t0, 1e-3)
        s_dot = (s1 - s0) / dt
        omega = TWO_PI * cadence
        phi = math.atan2(s, s_dot / omega) if omega > 1e-6 else 0.0
        return phi % TWO_PI

    def _decay_to_idle(self) -> None:
        self._state = "idle"
        self._cross_times.clear()
        self._last_sign = 0
