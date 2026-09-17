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
  they survive the subject walking toward/away from the camera. Landmarks are
  now MeTRAbs' absolute 3D coordinates in millimeters (camera frame: x right,
  y down, z forward/away -- see ``src/type_defs.Keypoint``) rather than the
  old MediaPipe-era normalized [0,1] image fractions, but every formula below
  is a ratio of same-unit quantities, so the math is unaffected by the unit
  change -- only the depth (``z``) trust below changes, because that channel
  went from unreliable to accurate.
* **Torso yaw** (``body_yaw_rad``) comes from the shoulder/hip line's *rotation
  in the horizontal plane*: the line's lateral extent shrinks as the subject
  turns (foreshortening) while the two endpoints separate in depth. Taking
  ``atan2(depth spread, lateral spread)`` recovers the yaw angle itself, not a
  vague "turn intent", and its sign is the one thing depth is reliable for
  (which shoulder is nearer, not how near). This is what lets the robot rotate
  its whole body instead of just its head: NAO has no torso-yaw joint, so the
  controller servos its measured heading onto this angle by stepping round.

  The lateral term is used **signed**, which is what makes the estimate cover
  the whole circle rather than a quarter of it. Turn past 90 degrees and your
  shoulders swap sides in the image, so the lateral term changes sign -- and that
  sign is exactly the front/back discriminator a single camera is otherwise
  missing. Using ``abs()`` folded the two halves together and bounded the estimate
  to +/-90 degrees, so the robot could never be asked to turn round.

  The lateral term is ``left.x - right.x``, not the other way about. Pose
  estimators conventionally label a person facing the camera with their
  *left* shoulder on the image's right (anatomical left/right, not
  screen-left/right -- MeTRAbs' ``coco_19`` follows the same COCO convention
  MediaPipe did), so ``right.x - left.x`` is negative in the facing-forward
  case -- which would report someone looking straight at the camera as being
  turned 180 degrees away. This was measured on recorded MediaPipe runs
  (negative on 67% of frames with both shoulders clearly visible, median
  -0.085, 99% of those frames also having both ears visible, i.e. a face-on
  view; the hips agreed, so it reflected a labelling convention rather than
  noise). MeTRAbs is expected to follow the same convention, but this has not
  been re-measured against real MeTRAbs output -- worth a quick sanity check
  ("stand facing the camera, confirm yaw reads ~0") the first time this runs
  on the target machine. It holds whether or not the preview is mirrored (the
  estimator cannot tell a mirrored subject from a real one, so it labels by
  appearance either way)::

      facing the camera   lateral > 0, dz ~ 0   ->  yaw ~   0 deg
      turned 90 deg       lateral ~ 0, dz != 0  ->  yaw ~ +/-90 deg
      facing away         lateral < 0, dz ~ 0   ->  yaw ~ +/-180 deg

  Robustness matters more here than anywhere else in this module, because a
  noisy yaw does not merely wobble the robot -- it makes the controller demand
  turn clips in alternating directions, which starves forward walking and trips
  the locomotion failure backoff. Recorded runs showed |yaw| spiking to the
  +/-90 bound on 7-18% of frames while the subject stood square to the camera.
  Three guards fix that: the shoulder span is compared against a self-calibrated
  reference so a collapsed or occluded detection is rejected rather than turned
  into a large angle; the estimate is smoothed on the *shortest arc* (it is a
  circular quantity now, so a plain EMA would take the long way round through
  180 degrees); and a frame that disagrees sharply with the running estimate has
  its confidence halved instead of being trusted.

The extractor is stateful (it keeps a short timestamped history) but is a pure
function of the landmark stream — no camera, no Webots, no RNG — so it is
unit-testable off-simulation and deterministic for recorded-video replay
(PRD Acceptance #3).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

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
    ``turn``        : [-1, 1] ``body_yaw_rad`` normalized by ``TURN_FULL_RAD``,
                      kept for controllers that only understand a turn *intent*.
    ``conf``        : [0, 1] fraction of required lower-body landmarks visible.
    ``body_yaw_rad``: signed torso yaw in radians; 0 = squarely facing the camera,
                      positive = rotated toward the subject's screen-left. This is
                      an *angle*, so the controller can close a heading loop on it.
    ``yaw_conf``    : [0, 1] confidence in ``body_yaw_rad``. Independent of
                      ``conf``: the yaw only needs the shoulders and hips, so it
                      stays usable when the legs leave the frame.
    ``cue_channel`` : which signal declared the march -- "knee" (marching on the
                      spot) or "stride" (actually walking), or "none" when idle.
                      Logged so a session can say WHY the robot did or did not
                      walk. See _Channel.
    """

    state: str
    cadence_hz: float
    phase: float
    swing_side: int
    intensity: float
    turn: float
    conf: float
    body_yaw_rad: float = 0.0
    yaw_conf: float = 0.0
    cue_channel: str = "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "cadence_hz": round(self.cadence_hz, 4),
            "phase": round(self.phase, 4),
            "swing_side": self.swing_side,
            "intensity": round(self.intensity, 4),
            "turn": round(self.turn, 4),
            "conf": round(self.conf, 3),
            "body_yaw_rad": round(self.body_yaw_rad, 4),
            "yaw_conf": round(self.yaw_conf, 3),
            "cue_channel": self.cue_channel,
        }


IDLE = GaitCommand("idle", 0.0, 0.0, 0, 0.0, 0.0, 0.0)

# Torso yaw that maps to a full-scale (+/-1) legacy ``turn`` value.
TURN_FULL_RAD = math.radians(60.0)
# MeTRAbs' depth channel is a real metric measurement (unlike MediaPipe's,
# which was foreshortening-only and unreliable), so it is trusted at full
# weight in the yaw solve -- no down-scaling needed. Kept as a named constant
# (rather than inlining 1.0) so the yaw formula's shape stays unchanged and
# this trust level is easy to find and re-tune if MeTRAbs' depth precision
# does not hold up in practice.
YAW_Z_TRUST = 1.0
# Landmark pairs the yaw is averaged over, hips weighted lower (they are noisier
# and clothing-dependent).
_YAW_PAIRS = (("left_shoulder", "right_shoulder", 1.0), ("left_hip", "right_hip", 0.6))
# A body-fixed segment keeps a near-constant length however the subject turns
# (the depth term takes over from the lateral one), so an observed length well
# below the self-calibrated reference means the landmarks are unreliable -- not
# that the subject is at some large angle. Below ``_YAW_SPAN_FLOOR`` of the
# reference the reading is discarded; it ramps to full trust at ``_YAW_SPAN_GOOD``.
_YAW_SPAN_FLOOR = 0.45
_YAW_SPAN_GOOD = 0.70
# Circular EMA rate for the reported yaw. Slow on purpose: this drives a stepping
# servo whose granularity is a 60 degree clip, so chasing per-frame noise buys
# nothing and costs clip thrash.
_YAW_EMA_ALPHA = 0.18
# A single frame disagreeing with the running estimate by more than this is more
# likely noise than a real rotation at camera frame rates, so it is de-weighted.
_YAW_JUMP_RAD = 0.60


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi). Yaw is circular now, so this is load-bearing."""
    return (angle + math.pi) % TWO_PI - math.pi


class _PeakHold:
    """Running maximum: rises quickly, decays very slowly.

    Learns a body-fixed segment's true length from the stream. A projected
    segment can only ever look *shorter* than it is, so the running peak
    converges on the real length; the slow decay lets it follow a genuinely
    different subject or camera distance instead of latching forever.
    """

    __slots__ = ("value", "rise", "decay")

    def __init__(self, rise: float = 0.30, decay: float = 0.004) -> None:
        self.value: float = 0.0
        self.rise = rise
        self.decay = decay

    def update(self, sample: float) -> float:
        if not math.isfinite(sample) or sample <= 0.0:
            return self.value
        if self.value <= 0.0:
            self.value = sample
        elif sample > self.value:
            self.value += self.rise * (sample - self.value)
        else:
            self.value += self.decay * (sample - self.value)
        return self.value


class _Channel:
    """One oscillating gait signal, with its own history, crossings and cadence.

    There are two, because a human in front of this camera does two different
    things and only one of them was ever measured:

    * ``knee``   -- the left/right knee-height differential. This is a
                    MARCH-IN-PLACE cue: it is large when someone lifts their
                    knees alternately on the spot.
    * ``stride`` -- the signed ankle separation along the body's forward axis.
                    This is a WALKING cue, and it is the one that matters.
                    Measured over a recorded session in which the subject
                    translated 3.25 m (pelvis depth 2,509 -> 5,759 mm): the knee
                    differential had a median of 8 mm (p95 47) while the ankle
                    separation had a median of 102 mm (p95 338, max 521) -- 12
                    times the signal, on landmarks visible in 97.0% of the frames
                    where the hips and knees were.

    The amplitude window and the CROSSING window are deliberately different.
    Requiring two crossings inside the 1.3 s amplitude window imposes a cadence
    floor of 1/(2*1.3) = 0.385 Hz -- 2.6 times plan_action's own
    walk_cadence_min_hz of 0.15 -- and the recorded human sits right on it: the
    histogram of crossings-per-window over amplitude-sufficient frames peaks at
    exactly ONE (498 frames with 1, 47 with 2). That single gate, not the
    amplitude and not the confidence, is what held "march" to 1.14% of frames.
    """

    def __init__(self, amp_window_s: float, cross_window_s: float,
                 deadband: float = 0.02, deadband_frac: float = 0.20) -> None:
        self.amp_window_s = float(amp_window_s)
        self.cross_window_s = float(cross_window_s)
        self.deadband = float(deadband)
        self.deadband_frac = float(deadband_frac)
        self.hist: deque[tuple[float, float]] = deque()
        self.cross_times: deque[float] = deque(maxlen=12)
        self.last_sign: int = 0
        self.value: float = 0.0

    def clear(self) -> None:
        self.hist.clear()
        self.cross_times.clear()
        self.last_sign = 0
        self.value = 0.0

    def push(self, t: float, s: float) -> None:
        self.value = s
        self.hist.append((t, s))
        cutoff = t - max(self.amp_window_s, self.cross_window_s)
        while self.hist and self.hist[0][0] < cutoff:
            self.hist.popleft()

        # Crossings are counted against the signal's OWN running middle, not
        # against zero, and with a deadband proportional to the amplitude. A
        # walking human's stride signal is not centred on zero (stance width,
        # camera obliquity and a limp all bias it), and a fixed 0.02 deadband is
        # either noise-blind or signal-blind depending on how far away they are
        # standing.
        amp = self.amplitude()
        ref = self.median()
        dead = max(self.deadband, self.deadband_frac * amp)
        centred = s - ref
        sign = 1 if centred > dead else (-1 if centred < -dead else 0)
        if sign != 0 and self.last_sign != 0 and sign != self.last_sign:
            self.cross_times.append(t)
        if sign != 0:
            self.last_sign = sign
        cutoff = t - self.cross_window_s
        while self.cross_times and self.cross_times[0] < cutoff:
            self.cross_times.popleft()

    def _window(self, seconds: float) -> list[float]:
        if not self.hist:
            return []
        newest = self.hist[-1][0]
        return [v for ts, v in self.hist if ts >= newest - seconds]

    def amplitude(self) -> float:
        return self.amplitude_in(self.amp_window_s)

    def amplitude_in(self, seconds: float) -> float:
        """Peak-to-peak over the last ``seconds``.

        Split out from :meth:`amplitude` because starting and stopping want
        different windows. Starting wants the long one: it is the evidence that
        a real gait is under way. Stopping wants the SHORTEST window that still
        spans a half-step, because ``max - min`` cannot fall until the whole
        window has emptied of motion -- so the amplitude window IS the stop
        latency. Measured on the 2026-09-08 session, ``amp_window_s`` of 1.3 s
        produced a 1218 ms mean stop latency, which is most of the "robot keeps
        walking after I stop" complaint.
        """
        vals = self._window(seconds)
        if len(vals) < 3:
            return 0.0
        return max(vals) - min(vals)

    def median(self) -> float:
        vals = sorted(self._window(self.cross_window_s))
        if not vals:
            return 0.0
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])

    def crossings(self) -> int:
        return len(self.cross_times)

    def cadence_hz(self, cadence_max_hz: float) -> float:
        """Full-cycle (L+R) frequency from the half-step crossing intervals."""
        if len(self.cross_times) < 2:
            return 0.0
        times = list(self.cross_times)
        intervals = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
        if not intervals:
            return 0.0
        half_period = sum(intervals) / len(intervals)
        if half_period <= 1e-3:
            return 0.0
        return _clamp(1.0 / (2.0 * half_period), 0.0, cadence_max_hz)


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
    stop_window_s:
        Amplitude window used for the STOP decision only, and therefore the
        stop latency itself -- ``max - min`` cannot fall until the window has
        emptied of motion. Do not lower this below ~0.8 s. Replayed against
        logs/run_20260908_130043 (3351 frames, 24 ground-truth walking bouts
        from acausal ankle speed), shortening it trades a faster stop for a
        FRAGMENTED walk, and each fragment is a clip stop, a fresh prepare ramp
        and another clip->pose handover -- the handover being what caused all
        three falls in that session:

            stop_window_s   stop latency   false march   march segments/bout
            1.3 (was)         1210 ms        51.4 s          0.88
            0.8 (now)          487 ms        34.2 s          0.96
            0.6                177 ms        30.2 s          1.50   <-- chops
            0.5                 71 ms        27.1 s          1.83   <-- chops

        0.8 s is the knee of that curve: it nearly halves the stop latency and
        removes a third of the false march while leaving continuity alone. A
        genuine walk here has a half-step every 0.807-0.886 s, so a window much
        under that cannot span one and starts reading mid-stride as a stop.
    conf_grace_frames:
        Consecutive sub-``conf_min`` frames tolerated before the cadence
        evidence is discarded. ``conf`` is a quantised visibility fraction, so
        one mis-detected ankle in an otherwise clean walk used to wipe
        ``cross_times`` and make the robot re-earn the whole ``start_cycles``
        gate (~1.7 s). Idle is still REPORTED on every low-confidence frame;
        only the forgetting is delayed.
    """

    def __init__(
        self,
        *,
        window_s: float = 1.3,
        cross_window_s: float = 3.0,
        stop_window_s: float = 0.8,
        amp_start: float = 0.08,
        amp_stop: float = 0.05,
        conf_min: float = 0.6,
        cadence_max_hz: float = 2.5,
        start_cycles: int = 2,
        conf_grace_frames: int = 2,
    ) -> None:
        self.window_s = float(window_s)
        self.cross_window_s = float(cross_window_s)
        # Never longer than the amplitude window: a stop window that outran it
        # would read peak-to-peak over samples the channel has already dropped.
        self.stop_window_s = min(float(stop_window_s), float(window_s))
        self.amp_start = float(amp_start)
        self.amp_stop = float(amp_stop)
        self.conf_min = float(conf_min)
        self.cadence_max_hz = float(cadence_max_hz)
        self.start_cycles = int(start_cycles)
        self.conf_grace_frames = max(0, int(conf_grace_frames))
        self._low_conf_run = 0

        # One channel per thing a human might be doing: marching on the spot
        # (knee-height differential) and actually walking (ankle separation along
        # the body's forward axis). See _Channel.
        self._knee = _Channel(self.window_s, self.cross_window_s)
        self._stride = _Channel(self.window_s, self.cross_window_s)
        self._channel: str = "none"       # which one declared the march
        self._scale_ema: float | None = None  # smoothed body width
        self._leg_ema: float | None = None    # smoothed hip->ankle length
        self._yaw_ema: float = 0.0                # smoothed torso yaw (rad)
        self._yaw_seen: bool = False
        # Self-calibrated reference length of each yaw segment (shoulders, hips).
        self._yaw_span = [_PeakHold() for _ in _YAW_PAIRS]
        self._state: str = "idle"

    # -- public API ---------------------------------------------------------
    def update(self, pose: PoseFrame) -> GaitCommand:
        """Ingest one pose frame and return the current gait command."""
        kps = pose.keypoints
        # Torso yaw is computed FIRST and unconditionally: it needs only the
        # shoulders and hips, and the robot must be able to turn to face you
        # while standing perfectly still -- gating it behind the marching
        # confidence is exactly why body rotation used to move nothing but the
        # head.
        yaw, yaw_conf = self._update_yaw(kps)
        turn = _clamp(yaw / TURN_FULL_RAD, -1.0, 1.0)

        conf = self._confidence(kps)
        if conf < self.conf_min:
            # Legs not reliably visible: report idle for this frame -- we cannot
            # assert a march we cannot see -- but do NOT throw the cadence
            # evidence away on the strength of one bad frame. ``conf`` is a
            # quantised visibility fraction, so a single mis-detected ankle
            # drops it below ``conf_min`` for one frame in an otherwise clean
            # walk, and wiping ``cross_times`` there made the robot re-pay the
            # full two-crossing start gate (~1.7 s) after every blink. That is
            # the "walks, stops, walks again" stutter. The controller's walk
            # latch is what bridges these one-frame idles.
            self._low_conf_run += 1
            if self._low_conf_run > self.conf_grace_frames:
                self._decay_to_idle()
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, turn, conf, yaw, yaw_conf)
        self._low_conf_run = 0

        t = pose.timestamp_s
        scale = self._body_scale(kps)
        self._knee.push(t, self._knee_diff_signal(kps, scale))
        stride = self._stride_signal(kps)
        if stride is not None:
            self._stride.push(t, stride)

        # EITHER channel may declare the march, and the one that does supplies
        # the phase and swing side. The knee channel is preferred when both
        # qualify: its sign maps directly onto which knee is up, which is what
        # the swing side means.
        def qualifies(ch: _Channel) -> bool:
            return (ch.amplitude() >= self.amp_start
                    and ch.crossings() >= self.start_cycles
                    and ch.cadence_hz(self.cadence_max_hz) > 0.0)

        if self._state == "march":
            active = self._knee if self._channel == "knee" else self._stride
            # STOPPING is judged on RECENT amplitude only. The cadence estimate
            # is deliberately not part of this test: it is derived from crossings
            # kept for ``cross_window_s`` (3.0 s), so it stays positive for three
            # seconds after the human plants their feet, and an AND with it could
            # only ever delay the stop, never hasten it.
            holding = active.amplitude_in(self.stop_window_s) >= self.amp_stop
            if not holding:
                # The other channel may still be carrying it (a walk that turns
                # into a march on the spot, say) -- do not stop if it is. But it
                # must be carrying it NOW: ``qualifies`` is the START gate, and
                # its crossing count spans 3.0 s, so using it here let a channel
                # that had stopped moving a full window ago veto the stop. That
                # handover is what kept "march" alive for 61.4 s of a 238 s
                # session in which the human was walking for only half of it.
                other = self._stride if active is self._knee else self._knee
                if qualifies(other) and other.amplitude_in(self.stop_window_s) >= self.amp_stop:
                    self._channel = "stride" if other is self._stride else "knee"
                else:
                    self._state = "idle"
                    self._channel = "none"
        else:
            if qualifies(self._knee):
                self._state, self._channel = "march", "knee"
            elif qualifies(self._stride):
                self._state, self._channel = "march", "stride"

        if self._state != "march":
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, turn, conf, yaw, yaw_conf,
                               cue_channel="none")

        ch = self._knee if self._channel == "knee" else self._stride
        s = ch.value
        amp = ch.amplitude()
        cadence = ch.cadence_hz(self.cadence_max_hz)
        phase = self._phase(s, cadence)
        swing = 1 if s > 0.01 else (-1 if s < -0.01 else 0)
        # Map amplitude to a [0,1] intensity (amp_start..~3x amp_start -> 0..1).
        intensity = _clamp((amp - self.amp_stop) / (3.0 * self.amp_start), 0.0, 1.0)
        cadence = _clamp(cadence, 0.0, self.cadence_max_hz)
        return GaitCommand(
            "march", cadence, phase, swing, intensity, turn, conf, yaw, yaw_conf,
            cue_channel=self._channel,
        )

    def reset(self) -> None:
        self._knee.clear()
        self._stride.clear()
        self._channel = "none"
        self._scale_ema = None
        self._leg_ema = None
        self._yaw_ema = 0.0
        self._yaw_seen = False
        self._yaw_span = [_PeakHold() for _ in _YAW_PAIRS]
        self._state = "idle"

    # -- internals ----------------------------------------------------------
    def _confidence(self, kps: dict[str, Keypoint]) -> float:
        vis = [kps[n].visibility for n in _REQUIRED if n in kps]
        if len(vis) < len(_REQUIRED):
            return 0.0
        return sum(1.0 for v in vis if v >= 0.5) / len(_REQUIRED)

    def _body_scale(self, kps: dict[str, Keypoint]) -> float:
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

    def _knee_diff_signal(self, kps: dict[str, Keypoint], scale: float) -> float:
        """(left-knee height − right-knee height) / scale. Image y is DOWN, so a
        *raised* knee has a smaller y; height = hip_y − knee_y is larger when the
        knee is up. The differential is anti-phase between legs -> immune to arm
        swing, large only during alternating marching."""
        hip_y = 0.5 * (kps["left_hip"].y + kps["right_hip"].y)
        left_h = hip_y - kps["left_knee"].y
        right_h = hip_y - kps["right_knee"].y
        return (left_h - right_h) / scale

    def _leg_length(self, kps: dict[str, Keypoint]) -> float:
        """Smoothed hip->ankle length, the natural normaliser for a stride."""
        best = 0.0
        for hip, ankle in (("left_hip", "left_ankle"), ("right_hip", "right_ankle")):
            if hip in kps and ankle in kps:
                best = max(best, math.dist(
                    (kps[hip].x, kps[hip].y, kps[hip].z),
                    (kps[ankle].x, kps[ankle].y, kps[ankle].z)))
        if best > 1e-3:
            self._leg_ema = best if self._leg_ema is None else (
                self._leg_ema + 0.05 * (best - self._leg_ema))
        return max(self._leg_ema or 0.0, 1e-3)

    def _stride_signal(self, kps: dict[str, Keypoint]) -> float | None:
        """Signed ankle separation along the body's FORWARD axis, in leg lengths.

        This is the channel that sees actual walking. The forward axis is taken
        from the shoulder line -- the same vector the yaw estimate already builds
        -- rotated 90 degrees in the ground plane, so the measurement follows the
        subject however they are facing rather than assuming they walk across the
        image. Returns None when either ankle is not reliably visible, in which
        case the knee channel carries the cue alone.
        """
        need = ("left_ankle", "right_ankle", "left_shoulder", "right_shoulder")
        if any(n not in kps for n in need):
            return None
        if min(kps["left_ankle"].visibility, kps["right_ankle"].visibility) < 0.5:
            return None
        ux = kps["left_shoulder"].x - kps["right_shoulder"].x
        uz = kps["left_shoulder"].z - kps["right_shoulder"].z
        span = math.hypot(ux, uz)
        if span < 1e-6:
            return None
        # Ground-plane normal to the shoulder line = the way the body faces.
        nx, nz = uz / span, -ux / span
        dx = kps["left_ankle"].x - kps["right_ankle"].x
        dz = kps["left_ankle"].z - kps["right_ankle"].z
        return (dx * nx + dz * nz) / self._leg_length(kps)

    def _update_yaw(self, kps: dict[str, Keypoint]) -> tuple[float, float]:
        """Smoothed torso yaw in radians (full circle) and its confidence.

        The shoulder (and hip) line is a body-fixed horizontal segment, so its
        image-plane extent and its depth spread are the two legs of a right
        triangle whose angle *is* the yaw::

            yaw = atan2(-depth spread, lateral extent)

        with the lateral term taken **signed** so the estimate covers the whole
        circle -- see the module docstring for the sign table and for why the
        robustness guards below exist.
        """
        best_conf = 0.0
        vec_x = 0.0     # accumulate as a VECTOR, not an angle: averaging angles
        vec_y = 0.0     # across the +/-180 seam gives nonsense.
        total_w = 0.0
        for index, (left_name, right_name, weight) in enumerate(_YAW_PAIRS):
            left = kps.get(left_name)
            right = kps.get(right_name)
            if left is None or right is None:
                continue
            vis = min(left.visibility, right.visibility)
            if vis < 0.5:
                continue
            # See the module docstring: MediaPipe puts the LEFT shoulder on the
            # image's right for a subject facing the camera, so this ordering is
            # what makes "facing forward" read as 0 rather than 180 degrees.
            lateral = left.x - right.x
            depth = (right.z - left.z) * YAW_Z_TRUST
            span = math.hypot(lateral, depth)
            if span < 1e-4:
                continue
            # A body-fixed segment has a near-constant length at any yaw, so the
            # observed length against its learned reference tells us whether to
            # believe this frame at all.
            reference = self._yaw_span[index].update(span)
            trust = _clamp(
                (span / max(reference, 1e-6) - _YAW_SPAN_FLOOR)
                / (_YAW_SPAN_GOOD - _YAW_SPAN_FLOOR), 0.0, 1.0
            )
            if trust <= 0.0:
                continue
            w = weight * trust
            vec_x += w * lateral / span
            vec_y += w * -depth / span
            total_w += w
            best_conf = max(best_conf, vis * trust)

        if total_w <= 0.0 or math.hypot(vec_x, vec_y) < 1e-9:
            return self._yaw_ema, 0.0

        raw = math.atan2(vec_y, vec_x)
        if not self._yaw_seen:
            self._yaw_ema = raw
            self._yaw_seen = True
            return self._yaw_ema, _clamp(best_conf, 0.0, 1.0)

        # Smooth along the SHORTEST arc: a plain EMA would travel the long way
        # round whenever the subject crosses the +/-180 seam.
        delta = _wrap_pi(raw - self._yaw_ema)
        if abs(delta) > _YAW_JUMP_RAD:
            # Too big a jump for one camera frame to be a real rotation. Follow it
            # (it may be genuine and we must not latch up) but say we are unsure.
            best_conf *= 0.5
        self._yaw_ema = _wrap_pi(self._yaw_ema + _YAW_EMA_ALPHA * delta)
        return self._yaw_ema, _clamp(best_conf, 0.0, 1.0)

    def _phase(self, s: float, cadence: float) -> float:
        """Gait phase in [0, 2*pi) from the signal and its derivative.

        For s = A*sin(phi), phi = atan2(s, s_dot/omega). We estimate s_dot from
        the last two samples of whichever channel declared the march, and omega
        from the cadence; the on-robot engine only uses this to phase-lock its own
        integrated clock, so an approximate value is fine.
        """
        hist = (self._knee if self._channel == "knee" else self._stride).hist
        if cadence <= 0.0 or len(hist) < 2:
            return 0.0
        (t0, s0), (t1, s1) = hist[-2], hist[-1]
        dt = max(t1 - t0, 1e-3)
        s_dot = (s1 - s0) / dt
        omega = TWO_PI * cadence
        phi = math.atan2(s, s_dot / omega) if omega > 1e-6 else 0.0
        return phi % TWO_PI

    def _decay_to_idle(self) -> None:
        self._state = "idle"
        self._channel = "none"
        self._knee.cross_times.clear()
        self._knee.last_sign = 0
        self._stride.cross_times.clear()
        self._stride.last_sign = 0
