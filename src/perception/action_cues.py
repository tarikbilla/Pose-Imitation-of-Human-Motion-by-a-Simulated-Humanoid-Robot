"""Which lower-body clip is this human asking for? A fail-safe action classifier.

The robot answers a human motion with a pre-balanced clip, and a clip is a
COMMITMENT: measured over one recorded session the robot spent 71.8% of the
window inside one, mean 6.84 s. So a misread here is not a small error that the
next frame corrects -- it is several seconds of the robot confidently doing the
wrong thing, open loop, with no way to take it back.

Every symptom this project has recorded on the leg side traces to how that
decision was made rather than to the clip that followed it:

    walks 5-7 s late     the start gate needed two zero-crossings of a periodic
                         signal, so it could not fire until the human had taken
                         two full steps
    stops 5-7 s late     the stop was judged on the AMPLITUDE of a windowed
                         signal, and an amplitude cannot fall until the window
                         has emptied of motion -- the latency was the window
    walks with no input  51.4 s of false march in 273 s (18.8%), from a 3.0 s
                         crossing memory that outlived the walking, plus a decay
                         path that wiped all evidence on ONE low-confidence frame
    falls                the clip-to-pose handover, since fixed in the clips
                         themselves (they now start and end standing)

What this module does differently
---------------------------------
**It measures the thing, instead of a proxy for it.** The old cue asked "is a
knee-height signal oscillating", which is a proxy for walking and can be fooled
by a stale window. This asks "is the pelvis actually translating along the body's
own forward axis", which is a direct measurement of the thing the robot is being
asked to copy. That single change is what removes the stop latency at its root: a
crossing COUNT can only decay as its window empties, while a velocity is near
zero on the very frame the human stops.

Calibrated, not guessed. On run_20260916_135555 (3436 frames, 269 s), restricted
to 622 frames where the pelvis provably did not move (under 40 mm across a 2 s
window), the short-window speed reads:

    parked   p50 0.021   p95 0.068   p99 0.085   max 0.112 m/s
    moving   p50 0.067   p90 0.317   p99 0.655   max 0.890 m/s

so :data:`WALK_ENTER_MPS` = 0.12 m/s fires on **0.0%** of those stationary
frames. The exit threshold is deliberately lower, not equal -- see
:class:`_Debounce`.

The three rules that make it fail safe
---------------------------------------
1. **Unobserved is its own answer.** When the landmarks an action needs are not
   visible, the verdict is ``"unknown"`` -- never ``"idle"`` and never the
   previous action. Those are three different things and collapsing them is how a
   robot keeps walking at a human who has left the frame. The controller is told
   plainly that nobody knows, and stands.

2. **Evidence expires.** Absence of contradiction is not evidence. Every action
   must be positively supported on the frames it is claimed, and support that
   stops arriving decays on a clock rather than persisting until something
   overwrites it.

3. **One bad frame decides nothing, and neither does one good one.** Entering an
   action needs sustained evidence; leaving it needs less. Both directions are
   debounced, and a chosen action holds for a minimum dwell, because the cost of
   flicker is not a wrong frame -- it is a clip restart, and restarts were what
   chopped one walk into fragments and put the robot on the floor.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from src.type_defs import Keypoint, PoseFrame

# --- Locomotion -------------------------------------------------------------
# Measured: see the module docstring. ENTER clears the noise floor completely;
# EXIT sits just above it, so a walk survives a momentary dip but a genuine stop
# (which reads ~0.02 m/s) ends it on the next frame the debouncer allows.
WALK_ENTER_MPS = 0.12
WALK_EXIT_MPS = 0.07
SIDE_ENTER_MPS = 0.12
SIDE_EXIT_MPS = 0.07

# Two windows, because speed and DIRECTION are not the same question and cannot
# share an answer.
#
# SPEED decides start and stop, so it wants the shortest window that is not just
# noise -- at 13 FPS this is about four frames, and any longer is latency the
# human will feel as the robot being slow to stop.
#
# DIRECTION cannot use that window. The pelvis surges and slows WITHIN each gait
# cycle, so a short window reads the sway rather than the travel: measured on
# run_20260916_135555, restricted to frames genuinely moving (|v| over 1.0 s
# above 0.12 m/s), a 0.30 s window disagrees with a 1.0 s window about which way
# the human is going in **10.5%** of frames. At 0.80 s that falls to 0.6%.
#
# That disagreement is not a cosmetic error. Forward and backward are different
# clips, so every sign flip is a clip restart -- and restarts are what fragment
# one walk into pieces. Reading direction over the longer window costs nothing
# that matters, because the decision it feeds (which of two clips) only has to be
# made once per walk, while the decision the short window feeds (walk at all)
# has to be made continuously.
VELOCITY_WINDOW_S = 0.30
DIRECTION_WINDOW_S = 0.80

# The second witness, and the one that matters most.
#
# Speed alone is not evidence of travel. A single mis-placed landmark, or one
# sharp lean, produces a velocity spike that clears any threshold for long enough
# to satisfy a debouncer -- and the robot answers it with several seconds of
# walking. Measured on run_20260916_135555 with speed as the only witness, **40
# of 74** locomotion episodes involved under 60 mm of actual pelvis travel: more
# than half the clips the robot would have played were answers to someone who had
# not gone anywhere.
#
# So a locomotion action additionally requires net DISPLACEMENT across the
# direction window. Displacement cannot be faked by a spike, because it is the
# integral the spike would have to sustain: 0.08 m over 0.8 s is the same
# 0.1 m/s held continuously, which is what walking is and what a jitter is not.
#
# This is deliberately a second INDEPENDENT measurement rather than a stricter
# threshold on the first. Raising the speed gate would reject slow walking along
# with the spikes; requiring both asks the two questions that actually differ --
# "is it moving now" and "has it moved at all".
MIN_TRAVEL_M = 0.08

# --- Posture ----------------------------------------------------------------
# Crouch as a fraction of the subject's own standing hip height, so it is free of
# body size and camera distance.
SQUAT_ENTER = 0.10
SQUAT_EXIT = 0.05

# Foot lift as a fraction of leg length. A raised foot is unambiguous; the floor
# is set by how well MeTRAbs pins an ankle, not by anatomy.
LIFT_ENTER = 0.08
LIFT_EXIT = 0.04

# --- Timing -----------------------------------------------------------------
# Asymmetric on purpose. Entering is a commitment and is paid for with evidence;
# leaving is cheap because standing still is the safe answer.
# Measured on run_20260916_135555 against an ACAUSAL ground truth (pelvis speed
# over a 1.5 s window centred on each instant -- something an evaluator may use
# and a live classifier may not), across 16 walking bouts of at least 0.8 s:
#
#   configuration              missed   start    stop   false walking
#   the OLD cue (recorded)        --   1064 ms  1218 ms     18.8%
#   speed witness only          0/16    461 ms   372 ms      3.9%
#   + displacement 80 mm        0/16    790 ms   165 ms      1.7%   <- shipped
#   + displacement 120 mm       0/16    988 ms   165 ms      1.1%
#
# The displacement witness buys an 11x cut in false walking for 330 ms of start
# latency, and start latency is the cheap direction: being slightly late to walk
# costs a moment, while walking at a human who is not walking costs a clip.
#
# EXIT is longer than it looks like it should be, and that is not a slower stop:
# the stop is carried by the displacement witness collapsing, which it does at
# once, so 0.12 s and 0.50 s both measure 165 ms of stop latency. What the longer
# hold buys is frame-level recall DURING a bout (63.5% -> 68.0%), by bridging the
# dips the pelvis makes within each stride.
ENTER_HOLD_S = 0.25
EXIT_HOLD_S = 0.30

# Once an action is chosen it holds for at least this long. Not politeness: the
# cost of changing the answer is a clip restart, and restarts are what fragmented
# one walk into 1.5-1.8 pieces, each paying its own ramp and handover.
MIN_DWELL_S = 0.60

# Frames of non-observation tolerated before the evidence is thrown away. One
# dropped landmark must not wipe the history (that exact bug produced 18.8% false
# march), but a human who has left must not leave a usable opinion behind either.
OBSERVE_GRACE_S = 0.40

# Past this, the last observation is too old to act on whatever it said.
STALE_S = 0.75

ACTIONS = (
    "unknown", "idle",
    "walk_forward", "walk_backward",
    "step_left", "step_right",
    "raise_left", "raise_right",
    "squat",
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class ActionCommand:
    """What the human is doing, and how sure we are.

    ``action`` is one of :data:`ACTIONS`. ``"unknown"`` and ``"idle"`` are NOT
    the same: idle means "seen, and standing still", unknown means "not seen".
    The robot's response to them differs -- it can hold a pose for the first and
    must stand down for the second -- so the distinction is carried all the way
    through rather than collapsed here.

    Every supporting measurement is carried alongside, because a session that
    only records the verdict cannot answer why the verdict was wrong.
    """
    action: str = "unknown"
    confidence: float = 0.0
    reason: str = "no observation yet"
    forward_mps: float = 0.0     # + = the subject's own forward
    lateral_mps: float = 0.0     # + = the subject's own left
    yaw_rate_dps: float = 0.0
    crouch: float = 0.0          # 0 = standing tall
    lift: float = 0.0            # 0..1, the raised foot's clearance
    lift_side: int = 0           # +1 left raised, -1 right, 0 neither
    observed: bool = False
    held_s: float = 0.0          # how long this action has been the answer

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "conf": round(self.confidence, 3),
            "reason": self.reason,
            "forward_mps": round(self.forward_mps, 4),
            "lateral_mps": round(self.lateral_mps, 4),
            "yaw_rate_dps": round(self.yaw_rate_dps, 2),
            "crouch": round(self.crouch, 3),
            "lift": round(self.lift, 3),
            "lift_side": self.lift_side,
            "observed": self.observed,
            "held_s": round(self.held_s, 2),
        }


class _Debounce:
    """Latch that is slow to switch on and quicker to switch off.

    One threshold with one comparison flickers: the signal sits on the boundary
    and the answer changes every frame, which downstream is a clip restart. Two
    thresholds with two hold times means the answer can only change when the
    evidence has actually been consistent for a while, and the two times differ
    because the two mistakes are not equally bad -- failing to act is safe,
    acting wrongly is several seconds of open-loop commitment.
    """

    __slots__ = ("enter_s", "exit_s", "state", "_since")

    def __init__(self, enter_s: float = ENTER_HOLD_S, exit_s: float = EXIT_HOLD_S):
        self.enter_s = enter_s
        self.exit_s = exit_s
        self.state = False
        self._since: float | None = None

    def update(self, now: float, evidence: bool) -> bool:
        if evidence == self.state:
            self._since = None
            return self.state
        if self._since is None:
            self._since = now
        needed = self.exit_s if self.state else self.enter_s
        if now - self._since >= needed:
            self.state = evidence
            self._since = None
        return self.state

    def reset(self) -> None:
        self.state = False
        self._since = None


class _PeakHold:
    """Running maximum that rises fast and decays very slowly.

    The subject's standing hip height cannot be known in advance and cannot be
    measured from one frame either, because any single frame may be a crouch. It
    CAN be bounded: crouching only ever brings the hips closer to the floor, so
    the running peak converges on the true standing height. The slow decay lets
    the estimate follow a genuinely different subject rather than latching on the
    first one forever.
    """

    __slots__ = ("value", "rise", "decay")

    def __init__(self, rise: float = 0.30, decay: float = 0.003):
        self.value = 0.0
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


@dataclass
class ActionCue:
    """Turns a stream of pose frames into a debounced lower-body action."""

    walk_enter_mps: float = WALK_ENTER_MPS
    walk_exit_mps: float = WALK_EXIT_MPS
    side_enter_mps: float = SIDE_ENTER_MPS
    side_exit_mps: float = SIDE_EXIT_MPS
    squat_enter: float = SQUAT_ENTER
    squat_exit: float = SQUAT_EXIT
    lift_enter: float = LIFT_ENTER
    lift_exit: float = LIFT_EXIT
    velocity_window_s: float = VELOCITY_WINDOW_S
    direction_window_s: float = DIRECTION_WINDOW_S
    min_travel_m: float = MIN_TRAVEL_M
    enter_hold_s: float = ENTER_HOLD_S
    exit_hold_s: float = EXIT_HOLD_S
    min_dwell_s: float = MIN_DWELL_S
    observe_grace_s: float = OBSERVE_GRACE_S
    stale_s: float = STALE_S

    _history: deque = field(default_factory=lambda: deque(maxlen=128), init=False)
    _standing_hip: _PeakHold = field(default_factory=_PeakHold, init=False)
    _gates: dict = field(default_factory=dict, init=False)
    _action: str = field(default="unknown", init=False)
    _action_since: float | None = field(default=None, init=False)
    _last_seen: float | None = field(default=None, init=False)
    _last_command: ActionCommand = field(default_factory=ActionCommand, init=False)

    def __post_init__(self) -> None:
        for name in ("walk_forward", "walk_backward", "step_left", "step_right",
                     "squat", "raise_left", "raise_right"):
            self._gates[name] = _Debounce(self.enter_hold_s, self.exit_hold_s)

    # -- public ------------------------------------------------------------
    def reset(self) -> None:
        self._history.clear()
        self._gates and [g.reset() for g in self._gates.values()]
        self._action = "unknown"
        self._action_since = None
        self._last_seen = None
        self._last_command = ActionCommand()

    def update(self, pose: PoseFrame) -> ActionCommand:
        now = float(pose.timestamp_s)
        measured = self._measure(pose.keypoints, now)

        if measured is None:
            return self._unobserved(now)

        self._last_seen = now
        self._history.append(measured)
        features = self._features(now)
        if features is None:
            # Seen, but not for long enough to have a velocity yet. Idle is the
            # honest answer: we can see the human and they have not yet done
            # anything we can measure.
            return self._settle(now, "idle", 0.4, "gathering motion history",
                                measured, 0.0, 0.0, 0.0)


        (forward, lateral, yaw_rate, slow_forward, slow_lateral,
         travel_forward, travel_lateral) = features
        # Speed says whether to move; the slower window says which way. Using
        # the fast sign here is what made walk_forward and walk_backward
        # alternate within a single stride.
        signed_forward = math.copysign(abs(forward), slow_forward or forward)
        signed_lateral = math.copysign(abs(lateral), slow_lateral or lateral)
        crouch, lift, lift_side = measured[4], measured[5], measured[6]

        # --- raw evidence, one per action --------------------------------
        travelling = max(abs(forward), abs(lateral))
        # Both witnesses, per axis: moving NOW and having actually gone somewhere.
        went_forward = travel_forward >= self.min_travel_m
        went_backward = -travel_forward >= self.min_travel_m
        went_left = travel_lateral >= self.min_travel_m
        went_right = -travel_lateral >= self.min_travel_m

        def walk_gate(name):
            return self._gate_level(name, self.walk_enter_mps, self.walk_exit_mps)

        def side_gate(name):
            return self._gate_level(name, self.side_enter_mps, self.side_exit_mps)

        raw = {
            "walk_forward": (signed_forward >= walk_gate("walk_forward")
                             and went_forward),
            "walk_backward": (-signed_forward >= walk_gate("walk_backward")
                              and went_backward),
            "step_left": (signed_lateral >= side_gate("step_left")
                          and went_left),
            "step_right": (-signed_lateral >= side_gate("step_right")
                           and went_right),
            # Posture actions require the human to be STILL. A crouch measured
            # mid-stride is a stride, not a squat, and asking the robot to squat
            # while the human walks is how two clips end up fighting.
            "squat": (crouch >= self._gate_level("squat", self.squat_enter,
                                                 self.squat_exit)
                      and travelling < self.walk_exit_mps),
            "raise_left": (lift_side > 0
                           and lift >= self._gate_level("raise_left", self.lift_enter,
                                                        self.lift_exit)
                           and travelling < self.walk_exit_mps),
            "raise_right": (lift_side < 0
                            and lift >= self._gate_level("raise_right", self.lift_enter,
                                                         self.lift_exit)
                            and travelling < self.walk_exit_mps),
        }
        latched = {name: gate.update(now, raw[name])
                   for name, gate in self._gates.items()}

        # --- pick one -----------------------------------------------------
        # Travel beats posture: a human who is moving across the floor wants the
        # robot to move across the floor, and a crouch read during that is part
        # of the stride. Within travel, the larger component wins, so a diagonal
        # resolves to its dominant axis instead of alternating between two clips.
        chosen, why = "idle", "still"
        if latched["walk_forward"] or latched["walk_backward"] or \
                latched["step_left"] or latched["step_right"]:
            if abs(forward) >= abs(lateral):
                chosen = "walk_forward" if signed_forward > 0 else "walk_backward"
                why = f"pelvis {signed_forward:+.2f} m/s along its own forward axis"
            else:
                chosen = "step_left" if signed_lateral > 0 else "step_right"
                why = f"pelvis {signed_lateral:+.2f} m/s sideways"
            if not latched[chosen]:
                # The dominant axis has not earned its gate yet; do not borrow
                # another axis's evidence to fire this one.
                chosen, why = "idle", "motion not yet sustained on either axis"
        elif latched["raise_left"] or latched["raise_right"]:
            chosen = "raise_left" if latched["raise_left"] else "raise_right"
            why = f"one foot {lift:.2f} leg-lengths clear, standing"
        elif latched["squat"]:
            chosen, why = "squat", f"hips {crouch:.2f} below standing, feet planted"

        confidence = self._confidence(chosen, forward, lateral, crouch, lift)
        return self._settle(now, chosen, confidence, why, measured,
                            signed_forward, signed_lateral, yaw_rate)

    # -- internals ---------------------------------------------------------
    def _gate_level(self, name: str, enter: float, exit_: float) -> float:
        """Enter threshold normally; the lower exit threshold once latched."""
        return exit_ if self._gates[name].state else enter

    def _unobserved(self, now: float) -> ActionCommand:
        """No usable landmarks this frame.

        Tolerating a short gap is essential -- a single dropped ankle must not
        erase the evidence, which is precisely the bug that produced 18.8% false
        march. Tolerating an unbounded gap is the opposite failure, so the grace
        period is short and what follows it is a full reset rather than a decay.
        """
        stale_for = math.inf if self._last_seen is None else now - self._last_seen
        if stale_for <= self.observe_grace_s:
            held = 0.0 if self._action_since is None else now - self._action_since
            return ActionCommand(
                action=self._last_command.action,
                confidence=self._last_command.confidence * 0.5,
                reason=f"landmarks lost {stale_for*1000:.0f} ms ago, inside grace",
                forward_mps=self._last_command.forward_mps,
                lateral_mps=self._last_command.lateral_mps,
                observed=False, held_s=held,
            )
        for gate in self._gates.values():
            gate.reset()
        self._history.clear()
        self._action = "unknown"
        self._action_since = None
        self._last_command = ActionCommand(
            action="unknown", confidence=0.0,
            reason=("nobody in frame" if stale_for is math.inf
                    else f"no landmarks for {stale_for:.1f} s"),
            observed=False)
        return self._last_command

    def _settle(self, now, chosen, confidence, why, measured,
                forward, lateral, yaw_rate) -> ActionCommand:
        """Apply the minimum dwell and emit."""
        if self._action_since is None:
            self._action, self._action_since = chosen, now
        elif chosen != self._action:
            held = now - self._action_since
            # Leaving for "idle" is always allowed: stopping is the safe
            # direction and must never wait on a dwell timer. Swapping one
            # ACTIVE clip for another does wait, because that is a restart.
            if chosen == "idle" or held >= self.min_dwell_s:
                self._action, self._action_since = chosen, now
            else:
                why = (f"holding {self._action} for another "
                       f"{self.min_dwell_s - held:.2f}s (min dwell)")
                chosen = self._action

        self._last_command = ActionCommand(
            action=chosen, confidence=confidence, reason=why,
            forward_mps=forward, lateral_mps=lateral, yaw_rate_dps=yaw_rate,
            crouch=measured[4], lift=measured[5], lift_side=measured[6],
            observed=True,
            held_s=0.0 if self._action_since is None else now - self._action_since,
        )
        return self._last_command

    def _confidence(self, action, forward, lateral, crouch, lift) -> float:
        """How far past its own gate the winning evidence sits, in [0, 1].

        Reported rather than thresholded, so the controller can require more
        certainty for an expensive action than for a cheap one without this
        module having to know which is which.
        """
        if action in ("idle", "unknown"):
            return 1.0 if action == "idle" else 0.0
        if action.startswith("walk"):
            return _clamp(abs(forward) / (2.0 * self.walk_enter_mps), 0.0, 1.0)
        if action.startswith("step"):
            return _clamp(abs(lateral) / (2.0 * self.side_enter_mps), 0.0, 1.0)
        if action == "squat":
            return _clamp(crouch / (2.0 * self.squat_enter), 0.0, 1.0)
        return _clamp(lift / (2.0 * self.lift_enter), 0.0, 1.0)

    def _measure(self, kps: dict[str, Keypoint], now: float):
        """Per-frame geometry, or None when the frame cannot support a verdict.

        Requires hips and shoulders: the hips are what travel is measured from
        and the shoulders give the body-forward axis, so without both there is no
        frame to measure travel IN. Ankles are optional -- losing them costs the
        lift and crouch readings but not the ability to see someone walk.
        """
        def seen(name: str) -> Keypoint | None:
            kp = kps.get(name)
            return kp if kp is not None and kp.visibility >= 0.5 else None

        lh, rh = seen("left_hip"), seen("right_hip")
        ls, rs = seen("left_shoulder"), seen("right_shoulder")
        if not (lh and rh and ls and rs):
            return None

        pelvis_x = 0.5 * (lh.x + rh.x)
        pelvis_z = 0.5 * (lh.z + rh.z)
        pelvis_y = 0.5 * (lh.y + rh.y)

        ux, uz = ls.x - rs.x, ls.z - rs.z
        span = math.hypot(ux, uz)
        if span < 1e-6:
            return None
        # Ground-plane normal to the shoulder line is the way the body faces --
        # the same construction gait_cues uses for its stride channel, so the two
        # cannot disagree about which way "forward" is.
        fwd_x, fwd_z = uz / span, -ux / span
        left_x, left_z = -fwd_z, fwd_x
        yaw = math.atan2(-uz, ux)

        la, ra = seen("left_ankle"), seen("right_ankle")
        crouch = lift = 0.0
        lift_side = 0
        if la and ra:
            # y is DOWN in the camera frame, so a larger y is closer to the floor.
            ground = max(la.y, ra.y)
            hip_height = ground - pelvis_y
            standing = self._standing_hip.update(hip_height)
            if standing > 1e-6:
                crouch = _clamp(1.0 - hip_height / standing, 0.0, 1.0)
            leg = max(hip_height, 1e-6)
            clearance = (ra.y - la.y) / leg      # + = left foot higher
            if abs(clearance) >= 1e-6:
                lift = abs(clearance)
                lift_side = 1 if clearance > 0 else -1

        return (now, pelvis_x, pelvis_z, yaw, crouch, lift, lift_side,
                fwd_x, fwd_z, left_x, left_z)

    def _velocity_over(self, window_s: float):
        """Body-frame ``(forward, lateral, yaw_rate)`` across ``window_s``.

        Projected onto the body axes of the CURRENT frame, so the reading follows
        a subject who turns while walking instead of assuming they move across
        the image.
        """
        latest = self._history[-1]
        oldest = None
        for sample in reversed(self._history):
            oldest = sample
            if latest[0] - sample[0] >= window_s:
                break
        if oldest is None or latest is oldest:
            return None
        dt = latest[0] - oldest[0]
        if dt <= 1e-6:
            return None
        dx = (latest[1] - oldest[1]) / 1000.0     # mm -> m
        dz = (latest[2] - oldest[2]) / 1000.0
        fwd_x, fwd_z, left_x, left_z = latest[7], latest[8], latest[9], latest[10]
        dyaw = (latest[3] - oldest[3] + math.pi) % (2.0 * math.pi) - math.pi
        return ((dx * fwd_x + dz * fwd_z) / dt,
                (dx * left_x + dz * left_z) / dt,
                math.degrees(dyaw) / dt)

    def _features(self, now: float):
        """``(forward, lateral, yaw_rate, slow_forward, slow_lateral)``.

        The fast pair decides whether there is motion; the slow pair decides
        which way it goes. See DIRECTION_WINDOW_S for why one window cannot do
        both jobs.
        """
        if len(self._history) < 2:
            return None
        fast = self._velocity_over(self.velocity_window_s)
        if fast is None:
            return None
        slow = self._velocity_over(self.direction_window_s)
        # Before the slow window has filled, fall back to the fast sign rather
        # than refusing to answer: at the very start of a walk that is the only
        # evidence there is, and the debouncer still has to be satisfied.
        slow_forward = fast[0] if slow is None else slow[0]
        slow_lateral = fast[1] if slow is None else slow[1]
        # Net displacement over the direction window, in metres. Reconstructed
        # from the mean velocity and the window it was measured over rather than
        # differenced again, so the two can never disagree about the same motion.
        span = self.direction_window_s if slow is not None else self.velocity_window_s
        return (fast[0], fast[1], fast[2], slow_forward, slow_lateral,
                slow_forward * span, slow_lateral * span)
