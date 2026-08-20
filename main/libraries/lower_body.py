"""Balance-aware lower-body controller: human leg pose -> safe NAO leg motion.

``nao_retarget.LowerBodyRetargeter`` answers *what the human's legs are doing*.
This module answers the separate, robot-side question: **how much of that may the
robot actually execute right now without falling over?**

Why the two are separated
-------------------------
A camera cannot see whether NAO's centre of mass is over a foot -- that depends on
the robot's own configuration and mass distribution. So the retargeter stays a
pure kinematic observer, and every stability decision is made here from the
robot's own state (``balance.NaoCoMModel`` forward kinematics, the InertialUnit
tilt, and the foot force sensors when the model provides them).

The step sequence
-----------------
Lifting a foot on a free-standing biped is not one action but three, and the
previous code skipped the first two -- which is why raising a leg in front of the
camera produced nothing:

1. **LOAD**  - lean the body so the centre of mass moves over the *stance* foot.
   The lean direction is not hard-coded: we probe both signs against the CoM
   model and keep the one that actually raises the stance-foot margin (a
   hard-coded lean sign is the classic way to make a balance controller tip
   *faster*, and this makes that failure impossible).
2. **SINGLE** - only once the model reports positive stance margin (and the foot
   force sensors, when present, confirm the transfer) does the swing leg start to
   follow the human's leg, its authority scaled *continuously* by that margin.
3. **UNLOAD** - the human lowers the leg, or the margin/tilt safety closes the
   gate; both blends ramp back down to the symmetric crouch.

Everything is expressed as two rate-limited blends (``shift`` and ``lift``), so
there are no discrete jumps and no state to get stuck in: the controller can
always ramp back to the exact symmetric crouch that is the project's proven
no-fall baseline.

Symmetric / antisymmetric split
-------------------------------
The commanded posture is ``crouch_posture(u)`` -- the statically-balanced squat
whose ``Hip + Knee + Ankle == 0`` keeps the torso vertical and the soles flat --
plus the human's *deviation from* that posture, per leg, authority-weighted.

The weight is not one number, because two very different things live in those
deviations. Splitting each channel into its mirror-symmetric and antisymmetric
parts separates them exactly:

* **Mirror-symmetric** -- both legs abducting outward by the same amount (a wider
  stance), or both flexing equally (a squat). By symmetry these move the centre of
  mass *not at all*, and a wider stance makes the support polygon **bigger**. They
  are therefore safer than standing, and get full authority. Gating them was a
  plain mistake: it turned a 40 deg human stance into a 12 deg robot one.
* **Antisymmetric** -- both legs rolled the same way (a lean), or one leg forward
  and one back (a split stance). These do move the centre of mass over the feet,
  so they stay gated.

While a foot is genuinely off the ground the split is dropped: the swing leg is
unloaded and therefore free to take its own pose at whatever authority the safety
gate allows, and the stance leg is left near the balanced crouch because it is
carrying the whole robot.

Standing still yields a deviation of exactly zero either way, so the controller
degrades to the proven baseline rather than to noise.

What actually limits a wide stance
----------------------------------
Not the hip. NAO's ``HipRoll`` reaches 45.3 deg, but ``AnkleRoll`` only reaches
22.8 deg -- and the ankle is what levels the sole against the hip's abduction.
Past 22.8 deg the sole can no longer be kept flat.

Refusing to go past that point turned out to be too strict: recorded runs show
subjects spreading to ~25 deg routinely and 34 deg at the extreme, so the robot
saturated just below the human and it read as "it spreads, but not as much as
me". So a small, explicit ``sole_tilt_budget`` is spent instead: the hip may
abduct that much further than the ankle can level, leaving each sole a few
degrees off flat and the robot standing on the inner part of each foot, which is
a good trade for the extra width. The budget is then enforced as a *post-
condition* on the commanded angles (:meth:`_limit_sole_tilt`), so it holds no
matter how the symmetric, antisymmetric and balance terms happen to add up -- and
when something has to give, it is stance width, never sole contact.

Pure Python + the (optional) NumPy CoM model, no Webots import, so all of the
sequencing and gating logic is unit-testable off-simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from nao_retarget import LegTarget, LowerBodyObservation, crouch_posture
from pose_control_utils import JointLimiter, get_default_motor_configs

_CONFIGS = get_default_motor_configs()
# How far the ankle can roll to level a sole against the hip's abduction. Smaller
# than the hip's own range, which is what makes it -- not the hip -- the binding
# constraint on stance width.
ANKLE_ROLL_LIMIT = min(abs(_CONFIGS["LAnkleRoll"].min_angle),
                       _CONFIGS["RAnkleRoll"].max_angle)

# The 12 leg joints this controller owns while it is active.
LEG_JOINTS = (
    "LHipYawPitch", "RHipYawPitch",
    "LHipRoll", "RHipRoll",
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
    "LAnkleRoll", "RAnkleRoll",
)

MODE_DOUBLE = "double"
MODE_LOAD = "load"
MODE_SINGLE = "single"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class LowerBodyParams:
    """Tuning for :class:`LowerBodyController` (radians / seconds)."""

    # -- posture -----------------------------------------------------------
    base_crouch_u: float = 0.10     # knees never fully locked: leaves the
                                    # balance loop some authority to work with
    # Deepest squat commanded. See nao_retarget.MAX_CROUCH: the crouch posture is
    # statically balanced at any depth, so this is a range limit, not a safety one.
    max_crouch_u: float = 0.70

    # -- how much human detail we allow through, per channel ---------------
    max_hip_pitch_dev: float = 1.25
    max_hip_roll_dev: float = 0.7906  # rad ~ 45.3 deg, NAO HipRoll hardware limit
    max_knee_dev: float = 1.45
    max_ankle_dev: float = 0.60
    # Mirror-symmetric poses (wider stance, deeper squat) are CoM-neutral and
    # widen the support polygon, so they pass at full authority.
    symmetric_gain: float = 1.0
    # How far out of flat each sole is allowed to end up. A sole is flat when
    # AnkleRoll == -HipRoll, and the ankle's range is smaller than the hip's, so a
    # wide stance necessarily tilts the soles a little. Spending a small, bounded
    # amount of tilt buys real stance width: at 0.15 rad the outer edge of a
    # 76 mm-wide foot lifts about 11 mm, so the robot stands on the inner part of
    # each sole while the two soles are far further apart than before.
    sole_tilt_budget: float = 0.15
    # Largest both-legs-outward abduction we ask for: what the ankle can level,
    # plus that tilt budget. Recorded runs show subjects reaching ~25 deg (p99)
    # and 34 deg (peak), so the ankle limit alone (22.8 deg) saturated just below
    # what people actually do -- which read as "it spreads, but not like me".
    max_abduction: float = ANKLE_ROLL_LIMIT + 0.15
    # Antisymmetric poses (a lean, a split stance) do move the CoM, and in double
    # support there is no single-foot polygon to verify them against, so they are
    # deliberately limited.
    asymmetric_gain: float = 0.35

    # -- weight transfer ---------------------------------------------------
    shift_rad: float = 0.16         # lean amplitude that loads the stance foot
    shift_rate: float = 1.1         # 1/s ramp of the shift blend
    shift_ready: float = 0.75       # shift blend required before lifting starts
    lift_rate: float = 1.8          # 1/s ramp of the lift blend
    lift_start: float = 0.15        # human lift fraction that requests a step
    lift_stop: float = 0.07         # hysteresis: below this the foot comes down

    # -- safety gates ------------------------------------------------------
    margin_min: float = 0.002       # m; stance margin needed to begin lifting
    # Margin at which the full requested lift is allowed. A completed weight
    # transfer yields ~0.015 m, so anything larger silently caps the lift below
    # the human's -- which read as "the leg only moves a little".
    margin_full: float = 0.012
    fsr_load_frac: float = 0.55     # stance share at which the FSRs fully confirm
    # Authority retained when the foot sensors do NOT confirm the transfer. The
    # FSRs are a *confirmation*, never a veto: a sensor that reads a constant
    # 50/50 (uncalibrated, or a proto whose soles barely redistribute) would
    # otherwise forbid every step forever, which is exactly how a leg lift ends
    # up doing nothing at all. The CoM model has already agreed by this point and
    # the tilt abort is the real safety net.
    fsr_min_gain: float = 0.40
    fsr_total_min: float = 1.0      # N; below this the FSR reading is ignored
    tilt_abort_rad: float = 0.28    # |IMU roll/pitch| beyond this -> stand down
    conf_min: float = 0.50          # min lower-body landmark confidence

    # -- standing turn -----------------------------------------------------
    # Shared-hip-yaw bias used for immediate visual feedback while a stepping
    # turn is unavailable. Deliberately tiny: NAO's HipYawPitch axis is canted
    # 45 deg, so it splays the legs as well as yawing the pelvis.
    max_yaw_bias: float = 0.12

    # -- misc --------------------------------------------------------------
    max_dt_s: float = 0.1
    # Lift authority when no CoM model is available (numpy missing). We do not
    # refuse to move -- that is what made the legs look dead -- but we cap the
    # motion hard and the tilt abort remains the safety net.
    ungated_lift_cap: float = 0.35
    # Probe amplitude used to discover the lean sign from the CoM model.
    probe_rad: float = 0.10
    probe_refresh_s: float = 0.25


@dataclass
class LowerBodyState:
    shift: float = 0.0        # [0, 1] weight-transfer blend
    lift: float = 0.0         # [0, 1] swing-leg authority
    stance: str = ""          # "L" / "R" / "" (double support)
    last_now: Optional[float] = None


class LowerBodyController:
    """Turns a :class:`LowerBodyObservation` into safe NAO leg targets.

    Usage (per simulation step, whether or not a fresh camera frame arrived)::

        targets, meta = controller.step(now_s, observation,
                                        torso_rp=(roll, pitch),
                                        fsr={"L": fz, "R": fz},
                                        measured=current_joint_angles)

    ``meta["balance_ok"]`` tells the caller whether the *symmetric* CoM balance
    correction from ``balance.BalanceController`` is still a valid thing to add
    on top (it is not, once we are deliberately leaning onto one foot).
    """

    def __init__(
        self,
        params: Optional[LowerBodyParams] = None,
        *,
        com_model: Optional[object] = None,
        limiter: Optional[JointLimiter] = None,
    ) -> None:
        self.params = params or LowerBodyParams()
        self.limiter = limiter or JointLimiter(get_default_motor_configs())
        self.com_model = com_model
        self.state = LowerBodyState()
        self._probe: Dict[str, Tuple[float, float]] = {}  # stance -> (dir, t)
        self._last_obs = LowerBodyObservation()
        self._fsr_share: Optional[float] = None

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Drop all blend state (call after an external whole-body motion).

        A motion clip leaves the robot in a completely different configuration,
        so a half-finished weight transfer from before the clip must not be
        resumed on top of it.
        """
        self.state = LowerBodyState()
        self._probe.clear()

    def set_observation(self, obs: Optional[LowerBodyObservation]) -> None:
        """Latch the newest camera observation (control runs at sim rate)."""
        if obs is not None:
            self._last_obs = obs

    def stand_down(self) -> None:
        """Forget the latched observation so the sequencer ramps to the crouch.

        Call this when tracking goes stale. The latch exists so control keeps
        running at simulation rate between camera frames -- but without an
        expiry the controller would go on acting on a snapshot of a human who
        has left, holding a one-legged stance indefinitely. The blends still ramp
        down rather than snapping, so the foot is set down, not dropped.
        """
        self._last_obs = LowerBodyObservation()

    def step(
        self,
        now_s: float,
        obs: Optional[LowerBodyObservation] = None,
        *,
        torso_rp: Tuple[float, float] = (0.0, 0.0),
        fsr: Optional[Dict[str, float]] = None,
        measured: Optional[Dict[str, float]] = None,
        yaw_bias: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Advance the sequencer one control step and emit leg targets."""
        p = self.params
        st = self.state
        if obs is not None:
            self._last_obs = obs
        obs = self._last_obs

        dt = 0.0 if st.last_now is None else _clamp(now_s - st.last_now, 0.0, p.max_dt_s)
        st.last_now = now_s

        roll, pitch = torso_rp
        tilt_ok = abs(roll) < p.tilt_abort_rad and abs(pitch) < p.tilt_abort_rad
        usable = bool(obs.valid and obs.confidence >= p.conf_min and tilt_ok)

        swing = self._requested_swing(obs) if usable else ""
        if swing:
            st.stance = "R" if swing == "L" else "L"
        elif st.lift <= 1e-3 and st.shift <= 1e-3:
            st.stance = ""

        # --- weight transfer blend ---------------------------------------
        shift_target = 1.0 if swing else 0.0
        st.shift = _approach(st.shift, shift_target, p.shift_rate * dt)

        # --- lift blend, gated by the robot's own balance state -----------
        gate, margin = self._lift_gate(st.stance, measured, fsr)
        human_lift = 0.0
        if swing:
            leg = obs.leg(swing)
            human_lift = leg.lift if leg is not None else 0.0
        ready = st.shift >= p.shift_ready
        lift_target = (human_lift * gate) if (swing and ready) else 0.0
        st.lift = _approach(st.lift, lift_target, p.lift_rate * dt)

        mode = (
            MODE_SINGLE if st.lift > 0.05
            else (MODE_LOAD if st.shift > 0.05 else MODE_DOUBLE)
        )

        # Keep the lean sign fresh for whichever foot is currently the stance
        # foot (cheap: cached for ``probe_refresh_s``).
        if st.stance:
            self._shift_direction(st.stance, measured, now_s)

        targets = self._compose(
            obs, swing, yaw_bias if mode == MODE_DOUBLE else 0.0, usable
        )
        clamped = {n: self.limiter.clamp_angle(n, v) for n, v in targets.items()}
        self._limit_sole_tilt(clamped)
        meta: Dict[str, object] = {
            "why": self._explain(obs, tilt_ok, swing, gate, human_lift),
            "lift_source": obs.lift_source,
            "mode": mode,
            "single_support": mode == MODE_SINGLE,
            "balance_ok": mode == MODE_DOUBLE and st.shift < 0.05,
            "swing_side": swing,
            "stance_side": st.stance,
            "shift": round(st.shift, 4),
            "lift": round(st.lift, 4),
            "gate": round(gate, 4),
            "stance_margin": round(margin, 5),
            "fsr_share": (None if self._fsr_share is None
                          else round(self._fsr_share, 3)),
            "human_lift": round(human_lift, 4),
            "crouch_u": round(self._crouch_u(obs), 4),
            "crouch_cue": round(obs.crouch_u, 4),
            "tilt_ok": tilt_ok,
            "tracking": usable,
        }
        return clamped, meta

    # ------------------------------------------------------------- internals
    def _explain(self, obs: LowerBodyObservation, tilt_ok: bool, swing: str,
                 gate: float, human_lift: float) -> str:
        """One short phrase naming the current limiting factor.

        Worth its keep: "the legs are not moving" has half a dozen legitimate
        causes (nobody in frame, legs cropped, low confidence, the CoM not yet
        over the stance foot) and they are indistinguishable from a bug unless
        the controller says which one it is.
        """
        p = self.params
        st = self.state
        if not obs.valid:
            return "no lower-body landmarks in frame"
        if obs.confidence < p.conf_min:
            return f"lower-body confidence {obs.confidence:.2f} < {p.conf_min:.2f}"
        if not tilt_ok:
            return "torso tilted past the safety limit; standing down"
        if obs.lift_source == "none":
            return "knees and feet both out of frame; leg lift cannot be seen"
        if not swing:
            if human_lift <= 0.0 and st.lift <= 1e-3:
                return "tracking (no leg lift requested)"
            return "returning the foot to the ground"
        if st.shift < p.shift_ready:
            return f"transferring weight onto the {st.stance or '?'} foot"
        if gate <= 0.0:
            return "holding: centre of mass not yet over the stance foot"
        if self._fsr_share is not None and self._fsr_share < p.fsr_load_frac:
            return (f"stepping at reduced authority: foot sensors report only "
                    f"{self._fsr_share * 100:.0f}% of the load on the "
                    f"{st.stance} foot")
        return f"stepping ({st.lift * 100:.0f}% of the requested lift)"

    def _crouch_u(self, obs: LowerBodyObservation) -> float:
        """Symmetric squat depth to command, in radians.

        A squat has exactly ONE degree of freedom: the crouch posture ties
        ``Hip = -d``, ``Knee = +2d``, ``Ankle = -d`` together so the torso stays
        vertical and the soles flat. So the depth is read off the leg solve as a
        single number and the three joints are rebuilt from it, rather than
        letting three independently-clamped channels drift out of that relation
        (and rather than adding the squat twice -- once here and again as a
        "symmetric pitch deviation", which is what it used to do).

        The depth is the SMALLEST reading available, across two axes and both
        legs, because every other reading can be inflated by something that is
        not a squat:

        * hip vs. knee -- bending at the WAIST flexes the hip relative to the
          torso while the knees stay straight, and NAO has no torso joint to
          render that with. Requiring the knee to agree turns a waist hinge into
          no squat, correctly.
        * left vs. right -- a raised leg is deeply flexed at both hip and knee,
          so averaging the two legs made lifting one knee also squat the robot.
          The straighter leg is the one bearing the weight, and it is the one
          that says how low the body actually is.
        """
        p = self.params
        legs = [lg for lg in (obs.left, obs.right) if lg is not None] if obs.valid else []
        depth = 0.0
        if legs:
            depth = min(min(-lg.hip_pitch, 0.5 * lg.knee_pitch) for lg in legs)
        # base_crouch_u is a floor, not a target: fully locked knees leave the
        # balance loop nothing to work with (KneePitch bottoms out at -5.3 deg).
        return _clamp(max(depth, p.base_crouch_u), 0.0, p.max_crouch_u)

    def _requested_swing(self, obs: LowerBodyObservation) -> str:
        """Which foot (if any) the human is asking the robot to lift."""
        p = self.params
        lifts = {s: (obs.leg(s).lift if obs.leg(s) is not None else 0.0) for s in ("L", "R")}
        side = "L" if lifts["L"] >= lifts["R"] else "R"
        # Hysteresis: it takes a clear lift to start a step and a clear return
        # to the ground to end it, so a foot hovering near the threshold does
        # not chatter the weight transfer.
        already = self.state.lift > 1e-3 or self.state.shift > 1e-3
        threshold = p.lift_stop if already else p.lift_start
        if lifts[side] < threshold:
            return ""
        # Both feet "lifted" is not a step -- it is a jump, or bad tracking.
        other = "R" if side == "L" else "L"
        if lifts[other] >= threshold and abs(lifts[side] - lifts[other]) < 0.08:
            return ""
        return side

    def _lift_gate(
        self,
        stance: str,
        measured: Optional[Dict[str, float]],
        fsr: Optional[Dict[str, float]],
    ) -> Tuple[float, float]:
        """Return ``(gate, stance_margin_m)`` -- how much lift is safe right now.

        ``gate`` scales the swing leg's authority continuously from 0 (CoM not
        over the stance foot: do not unload it) to 1 (comfortably over it), so
        the step degrades smoothly instead of snapping on and off.
        """
        p = self.params
        self._fsr_share = None
        if not stance:
            return 0.0, 0.0

        # --- model gate: is the CoM over the stance foot? ------------------
        margin = 0.0
        if self.com_model is None or measured is None:
            gate = p.ungated_lift_cap   # nothing to prove safety with; move a little
        else:
            try:
                margin = float(self.com_model.stance_margin(measured, stance))
            except Exception:  # noqa: BLE001 - no stance_margin / bad state
                gate = p.ungated_lift_cap
            else:
                span = max(p.margin_full - p.margin_min, 1e-6)
                gate = _clamp((margin - p.margin_min) / span, 0.0, 1.0)

        # --- foot sensors: a confirmation, scaled -- never a veto ----------
        if fsr:
            total = float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
            if total > p.fsr_total_min:
                share = float(fsr.get(stance, 0.0)) / total
                self._fsr_share = share
                span = max(p.fsr_load_frac - 0.5, 1e-6)
                confirm = _clamp((share - 0.5) / span, 0.0, 1.0)
                gate *= p.fsr_min_gain + (1.0 - p.fsr_min_gain) * confirm
        return gate, margin

    def _shift_direction(self, stance: str, measured: Optional[Dict[str, float]],
                         now_s: float) -> float:
        """Sign of the same-sign roll lean that loads ``stance``.

        Discovered from the CoM model instead of hard-coded (see the module
        docstring). Cached briefly because it only changes with the posture.
        """
        p = self.params
        cached = self._probe.get(stance)
        if cached is not None and (now_s - cached[1]) < p.probe_refresh_s:
            return cached[0]

        # Documented fallback if there is no model to ask: leaning the hips
        # toward +roll carries the pelvis away from that side, so loading the
        # left foot needs a negative same-sign roll.
        direction = -1.0 if stance == "L" else 1.0
        if self.com_model is not None and measured:
            best_margin = -math.inf
            for d in (1.0, -1.0):
                probe = dict(measured)
                for j in ("LHipRoll", "RHipRoll"):
                    probe[j] = probe.get(j, 0.0) + d * p.probe_rad
                for j in ("LAnkleRoll", "RAnkleRoll"):
                    probe[j] = probe.get(j, 0.0) - d * p.probe_rad
                try:
                    m = float(self.com_model.stance_margin(probe, stance))
                except Exception:  # noqa: BLE001
                    best_margin = -math.inf
                    break
                if m > best_margin:
                    best_margin, direction = m, d
        self._probe[stance] = (direction, now_s)
        return direction

    def _compose(
        self, obs: LowerBodyObservation, swing: str, yaw_bias: float, usable: bool
    ) -> Dict[str, float]:
        """Symmetric crouch + authority-weighted per-leg human deviation + lean.

        ``usable`` gates only the ANTIsymmetric part of the double-support
        deviation (see ``_apply_symmetric``): a lean/stride asymmetry is what
        tips the robot, so it must not keep being reapplied once tilt_ok is
        already False, the very case the "standing down" gate exists for. The
        symmetric part (wider stance) is CoM-neutral and stays at full
        authority regardless, per the module docstring.
        """
        p = self.params
        st = self.state
        u = self._crouch_u(obs)
        targets = dict(crouch_posture(u))

        left = obs.leg("L") if obs.valid else None
        right = obs.leg("R") if obs.valid else None
        dev_l = self._deviation(left, "L", u) if left is not None else None
        dev_r = self._deviation(right, "R", u) if right is not None else None

        stepping = bool(swing) and st.lift > 1e-4
        if stepping:
            # A foot that is off the ground is unloaded, so it is free to take the
            # human's own pose at whatever authority the safety gate allows. The
            # stance leg is carrying the robot, so it stays near the balanced
            # crouch -- sharing the swing leg's pose onto it would move the very
            # foot the CoM is standing on.
            stance = "R" if swing == "L" else "L"
            for side, dev, weight in (
                (swing, dev_l if swing == "L" else dev_r, st.lift),
                (stance, dev_l if stance == "L" else dev_r, p.asymmetric_gain),
            ):
                if dev is None or weight <= 1e-4:
                    continue
                for name, value in dev.items():
                    targets[name] = targets.get(name, 0.0) + weight * value
        else:
            self._apply_symmetric(targets, dev_l, dev_r, usable)

        if st.shift > 1e-4 and st.stance:
            lean = p.shift_rad * st.shift * self._shift_dir_cached(st.stance)
            for name in ("LHipRoll", "RHipRoll"):
                targets[name] = targets.get(name, 0.0) + lean
            for name in ("LAnkleRoll", "RAnkleRoll"):
                targets[name] = targets.get(name, 0.0) - lean

        if abs(yaw_bias) > 1e-4:
            bias = _clamp(yaw_bias, -p.max_yaw_bias, p.max_yaw_bias)
            targets["LHipYawPitch"] = targets.get("LHipYawPitch", 0.0) + bias
            targets["RHipYawPitch"] = targets.get("RHipYawPitch", 0.0) + bias
        return targets

    # NAO's left/right sign conventions differ per axis: the ROLL channels are
    # mirrored (LHipRoll positive and RHipRoll negative both mean "outward"), the
    # PITCH channels are not. So a mirror-symmetric human pose shows up as
    # numerically opposite roll values and numerically equal pitch values.
    _MIRRORED_CHANNELS = ("HipRoll", "AnkleRoll")
    _ALIGNED_CHANNELS = ("HipPitch", "KneePitch", "AnklePitch")

    def _apply_symmetric(
        self,
        targets: Dict[str, float],
        dev_l: Optional[Dict[str, float]],
        dev_r: Optional[Dict[str, float]],
        usable: bool,
    ) -> None:
        """Add the human's double-support deviation, split by symmetry.

        The mirror-symmetric half (wider stance, deeper squat) is CoM-neutral and
        enlarges the support polygon, so it passes at full authority regardless of
        ``usable``; the antisymmetric half (a lean, a split stance -- exactly the
        kind of asymmetry that tips the robot) moves the CoM and is gated to zero
        whenever ``usable`` is False, e.g. because tilt_ok already tripped. It must
        not keep reapplying a lean-inducing signal once the robot is already
        "standing down" for being too tilted -- that starves the CoM balance
        correction of ever catching up. See the module docstring.
        """
        p = self.params
        asym_gain = p.asymmetric_gain if usable else 0.0
        for channel, mirrored in (
            *((c, True) for c in self._MIRRORED_CHANNELS),
            *((c, False) for c in self._ALIGNED_CHANNELS),
        ):
            left = dev_l.get("L" + channel) if dev_l else None
            right = dev_r.get("R" + channel) if dev_r else None
            if left is None or right is None:
                # Only one leg in view: there is no way to tell a symmetric pose
                # from a lean, so read the whole thing as antisymmetric, which is
                # the conservative interpretation.
                for side, value in (("L", left), ("R", right)):
                    if value is not None:
                        targets[side + channel] = (
                            targets.get(side + channel, 0.0)
                            + asym_gain * value
                        )
                continue

            if mirrored:
                symmetric = 0.5 * (left - right)   # both legs outward by this much
                antisym = 0.5 * (left + right)     # both rolled the same way = lean
                symmetric = _clamp(symmetric, -p.max_abduction, p.max_abduction)
                l_sym, r_sym = symmetric, -symmetric
            else:
                # The symmetric part of a pitch channel IS the squat, and the
                # crouch posture already carries it (see _crouch_u). Adding it
                # again double-counted the squat and let a straight-legged human
                # cancel the base crouch entirely, locking the knees.
                antisym = 0.5 * (left - right)     # one forward, one back
                l_sym = r_sym = 0.0

            targets["L" + channel] = (
                targets.get("L" + channel, 0.0)
                + p.symmetric_gain * l_sym + asym_gain * antisym
            )
            targets["R" + channel] = (
                targets.get("R" + channel, 0.0)
                + p.symmetric_gain * r_sym
                + asym_gain * (antisym if mirrored else -antisym)
            )

    def apply_sole_tilt_limit(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Keep each sole within ``sole_tilt_budget`` of flat, in place.

        A sole is flat when ``AnkleRoll == -HipRoll``. When the budget is exceeded
        the HIP gives way, not the ankle: standing on the edge of a foot is what
        tips the robot, and losing a little stance width is cheap by comparison.

        Call this **last**, immediately before commanding the motors. The layer
        applies it to its own output, but the caller then folds in the CoM balance
        correction, whose roll terms are not tilt-neutral (``balance.py`` searches
        hip and ankle roll independently) -- so the guarantee only actually holds
        at the final commanded values if it is re-applied there. Idempotent.
        """
        self._limit_sole_tilt(targets)
        return targets

    def _limit_sole_tilt(self, targets: Dict[str, float]) -> None:
        """In-place implementation of :meth:`apply_sole_tilt_limit`."""
        budget = self.params.sole_tilt_budget
        for side in ("L", "R"):
            hip_name, ankle_name = f"{side}HipRoll", f"{side}AnkleRoll"
            hip = targets.get(hip_name)
            ankle = targets.get(ankle_name)
            if hip is None or ankle is None:
                continue
            tilt = hip + ankle
            if abs(tilt) <= budget:
                continue
            targets[hip_name] = self.limiter.clamp_angle(
                hip_name, hip - (tilt - math.copysign(budget, tilt))
            )

    def _shift_dir_cached(self, stance: str) -> float:
        cached = self._probe.get(stance)
        return cached[0] if cached is not None else (-1.0 if stance == "L" else 1.0)

    def _deviation(self, leg: LegTarget, side: str, u: float) -> Dict[str, float]:
        """The human leg pose minus the symmetric crouch, per-channel capped."""
        p = self.params
        base_hip, base_knee, base_ankle = -u, 2.0 * u, -u
        return {
            f"{side}HipPitch": _clamp(leg.hip_pitch - base_hip,
                                      -p.max_hip_pitch_dev, p.max_hip_pitch_dev),
            f"{side}HipRoll": _clamp(leg.hip_roll,
                                     -p.max_hip_roll_dev, p.max_hip_roll_dev),
            f"{side}KneePitch": _clamp(leg.knee_pitch - base_knee,
                                       -p.max_knee_dev, p.max_knee_dev),
            f"{side}AnklePitch": _clamp(leg.ankle_pitch - base_ankle,
                                        -p.max_ankle_dev, p.max_ankle_dev),
            f"{side}AnkleRoll": _clamp(leg.ankle_roll,
                                       -p.max_hip_roll_dev, p.max_hip_roll_dev),
        }


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta``."""
    if max_delta <= 0.0:
        return current
    return _clamp(current + _clamp(target - current, -max_delta, max_delta), 0.0, 1.0)


def default_lower_body_params() -> LowerBodyParams:
    return LowerBodyParams()
