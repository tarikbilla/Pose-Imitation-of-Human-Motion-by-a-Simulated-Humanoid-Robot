"""On-robot walk engine: a gait *command* -> balance-stable NAO leg motion.

The human pose pipeline cannot hand the robot leg *angles* to walk — monocular
depth is unreliable and copying leg angles topples a free-standing NAO. Instead
``src/perception/gait_cues`` distils the human's motion into a small gait command
(cadence / phase / swing-side / intensity / stop), and THIS engine turns that
command into leg joint targets the NAO can actually execute while staying up.

Two tiers, behind config flags
-------------------------------
* **Tier A — double-support "weight-shift march"** (the safe, graded default).
  The robot pumps its knees alternately and shifts its CoM laterally *in time
  with the human's cadence and phase*, but **never fully unloads a foot**. Both
  feet stay loaded, so the existing symmetric CoM balance loop (``balance.py``)
  remains valid and the no-fall baseline is never regressed. It reads clearly as
  marching / walking-in-place and tracks the human precisely on the only signals
  a frontal monocular camera observes continuously (cadence, phase, stop).

* **Tier B — single-support stepping** (experimental, flag-gated, default OFF).
  True foot lift / weight transfer. Only ever lifts a foot when the model
  predicts the CoM is safely over the *stance* foot (and, when wired, the foot
  force sensors confirm the weight transfer). On any doubt it collapses the step
  height to zero and stays in the Tier-A double-support regime. Single-support on
  a free-standing NAO with on-board sensing only is genuinely unproven, so this
  tier is isolated from the demo until proven in simulation.

Design invariants
-----------------
* The engine integrates its **own phase clock from simulation time** (not from
  pose frames), so the gait keeps a steady cadence even when pose packets stall,
  and it *phase-locks* to the human with a bounded slew so left/right stays
  aligned with the human's steps.
* When the command says stop (idle / low confidence / stale) or the torso tilts
  past a safety threshold, an ``amp_gain`` ramps smoothly to 0, returning the
  legs to the exact symmetric crouch — it **never freezes mid-step**.
* Pure Python + NumPy, **no Webots import**, so the gait math is unit-testable
  off-simulation exactly like ``balance.py``. The driver owns the motors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from pose_control_utils import JointLimiter, get_default_motor_configs

try:  # CoM model is used only for the Tier-B stance-aware safety gate.
    from balance import NaoCoMModel
except Exception:  # noqa: BLE001 - keep the engine importable without numpy
    NaoCoMModel = None  # type: ignore

TWO_PI = 2.0 * math.pi

# The 12 leg joints the engine may command.
LEG_JOINTS = (
    "LHipYawPitch", "RHipYawPitch",
    "LHipRoll", "RHipRoll",
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
    "LAnkleRoll", "RAnkleRoll",
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % TWO_PI - math.pi


@dataclass
class GaitParams:
    """Tuning for :class:`GaitEngine` (all radians / seconds unless noted)."""
    # Posture
    base_crouch_u: float = 0.15     # symmetric standing crouch (Hip=-u,Knee=2u,Ankle=-u)
    # Tier A (double-support march) amplitudes — kept small so both feet stay loaded
    knee_bob_rad: float = 0.18      # extra knee/hip flex of the unloaded leg per step
    sway_rad: float = 0.06          # lateral CoM sway (same-sign roll on both legs)
    # Tier B (single-support step) amplitudes — only used when tier == "step"
    step_height_rad: float = 0.35   # swing-leg knee flex at mid-swing (foot clearance)
    step_shift_rad: float = 0.12    # asymmetric roll to load the stance foot
    # Turning (experimental, low gain) — shared HipYawPitch
    turn_rad: float = 0.10
    # Timing / dynamics
    cadence_max_hz: float = 0.9     # cap on the robot's gait-cycle frequency
    cadence_slew_hz_per_s: float = 1.5  # how fast cadence can change
    phase_lock_gain: float = 2.0    # rad/s bounded pull of robot phase -> human phase
    amp_rise_s: float = 0.6         # ramp-up time of amp_gain (slow to start)
    amp_decay_s: float = 0.35       # ramp-down time of amp_gain (fast to stop)
    # Safety
    tilt_abort_rad: float = 0.35    # |IMU roll/pitch| beyond this -> decay to crouch
    conf_min: float = 0.6           # min command confidence to keep walking
    # Tier B gates
    stance_margin_min: float = 0.005  # m; CoM must be this far inside stance polygon
    fsr_load_frac: float = 0.6      # stance foot must carry this fraction of body weight
    max_dt_s: float = 0.1           # clamp on the per-step time delta


@dataclass
class GaitState:
    phase: float = 0.0       # robot gait phase [0, 2pi)
    amp_gain: float = 0.0    # [0, 1] blend crouch(0) <-> full gait(1)
    cadence: float = 0.0     # current robot cadence (Hz)
    last_now: Optional[float] = None


class GaitEngine:
    """Turns a gait command + sim time into NAO leg targets (radians).

    Usage (per simulation step)::

        engine.set_command(gait_dict)            # when a fresh command arrives
        targets, meta = engine.step(now_s, tier="march", torso_rp=(roll, pitch))

    ``targets`` is a dict of leg-joint -> angle (clamped to NAO limits). ``meta``
    reports ``single_support`` (bool), ``swing_side`` (-1/0/+1), ``amp_gain``,
    ``phase`` and ``cadence`` so the driver can hand roll authority to the gait
    during single support and so callers can log/telemeter.
    """

    def __init__(
        self,
        params: Optional[GaitParams] = None,
        *,
        com_model: Optional[object] = None,
        limiter: Optional[JointLimiter] = None,
    ) -> None:
        self.params = params or GaitParams()
        self.limiter = limiter or JointLimiter(get_default_motor_configs())
        self.state = GaitState()
        # CoM model for the Tier-B stance gate; optional (degrades to no-lift).
        if com_model is not None:
            self.com_model = com_model
        elif NaoCoMModel is not None:
            try:
                self.com_model = NaoCoMModel()
            except Exception:  # noqa: BLE001
                self.com_model = None
        else:
            self.com_model = None

        # Latest command (defaults = idle/stand).
        self._cmd_state = "idle"
        self._cmd_cadence = 0.0
        self._cmd_phase = 0.0
        self._cmd_swing = 0
        self._cmd_intensity = 0.0
        self._cmd_turn = 0.0
        self._cmd_conf = 0.0

    # -- command ingestion --------------------------------------------------
    def set_command(self, gait: Optional[Dict[str, object]]) -> None:
        """Update the target gait from a (possibly partial) command dict."""
        if not gait:
            self._cmd_state = "idle"
            self._cmd_cadence = 0.0
            self._cmd_intensity = 0.0
            self._cmd_conf = 0.0
            return
        self._cmd_state = str(gait.get("state", "idle"))
        self._cmd_cadence = float(gait.get("cadence_hz", 0.0) or 0.0)
        self._cmd_phase = float(gait.get("phase", 0.0) or 0.0) % TWO_PI
        self._cmd_swing = int(gait.get("swing_side", 0) or 0)
        self._cmd_intensity = _clamp(float(gait.get("intensity", 0.0) or 0.0), 0.0, 1.0)
        self._cmd_turn = _clamp(float(gait.get("turn", 0.0) or 0.0), -1.0, 1.0)
        self._cmd_conf = _clamp(float(gait.get("conf", 0.0) or 0.0), 0.0, 1.0)

    # -- per-step update ----------------------------------------------------
    def step(
        self,
        now_s: float,
        *,
        tier: str = "march",
        torso_rp: Tuple[float, float] = (0.0, 0.0),
        fsr: Optional[Dict[str, float]] = None,
        measured: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Advance the gait one simulation step and emit leg targets.

        ``tier``      : "stand" | "march" (Tier A) | "step" (Tier B).
        ``torso_rp``  : measured (roll, pitch) from the IMU, for the tilt abort.
        ``fsr``       : optional {"L": fz, "R": fz} foot loads (N) for Tier B gate.
        ``measured``  : optional current joint angles for the Tier-B CoM gate.
        """
        p = self.params
        st = self.state
        dt = 0.0 if st.last_now is None else _clamp(now_s - st.last_now, 0.0, p.max_dt_s)
        st.last_now = now_s

        roll, pitch = torso_rp
        tilt_ok = abs(roll) < p.tilt_abort_rad and abs(pitch) < p.tilt_abort_rad
        active = (
            tier in ("march", "step")
            and self._cmd_state == "march"
            and self._cmd_cadence > 0.0
            and self._cmd_conf >= p.conf_min
            and tilt_ok
        )

        # Cadence slew toward the human's (bounded rate).
        target_cad = _clamp(self._cmd_cadence if active else 0.0, 0.0, p.cadence_max_hz)
        max_step = p.cadence_slew_hz_per_s * dt
        st.cadence += _clamp(target_cad - st.cadence, -max_step, max_step)

        # Advance the robot's own phase, then phase-lock toward the human's.
        st.phase = (st.phase + TWO_PI * st.cadence * dt) % TWO_PI
        if active and dt > 0.0:
            err = _wrap_pi(self._cmd_phase - st.phase)
            lock = _clamp(err, -p.phase_lock_gain * dt, p.phase_lock_gain * dt)
            st.phase = (st.phase + lock) % TWO_PI

        # amp_gain ramp (slow up, fast down); never frozen mid-step on stop.
        amp_target = 1.0 if active else 0.0
        rate = (1.0 / p.amp_rise_s) if amp_target > st.amp_gain else (1.0 / p.amp_decay_s)
        st.amp_gain += _clamp(amp_target - st.amp_gain, -rate * dt, rate * dt)
        st.amp_gain = _clamp(st.amp_gain, 0.0, 1.0)

        eff = st.amp_gain * (0.4 + 0.6 * self._cmd_intensity)

        if tier == "step":
            targets, meta = self._tier_b_step(eff, measured, fsr)
        else:  # "march" (Tier A) and "stand" both use the double-support path
            targets, meta = self._tier_a_march(eff if tier == "march" else 0.0)

        # Turn bias (shared hip-yaw), experimental and small.
        turn = p.turn_rad * self._cmd_turn * st.amp_gain
        targets["LHipYawPitch"] = targets.get("LHipYawPitch", 0.0) + turn
        targets["RHipYawPitch"] = targets.get("RHipYawPitch", 0.0) + turn

        clamped = {n: self.limiter.clamp_angle(n, v) for n, v in targets.items()}
        meta.update({
            "amp_gain": st.amp_gain,
            "phase": st.phase,
            "cadence": st.cadence,
        })
        return clamped, meta

    # -- postures -----------------------------------------------------------
    def _crouch(self, u: float) -> Dict[str, float]:
        """Symmetric statically-balanced crouch (matches nao_retarget._lower_body)."""
        return {
            "LHipPitch": -u, "RHipPitch": -u,
            "LKneePitch": 2.0 * u, "RKneePitch": 2.0 * u,
            "LAnklePitch": -u, "RAnklePitch": -u,
            "LHipRoll": 0.0, "RHipRoll": 0.0,
            "LAnkleRoll": 0.0, "RAnkleRoll": 0.0,
            "LHipYawPitch": 0.0, "RHipYawPitch": 0.0,
        }

    def _tier_a_march(self, eff: float) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Double-support march: alternating knee pump + small lateral sway.

        Both feet stay loaded (single_support is always False), so the existing
        symmetric balance loop stays valid. At ``eff == 0`` the output is exactly
        the symmetric crouch.
        """
        p = self.params
        u0 = p.base_crouch_u
        theta = self.state.phase
        s = math.sin(theta)

        # Alternating knee pump: each leg flexes more while it is the unloaded
        # (light) leg. sin>0 -> lean right, left leg light -> bob left knee.
        left_bob = p.knee_bob_rad * eff * max(0.0, s)
        right_bob = p.knee_bob_rad * eff * max(0.0, -s)
        lu = u0 + left_bob
        ru = u0 + right_bob

        # Lateral sway shared (same-sign) on both legs -> shift weight, no splay.
        sway = p.sway_rad * eff * s

        targets = {
            "LHipPitch": -lu, "RHipPitch": -ru,
            "LKneePitch": 2.0 * lu, "RKneePitch": 2.0 * ru,
            "LAnklePitch": -lu, "RAnklePitch": -ru,
            "LHipRoll": sway, "RHipRoll": sway,
            "LAnkleRoll": sway, "RAnkleRoll": sway,
            "LHipYawPitch": 0.0, "RHipYawPitch": 0.0,
        }
        swing = 1 if s > 0.05 else (-1 if s < -0.05 else 0)
        return targets, {"single_support": False, "swing_side": swing, "tier": "march"}

    def _tier_b_step(
        self,
        eff: float,
        measured: Optional[Dict[str, float]],
        fsr: Optional[Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Single-support stepping (experimental). Lifts a foot ONLY when the CoM
        is predicted safely over the stance foot (and FSRs, if present, confirm
        the load). Otherwise step height collapses to 0 -> safe double support.
        """
        p = self.params
        u0 = p.base_crouch_u
        theta = self.state.phase
        # First half-cycle: stance LEFT, swing RIGHT; second half: swapped.
        stance = "L" if math.sin(theta) >= 0.0 else "R"
        swing = "R" if stance == "L" else "L"
        # Swing window: a raised-cosine bump peaking mid-swing (0 at the phase
        # boundaries so the foot is planted when support changes).
        half = (theta % math.pi) / math.pi  # 0..1 within the current half-cycle
        lift_window = 0.5 * (1.0 - math.cos(TWO_PI * half))  # 0 at ends, 1 mid-swing

        # Shift weight onto the stance foot (asymmetric, same-sign world lean).
        shift_dir = 1.0 if stance == "L" else -1.0
        shift = p.step_shift_rad * eff * shift_dir

        targets = self._crouch(u0)
        targets["LHipRoll"] = shift
        targets["RHipRoll"] = shift
        targets["LAnkleRoll"] = shift
        targets["RAnkleRoll"] = shift

        # Safety gate: may we actually lift the swing foot this step?
        gate_ok, margin = self._stance_gate(stance, measured, fsr)
        lift = p.step_height_rad * eff * lift_window if gate_ok else 0.0

        # Apply the swing-leg lift (extra knee flex + hip flex + ankle follow).
        su = u0 + lift
        targets[f"{swing}HipPitch"] = -su
        targets[f"{swing}KneePitch"] = 2.0 * su
        targets[f"{swing}AnklePitch"] = -su

        single_support = lift > 1e-3
        meta = {
            "single_support": single_support,
            "swing_side": (1 if swing == "L" else -1),
            "stance": stance,
            "stance_margin": margin,
            "gate_ok": gate_ok,
            "tier": "step",
        }
        return targets, meta

    def _stance_gate(
        self,
        stance: str,
        measured: Optional[Dict[str, float]],
        fsr: Optional[Dict[str, float]],
    ) -> Tuple[bool, float]:
        """Return (may_lift, stance_margin_m). Conservative: any doubt -> no lift."""
        p = self.params
        # FSR confirmation of weight transfer, when wired.
        if fsr:
            total = float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
            if total > 1e-6:
                stance_frac = float(fsr.get(stance, 0.0)) / total
                if stance_frac < p.fsr_load_frac:
                    return False, 0.0
        # Model CoM must project inside the stance-foot polygon with margin.
        if self.com_model is None or measured is None:
            # Without a model/measurements we cannot prove safety -> don't lift.
            return False, 0.0
        try:
            margin = self.com_model.stance_margin(measured, stance)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - older model without stance_margin
            return False, 0.0
        return (margin >= p.stance_margin_min), float(margin)


def default_gait_params() -> GaitParams:
    return GaitParams()
