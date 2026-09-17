from __future__ import annotations

from src.utils.filtering import ExponentialSmoother
from src.utils.fps import AdaptiveFPSController


def test_smoother_converges() -> None:
    smoother = ExponentialSmoother(alpha=0.5)
    out = smoother.update({"a": 1.0})
    assert out["a"] == 1.0
    out = smoother.update({"a": 3.0})
    assert out["a"] == 2.0


def test_fps_controller_backs_off_on_high_latency() -> None:
    ctl = AdaptiveFPSController(min_fps=25, max_fps=100, latency_budget_ms=100,
                               step_fps=10, _current_fps=80)
    for _ in range(40):
        ctl.update(measured_latency_ms=300.0)
    assert ctl.current_fps == 25


def test_fps_controller_ramps_up_when_fast() -> None:
    ctl = AdaptiveFPSController(min_fps=25, max_fps=100, latency_budget_ms=200,
                               step_fps=10, _current_fps=30)
    for _ in range(40):
        ctl.update(measured_latency_ms=20.0)
    assert ctl.current_fps == 100


# ---------------------------------------------------------------------------
# ArmTracker (2026-09-16). The arms were the slowest thing on the robot: the
# per-frame EMA spent 120 ms of the measured 180 ms between the pose log and the
# commanded angle, and its delay was set by the camera's frame rate rather than
# by any number anyone had chosen.
# ---------------------------------------------------------------------------
class TestArmTracker:
    def _ramp(self, tracker, joint, rate, frame_dt, frames, sim_dt=0.02):
        """Feed a constant-velocity target at ``frame_dt`` while advancing at
        ``sim_dt``. Returns ``[(t, commanded, true_target)]``."""
        out = []
        t = 0.0
        next_frame = 0.0
        tracker.reset(joint, 0.0)
        while t <= frames * frame_dt:
            if t >= next_frame - 1e-9:
                tracker.observe(joint, rate * t, t)
                next_frame += frame_dt
            out.append((t, tracker.advance(joint, t, sim_dt), rate * t))
            t += sim_dt
        return out

    def test_the_response_does_not_depend_on_the_frame_rate(self):
        """The whole point of a time constant. The old EMA was stepped once per
        UDP frame, so the same alpha meant 127 ms at 12 FPS and 250 ms at 6 --
        the robot got laggier exactly when the pipeline was already struggling.
        """
        from pose_control_utils import ArmTracker

        settled = []
        for frame_dt in (1 / 30.0, 1 / 12.0, 1 / 6.0):
            tracker = ArmTracker(tau_s=0.07, lead_s=0.0)
            tracker.reset("LElbowRoll", 0.0)
            t = 0.0
            next_frame = 0.0
            while t < 1.0:
                if t >= next_frame - 1e-9:
                    tracker.observe("LElbowRoll", 1.0, t)
                    next_frame += frame_dt
                value = tracker.advance("LElbowRoll", t, 0.02)
                if t >= 0.2 - 1e-9 and t < 0.2 + 0.02:
                    settled.append(value)
                t += 0.02
        spread = max(settled) - min(settled)
        assert spread < 0.05, (
            f"step response varies by {spread:.3f} across 30/12/6 FPS; the "
            "response must be set by tau_s, not by the camera")

    def test_a_steady_motion_is_tracked_without_lag(self):
        """A constant-velocity gesture is where lag is visible and where the
        prediction has to earn its keep: with lead_s matched to the loop delay
        the command should sit ON the target, not behind it."""
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.07, max_lead_rad=1.0)
        samples = self._ramp(tracker, "LElbowRoll", rate=1.0, frame_dt=1 / 12.0, frames=24)
        tail = [(cmd, truth) for t, cmd, truth in samples if t > 1.0]
        worst = max(abs(cmd - truth) for cmd, truth in tail)
        assert worst < 0.02, f"steady-state error {worst:.4f} rad on a 1 rad/s ramp"

    def test_prediction_beats_no_prediction_on_the_same_ramp(self):
        from pose_control_utils import ArmTracker

        errs = {}
        for lead in (0.0, 0.07):
            tracker = ArmTracker(tau_s=0.07, lead_s=lead, max_lead_rad=1.0)
            samples = self._ramp(tracker, "LElbowRoll", 1.0, 1 / 12.0, 24)
            tail = [abs(c - g) for t, c, g in samples if t > 1.0]
            errs[lead] = sum(tail) / len(tail)
        assert errs[0.07] < errs[0.0] * 0.25, (
            f"lead 0.07 must cancel most of the delay: {errs} rad mean error")

    def test_the_arm_keeps_moving_between_camera_frames(self):
        """The simulation steps at 50 Hz and the camera runs at ~12. Without
        per-tick advancement the arm held for four steps and jumped on the
        fifth, which reads as stepping rather than motion."""
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.2, max_lead_rad=1.0)
        samples = self._ramp(tracker, "LShoulderPitch", 1.0, 1 / 12.0, 24)
        tail = [cmd for t, cmd, _ in samples if t > 1.0]
        held = sum(1 for a, b in zip(tail, tail[1:], strict=False) if abs(b - a) < 1e-9)
        assert held == 0, f"{held}/{len(tail)} simulation steps did not move the arm"

    def test_extrapolation_is_bounded(self):
        """A single bad landmark must not fling the arm across its range."""
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.2, max_lead_rad=0.35)
        tracker.reset("LElbowRoll", 0.0)
        tracker.observe("LElbowRoll", 0.0, 0.0)
        tracker.observe("LElbowRoll", 3.0, 0.02)   # a 150 rad/s "gesture"
        peak = max(abs(tracker.advance("LElbowRoll", 0.02 + i * 0.02, 0.02))
                   for i in range(30))
        assert peak <= 3.0 + 0.35 + 1e-6, f"extrapolated to {peak:.2f} rad"

    def test_a_lost_human_leaves_a_still_arm_not_a_drifting_one(self):
        """Observations stop; the prediction must coast briefly and then hold.
        An arm that kept integrating its last velocity would walk itself into a
        joint limit while nobody was in front of the camera."""
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.2, max_extrapolate_s=0.2,
                             max_lead_rad=1.0)
        tracker.reset("LElbowRoll", 0.0)
        for i in range(12):                       # a real 1 rad/s motion
            tracker.observe("LElbowRoll", i * (1 / 12.0), i / 12.0)
            tracker.advance("LElbowRoll", i / 12.0, 0.02)
        last_obs_t = 11 / 12.0
        tail = [tracker.advance("LElbowRoll", last_obs_t + i * 0.02, 0.02)
                for i in range(1, 200)]
        assert abs(tail[-1] - tail[-10]) < 1e-6, "the arm is still drifting"
        assert tail[-1] <= 11 / 12.0 + 0.25, (
            f"coasted to {tail[-1]:.3f} rad past a target of {11/12.0:.3f}")

    def test_a_zero_dt_observation_cannot_manufacture_a_velocity(self):
        """Two frames landing on one simulation step used to divide by ~0."""
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.2, max_lead_rad=10.0)
        tracker.reset("LElbowRoll", 0.0)
        tracker.observe("LElbowRoll", 0.0, 1.0)
        tracker.observe("LElbowRoll", 0.5, 1.0)   # same timestamp
        value = tracker.advance("LElbowRoll", 1.0, 0.02)
        assert abs(value) < 0.5, f"a zero-dt frame produced {value:.3f} rad"

    def test_forget_drops_the_velocity_but_keeps_the_position(self):
        from pose_control_utils import ArmTracker

        tracker = ArmTracker(tau_s=0.07, lead_s=0.2, max_lead_rad=1.0)
        tracker.reset("LElbowRoll", 0.0)
        tracker.observe("LElbowRoll", 0.0, 0.0)
        tracker.observe("LElbowRoll", 0.5, 0.1)
        before = tracker.advance("LElbowRoll", 0.1, 0.02)
        tracker.forget("LElbowRoll")
        after = tracker.advance("LElbowRoll", 0.12, 0.02)
        assert abs(after - before) < 0.2, "position jumped when velocity was dropped"
