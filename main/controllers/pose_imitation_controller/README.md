# Webots NAO Pose Imitation Controller

Real-time control of the simulated **NAO (H25)** humanoid from human pose
tracking: arms, head, legs, and genuine locomotion — the robot squats when you
squat, lifts the leg you lift, walks across the floor when you walk, and turns
its whole body to face where you face.

---

## 1. Architecture

The Python pipeline is a generic *pose source*. Everything NAO-specific (joint
axes, signs, limits, balance, gait) lives here, on the robot side.

```
Python pipeline (src/)                    Webots controller (this folder)
──────────────────────                    ────────────────────────────────
MeTRAbs 3D landmarks (mm, GPU)            pose_imitation_controller.py
  ├─ raw landmarks ───────── UDP 8765 ──►    ├─ arms + head
  ├─ gait cues (cadence,                     │    nao_retarget.retarget_upper_body
  │   phase, body_yaw_rad)                   └─ legs: EXACTLY ONE of ↓
  └─ legacy joint angles                          1. locomotion  (motion clips)
     (fallback)                                   2. march engine (in place)
                                                  3. pose imitation (per-leg)
                                                  4. stand (balance only)
                                                        │
                                                   NaoPoseDriver
                                                   clamp → smooth → setPosition
```

Landmarks are now absolute 3D (millimeters, camera frame -- see
`src/perception/pose_estimator.py`), not MediaPipe's normalized 2D image
coordinates plus a weak depth channel. That changes what `nao_retarget.py` has
to do (§2) but nothing downstream of it -- `lower_body.py`, `balance.py`,
`gait.py` and `walk_motion.py` only ever see already-retargeted NAO joint
angles or the abstract gait-command dict, never raw landmarks.

**Exactly one layer commands the 12 leg joints on any given simulation step.**
Two at once means they fight each other and the robot falls; that single rule is
what the arbiter in `pose_imitation_controller._drive_legs` exists to enforce.

| Library | Responsibility | Webots-free? |
|---|---|---|
| [`nao_retarget.py`](../../libraries/nao_retarget.py) | landmarks → NAO angles (arms, head, and the closed-form per-leg solve) | ✅ |
| [`lower_body.py`](../../libraries/lower_body.py) | *may* the robot execute this leg pose? weight-shift / lift sequencer | ✅ |
| [`gait.py`](../../libraries/gait.py) | in-place march engine (gait command → leg motion) | ✅ |
| [`balance.py`](../../libraries/balance.py) | model-based CoM balance (FK + link masses + Fibonacci search), the support polygon, and the tilt sign conventions | ✅ |
| [`walk_motion.py`](../../libraries/walk_motion.py) | motion-clip discovery, yaw servo, locomotion planning | ✅ |
| [`pose_control_utils.py`](../../libraries/pose_control_utils.py) | `NaoPoseDriver`: limits, smoothing, velocity caps, logging | ✅ |
| `pose_imitation_controller.py` | Webots glue: sockets, devices, motion playback, arbitration | ❌ |

Every library is Webots-free on purpose, so all of the maths is unit-tested on a
dev machine that has no Webots installed (`pytest -q`).

---

## 2. What follows the human

### Upper body

| Human motion | NAO joints | How |
|---|---|---|
| Arm up / down | `ShoulderPitch` | vertical component of the upper-arm direction |
| Arm out sideways | `ShoulderRoll` | lateral component of the same direction |
| Elbow bend | `ElbowRoll` | angle between upper-arm and forearm |
| Head turn / nod | `HeadYaw`, `HeadPitch` | nose vs. shoulder midline |

NAO's 2-DOF shoulder is recovered from the observed 3D arm direction by the same
swing-twist decomposition used for the legs below (reference direction: arm
pointing straight forward).

### Legs — the closed-form per-leg solve

MeTRAbs gives real 3D, so this is no longer a *reconstruction* from a 2D
projection plus a noisy depth sign (the MediaPipe-era version of this section)
-- it is an exact closed-form decomposition of a real 3D bone direction.

Because the subject may now face any direction relative to the camera (not just
frontally, which the old 2D-projection approach implicitly assumed), each frame
`nao_retarget.py` first builds an orthonormal **torso-local basis**
(`right`, `up`, `forward`) from the shoulder line and the hip-to-shoulder line.
Every bone vector is projected onto this basis before solving, so the angles
that come out are relative to the subject's own body, not the camera.

For a thigh at hip roll `φ` and hip pitch `θ`, its UNIT direction in that
torso-local frame is

```
(lat, up, fwd) = ( sin φ·cos θ , −cos φ·cos θ , sin θ )
                    ^lateral      ^vertical       ^forward
```

which is now directly measured (all three components), not just two of three
approximated from foreshortening. The closed-form inverse ("swing-twist
decomposition", `nao_retarget._swing_twist`) is:

```
d  = −up             = cos φ·cos θ
φ  = atan2(lat, d)                      ← abduction (HipRoll)
θ  = ± acos(d / cos φ), sign from fwd   ← flexion (HipPitch)
```

The shank shares the hip roll and adds `KneePitch` about the same axis, so the
identical solve on knee→ankle yields `θ_hip + θ_knee`; the sole is then levelled
by `AnklePitch = −(θ_hip + θ_knee)` and `AnkleRoll = −φ`. Segment lengths (thigh,
shank, torso) are measured directly in millimeters each frame and lightly
smoothed for jitter -- there is no foreshortening left to correct for, so
unlike the old MediaPipe-era peak-hold scheme there is nothing to *learn*, only
to smooth.

Foot-lift and crouch detection deliberately stay in camera-frame vertical
(assuming a roughly level camera) rather than the torso-local frame: "which
foot is on the ground" is a real-world-verticality question, and answering it
from the torso's own up axis would make a forward lean read as a foot lift.

#### When the camera crops your legs

Lift detection needs no knowledge of where the floor is: whichever of your two
**feet** is lower defines the ground line, so the other foot's rise above it is
the lift. That needs both feet in frame — and standing close to a webcam usually
crops you at the shins, which made the lift signal read exactly zero however
high you lifted a leg. So the same relative trick falls back to the **knees**,
which are in frame whenever the hips are (it is also the signal the Python-side
gait detector uses, for the same reason). The status line reports which cue is
live as `lift-cue=feet|knees|none`.

With no ankle in view the knee *bend* is genuinely unobservable, so a lift then
shows as hip flexion — a raised straight leg rather than a folded knee. That is
the honest reading of what the camera can see. Segment lengths self-calibrate the
same way: if an ankle is never seen, the shank borrows the thigh (they are within
a few percent in both the human and NAO) rather than declaring the subject
uncalibrated and discarding the whole lower body.

#### How much of your pose gets through

Two very different things live in "the legs are at different angles", and gating
them the same way was wrong:

* **Mirror-symmetric** — both legs abducting outward (a wider stance), or both
  flexing equally (a squat). By symmetry these move the centre of mass *not at
  all*, and a wider stance makes the support polygon **bigger**. They are safer
  than standing, so they pass at **full authority, 1:1 with you**.
* **Antisymmetric** — both legs rolled the same way (a lean), or one leg forward
  and one back. These do move the CoM over the feet. They are followed 1:1 too,
  but the robot **shifts its pelvis to make the pose holdable** (§3a) rather than
  attenuating it, and three guards bound them: the lean is capped at
  `max_lean_dev` (0.45 rad, ~72 mm of CoM travel), the antisymmetric channels
  are rate-limited (`asym_rate_limit`, 1.2 rad/s — a marching human's leg swing
  copied onto two planted feet is otherwise a rocking excitation), and when the
  pelvis has run out of travel the pose is scaled back as a last resort
  (`_limit_asymmetry`, slewed so it cannot flap).

While a foot is genuinely off the ground the split is dropped: the swing leg is
unloaded and free to take your pose at whatever the safety gate allows, and the
stance leg stays near the balanced crouch because it is carrying the robot.

| You do | Robot does | Limited by |
|---|---|---|
| Squat | 1:1 to 40° hip / 80° knee | knee range, not balance — see below |
| **Walk forward** | pre-balanced walk clips, one per 2.6 s | the cue must see you walking — §4 and the `stride` channel |
| **Turn** | pre-balanced turn clips, closed loop on the true heading | §5 |
| Spread your legs | 1:1 to ~30°, saturating at 31.4° per leg | the **ankle** plus a bounded sole tilt — see below |
| Lean sideways | 1:1 up to 0.45 rad, pelvis shifted to hold it | `max_lean_dev`, then pelvis travel (§3a) |
| Split stance (one leg fwd) | 1:1, rate-limited to 1.2 rad/s | pelvis travel (§3a) |
| **Raise one leg** | full lift once the weight has transferred | the CoM model (§3) |
| Walk / march | walk clips, or the march engine | §4 |
| **Turn your body** | stepping turn (heading servo) | §5 |

**The squat cap is a joint-range limit, not a stability one.** NAO's thigh and
shank are within 3 mm of the same length, so the crouch posture
(`Hip = −d`, `Knee = +2d`, `Ankle = −d`) keeps the ankle under the hip — and the
CoM over the foot — at *any* depth, with the torso vertical and the soles flat
throughout. The real ceiling is the knee's own 121° range.

**A wide stance is capped by the ankle, plus a deliberate tilt budget.**
`HipRoll` reaches 45.3° but `AnkleRoll` only 22.8°, and the ankle is what levels
the sole against the hip's abduction — so past 22.8° the sole cannot be kept flat.

Refusing to go further turned out to be too strict. Recorded runs show subjects
spreading to ~25° routinely and 34° at the extreme, so the robot saturated just
below the human and it read as *"it spreads, but not as much as me"*. A small
explicit `sole_tilt_budget` (0.05 rad) is spent instead: the hip may abduct that
much further than the ankle can level, leaving each sole a couple of degrees off
flat (the outer edge of a 76 mm foot lifts about 4 mm). It used to be 0.15 rad;
at that angle only the inner corners of each sole touch, the contact-filtered
support polygon shrinks to the strip between them, and the 2026-09-03 logs show
the robot standing on its foot edges right before every lateral fall.

The budget is enforced as a **post-condition on the final commanded angles**.
The hip gives way first; whatever the hip's hardware stop will not absorb comes
off the ankle (`LHipRoll` bottoms out at −21.7° where `RHipRoll` reaches −45.3°,
and with the hip pinned the old version left the ankle running — recorded as a
sole 0.18 rad off flat). When the budget binds, stance width gives way, never sole
contact.

**The squat is one degree of freedom, read off the solve.** Depth is the smallest
reading across two axes and both legs: hip vs. knee (bending at the *waist* also
flexes the hip while the knees stay straight, and NAO has no torso joint for
that), and left vs. right (a raised leg is deeply flexed at both, so averaging
the legs made lifting one knee also squat the robot — the straighter leg is the
one bearing the weight).

---

## 3. Lifting one foot: LOAD → SINGLE → UNLOAD

Raising a foot on a free-standing biped is three actions, not one. Skipping the
first two is why a raised leg used to produce no visible response at all.

```
    human raises a foot
            │
            ▼
   ┌──── LOAD ────┐   lean so the CoM moves over the STANCE foot.
   │              │   The lean sign is PROBED against balance.NaoCoMModel,
   │              │   not hard-coded — a wrong lean sign makes a balance
   │              │   loop tip faster, and probing makes that impossible.
   │              ▼
   │      stance_margin > 0 ?   ← forward kinematics + link masses,
   │              │               plus the foot force sensors when present
   │              ▼
   │  ┌─── SINGLE ───┐  the swing leg follows the human's leg, its
   │  │              │  authority scaled CONTINUOUSLY by that margin
   │  ▼              ▼
   └── UNLOAD ◄──── human lowers the foot / margin lost / torso tilts
            │
            ▼
     symmetric crouch (the proven no-fall baseline)
```

Both `shift` and `lift` are rate-limited blends in `[0, 1]`, so there are no
discrete jumps and no state that can get stuck: the controller can always ramp
back to the exact symmetric crouch.

The commanded posture is `crouch_posture(u)` — the squat whose
`Hip + Knee + Ankle = 0` keeps the torso vertical and the soles flat — **plus**
the human's per-leg *deviation from* that posture, authority-weighted. So the
symmetric part never leaves the proven-stable family, only the asymmetric detail
is gated, and a subject standing still produces a deviation of exactly zero.

Safety gates that stand the robot down: torso tilt past `tilt_abort_rad`,
lower-body landmark confidence below `conf_min`, and "both feet up" (a jump, or
bad tracking — never a step).

The **foot force sensors are a confirmation, not a veto**. They scale the lift
between `fsr_min_gain` and 1.0 as the stance foot's load share rises to
`fsr_load_frac`. They used to veto outright, which meant a sensor reading a
constant 50/50 — uncalibrated, or a proto whose soles barely redistribute —
forbade every step forever: a silent, permanent "raising my leg does nothing".
By the time this gate runs the CoM model has already agreed the weight is over
the stance foot, and the tilt abort is the real safety net. Likewise, without a
CoM model at all (no NumPy) the lift is hard-capped by `ungated_lift_cap` rather
than cancelled.

`margin_full` matters more than it looks: a completed weight transfer yields
about 0.015 m of stance margin, so setting it any higher silently caps the lift
below what you asked for and reads as "the leg only moves a little".

Tuning lives in `LowerBodyParams` in [`lower_body.py`](../../libraries/lower_body.py).

### 3a. One centre-of-mass manager

Two things move the pelvis to keep the CoM over the feet, and they act on the
**same** degree of freedom (hip +c / ankle −c on both legs, which translates the
pelvis with the soles kept flat):

| Term | Answers | Computed from | Where |
|---|---|---|---|
| feed-forward | *is the COMMANDED pose statically holdable?* | forward kinematics of the commanded targets, **no tilt term** | `lower_body._shift_com` |
| feedback | *is the robot actually tipping?* | the MEASURED posture, the InertialUnit tilt **plus 0.12 s of gyro lead**, rate-limited to 0.8 rad/s | `balance.BalanceController` |

The driver computes the feedback first and hands it **into**
`LowerBodyController.step(feedback=...)`, where it is summed with the
feed-forward term, clamped **once** (`com_shift_max_pitch` 0.30 / `_roll` 0.25
rad) and rate-limited (`com_shift_max_step`, 1 rad/s). When the budget is short
the feed-forward term yields and the pose is scaled back; the feedback loop is
the safety net.

They used to be independent: this layer evaluated the IMU tilt too, and the
driver added the feedback on top afterwards with its own clamp. Two controllers
answered every tilt in full, their clamps summed to 0.55 rad (~85 mm of CoM
travel at 16 mm per 0.1 rad, against a 40–60 mm margin), and every fall in the
2026-09-03 logs shows both at their clamps: `HipPitch` +0.45 / `AnklePitch`
−0.65. `scripts/analyze_run.py` now flags that pattern ("the pelvis shift
exceeds what ONE clamp allows").

**Which way is up.** The tilt term's signs are *derived*, not tuned:
`Nao.proto` mounts the InertialUnit rolled +90° about the torso's x axis (raw
roll reads +π/2 upright), and Webots' ENU API decomposes attitude as
Z(yaw)·Y(pitch)·X(roll), so the reported pitch is the torso's rotation about its
own left axis — **a forward tilt reads positive**, and a right tilt reads as
positive roll. The Gyro node is mounted unrotated and agrees (sign agreement 81%
/ 85% on recorded data). `balance.TILT_PITCH_SIGN` shipped as −1 for a while,
inferred from one session whose fall direction was assumed; inverted, it turned
both shifters into positive feedback in pitch.
`tests/test_balance.py::test_pitch_sign_matches_webots_convention_for_the_mounted_sensor`
re-derives the signs from the mounting so they cannot regress, and
`analyze_run.py` cross-checks every log's IMU against its gyro.

### Tuning for more pose fidelity

If you want the robot to follow you harder, these are the knobs, most useful
first — each trades stability margin for faithfulness:

| Knob | Raise it to… | Cost |
|---|---|---|
| `max_lean_dev` (0.45) | lean further | more CoM travel for the pelvis shift to pay back; past ~0.55 the robot would have to step |
| `asym_rate_limit` (1.2 rad/s) | follow fast leg swings | rocking: both feet are planted, so a fast split stance shakes the robot |
| `max_crouch_u` (0.70) | squat deeper | approaches the knee's 121° limit |
| `sole_tilt_budget` (0.05) | spread wider still | the soles sit off flat, onto their inner edges; at 0.15 the robot stood on its foot edges before every lateral fall |
| `fsr_min_gain` (0.40) | trust the CoM model over the foot sensors | loses the load-transfer cross-check |
| `margin_min` (0.002) | start lifting sooner | starts unloading a foot with less margin |

Still **not** driven from the camera: `ElbowYaw` (forearm twist) and `WristYaw`
are held at their rest angles, so arm *rotation* is not imitated — only the arm's
direction and elbow bend. That is a known gap, not a fault.

---

## 4. Real locomotion

> **The one-line bug that cost this project walking and turning.** Webots'
> R2025a Python binding is `def play(self): wb.wbu_motion_play(self._ref)` — no
> return statement — so `play()` is `None`, and this controller tested it as
> `if not motion.play(): return False`. `MotionPlayer.start()` therefore returned
> False on **every call ever made**: `leg_mode` is `"pose"` in 100.0% of the
> frames of every recorded session (1.25 M rows), and four sessions logged
> `clip_status = "start REFUSED by Webots"` for 198–964 frames each. Worse, the
> clip *had* started — `wbu_motion_play` does not care what Python does with its
> return value, and the controller library applies a playing clip's keyframes
> every step — so the clip drove the legs while this code believed nothing was
> playing and kept commanding them itself. Log 1788428293 has `LKneePitch`
> **measured** at the clip's own first keyframe, 1.042 rad, while the controller
> commanded 0.20–0.52. Two commanders, one joint set, for 19 s. Playback is now
> confirmed by asking the clip whether it is running, which is a question the API
> does answer.

### Preparing to walk: the legs arrive before the clip does

Every one of Cyberbotics' walk and turn clips **opens in a deep, sole-flat
crouch** — `Forwards.motion` starts at `LHipPitch -0.505, LKneePitch +1.042,
LAnklePitch -0.537`, which sum to zero, so the torso is vertical and the soles
flat, and which is a squat of about 0.51 rad. This controller stands at
`base_crouch_u = 0.10` (knee 0.20). Playback commands its first keyframe on its
very first step, and `release_to_motion` has by then *lifted the velocity caps* —
so handing over from standing asks the knees for 0.84 rad in one 20 ms step and
the robot squats out from under itself.

So the arbiter gains a state: `prepare:<action>`. It reads the clip's own first
keyframe (`walk_motion.motion_first_pose`) and ramps the legs there via
`NaoPoseDriver.approach_leg_pose`, then plays the clip only once the **measured**
joints have arrived. Two details matter:

* **Every joint arrives together.** The knee travels 0.842 rad, the hip 0.405 and
  the ankle 0.437. One flat rate is not a synchronised move — and a sole's
  attitude is Hip + Knee + Ankle, so desynchronised joints tip the feet.
  Simulated against this repo's own CoM model, a flat rate drives that sum to
  0.405 rad (**eight times** the 0.05 sole-tilt budget) and the fore/aft support
  margin to **−0.064 m**: the ramp added to make the handover safe took the
  centre of mass off the feet on its own. Each joint's rate is therefore scaled
  by its share of the longest travel: sum 0.0000 rad, margin +0.0596 m, same
  0.56 s.
* **Coming back is a ramp too.** A clip *ends* in the same 0.51 rad squat, so the
  lower body's crouch limiter is seeded from the posture the clip left
  (`LowerBodyController.seed_crouch_from`) and comes down at
  `crouch_rate_limit`. Otherwise the first post-clip step commands the whole
  0.41 rad of knee travel at once — the same jolt, in reverse.

If the legs cannot reach the stance within `CLIP_PREPARE_TIMEOUT_S` (2.5 s
against 0.56 s of travel) the attempt counts as a locomotion failure, and the
existing `MOTION_MAX_FAILURES` policy retires the clips for the session rather
than ramping in place forever.

The watchdog budget now comes from the clip's **own** duration with no ceiling.
`TurnLeft180` runs 9.0 s, and capping it at `MOTION_WATCHDOG_S` (8.0 s)
guaranteed it overran, was dropped as broken and counted a failure — so the one
clip that can turn the robot right round in a single action could never be used.



The robot **actually translates across the floor** by playing Webots' own
pre-balanced NAO `.motion` clips (`Forwards`, `TurnLeft60`, …). Those clips are
tuned by Cyberbotics for this exact robot; an online gait good enough to walk a
free-standing NAO is a research project in itself, and a Supervisor
base-teleport explodes the physics.

Two details make the difference between a walk and a stumble:

1. **Velocity caps must be lifted.** `Motion` playback works by calling
   `setPosition` on every joint each step. A motor still limited to ~25 % of its
   maximum velocity cannot reach those keyframes, so the pre-balanced gait
   arrives late at every foot placement and the robot topples.
   `NaoPoseDriver.release_to_motion()` raises the caps and suspends per-joint
   commanding; `reclaim_from_motion()` reseeds the smoothers from the position
   sensors so control returns without a jolt.
2. **How a clip ends decides how well it walks.** A clip boundary is a balanced
   double-support pose, so the safe default is to play the whole clip. But that
   makes clip length the latency of "stop walking", and it makes the robot pay
   the clip's start/stop transient for every stride. That transient is 49 % of
   `Forwards.motion`'s 2.60 s, and during the closing settle the torso travels
   17 mm **backward** — which is why chained clips measured 0.036 m/s against
   NAO's documented ~0.10, and why it looked like stepping rather than walking.

   There are three exits, in order of preference:

   | Exit | When | Cost |
   |---|---|---|
   | **Leave the gait cycle** (`GAIT_CYCLE`) | the clip has a detectable limit cycle in it — of the clips Webots ships, only `Forwards50.motion` | jump once at the phase where it is free (0.0010 rad = 0.05 rad/s), then 1.44 s of the clip's own deceleration |
   | **Stop at a safe keyframe** (`CLIP_EXIT_TOLERANCE_S`) | any other clip, once nothing wants locomotion | wait up to 1.04 s (0.52 s for the turn clips) for a keyframe that is statically holdable **and** not carrying momentum |
   | **Tilt abort** | the robot is going over | immediate, mid-stride |

   The momentum half of that second test is easy to miss and matters: stopping
   the legs does not stop the robot. The body keeps its velocity, and the capture
   point sits `v/ω` ahead of the CoM (ω = √(g/h) ≈ 5.9 rad/s at this crouch), so
   0.18 m/s throws it 30 mm past the CoM against a margin budget of 40–60 mm.
   13 of the 46 poses in `Forwards.motion` that pass the *static* tests are
   moving that fast.

### Cyclic gait: one clip, many strides

Cyberbotics' clips are animations — squat, accelerate, stride, decelerate,
stand. The stride is fine; the transient around it is what costs. But the
*middle* of the long walk clip is a true limit cycle, and exactly so:

    max|q(2.84 s) − q(1.80 s)| = 0.0000 rad   over all 12 leg joints

and the periodicity holds to that precision from 1.80 s to 5.32 s. So the
segment can be rewound with `Motion.setTime()` without commanding any joint
motion at the seam, and one clip becomes a gait generator that runs for as long
as the human keeps walking. `walk_motion.gait_cycle()` derives the schedule from
the clip's own keyframes every time — never a table — so it cannot drift from
the files Webots actually ships, and a clip that is not cyclic is *detected* as
not cyclic rather than looped on faith:

| Field | `Forwards50.motion` | Meaning |
|---|---|---|
| `enter_s` | 1.38 s | playback starts here; the prepare-ramp does the opening squat instead, rate-limited and under balance supervision |
| `loop_start_s` … `loop_end_s` | 2.95 → 4.23 s | rewound by `period_s` on reaching the end |
| `period_s` | 1.28 s | one full stride, two stance exchanges |
| `advance_m` | 0.093 m | → **0.073 m/s sustained** (forward kinematics) |
| `exit_from_s` → `exit_to_s` | 3.49 → 7.27 s | the stop jump, taken on *crossing* that phase |
| `exit_cost_rad` | 0.0010 rad | 0.05 rad/s over one control step |
| `tail_s` | 2.46 s | Cyberbotics' whole feet-together deceleration |
| `rest_s` | 8.45 s | where the tail has *actually* come to rest — see below |
| `settle_s` | 1.18 s | `rest_s − exit_to_s`: the part of the tail that is really decelerating |
| `stop_latency_s` | **2.46 s** | `period_s + settle_s`, the worst case from "stop" to standing |

The last three are why stopping is not the whole 3.73 s the tail implies. After
`rest_s` the clip commands no further travel — the remaining 1.28 s is it
standing up out of its own walk crouch, a posture the lower-body layer has to
ramp out of anyway the moment it gets the legs back. Riding that out bought
nothing and cost 1.28 s of "the robot will not answer me" on every stop, so the
tail is trimmed there (`tails_trimmed` in the trajectory log counts it).

Two requirements keep this honest. The seam must be exact — 1 µrad, because a
seam is a teleport executed in one 20 ms step with the caps lifted, and
`Forwards.motion`'s best available seam is 0.107 rad (a 5.35 rad/s jolt), which
is why it cannot be cycled. And the cycle must **translate** the robot
(≥ 0.02 m): every clip has some periodicity — a turn rotates through repeated
steps, a side-step shuffles — and looping those would spin or drift the robot
indefinitely, while turning is closed-loop on the heading and needs discrete,
countable clips.

Because a cyclic clip is rewound before it can ever report itself "over", it sits
permanently in the state the watchdog exists to catch. Each completed stride is
therefore treated as proof of life and extends the deadline; if rewinding stops
happening, the deadline arrives exactly as it always did.

What this does **not** claim: the stride is the same stride either clip walks —
both take a 0.051 m half-step and both peak at 0.18 m/s. Cycling does not make
the steps faster. It removes the start/stop transient between them, which is
where the time was going. For a 10 s walk that is 47 stand/squat cycles reduced
to 1.

Motor headroom, for the record: across the clip's 2 376 joint-intervals
**nothing exceeds the declared maximum**. The peak is `RKneePitch` at 83.5 % of
its rated 6.40 rad/s, and inside the cycled window the peak is `LKneePitch` at
82.0 %. Pinned by `test_the_cycled_window_stays_within_the_motors_declared_speed`.

That 84 % is also the answer to "why is it only 0.073 m/s" — the clip is already
near the motors' ceiling, so it cannot be retimed faster. Walking speed is
hardware-bound here, not tuning-bound.

If no clips are found on disk the controller says so in the log and falls back
to the **march engine** (`gait.py`), which tracks your cadence, phase and stop
but marches in place.

### Playback can never hold the body

Because a clip owns the *whole* body, anything that stops it from reporting
"over" would freeze the entire robot, not just the legs. Three guards make that
impossible:

| Guard | Effect |
|---|---|
| **Watchdog** (`MOTION_WATCHDOG_S`, and the clip's own `getDuration()` when Webots reports it) | Suspension is always time-bounded. On expiry the body is taken back and that clip is never used again. |
| **Failure backoff** (`MOTION_MAX_FAILURES`) | A clip that trips the watchdog, hits the tilt abort, or finishes with the robot tipped counts as a failure. After a few, clips are abandoned for the session and the controller says so — falling over repeatedly is worse than never walking. |
| **Settled-start check** (`MOTION_START_MAX_TILT_RAD`) | A clip is only started from an upright, calm robot. Starting one mid-wobble is how a walk becomes a fall. |

Likewise, a raised exception in a control step no longer ends the loop: it is
logged, the body is forced back under our control, and the loop carries on
(`MAX_CONSECUTIVE_ERRORS` bounds how long that can go on). Before this, one
transient error broke out of `run()` into cleanup, which zeroed every motor
velocity — a permanently dead robot from a single bad frame.

---

## 5. Turning: a heading servo, not a gesture

> **Turning needs a heading, and this robot has none of its own.** The proto
> disables the InertialUnit's yaw axis *and* the Gyro's z axis, so neither the
> angle nor the rate is measurable — and what the InertialUnit returns in place
> of a heading is a copy of its own half-scale pitch (§9). The loop now closes on
> `Supervisor.getSelf().getOrientation()`: the robot's forward axis is +x in its
> own frame, so the heading is `atan2(m[3], m[0])` of the row-major matrix,
> counter-clockwise-positive, which is the sign `YawServo` and `TURN_SIGN`
> already expect. Set `HEADING_SOURCE = "off"` to go back to no turning.
>
> `overshoot_frac` was also 0.5, which is exactly the value at which a
> discrete-clip turn **cannot converge**: firing at |error| = f·nominal leaves
> |error − nominal| = |1 − 1/f|·|error|, which is ≥ |error| for every f ≤ 0.5, so
> a clip turned the robot from +e to −e forever. At 0.65 each turn cuts the error
> to at most 0.54 of what it was.



NAO has no torso-yaw joint, so "the human turned round" cannot be imitated by a
joint angle — the robot has to step round. And it cannot be done open-loop
("turned left → play one left-turn clip"), because clip and human turn by
different amounts and the error accumulates.

```
 human torso yaw  ──┐
 (gait_cues: atan2  │   desired = robot_yaw_at_latch
  of the shoulder   ├──►           + (human_yaw − human_yaw_at_latch)
  line's depth      │   error   = wrap_pi(desired − InertialUnit yaw)
  spread over its   │                    │
  lateral extent)   │                    ▼
 robot IMU yaw ─────┘        plan_action → turn_left / turn_right clip
```

Because the error is measured against the robot's **real** heading every step,
the rotation converges despite coarse clips and noisy tracking, and it works
while standing perfectly still. `plan_action` also refuses a clip that would
overshoot (a 60° clip is not fired at a 15° error), so the residual heading error
is bounded by half a clip. Aligning the heading takes priority over walking
forward: walking off along the wrong heading is much harder to undo.

The servo tracks rotation **relative to where it first saw you**, so standing
habitually a little off-square is absorbed by the latch rather than corrected
forever.

### The yaw covers the whole circle

The lateral term is used **signed**. Turn past 90° and your shoulders swap sides
in the image, and that sign is exactly the front/back discriminator a single
camera is otherwise missing:

| You are | lateral | depth | yaw |
|---|---|---|---|
| facing the camera | > 0 | ~0 | ~0° |
| turned 90° | ~0 | ≠ 0 | ±90° |
| facing away | < 0 | ~0 | ±180° |

An earlier version took `abs()` of the lateral term, which folded the two halves
together and bounded the estimate to ±90° — so the robot could never be asked to
turn round, however far you turned.

The term is `left.x − right.x`, not the other way about: pose estimators
conventionally label a person facing the camera with their *left* shoulder on
the image's **right** (anatomical left/right; MeTRAbs' `coco_19` follows the
same COCO convention MediaPipe did). Measured on recorded MediaPipe runs,
`right.x − left.x` was negative on 67% of frames with both shoulders clearly
visible (median −0.085), and 99% of those frames had both ears visible — a
face-on view. The hips agreed, so it reflected a labelling convention rather
than noise, and it held whether or not the preview was mirrored (the estimator
cannot tell a mirrored subject from a real one, so it labels by appearance
either way). With the sign inverted, someone looking straight at the camera
was reported as turned 180° away. **This has not been re-measured against
real MeTRAbs output** (written without GPU access) — worth a quick sanity
check ("stand facing the camera, confirm yaw reads ~0°") the first time this
runs on the target machine; see `gait_cues.py`'s module docstring.

### Why the yaw is filtered so carefully

A noisy heading does not merely wobble the robot — it makes the controller demand
turn clips in **alternating directions**, which starves forward walking and trips
the locomotion failure backoff, so the robot ends up doing nothing at all. That
is exactly what recorded runs showed: `|yaw|` spiking to the ±90° bound on 7–18%
of frames while the subject stood square to the camera, and the planner asking
for `turn_left` 1438× against `forward` 17× in one run.

Three guards, all in `gait_cues._update_yaw`:

1. **Span gating.** A body-fixed segment keeps a near-constant length however you
   turn (the depth term takes over from the lateral one). The observed length is
   compared against a self-calibrated peak-hold reference, so a collapsed or
   occluded detection is rejected outright instead of becoming a large angle.
2. **Shortest-arc smoothing.** Yaw is circular now, so a plain EMA would travel
   the long way round through 180°. The estimate is smoothed along the shortest
   arc and the pair contributions are averaged as *vectors*, not angles.
3. **Jump de-weighting.** A frame disagreeing with the running estimate by more
   than 0.6 rad is more likely noise than a real rotation at camera frame rates,
   so it is followed (it might be genuine — never latch up) but its confidence is
   halved.

Replayed over the same recorded runs, turn requests drop from 1438+53 and
846+1206 to 9+8 and 11+15, `forward` rises to 13–15% of decisions, and the
planner sits in "stand" 83–86% of the time.

While the stepping turn has not fired yet, a small capped `HipYawPitch` bias
gives immediate visual feedback. It is deliberately tiny — NAO's `HipYawPitch`
axis is canted 45°, so it splays the legs as well as yawing the pelvis — and it
is suppressed mid-step.

If the robot turns the *wrong way* for your camera setup, flip `TURN_SIGN` at the
top of the controller. (One sign knob, rather than a sign buried in the geometry:
with the pipeline's default mirrored selfie preview the robot mirrors the
on-screen figure, which is consistent with how the arms are mapped.)

---

## 6. Configuration

### `LEG_CONTROL` — the one knob that matters

At the top of `pose_imitation_controller.py`:

| Value | Behaviour |
|---|---|
| `"auto"` *(default)* | Full stack: motion clips for real walking/turning, march engine when no clips exist, per-leg pose imitation otherwise. |
| `"pose"` | Per-leg pose imitation only. Squat and single-leg lift work; the robot never leaves its spot. |
| `"engine"` | March engine + pose imitation, never the clips. Marches in place. |
| `"off"` | Legs held in the standing posture. Upper body only. |

### Other tunables

| Name | Meaning |
|---|---|
| `DRIVE_HEAD` | head follows the human head |
| `SWAP_SIDES` | mirror-image mapping (set `True` if left/right feels reversed) |
| `SMOOTHING_ALPHA` | EMA on arm/head targets (higher = snappier) |
| `VELOCITY_SCALE`, `LEG_VELOCITY_FACTOR` | fraction of hardware max velocity |
| `GAIT_SMOOTHING_ALPHA`, `GAIT_LEG_VELOCITY_FACTOR` | the same, for legs while stepping/marching |
| `ENABLE_BALANCE` | model-based CoM feedback |
| `WALK_TIER` | march engine tier (`"march"` / `"step"` / `"stand"`) |
| `TURN_SIGN` | flip the turn direction |
| `TILT_ABORT_RAD`, `TILT_RATE_LEAD_S` | fall detection (the gyro term predicts ahead) |
| `MOTION_WATCHDOG_S` | hard ceiling on how long a clip may own the body |
| `MOTION_MAX_FAILURES` | bad locomotion attempts before clips are abandoned |
| `MOTION_START_MAX_TILT_RAD` | how upright the robot must be to start a clip |
| `MAX_CONSECUTIVE_ERRORS` | failed control steps tolerated before giving up |
| `LOCOMOTION` | `LocomotionParams`: yaw gates, overshoot guard, walk thresholds |
| `MOTION_SEARCH_DIRS_EXTRA` | extra folders to search for `.motion` clips |

Deeper tuning: `LowerBodyParams` in `lower_body.py`, `GaitParams` in `gait.py`,
`BalanceParams` in `balance.py`.

Python side: `configs/default.yaml` (`walk.enabled` is the master switch for
streaming gait/yaw cues at all).

---

## 7. World requirements (do not skip)

`main/worlds/…​.wbt` **must** define the foot/floor contact pair:

```
WorldInfo {
  basicTimeStep 20
  contactProperties [
    ContactProperties {
      material2 "NAO foot material"
      coulombFriction [ 8 ]
      bounce 0
      bounceVelocity 0.003
    }
  ]
}
```

`Nao.proto` tags its soles with the contact material `"NAO foot material"`. With
no matching `ContactProperties`, the sole/floor pair silently falls back to
Webots' default contact (`coulombFriction 1`, bouncy) — the feet **slide and
jitter**, the robot cannot load one foot or take a step, and the leg controller
looks *frozen* because every leg command is absorbed by foot slip. This is the
single most common cause of "the legs do not respond".

`basicTimeStep` must be ≤ 20 ms. Webots' default of 32 ms is too coarse for NAO
leg control and destabilises the pre-balanced walk clips; the controller logs a
warning if it finds a larger value.

---

## 8. Communication protocol

UDP JSON on **port 8765**:

```json
{
  "timestamp_s": 1234567890.123,
  "frame_index": 45,
  "joint_angles_rad": { "LShoulderPitch": 0.5, "RElbowRoll": -1.1 },
  "keypoints": {
    "left_shoulder": [-160.2, -580.4, 1980.1, 0.99],
    "left_hip":      [-108.6, 15.2, 1975.3, 0.98],
    "left_knee":     [-114.8, 428.7, 2010.5, 0.97],
    "left_ankle":    [-119.3, 826.9, 2005.1, 0.95]
  },
  "gait": {
    "state": "march", "cadence_hz": 0.95, "phase": 1.83, "swing_side": 1,
    "intensity": 0.7, "turn": 0.4, "conf": 0.98,
    "body_yaw_rad": 0.42, "yaw_conf": 0.99
  }
}
```

- **`keypoints`** *(preferred)* — MeTRAbs landmarks `name → [x, y, z, visibility]`
  in absolute METRIC camera-frame coordinates (millimeters; `x` right, `y` down,
  `z` forward/away from the camera) -- see `src/type_defs.Keypoint`. This
  replaces MediaPipe's normalized [0,1] image coordinates plus a weak depth
  channel. `visibility` is a PROXY (in-frame/in-box confidence), not a true
  per-joint occlusion estimate -- MeTRAbs has no per-joint confidence output.
  The controller retargets these itself. 19 landmarks are streamed (head,
  shoulders, elbows, wrists, hips, knees, ankles, neck, pelvis) to keep packets
  small. MeTRAbs' `coco_19` skeleton has no separate heel/toe landmarks (unlike
  MediaPipe's 33-point set) -- the controller's ground-line/lift detection falls
  back to ankle-only (see `nao_retarget.LowerBodyRetargeter._foot_height`).
- **`joint_angles_rad`** *(fallback)* — used only when no `keypoints` are present.
- **`gait`** *(optional)* — cadence/phase/stop for the march engine, plus
  `body_yaw_rad` (**an angle**, so the controller can close a heading loop on it)
  and `yaw_conf`, which is independent of `conf` because the yaw needs only the
  shoulders and hips and stays usable when the legs leave the frame.

Additive and backward compatible: an older controller ignores fields it does not
know.

---

## 9. Sensors used

| Device | Used for |
|---|---|
| `inertial unit` | the torso's attitude: **roll** as reported, **pitch** as reported × `IMU_PITCH_SCALE` (the device halves it, but the scale ships at 1.0 — see below). Its yaw is not a heading and is unused |
| `accelerometer` | gravity, computed and logged every step as the independent witness to that half-scale pitch — but *not* the control source (see below) |
| `gyro` | tilt *rate*: the lead term in fall detection and in the balance loop (`BalanceParams.tilt_lead_s`). Its z axis is disabled in the proto, so there is no yaw rate |
| `LFsr`, `RFsr` | per-foot load: confirming a weight transfer before a lift, and masking an unloaded foot out of the support polygon (`balance.contact_from_fsr`) |
| `<joint>S` | position sensors: achieved angles, stuck-motor detection, trajectory log |

### Why the attitude comes from gravity, not the InertialUnit

Read the proto:

```
DEF INERTIAL_UNIT InertialUnit { rotation 1 0 0 1.5708   yAxis FALSE }
DEF GYRO          Gyro         { ...                     zAxis FALSE }
```

The mount (+90° about x) is only an offset — that is what the learned tilt zero
handles, and why the raw roll reads +1.571 on an upright robot. **`yAxis FALSE`
is the problem.** It is meant to switch the yaw output off, and Webots implements
it by converting the sensor's attitude to axis–angle, zeroing the world-z
component of the axis, and re-deriving roll/pitch/yaw from the result
(`WbInertialUnit::computeValue`). On an upright sensor that is harmless. On one
rolled 90° it corrupts the axes it was never asked to touch. Composing
`R_world_sensor = R_torso · Rx(π/2)` and running Webots' own ENU decomposition
through that zeroing gives, for a pure torso pitch *b*, and for a pure body
rotation *ψ* with no tilt at all:

| Torso really does | Reported roll | Reported pitch | Reported yaw |
|---|---|---|---|
| pitch forward *b* | 0 | **b / 2** | **b / 2** |
| rotate *ψ* | 0 | **ψ / 2** | ψ / 2 |
| roll right *b* | **b** | 0 | 0 |

So the pitch channel reads **half** the real pitch and cannot be distinguished
from a body rotation, and the yaw channel is not a heading at all — it is a copy
of that same half-pitch. Both predictions hold on every recorded session: across
41 logs `imu_yaw` tracks `imu_pitch_raw` at r ≥ 0.994 with slope ≈ 1.0, and a
regression of `imu_pitch` on the torso pitch implied by the loaded foot's own
forward kinematics has slope +0.48. Roll is unaffected (+0.98).

The honest correction is therefore to **double** the pitch on the way in, and
`IMU_PITCH_SCALE` exists for exactly that — set to **1.0**, deliberately, because
the one experiment available says this loop is closer to its stability limit than
to its authority limit. On 2026-09-04 the attitude came from gravity, which
reports the *full* pitch; that is the same doubling, and the fore/aft loop limit
cycled — ±0.25 rad of pitch within the first second of each episode, the
correction pinned at its clamp, the robot down inside three seconds, six episodes
running. On the half-scale reading the same controller stood still for 900
seconds. Under-correcting is sluggish and stable; over-correcting is a fall, so
the loop stays conservative until a session with a human actually in frame shows
the fore/aft response is too slow. Then raise it once, to 2.0, and watch the
pitch trace — `analyze_run.py` reports the `acc_pitch`-on-`imu_pitch` slope
(≈ +2 on quiet frames) as the measurement of how far it can go.

One caveat travels with raising it: the same masking leaks a body rotation into
this channel one-for-one, so a heading change ψ reads as ψ/2 of pitch and would
be doubled back to ψ. With the heading loop off the robot no longer yaws itself
and the measured residual is small (`imu_yaw` within ±0.03 rad p95 while standing
⇒ ~9 mm of modelled CoM error against a 25 mm deadband).

#### Gravity: measured, logged, and deliberately not acted on

The accelerometer looks like the better sensor, and on paper it is: all three
axes enabled, a plain 180° mount, and gravity is an absolute reference that needs
no learned zero and cannot be mistaken for a heading. Webots documents the
convention (`acceleration = −gravity`; "at rest ... 1 g along the vertical
axis"), so an upright torso reads `(0, 0, −9.81)` and the attitude is two lines:

```
u = (a_x, −a_y, −a_z)  =  (−sin θ,  sin φ cos θ,  cos φ cos θ)
pitch θ = atan2(−u_x, hypot(u_y, u_z))          roll φ = atan2(u_y, u_z)
```

That derivation is right — with the robot quiet it matches the InertialUnit's
roll to four decimals. **And controlling from it puts the robot on the floor**,
because an accelerometer does not measure attitude; it measures gravity *plus the
robot's own acceleration*, so inside a position loop it feeds a second derivative
back as a position:

> the loop shifts the pelvis → the torso accelerates sideways → the accelerometer
> reports that acceleration as tilt → the loop shifts further

Measured, in `webots_joint_trajectory_1788521290` — ten episodes, nobody in front
of the camera in any of them, so the legs were not imitating anything:

| Episodes | Attitude from | Outcome |
|---|---|---|
| 0–6 | gravity | fell within ~2 s, six times over; median head height **0.19 m** |
| 7–9 | the InertialUnit | stood **40 s** and then **298 s**; head 0.46 m |

The cross-check is what ended that: the roll disagreement latched and the path
disabled itself (`ACC_DISAGREE_RAD` / `_S`), which is the only reason that session
produced 298 seconds of standing instead of a tenth reload. So
`TILT_FROM_ACCELEROMETER` ships **off**, and switching it on needs the standard
fix rather than a longer filter — integrate the **gyro** for the fast attitude and
use gravity only to correct its slow drift, with a time constant of half a second
or more. Both gyro tilt axes are enabled on this proto, so the parts are there.

Meanwhile gravity is still computed and logged every step (`acc_roll`,
`acc_pitch`), alongside the tilt actually acted on (`ctl_roll`, `ctl_pitch`) and
the source (`tilt_source`), because it is the only independent measurement of the
half-scale pitch. `scripts/analyze_run.py` re-runs the yaw-is-pitch test and the
gravity-vs-InertialUnit regression on every log.

**Turning is therefore off** (`HEADING_FROM_IMU = False`). With the IMU's yaw
axis and the gyro's z axis both disabled, nothing on this robot observes its own
rotation, so there is nothing to close the heading loop on. That loop was
steering on the pitch channel: standing still produced a median 21° heading
"error", and the hip-yaw bias it commanded tipped both soles — the `HipYawPitch`
axis is canted 45°, so it pitches the legs as it yaws them — which collapsed the
modelled support polygon onto the toe line and had the CoM compensation drive the
pelvis to its forward clamp on a dead-level robot. Re-enable it when there is a
real heading (a Supervisor read, or a proto with the yaw axes on); the servo and
clip planner are unchanged and tested.

NAO's foot sensors are 3-axis (`force-3d`) touch sensors, so their value comes
from **`getValues()`**, not `getValue()`. Reading them as scalars is why an
earlier version silently got no load information and the step gate never saw a
weight transfer. Both APIs are handled, so 1-axis protos work too.

---

## 10. Controllers

### `pose_imitation_controller.py` — **recommended**
The full stack described above. This is the one wired into the world file.

### `pose_imitation_controller_advanced.py`
Same `NaoPoseDriver` core, tuned for **inspecting motors**: smoother settings and
verbose per-joint diagnostics (commanded vs. measured, average error, stuck
flags). Deliberately no locomotion, so the robot stays on the spot and the
numbers stay interpretable — but the legs still do full per-leg pose imitation
through the same `LowerBodyController`, so what you measure here is what runs
there.

### Output: joint-trajectory log (FR-7 / US-3)
With `ENABLE_TRAJECTORY_LOG = True` (default), each run writes
`<project>/logs/webots_joint_trajectory_<epoch>.csv`: per simulation frame,
`wall_time_s, sim_time_s, frame_index` and, for every driven joint, a
`<joint>_cmd_rad` (commanded) and `<joint>_meas_rad` (achieved, from the position
sensor) column. This is what the evaluation step uses for per-joint MAE and
timing. Logging is fully defensive — any I/O error disables it without disturbing
the real-time loop. `logs/` is git-ignored.

---

## 11. Reading the controller log

### At startup — check this block first

```
====================================================================
NAO pose imitation controller ready
  timestep          : 20 ms
  motors / sensors  : 24 / 24  (12 leg joints)
  leg control       : auto
  arms + head       : ON
  leg pose imitation: ON (squat, single-leg lift)
  CoM balance       : ON
  march engine      : ON
  locomotion clips  : forward, turn_left, turn_right
  heading feedback  : ON (InertialUnit)
  foot force sensors: 2
====================================================================
```

Every "the robot does not move" report so far has had a cause that was visible
here — a missing device, a missing NumPy, no motion clips, a coarse timestep —
just buried further down the log. Anything reading `OFF` or `NONE FOUND` is
flagged inline with what it costs you.

### Per 100 frames

```
Frame 400 | sim 49.8 Hz | tracking | legs=pose | 8 joints applied
  legs: stepping (73% of the requested lift)
        mode=single stance=R shift=1.00 lift=0.73 gate=0.77 margin=+0.0145m crouch=0.10 lift-cue=feet
  heading error  -8 deg (servo latched)
```

- `legs=` — which layer is commanding: `pose`, `march:march`, `motion:forward`, `stand`
- `legs:` — **plain-language reason** for what the legs are (not) doing. "The legs
  are not moving" has half a dozen legitimate causes — nobody in frame, legs
  cropped, low confidence, the CoM not yet over the stance foot — and they are
  indistinguishable from a bug unless the controller names the one in play.
- `mode` — `double` / `load` / `single` (§3)
- `gate` — how much lift the CoM model currently permits (0 = do not unload)
- `margin` — signed distance of the CoM inside the stance foot; must be positive to step
- `lift-cue` — `feet` / `knees` / `none`: which landmarks the lift is read from
- `fsr_share` (in the reason text) — the stance foot's measured share of the load
- `heading error` — what the turn servo still wants to rotate

---

## 12. Troubleshooting

**The robot is completely frozen (nothing moves, not even the arms)**

Arms and head are driven independently of the legs, so *nothing* moving means
per-joint commanding is not reaching the motors at all. In order of likelihood:

1. **Read the startup block** (§11). If it never printed, the controller failed
   before its first step — look for a Python traceback in the Webots console. The
   usual cause is Webots' Python interpreter missing a package; set
   `Tools → Preferences → Python command` to the interpreter that has NumPy.
2. **Is a motion clip playing?** The status line shows `legs=motion:…` and
   `0 joints applied` while a clip owns the body. That is normal and bounded by
   the watchdog (§4); if you see it permanently, the watchdog message will say
   so and clips will be dropped.
3. **Is the simulation actually running?** Webots must be playing, not paused,
   and not in fast-forward-without-rendering if you are judging by eye.
4. **Are packets arriving?** The status line says `tracking` when frames are
   arriving and `STALE (holding)` when they are not. `STALE` plus a stationary
   robot means the pipeline is not reaching the controller — check the pipeline
   log for `Webots bridge sending to 127.0.0.1:8765` and that both processes are
   on the same machine.

**Legs do not move at all**
1. `WorldInfo.contactProperties` missing → §7. This is the usual cause.
2. `LEG_CONTROL = "off"` in the controller.
3. Log says `Lower-body pose imitation OFF (...)` → NumPy is missing from the
   Python interpreter Webots is using (`Tools → Preferences → Python command`).
4. Legs out of camera frame: watch `Visible: nn/33` in the pipeline HUD.

**Raised leg produces only a small response**
Read the `legs:` line — it names the limiting factor. `gate=0.00` means the CoM
model refuses to unload the foot: usually the robot has not finished leaning yet
(`shift < 0.75`). If it says *"stepping at reduced authority: foot sensors report
only 50% of the load…"*, the FSRs are not seeing the transfer; raise
`fsr_min_gain` toward 1.0 to trust the CoM model instead.

**Spreading my legs does not go as wide as mine**
It tracks you 1:1 to about 30° and saturates at 31.4° per leg (§2). Past that the
soles would sit too far off flat. Raise `sole_tilt_budget` if you want more and
can accept the robot standing further onto the inner edges of its feet. If it
stops well short of 30°, check the `legs:` reason — a one-sided spread is partly
antisymmetric and therefore partly gated.

**The robot never turns, or turns the wrong way and back again**
Check the `heading error` line. If it never settles, the yaw estimate is not
being trusted — most often because your torso is only partly in frame, so the
span gate rejects it (§5). If the robot turns *consistently* the wrong way, flip
`TURN_SIGN`; if it alternates, that is the yaw thrashing, not the sign.

**The robot never walks forward**
Forward walking needs the gait detector to see you marching, which needs your
**knees** in frame: `gait_cues` requires 3 of the 4 hip/knee landmarks above 0.5
visibility for `conf >= 0.6`. Recorded runs had the knees visible on only
53–65% of frames with a detected torso. Watch the `Gait: … conf` field in the
pipeline HUD and step back until it stays above 0.6.

**Raised leg produces nothing at all**
Check `lift-cue` in the log. `none` means neither your feet nor your knees are in
frame, so a leg lift is literally invisible — step back from the camera until at
least your knees are visible. `knees` means the feet are cropped: the lift is
detected, but it comes out as a raised straight leg rather than a folded knee
(§2), because the knee bend cannot be seen without an ankle.

**Robot marches instead of walking across the floor**
No `.motion` clips found. Check the startup log for `Locomotion clips found: …`.
Set `$WEBOTS_HOME`, or drop clips into
`main/controllers/pose_imitation_controller/motions/`.

**Robot turns the wrong way** → flip `TURN_SIGN`.

**Left/right mirrored** → set `SWAP_SIDES = True`.

**Robot falls while walking** → confirm `basicTimeStep` is 20 ms and the contact
properties are present; then lower `LOCOMOTION.walk_conf_min` so it walks less
eagerly, or use `LEG_CONTROL = "engine"` to keep it in place. After a few bad
attempts the controller abandons clips by itself and tells you.

**Robot not moving at all**
1. Is the simulation playing, and the controller running?
2. Pipeline log shows `Webots bridge sending to 127.0.0.1:8765`?
3. `netstat -uln | grep 8765` (Linux) — is the port bound?
4. Controller log shows `tracking`, not `STALE (holding)`?

**Jerky motion** → raise `SMOOTHING_ALPHA` for responsiveness or lower it for
smoothness; reduce camera resolution; run Webots without rendering (`--no-rendering`).

---

## 13. Tests

All the maths is Webots-free and unit-tested from the repo root:

```bash
pytest -q
```

| File | Covers |
|---|---|
| `tests/test_nao_retarget.py` | leg solve **round trip** (project known NAO angles → recover them), self-calibration, lift detection, mirrored roll signs, occlusion, garbage input |
| `tests/test_lower_body.py` | the step sequence, the model-probed lean direction, every safety gate, joint limits |
| `tests/test_walk_motion.py` | clip discovery, overshoot guard, hysteresis, yaw servo latch/relatch/wrapping |
| `tests/test_gait_cues.py` | cadence/phase/stop, and torso yaw sign, monotonicity and magnitude |
| `tests/test_gait.py`, `tests/test_balance.py` | march engine invariants, CoM model and Fibonacci search |
| `tests/test_controller_integration.py` | the real controller against a mocked Webots over real UDP: device setup, leg arbitration, motion hand-off, **freeze resistance** (a clip that never ends, a step that raises, repeated bad locomotion) and leg response with the feet cropped out of frame |
