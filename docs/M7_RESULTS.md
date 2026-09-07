# M7 — Walking Through the Divergent Component of Motion

Date 2026-09-05 · Branch `A3` · Raw data in `a3/results/legik_check.json`,
`lipm_check.json`, `walk_envelope.json`, `walk_profiles.json`, `walk_open.json`

Status: **the robot walks.** 14 steps, 2.6 m of locomotion, no fall,
reproducible across two runs. The 2 m demanded in plan §8 are reached. A
systematic lateral drift of 0.54 m over that distance remains open.

---

## 1. Why M5 had to fail

M5 planned steps as a state machine on top of a kinematic QP at position level.
Such a QP cannot generate angular momentum; it can only choose joint angles so
that the centre of mass stays quasi-statically inside the support polygon. That
is exactly what rules out walking, because when walking the centre of mass
leaves the instantaneous support area — that is not a disturbance but the
mechanism.

A humanoid has three strategies for keeping balance:

| Strategy | Mechanism | Present in M4/M5 |
|---|---|---|
| Ankle | shift the centre of pressure inside the foot | no |
| Hip / swing mass | generate angular momentum about the CoM | no |
| Step | foot placement | only quasi-statically |

M7 replaces the QP with the **divergent component of motion** (DCM, capture
point):

```
xi = c + c_dot / omega        with omega = sqrt(g / z_c)
xi_dot = omega * (xi - zmp)
```

`xi` is the unstable mode of the linear inverted pendulum. The stable part takes
care of itself; only `xi` has to be controlled. Because the dynamics are
divergent, they are solved **backwards** from the last planned foot position —
which produces a bounded centre-of-mass trajectory instead of a runaway one.

---

## 2. Foundations, verified individually

### 2.1 Leg inverse kinematics

Numeric, damped least squares over a dedicated chain across only the six leg
joints (`kinematics/legik.py`, checked with `tools/check_legik.py`, 2000 poses
per side).

| | left | right |
|---|---|---|
| Cross-check against full forward kinematics | **0.00 m** | **0.00 m** |
| Position error, median | 0.0133 mm | 0.0124 mm |
| Position error, p99 | 0.0494 mm | 0.0495 mm |
| Angle error, p99 | 0.0008° | 0.0007° |
| Fraction below 0.1 mm | **100 %** | **100 %** |
| Compute time when tracking | 0.31 ms | 0.35 ms |

**A finding from the way there.** The first version started the descent with
extended legs. There the leg Jacobian is singular and the method picks the wrong
branch: it pushed the knee against its limit at 0 and stopped with 54 mm of
residual error. It became visible through the sign — the descent wanted
`Kny −0.807`, the solution sat at `+0.791`. With a bent starting pose
(`DEFAULT_SEED`) and four restarts across different knee angles the error drops
by a factor of 4000. Forward kinematics and Jacobian were correct throughout;
both were cross-checked numerically (Jacobian against finite differences:
7.9 · 10⁻¹¹).

### 2.2 DCM trajectory generator

`kinematics/lipm.py` and `kinematics/gait.py`, checked with
`tools/check_lipm.py`.

| Gait | DCM jump at phase boundaries | ZMP back-solve p99 | Travel |
|---|---|---|---|
| in place, 8 steps | **0.00 m** | 0.42 mm | 0.00 m |
| forward 0.20 m, 10 steps | **0.00 m** | 0.67 mm | 1.75 m |
| forward 0.30 m, 10 steps | **0.00 m** | 0.89 mm | 2.62 m |

The sharpest test is the **ZMP back-solve**: from the integrated
centre-of-mass trajectory, the implicit centre of pressure is recovered through
`z = c − c_ddot / omega²` and compared against the planned one. The remaining
0.4 to 0.9 mm are numerical noise of the finite difference.

The capture step — where the foot must go so that the DCM is captured on
touchdown — is exact (residual 0.00 m).

---

## 3. The controller

```
gait parameters -> foot plan -> ZMP reference -> DCM backwards -> CoM trajectory
                                                                        |
measurement: CoM, CoM velocity (supervisor) -> DCM measured -> correction
                                                                        |
                          pelvis target pose -> foot poses relative -> leg IK
```

Two properties make this possible with **position control** at all:

**The centre of mass lies in the pelvis.** With profile `core` from M6 it sits
at [0.022, 0.000, 0.009] in the pelvis frame, that is practically at the pelvis
origin; in the shipped state it is 10 cm above. Pelvis control is therefore
effectively centre-of-mass control, and the planned CoM trajectory can be
commanded directly through the leg IK. That was not a design goal of M6 but a
side effect measured afterwards.

**Light legs.** `core` has 9 % mass per leg. The LIPM assumes massless legs;
this very modelling error otherwise breaks simple LIPM controllers.

### Two errors that made the controller unusable at first

**Target sole orientation in the world frame instead of the heading frame.** The
robot stands rotated 90° in the world. Requiring "foot parallel to the world
axes" therefore demands 1.57 rad of hip rotation; the joint can do 1.14. The IK
error rose to 654 mm and the robot fell. With the sole in the heading frame it
falls to **0.017 mm** — a factor of 38 000.

**Centre-of-mass reference from the measured velocity.** The first version
formed `c_ref = xi_ref − v_measured/omega`. That makes the reference chase the
measurement instead of being the plan. The correct way is to integrate it with
the plan dynamics: `c_ref_dot = omega (xi_ref − c_ref)`.

---

## 4. Results

### 4.1 Operating envelope, 10 steps, profile `core`

| Step length | Gain | Fall | Steps | Travel | Lateral drift | DCM error | IK error |
|---|---|---|---|---|---|---|---|
| 0.10 m | 0.0 | no | 10/10 | 0.79 m | 1.5 mm | 0.069 | 0.021 mm |
| 0.10 m | 0.3 | no | 10/10 | 0.87 m | 1.6 mm | **0.020** | 0.023 mm |
| 0.15 m | 0.0 | no | 10/10 | 1.23 m | 2.4 mm | 0.071 | 0.025 mm |
| 0.15 m | 0.3 | no | 10/10 | 1.28 m | 3.6 mm | **0.041** | 0.026 mm |
| 0.20 m | 0.0 | no | 10/10 | 1.68 m | 5.8 mm | 0.065 | 0.040 mm |
| 0.20 m | 0.3 | no | 10/10 | 1.69 m | 6.5 mm | **0.052** | 0.041 mm |

The DCM feedback lowers the tracking error by up to a factor of 3.5 without
touching standing safety.

### 4.2 The sign was measured, not chosen

| Sign | Gain | Result | DCM error, median | Lateral drift |
|---|---|---|---|---|
| **+1** | 0.5 | 13 steps | 0.060 | 0.32 m |
| **+1** | 1.0 | 11 steps | **0.041** | 0.24 m |
| −1 | 0.5 | fall after 2 steps | 0.163 | 0.52 m |
| −1 | 1.0 | fall after 1 step | 0.272 | 0.46 m |

This is the same discipline as with `TILT_PITCH_SIGN` in the NAO project and
with the sign errors from M1 and M3: a guessed sign is positive feedback that
looks like a controller that is merely too weak.

### 4.3 The mass distribution decides whether walking works

14 steps, 0.20 m, gain 0.3:

| Profile | Arm share | Fall | Steps | Travel |
|---|---|---|---|---|
| **`core`** | 4.0 % | **no** | **14/14** | **2.68 m** |
| `combo` | 6.0 % | yes | 13/14 | 2.40 m |
| `human` | 10.0 % | yes | 13/14 | 2.84 m |
| `spec` (as shipped) | 27.8 % | yes | 12/14 | 2.48 m |

Only `core` survives the full distance. The mass redistribution from M6 is
therefore not a convenience but a precondition — and the ranking agrees with the
static arm sway from M6 §2.

### 4.4 Acceptance

| Criterion (plan §8) | Result |
|---|---|
| 10 steps in place | **passed** — 14/14, drift 3 mm |
| 2 m of locomotion | **passed** — 2.61 m and 2.68 m across two runs |
| No fall | **passed** for `core` up to 14 steps |
| Foot actually lifts | **passed** — 45.5 mm of lift measured |

---

## 5. What is open

**Lateral drift of 0.54 m over 2.6 m of travel.** It is systematic (0.540 and
0.550 m across two runs), not random. It occurs **only when walking forward**:
14 steps in place produce 3 mm. It is therefore not a balance error but a
heading error — the foot plan lives in the initial heading frame while the robot
slowly veers off when walking forward. The fix is lateral step adjustment with
heading feedback.

**Step adjustment through the capture point is built but unusable.**
`A3_ADAPT_STEPS=1` places the foot on the full capture point; that destroys the
periodic gait and leads to a fall after three steps. The correct approach is to
correct only the *deviation* from the nominal plan, limited to a few
centimetres.

**Replanning at measured feet is built but worse.** `A3_REPLAN=1` re-anchors the
plan at every touchdown. The DCM set-point jumps in the process, and the jerk
costs more than the anchoring gains (fall after 5 to 7 steps against 14/14
without). A clean implementation would have to set up the new plan so that its
initial DCM matches the current one.

**The camera does not drive the gait yet.** All results here run with a
prescribed step length and cadence. Connecting it to perception — cadence and
step length from the human's foot lift — is the next step. That needs a
recording with walking motion; the existing one was believed to contain none
(see §8, where that assessment turned out to be wrong).

---

## 6. Reproduction

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\check_legik.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_lipm.py

C:\venvs\a3-pose\Scripts\python.exe tools\walk_trial.py ^
    --profiles core --step-length 0.20 --gain 0.3 --sign 1 ^
    --steps 14 --runs 2
```

Adjustments: `A3_STEP_LENGTH`, `A3_STEP_COUNT`, `A3_DCM_GAIN`, `A3_DCM_SIGN`,
`A3_SINGLE_S`, `A3_DOUBLE_S`, `A3_CLEARANCE`, `A3_MASS_PROFILE`,
`A3_ADAPT_STEPS`, `A3_REPLAN`.

---

## 7. Addendum: running operation instead of a preplanned step sequence

The version in section 3 planned a **fixed number** of steps of fixed length in
advance. For a live feed that is unusable: the robot must start walking when the
person does, without knowing how many steps or how far.

### Why the first replanning attempt failed

The first attempt to replan at every touchdown was markedly worse than not
replanning at all (fall after 5 to 7 steps against 14/14). The cause is
measurable: I planned with a **shrinking** remaining horizon. The closer the end
of the plan, the more strongly the backward recursion shifts the DCM set-point.

`tools/check_receding.py` measures the jump of the DCM set-point on replanning:

| Preview horizon | DCM jump |
|---|---|
| 1 step | **49.0 mm** |
| 2 steps | 7.0 mm |
| 3 steps | 0.45 mm |
| **4 steps** | **0.028 mm** |
| 6 steps | 0.0001 mm |

With a **constant** preview of four steps, replanning is practically continuous.
That is the difference between an offline-planned trajectory and a controller
with a receding horizon.

### Backward is not a special case

The same measurement with varying step length — forward 0.20 m, stop, backward
−0.18 m, stop, forward — gives jumps of 0.010 to 0.028 mm. Direction changes and
stopping fall out of a **signed step length**; no special handling and no
advance knowledge of the motion is needed.

Confirmed in simulation (script `0:0.18, 7:0.0, 10:-0.16`):

| Phase | Time | Steps | Travel |
|---|---|---|---|
| forward | 0.8 – 7.4 s | 9 | x 0.18 → **1.62 m** |
| standing | 7.4 – 10.4 s | 0 | double support, 8 contact points |
| backward | 10.8 – 17.4 s | 9 | x 1.46 → **0.18 m** |

The IK error stays at 0.03 mm through the first two phases.

### Lateral step adjustment

Limited correction of the landing position from the DCM error
(`A3_ADAPT_GAIN`, `A3_ADAPT_LIMIT`), 18 s continuous run at 0.18 m:

| Gain | Steps | Travel | Lateral drift | DCM error, max |
|---|---|---|---|---|
| 0.0 | 16 | 2.97 m | −0.231 m | 0.522 |
| 0.5 | 13 | 1.93 m | **+0.045 m** | 0.295 |
| 1.0 | 15 | 2.21 m | −0.180 m | **0.274** |

It halves the maximum DCM error and pushes lateral drift to a fifth, but does
not prevent the fall. Unlike the earlier attempt with the full capture point
(fall after three steps) it does no harm.

**Limit in continuous running: about 16 steps, or 2.9 m.** After that it tips.
Lateral drift remains the main suspect.

---

## 8. Gait analysis from the recording

The existing recording does contain walking motion — that was initially assessed
incorrectly. Torso length as a distance measure shows it unambiguously:

| Time | Torso | Motion |
|---|---|---|
| 40.0 → 45.6 s | 303 → 538 px | forward toward the camera |
| 45.6 → 50.4 s | 538 → 313 px | backward |
| 51.2 → 53.4 s | 318 → 525 px | forward |

Ankle confidence: median 0.93, minimum 0.82.

`perception/gait_detect.py` derives touchdown events, cadence and signed step
length from this. Foot lift is measured relative to its own smoothed baseline
and normalised to torso length; distance as the reciprocal of torso length.

| t | Foot | Cadence | Step measure |
|---|---|---|---|
| 42.53 | L | 0.50 | +0.013 |
| 43.23 | R | 1.43 | +0.069 |
| 44.20 | L | 1.03 | +0.119 |
| 45.03 | R | 1.20 | +0.148 |
| 51.67 | R | 1.20 | +0.044 |
| 52.43 | L | 1.30 | +0.147 |
| 53.17 | R | 1.36 | +0.178 |

The cadence of 1.2 to 1.4 Hz corresponds to 0.7 to 0.8 s per step, that is
normal walking.

**Two defects, open.** The backward phase between 45 and 50 s produces no
touchdown events — when walking backward the foot lift is smaller and the
thresholds do not trigger. And before second 38, where the subject only stands
and moves their arms, seven false triggers occur. Both must be fixed before the
detector is allowed to drive the gait; a false trigger while standing would make
the robot walk off for no reason.

**Not yet connected.** Detector and gait planner still run separately.
Connecting them is the next step.
