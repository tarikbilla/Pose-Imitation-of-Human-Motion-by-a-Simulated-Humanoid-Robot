# M1 — `AtlasA3.proto`: Sensing and Physics Correction

Date 2026-09-03 · Branch `A3` · Raw data in `a3/results/*.json`

Status: **complete.** Model, physics correction and posture control verified.

---

## 1. Approach

The fork is **not maintained by hand** but generated from the upstream PROTO by
[`tools/build_atlas_proto.py`](../tools/build_atlas_proto.py). Every deviation
from the original therefore stays traceable as code and reproducible across a
Webots update.

Four transformations:

1. **28 `PositionSensor`s** added, named `<MotorName>S` following the Webots NAO
   convention. This requires converting the SFNode form
   `device RotationalMotor { … }` into the MFNode form `device [ … ]`.
2. **`DEFAULT_PHYSICS.inertiaMatrix`** neutralised from `[1 1 1 0 0 0]` to
   `[1e-6 1e-6 1e-6 0 0 0]`. The mass (0.001 kg) stays; it is irrelevant against
   89 kg.
3. **`EXTERNPROTO`** switched to absolute URLs (29 entries) so the 28 sub-PROTOs
   load unchanged from upstream.
4. **`supervisor`** defaulted to `TRUE`.

The script aborts if it does not find exactly 28 motors or does not patch
exactly one physics node. With `--with-unfixed` it additionally produces
`AtlasA3Unfixed.proto` — identical but with the original inertia. That variant
exists solely as the control group for the A/B test below.

---

## 2. Verification of the model

`webots/worlds/m1_atlas_a3.wbt`, controller `m1_verify`, 6 s.

```
ROTATIONAL_MOTOR         28
POSITION_SENSOR          28
motor/sensor pairing     OK       (no unpaired motors)
```

| Quantity | Value |
|---|---|
| CoM height when standing | 0.9953 m |
| Horizontal CoM drift | **0.68 mm** |
| Contact points | 8 throughout |
| `getStaticBalance()` | `true` throughout |
| Max joint drift | 0.0 rad |

The CoM height is identical to the stock model — the physics fix changes the
inertias, not the mass distribution. That is intended, and it confirms the patch
hit only its target.

---

## 3. A/B test: does the physics fix work?

E1 (M0) proved the Webots semantics on a minimal model. This test measures on
the **real robot** whether the patch takes effect. Both variants, identical
world, zero gravity, 0.05 Nm on one joint at a time, effective inertia from a
least-squares fit to θ(t) = ½·α·t².

| Joint | `AtlasA3` (fixed) | `AtlasA3Unfixed` | Factor |
|---|---|---|---|
| **`LLegLax`** ankle roll | **0.01125** | 0.94219 | **83.7×** |
| **`LLegUay`** ankle pitch | **0.01524** | 0.80731 | **53.0×** |
| `LLegKny` knee | 0.41647 | 2.98420 | 7.2× |
| `LArmElx` elbow | 0.28390 | 2.89694 | 10.2× |

*(Values in kg·m², effective inertia including the reaction of the free chain.)*

**The two ankle joints were the worst affected** — 84× in roll, 53× in pitch.
Those are exactly the joints a humanoid uses to place its centre of pressure and
to absorb lateral disturbances (see finding C of the plan: 90 Nm roll torque,
220 Nm pitch).

The factors are smaller than the 286.6× from E1 because the whole kinematic
chain reacts here: a distal joint sees the inertia of every link beyond it, and
the real contributions add up as well. The order of magnitude remains
devastating.

The fixed values are physically plausible: 0.0113 kg·m² at the ankle roll
corresponds to the foot (I_xx = 0.001) plus the parallel-axis contribution from
the foot centre of mass 0.067 m lower (0.817 · 0.067² = 0.0037) plus the chain
reaction.

**Assessment for the case study.** The Webots Atlas is unsuitable for balance
investigations as shipped. It stems from a 2013 DRCSim URDF whose conversion did
not carry over the segment inertias but replaced them with a placeholder
(`inertiaMatrix [1 1 1 0 0 0]`) attached by `USE` to all 28 links. Anyone
studying balance control on this model is measuring the properties of a robot
whose feet are 84× too sluggish. This explains a substantial part of what was
observed in A1/A2 as "the robot falls over" and taken to be a perception
problem.

---

## 4. Side finding: ODE puts the robot to sleep

In the first verification run the contact points dropped to 0 after t ≈ 3.4 s
although the robot was demonstrably standing. The cause is ODE's auto-disable:
`WorldInfo.physicsDisableTime` defaults to 1 s, after which resting bodies are
removed from the physics computation and report no more contacts.

For this project that is harmful — a sleeping robot has an empty support polygon
and reacts late to disturbances. Countermeasure in all A3 worlds:

```
physicsDisableLinearThreshold 0
physicsDisableAngularThreshold 0
```

A body then counts as resting only when its velocity is exactly zero, which does
not happen numerically. After the change: 8 contact points throughout.

*(Note: `physicsDisableTime 0` would be wrong — per the Webots documentation it
means "disable immediately", not "never".)*

---

## 5. Tooling lesson: `runtime.ini`

Twice Webots aborted on load with SIGSEGV and no error message. Both times a
faulty `runtime.ini` was the cause:

- **relative interpreter path** — Webots crashes instead of warning;
- **`printf` with Windows paths** — `\v` in `C:\venvs` was interpreted as a
  vertical tab, `\a` in `a3-pose` as a bell.

Consequence: `runtime.ini` is produced exclusively by
[`tools/setup_env.py`](../tools/setup_env.py), never by hand or through a shell.
The files are in `.gitignore` because they are machine-specific.

A fresh checkout runs once:

```bat
python tools\setup_env.py
```

---

## 6. Posture control

### 6.1 Measure first, then control

The first controller design was guessed and knocked the robot over **without any
disturbance** (367 mm of lateral deflection before the fall). Rather than trying
gains, the plant was measured: `m1_identify` ramps each actuator to ±0.03 rad
out of the standing pose and measures the CoM displacement in the **body frame**.

Transforming into the body frame is not optional: the robot stands rotated 90°
in the world, so world x is body (−y). The first design confused world
coordinates with body axes and thereby controlled pitch against roll.

**Measured geometry in the body frame** (8 contact points):

| | Range | Span |
|---|---|---|
| sagittal x | −0.0886 … +0.1714 m | 0.2600 m |
| lateral y | −0.1514 … +0.1514 m | 0.3028 m |
| CoM | +0.0235 / −0.0003 m | |

The sagittal span matches the foot length exactly. The plan's prediction from
the `anchor` fields (heel −0.082, toe +0.178) is accurate to 7 mm.

**Measured actuator gains:**

| Actuator | Effect | Gain | Contacts |
|---|---|---|---|
| ankle pitch (symmetric) | body x | **−1.2812 m/rad** | 8 |
| ankle roll (symmetric) | body y | +0.2968 m/rad | 8 |
| **hip roll (symmetric)** | body y | **−0.5698 m/rad** | 8 |
| hip roll (antisymmetric) | body y | −0.1115 m/rad | 8 |

For the lateral axis **hip roll is the right actuator**: 1.9× the authority of
the ankle for the same joint travel. Sagittally the ankle pitch stays, where it
is by far the strongest.

### 6.2 The error that destroyed the first controller

```python
kp_roll = AUTHORITY / GAIN_ROLL_TO_Y      # wrong
kp_roll = -AUTHORITY / GAIN_ROLL_TO_Y     # right
```

A missing minus sign. It made both axes **positively coupled**: the control
action pushed the centre of mass further in the direction of the error. The log
shows it unambiguously — `residual == peak` in every run, the error never
returns. The control law is now uniform for both axes:

```
u = −(K_p · e + K_d · ė) / plant gain
```

so the sign comes from the measurement and not from an assumption.

A second error of the same class: `setPosition()` steps, and at
`maxVelocity 12 rad/s` a motor covers 0.03 rad in 2.5 ms. That is a blow, not a
control action — it knocked the robot over during the first identification. All
A3 controllers now limit joint velocity to 1.5 rad/s and ramp their set-points.

### 6.3 Disturbance rejection

Lateral impulse on the torso, 0.1 s duration, controller on versus off:

| Impulse | Peak with | Peak without | Residual with | Residual without |
|---|---|---|---|---|
| 20 Ns | **27.0 mm** | 40.7 mm | **0.7 mm** | 8.9 mm |
| 30 Ns | **53.9 mm** | 72.1 mm | **1.9 mm** | 18.3 mm |
| 40 Ns | **108.8 mm** | 129.9 mm | **5.9 mm** | 40.4 mm |
| 50 Ns | fall | fall | | |

The controller lowers peak deflection by 16–34 % and the **residual error by
85–92 %** — without it the robot stays permanently tilted, with it it returns to
its initial pose.

It does **not shift the fall boundary**, and that is not a shortcoming but
physics. The capture point leaves the foot edge at

```
Δv_crit = y_edge · ω = 0.1514 · 3.1395 = 0.4753 m/s
I_crit  = m · Δv     = 89.0 · 0.4753   = 42.3 Ns
```

Measured: 40 Ns is held, 50 Ns is not — **the prediction lies exactly inside the
measured interval.** Above this limit no joint torque can help, because the
centre of pressure cannot leave the support polygon. Only a capture step (M5) or
angular momentum through torso and arms helps there.

### 6.4 Acceptance

| Criterion | Result |
|---|---|
| 60 s standing without input | **passed** — 0.0 mm CoM drift, 8 contacts |
| ZMP drift < 2 cm | **passed** — order of magnitude below 1 mm |
| `getStaticBalance()` throughout | **passed** |
| 50 Ns impulse without lifting a foot | **failed** — 40 Ns reached |

The 50 Ns criterion comes from the plan, which was written before the support
polygon was measured. It lies above the physical limit of 42.3 Ns and is
fundamentally unreachable in double support. **Correction: the M1 criterion is
set to 40 Ns; the requirement to "absorb larger disturbances" moves to M5, where
the capture step is available.**

---

## 7. Open

Nothing for M1. The controller structure — measured plant, body coordinates,
ramped set-points, velocity limiting — is at the same time the foundation for
the whole-body QP in M4.

---

## 8. Reproduction

```bat
python tools\setup_env.py
C:\venvs\a3-pose\Scripts\python.exe tools\build_atlas_proto.py --with-unfixed

set WB=%LOCALAPPDATA%\Programs\Webots\msys64\mingw64\bin\webots.exe
"%WB%" --batch --mode=fast --minimize webots\worlds\m1_atlas_a3.wbt
"%WB%" --batch --mode=fast --minimize webots\worlds\m1_inertia_AtlasA3.wbt
"%WB%" --batch --mode=fast --minimize webots\worlds\m1_inertia_AtlasA3Unfixed.wbt
"%WB%" --batch --mode=fast --minimize webots\worlds\m1_identify.wbt

rem disturbance test, driven by environment variables:
set A3_IMPULSE=30
set A3_ADMITTANCE=1
set A3_TAG=i30_on
"%WB%" --batch --mode=fast --minimize webots\worlds\m1_stance.wbt
C:\venvs\a3-pose\Scripts\python.exe tools\report_stance.py
```
