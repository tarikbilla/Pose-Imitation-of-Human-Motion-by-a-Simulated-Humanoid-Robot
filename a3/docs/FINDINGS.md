# A3 — Consolidated Findings

Branch `A3` · Boston Dynamics Atlas in Webots R2025a · rebuild of the pose
imitation case study.

This document consolidates `PLAN.md`, `SUMMARY.md` and `M0_RESULTS.md` through
`M8_RESULTS.md` into a single English record. Every decision is stated together
with the problem that forced it and the measurement that settled it. Failures,
rejected approaches and errors in my own measurements are included, because
several of them cost more time than the engineering did.

---

## Table of contents

1. [What the system does](#1-what-the-system-does)
2. [Ground rules and constraints](#2-ground-rules-and-constraints)
3. [Architecture and why it is split this way](#3-architecture-and-why-it-is-split-this-way)
4. [M0 — Foundations: the inertia finding](#4-m0--foundations-the-inertia-finding)
5. [M1 — Model, sensing, posture control](#5-m1--model-sensing-posture-control)
6. [M2 — GPU perception](#6-m2--gpu-perception)
7. [M3 — Upper-body imitation](#7-m3--upper-body-imitation)
8. [M4 — Whole-body QP, and why it was rejected](#8-m4--whole-body-qp-and-why-it-was-rejected)
9. [M5 — Stepping, and why it could not work](#9-m5--stepping-and-why-it-could-not-work)
10. [M6 — Mass distribution](#10-m6--mass-distribution)
11. [M7 — Walking through the divergent component of motion](#11-m7--walking-through-the-divergent-component-of-motion)
12. [M8 — The puppet: imitation without physics](#12-m8--the-puppet-imitation-without-physics)
13. [Live operation](#13-live-operation)
14. [Errors in my own measurements](#14-errors-in-my-own-measurements)
15. [Decision log](#15-decision-log)
16. [What was not achieved](#16-what-was-not-achieved)
17. [Reproduction](#17-reproduction)

---

## 1. What the system does

A single monocular camera watches a person. The simulated Atlas reproduces
their pose, keeps its feet on the floor, and travels forward and backward the
distance the person travels. The chain runs live:

```
camera -> 2D pose (GPU) -> 3D lifting -> retargeting -> UDP -> Webots controller -> 28 motors
```

Two end states exist, and both are valid results for different questions:

| Track | Physics | What it demonstrates | Status |
|---|---|---|---|
| **M3 + M6 + M7** | full | Upper-body imitation at full amplitude on a robot with corrected mass distribution, plus walking via DCM: 14 steps, 2.6 m, no fall | complete |
| **M8** | none (kinematic) | 1:1 whole-body imitation including legs, with real foot-ground interaction and camera-driven travel | complete |

The M8 puppet is what the live demonstration runs on, because the project goal
was *pose imitation*, not balance research. The physics track is retained
because it answers the harder question and produced the mass and gait findings
that the puppet also relies on.

### Final measured state of the puppet (54 s recording)

| Quantity | Value |
|---|---|
| Forward travel | +0.407 m (camera target +0.580 m) |
| Travel range | 1.113 m |
| Travel tracking error, median | 19.3 mm |
| Ground penetration (measured on the simulation's own foot nodes) | −37.96 mm, in 4 of 3375 frames |
| Frames penetrating more than 10 mm | 0.12 % |
| Airborne | 1.29 % |
| Foot slide, p95 per frame | 0.156 / 0.133 mm |
| Joint tracking lag, p95 | 1.43° |
| Upper-body error, median | 0.937° |
| Pelvis height | 0.848 … 0.922 m |
| Steps (lift events) | 26 |

---

## 2. Ground rules and constraints

These were fixed before the work started and shaped nearly every decision.

- **No code from the previous repository.** The rebuild is independent.
- **No code comments**, except warnings and notes that prevent a regression.
- **Hardware: AMD Radeon RX 6800 XT.** No CUDA, therefore no IsaacGym, no
  TensorFlow-GPU, and no MeTRAbs — the estimator the original PRD assumed. The
  GPU path had to be DirectML through ONNX Runtime.
- **Simulation only.** The robot model may be modified; nothing has to survive
  contact with real hardware. This is what makes the mass redistribution in M6
  legitimate rather than cheating.
- Everything lives in the repository so it can be committed to branch `A3`.

The AMD constraint is the single most consequential one. It removed the
reference implementation path and forced an independent perception stack
(RTMDet + RTMPose + MotionBERT on DirectML) that had to be validated from
scratch.

---

## 3. Architecture and why it is split this way

### The camera must not sit inside the balance loop

The measured tipping time constant of the standing robot is **0.318 s** (M1).
The camera chain costs 80–150 ms. A balance controller closed through the
camera is therefore structurally unstable regardless of how accurate the pose
estimate is — the loop delay is a significant fraction of the divergence time
constant.

**Decision:** two rates, strictly separated.

| Path | Rate | Source of truth |
|---|---|---|
| Reference (what to imitate) | ~30 Hz | camera |
| Control (how to stay up) | 125 Hz | simulation-internal quantities only |

The reference path is allowed to be late and noisy. The control path never
waits for it.

### Transport

The perception process and the Webots controller are separate processes
communicating over UDP on localhost. Two reasons:

1. Webots controllers run inside Webots' own Python; the perception stack needs
   a different environment (ONNX Runtime with DirectML, rtmlib). Keeping them
   in one process would couple two dependency sets that conflict.
2. UDP drops rather than blocks. A slow consumer must never stall the producer,
   and a late pose is worthless anyway — the newest packet is the only one that
   matters. `transport/udp.py` therefore polls and keeps only the latest packet.

---

## 4. M0 — Foundations: the inertia finding

### The problem

The stock Webots Atlas PROTO defines one `Physics` node with
`inertiaMatrix [1 1 1 0 0 0]` and attaches it by `USE` to all 29 links. The
suspicion was that Webots merges this placeholder *in addition to* the real
link inertia rather than replacing it.

### The measurement

`m0_inertia_probe.wbt` places two hinge joints side by side carrying the same
real foot mass and inertia. One reproduces the Atlas nesting pattern exactly
(outer solid with `mass 0.001` and the placeholder inertia, no bounding object,
real foot as a child); the other has only the child. Both are accelerated at
0.1 Nm under zero gravity; angular acceleration comes from a least-squares fit
to θ(t) = ½·α·t².

| Probe | α [rad/s²] | I_eff [kg·m²] | Expected |
|---|---|---|---|
| reference | 29.122 | 0.003434 | 0.0035 (1.9 % deviation) |
| pattern | 0.1016 | 0.984126 | 0.0035 if correct |
| | | **ratio 286.6×** | |

The reference probe hits the known value to within 2 %, which validates the
method before it is used to make a claim. The Atlas pattern is **286.6× too
heavy in rotation**.

**Decision:** `AtlasA3.proto` must neutralise `DEFAULT_PHYSICS`. Without this,
every balance controller is built on a model whose feet are 287× too sluggish —
and the ankles are exactly the joints a humanoid balances over.

### GPU path

`tools/check_directml.py` runs eight chained 1024×1024 matrix multiplications
as an ONNX model, median of 20 runs after 3 warm-ups.

| Execution provider | Median | Throughput |
|---|---|---|
| DmlExecutionProvider | 4.87 ms | 3527 GFLOP/s |
| CPUExecutionProvider | 38.53 ms | 446 GFLOP/s |

**Speedup 7.9×.** DirectML does real GPU work on the RX 6800 XT, and the CPU
stays free for Webots' physics. The GPU path is viable.

### Environment findings that shaped the setup

1. **Windows path limit.** Installing `onnx` into a venv *inside* the repository
   fails with `WinError 206`: the repository name is long and
   `LongPathsEnabled` is `0`. The venv therefore lives at `C:\venvs\a3-pose`,
   outside the repository. `tools/setup_env.py` enforces and checks this.
2. **`runtime.ini` needs absolute paths.** With a relative interpreter path
   Webots crashes on load without any error message (SIGSEGV, exit 139). The
   generated files are in `.gitignore` because they are machine-specific.
3. **Webots does not forward controller `stdout` to the calling console on
   Windows**, even with `--stdout --stderr`. All measurement controllers
   therefore write JSON to `a3/results/`, which is a better basis for
   evaluation anyway.

---

## 5. M1 — Model, sensing, posture control

### What was built

`AtlasA3.proto` adds **28 `PositionSensor`s** named `<MotorName>S` (the stock
model has 28 motors and zero sensors, zero IMUs) and neutralises the placeholder
physics.

### Verification of the fix

Free-swing comparison of the fixed model against an unfixed control:

| Joint | `AtlasA3` (fixed) | `AtlasA3Unfixed` | Factor |
|---|---|---|---|
| `LLegLax` ankle roll | 0.01125 | 0.94219 | **83.7×** |
| `LLegUay` ankle pitch | 0.01524 | 0.80731 | **53.0×** |
| `LLegKny` knee | 0.41647 | 2.98420 | 7.2× |
| `LArmElx` elbow | 0.28390 | 2.89694 | 10.2× |

The two ankle joints were the worst affected, which is the finding that
mattered: those are the joints that carry balance.

Standing verification: CoM height 0.9953 m, horizontal CoM drift **0.68 mm**
over 60 s, 8 contact points throughout, `getStaticBalance()` true throughout,
maximum joint drift 0.0 rad.

### Measure the plant before designing the controller

The first admittance controller was written from assumption and **knocked the
robot over with no disturbance applied**. Two causes, both found only by
measuring:

- a sign error made the feedback positive rather than negative;
- hip roll moves the CoM laterally **1.9× more strongly than ankle roll**, so
  the assumed authority split was wrong.

Measured actuator gains (8 contact points, body frame):

| Actuator | Effect | Gain | Contacts |
|---|---|---|---|
| ankle pitch (symmetric) | body x | −1.2812 m/rad | 8 |
| ankle roll (symmetric) | body y | +0.2968 m/rad | 8 |
| hip roll (symmetric) | body y | −0.5698 m/rad | 8 |
| hip roll (antisymmetric) | body y | −0.1115 m/rad | 8 |

Measured support polygon: sagittal −0.0886 … +0.1714 m (span 0.2600 m), lateral
±0.1514 m (span 0.3028 m).

**Decision, and it became a rule for the whole project:** no controller is
written before its plant is measured. Every controller that was guessed failed;
every controller built on a measured gain worked on the first or second attempt.

### Disturbance rejection

| Impulse | Peak with control | Peak without | Residual with | Residual without |
|---|---|---|---|---|
| 20 Ns | 27.0 mm | 40.7 mm | 0.7 mm | 8.9 mm |
| 30 Ns | 53.9 mm | 72.1 mm | 1.9 mm | 18.3 mm |
| 40 Ns | 108.8 mm | 129.9 mm | 5.9 mm | 40.4 mm |
| 50 Ns | fall | fall | | |

40 Ns is rejected; 50 Ns is not. The capture-point prediction for the limit is
42.3 Ns, so the controller performs essentially at the theoretical bound. The
acceptance criterion asked for 50 Ns and was **not met** — recorded as a
failure rather than argued away.

---

## 6. M2 — GPU perception

### Model stack

| Mode | Detector | Pose | Resolution |
|---|---|---|---|
| `lightweight` | YOLOX-tiny | RTMPose-s | 256×192 |
| `balanced` (default) | YOLOX-m | RTMPose-m | 256×192 |
| `performance` | YOLOX-x | RTMPose-x | 384×288 |

Measured on the RX 6800 XT:

| Input | Mode | Median | FPS |
|---|---|---|---|
| 640×425 | balanced, DirectML | 33.9 ms | 29.5 |
| 640×425 | balanced, CPU | 271.8 ms | 3.7 |
| 1080×1920 | lightweight | 18.8 ms | 53.2 |
| 1080×1920 | balanced | 32.4 ms | 30.9 |

**Speedup 8.0×**, matching the synthetic benchmark from M0 almost exactly.

### The packaging trap

`rtmlib` does not know DirectML, so `perception/runtime.py` patches its backend
table. More importantly: installing `rtmlib` normally pulls in plain
`onnxruntime`, which **silently replaces** `onnxruntime-directml` and removes
`DmlExecutionProvider`. The pipeline then falls back to CPU with no error, only
an 8× slowdown. `tools/setup_env.py` installs rtmlib with `--no-deps`. This is
one of the comments deliberately kept in `requirements.txt`.

### Camera rotation

The camera is mounted portrait. Rotating immediately after capture, before any
inference, turns a 58° diagonal field of view into **51.6° vertical**, which
means the subject can stand at 1.97 m and still fill the frame with **1.8× more
pixels on the body**. Detection quality follows pixels on target, so this is
free accuracy. Mirroring is applied **only for display**, never to the data fed
to the retargeter, so left and right are never silently swapped.

### 3D lifter (MotionBERT)

**DirectML cannot run MotionBERT as exported.** The temporal attention builds
5-D operands `(B, H, N, T, C)` and DirectML has no 5-D MatMul. Folding the
leading axes into one keeps the mathematics identical and makes the graph run
on the GPU. (`tools/build_lifter.py`; this comment is kept in the source.)

| Window | DirectML | CPU (lite) | CPU (full) |
|---|---|---|---|
| 27 | **23 ms** | 50 ms | 119 ms |
| 81 | 108 ms | 163 ms | 376 ms |
| 243 | 135 ms | 625 ms | 1301 ms |

**The input normalisation was wrong at first**: I normalised to the image
rather than to the detection crop, which is what MotionBERT expects.

Accuracy gain over the 2D-only reconstruction, per segment:

| Segment | 2D | 3D | Gain |
|---|---|---|---|
| forearm L | 0.324 | 0.220 | 32 % |
| thigh L | 0.067 | 0.049 | 27 % |
| thigh R | 0.075 | 0.058 | 23 % |
| forearm R | 0.342 | 0.277 | 19 % |
| upper arm L | 0.166 | 0.143 | 14 % |
| shank L | 0.127 | 0.132 | −4 % |
| shank R | 0.129 | 0.129 | 0 % |

The gain is real but modest. **The lifter was adopted for the sign, not the
magnitude** — see M3.

---

## 7. M3 — Upper-body imitation

### Why not RTMW3D

RTMW3D outputs a depth channel. Tested against ground truth, the correlation
between 2D foreshortening and the reported |Δz| is **−0.138** for the left
upper arm and **+0.339** for the right. One of them has the wrong sign; neither
is usable. RTMW3D was rejected.

**Instead: segment foreshortening.** A limb of known length projected onto the
image is shorter the more it points along the optical axis. This is directly
measurable and physically sound.

**Its known limit, stated at the time:** `acos` returns no sign. Whether a
segment points toward or away from the camera cannot be recovered from
foreshortening alone. This limitation later caused three separate defects and
was the reason the 3D lifter was eventually adopted for the legs, torso and
head (M8).

### Arm IK with the slanted shoulder axis

The Atlas shoulder axis is not aligned with any body axis, so the analytic
solution has to account for it.

Two failures on the way:

- **Singularity.** With the elbow extended (`elx ≈ 0`), `ely` has no effect on
  the forearm direction and the solution is undefined.
- **Wrong assumption.** Computing the elbow angle analytically as the angle
  between upper arm and forearm ignores that the shoulder rotation changes the
  frame the forearm is expressed in.

| | before | after |
|---|---|---|
| Upper arm within 5° | 86 % | **100 %** (max 2.4°) |
| Forearm within 5° | 46 % | **100 %** (max 3.7°) |
| Median compute time | 13.5 ms | **2.1 ms** |

### The result, and the damping that had to be there

| | ARM_SCALE 0.55 | ARM_SCALE 1.0 |
|---|---|---|
| Fell | no | yes, after 32 s |
| Packets received | 1588 / 1588 | 1014 |
| Active imitation | 52.9 s | 31.9 s |
| CoM error x, median / max | 6 / 40 mm | 12 / 756 mm |
| Min contact points | 6 of 8 | 2 |

Imitating at full amplitude made the robot fall. The reason is not the
controller: **the arms weigh 24.7 kg of 89 kg, 28 % of total mass.** Moving
them is precisely what pose imitation does. This observation is what motivated
M6, and once the mass was redistributed the damping factor could be removed
entirely.

---

## 8. M4 — Whole-body QP, and why it was rejected

### What was built

A kinematic whole-body controller solving a quadratic program each tick: 28
variables, 30 constraints, **0.044 ms** per tick with OSQP. The kinematic model
was verified independently:

| Quantity | Model | Reference | Source |
|---|---|---|---|
| Total mass | 89.000 kg | 89.00 kg | M0 extraction |
| Hip to sole | 0.9221 m | 0.922 m | M0 geometry |
| Stance width | 0.1780 m | 0.178 m | M0 geometry |
| CoM Jacobian | — | numeric derivative | deviation 3.3 · 10⁻¹² |

Three failures on the way:

1. **The hard CoM constraint was infeasible.** "The CoM must stay inside the
   support polygon" cannot be satisfied within one tick when it is already
   outside, because the rate limit allows only ~0.014 rad per joint. The
   constraint was reformulated as "never move the CoM further outside than it
   already is", which is always solvable; recovery comes from the weighted
   tracking term.
2. **Position from the model rather than the supervisor.** The model supplies
   only the derivative; position must come from the simulation.
3. **The leg references knocked the robot over** and had to be dropped.

The CoM term is what carries the controller:

| | Fell | Packets | max |err_x| |
|---|---|---|---|
| without CoM term | yes, after 18.5 s | 584 | 0.721 |
| with CoM term | **no** | **1476** | **0.125** |

### Why it was rejected

M4 passed its stated acceptance criteria: no fall, squat follows the reference,
stance width follows, `ARM_SCALE` no longer needed.

**But it was accepted on the wrong quantity.** The criteria measured `fell`,
CoM error and QP status — that is *standing safety*, while the claim being made
was *imitation quality*. Measured afterwards on the quantity that actually
matters:

| | Joint error, median | max |
|---|---|---|
| M3 (direct control) | **5.0°** | 21° |
| M4 (whole-body QP) | **15.6°** | 49° |

**M4 is three times worse at the thing the project is about.** The QP spends
joint authority on balance and takes it away from imitation. It was rejected
and M3 remained the upper-body path.

This is the most instructive failure in the project: a component that passes
every test it was given can still be a regression, if the tests measure the
wrong thing.

---

## 9. M5 — Stepping, and why it could not work

A state machine — double support, weight shift, lift, place — on top of the M4
QP.

| | |
|---|---|
| Fell | no |
| Steps | 3 in 49 s |
| States | DS 254 · SHIFT 920 · LIFT 30 · PLACE 26 |
| Contacts during LIFT | 6–8 of 8 |
| Max lateral shift | 0.055 m |

**The foot never actually leaves the ground.** With 6 of 8 contact points still
touching, part of the sole is still down.

For genuine single support the CoM must move over the stance foot, which is
**89 mm** laterally (half the hip width, from M0). The controller reached
**26 mm**.

**My first explanation was wrong.** I recorded this as a physical limit of the
robot. It is not: 89 mm of lateral shift requires **0.123 rad of hip roll
against a 0.436 rad joint limit — 28 % of the available range.** The robot can
do three times what was asked. The QP simply never commanded it.

The real cause is structural: a **kinematic QP on position level cannot generate
angular momentum.** Humans swing the torso and arms when they march precisely to
generate it. Without centroidal dynamics there is no mechanism to produce the
momentum that carries the body over one foot.

**Decision:** abandon the state-machine approach and move to a method that
plans through the divergent dynamics (M7).

---

## 10. M6 — Mass distribution

### The finding

| Group | Atlas | Human (Winter) | Factor |
|---|---|---|---|
| Both arms | **27.8 %** | 10.0 % | **2.8×** |
| Both legs | 30.0 % | 32.2 % | 0.93× |
| Torso | 42.3 % | 49.7 % | 0.85× |
| Head | **0.0 %** | 8.1 % | — |

One Atlas arm weighs **12.35 kg**. This is not a simulation artefact; it is the
real Atlas, whose hydraulics and structure sit in the arms.

Consequence, measured over the reachable upper-body pose space: **22.6 % of all
reachable poses are statically unstable** in the shipped configuration — the CoM
lies outside the support polygon. No controller can prevent a fall in such a
pose. It is not a control problem.

Since only the simulation matters for this case study, the mass distribution is
a free parameter, and `tools/build_atlas_proto.py --masses` makes it one.

### Removing weight does not help; redistributing does

Halving every mass uniformly is **numerically identical** to the original. The
tipping condition is "CoM over the support polygon"; mass cancels out. This
answers the question "very low weight, maybe none at all?" — it changes nothing.
Only distribution acts.

| Profile | Arms | CoM height | Stable poses | Margin p05 | Arm sway |
|---|---|---|---|---|---|
| `spec` (shipped) | 27.8 % | 1.019 m | 77.4 % | −0.0510 m | 0.0737 m |
| `human` | 10.0 % | 1.004 m | 83.6 % | −0.0350 m | 0.0266 m |
| `legs` | 8.0 % | **0.831 m** | 100 % | +0.0262 m | 0.0212 m |
| `core` | **4.0 %** | 0.927 m | 100 % | **+0.0542 m** | **0.0106 m** |
| `combo` | 6.0 % | 0.861 m | 100 % | +0.0360 m | 0.0159 m |

Arm sway — how far pure arm motion drags the CoM — falls by a factor of 7, from
73.7 mm to 10.6 mm with `core`.

`core` concentrates mass in the torso, at roughly navel height, so that limb
motion barely moves the CoM. `legs` lowers the CoM instead. Both reach 100 %
stable poses; `core` has the larger margin.

### Honest negative result

On the actual test recording **the profiles are indistinguishable**: no profile
falls, and imitation error ranges only from 2.61° (`core`) to 2.79° (`legs`).
The recording simply does not contain poses extreme enough to separate them.
The difference only appears under amplified poses and, decisively, when walking
(M7).

**Decision:** `core` becomes the default, on the strength of the static margin
and the walking result, while recording that the everyday recording does not
justify the choice by itself.

### Implementation

The 28 sub-PROTOs are vendored into `webots/protos/vendor/` with rescaled mass
and inertia, preserving each `centerOfMass`. The build verifies 28 vendored
files, 29 physics nodes, 28 position sensors and the mass sum.

---

## 11. M7 — Walking through the divergent component of motion

### Why M5 had to fail

| Strategy | Mechanism | Present in M4/M5 |
|---|---|---|
| Ankle | shift centre of pressure inside the foot | no |
| Hip / swing mass | generate angular momentum about the CoM | no |
| Step | foot placement | only quasi-static |

None of the three momentum-generating strategies existed. M7 adds the third
properly and plans through the unstable dynamics rather than around them.

### Leg inverse kinematics, verified independently

| | left | right |
|---|---|---|
| Cross-check against full forward kinematics | 0.00 m | 0.00 m |
| Position error, median | 0.0133 mm | 0.0124 mm |
| Position error, p99 | 0.0494 mm | 0.0495 mm |
| Angle error, p99 | 0.0008° | 0.0007° |
| Fraction below 0.1 mm | 100 % | 100 % |
| Compute time when tracking | 0.31 ms | 0.35 ms |

**A failure on the way:** the first version started the descent at zero joint
angles, which is a singular configuration for the leg. The knee was driven into
its limit and 54 mm of residual error remained. Seeding with a bent knee and
adding restarts from several knee angles fixed it.

### DCM trajectory generator

The divergent component of motion ξ = c + ċ/ω satisfies ξ̇ = ω(ξ − zmp). It is
the unstable part of the linear inverted pendulum, and it can be planned by
backward recursion from a desired final state.

| Gait | DCM jump at phase boundaries | ZMP back-solve p99 | Travel |
|---|---|---|---|
| in place, 8 steps | 0.00 m | 0.42 mm | 0.00 m |
| forward 0.20 m, 10 steps | 0.00 m | 0.67 mm | 1.75 m |
| forward 0.30 m, 10 steps | 0.00 m | 0.89 mm | 2.62 m |

### Failures that made the controller unusable at first

- **Sole orientation commanded in the world frame** while the robot is rotated
  90° in the world. This demanded 1.57 rad of hip yaw against a 1.14 rad limit.
  IK error went from **654 mm to 0.017 mm** — a factor of 38 000 — once the
  orientation was expressed relative to the heading.
- **CoM reference derived from measured velocity** instead of the integrated
  plan dynamics, which fed measurement noise into the plan.
- **Replanning with a shrinking horizon** made walking worse: a 1-step preview
  gives a 49 mm DCM jump, a constant 4-step preview gives 0.028 mm.
- **Root placed from commanded angles** while the motors lag, producing a
  262 mm teleport. Using measured angles reduced it to 13.8 mm.
- **Heel-toe foot orientation over-constrains the 2-DOF ankle**: 20 mm residual
  against 0.019 mm with a flat sole.

### The sign was measured, not chosen

| Sign | Gain | Result | DCM error median | Lateral drift |
|---|---|---|---|---|
| **+1** | 0.5 | 13 steps | 0.060 | 0.32 m |
| **+1** | 1.0 | 11 steps | **0.041** | 0.24 m |
| −1 | 0.5 | fall after 2 steps | 0.163 | 0.52 m |
| −1 | 1.0 | fall after 1 step | 0.272 | 0.46 m |

### Mass distribution decides whether walking works

This is where the M6 profiles separate. The LIPM assumes massless legs and all
mass at the CoM; `core` (9 % per leg, CoM in the pelvis) matches that assumption
far better than the shipped distribution, and only the redistributed profiles
walk without falling.

### Receding horizon for live operation

A preplanned step count is unusable for a live feed: the robot does not know how
far or how many steps the person will walk. The generator therefore runs on a
constant 4-step preview and replans continuously, and **backward walking is not
a special case** — a negative step length is just another value.

---

## 12. M8 — The puppet: imitation without physics

### Why this path

The physics track imitates the upper body well and walks, but it cannot do both
at once: joint authority spent on balance is authority taken from imitation
(the M4 result, three times worse). The project goal is imitation.

**Decision:** remove physics entirely. All `Physics` nodes are stripped
(`build_atlas_proto.py --kinematic`), so there is no gravity and no contact
force. The robot matches the skeleton 1:1. What must be *added back* is the one
thing physics was providing for free: the feet must stay on the floor, and the
robot must travel when the person travels.

### Root motion from foot anchoring

With no physics, the pelvis position is a free variable that must be computed.
The rule is: the world position of the planted foot does not change.

```
pelvis_world = anchor − R · sole_local(stance foot)
```

Both feet are locked during double support, each with its own IK correction that
forces the sole onto its anchor. Plant and lift are decided from the
*uncorrected* 1:1 foot height, not the corrected one — otherwise a locked foot
can never rise and the robot never steps. That bug produced exactly zero lift
events until it was found.

### Why the foreshortening method fails for legs

Depth from foreshortening is unsigned. For arms this was tolerable; for legs,
torso and head it produced three separate defects:

- the ankle sat 34 cm in front of the hip;
- the torso held a permanent 50° forward lean;
- head pitch ranged 0 … +19.2° and was never negative, so looking down was
  impossible.

All three were fixed by taking legs, torso and head from the 3D lifter with a
slow bias removal. **This is the reason the lifter is in the pipeline** — not
the 20–30 % accuracy gain, but the sign.

### Torso limits inherited from M3

M3 clamped `BackMby` to ±0.10 rad to protect balance. In the puppet there is no
balance to protect, and the clamp was **12× too tight**: 66.4 % of frames sat
against it. Removed, replaced by the real joint limits.

### Workspace geometry decides what is possible

Measured reach envelope of the leg (0.9276 m from pelvis origin to sole):

| Pelvis height | Horizontal foot reach |
|---|---|
| 0.92 m | 119 mm |
| 0.90 m | 225 mm |
| 0.85 m | 372 mm |
| 0.80 m | 470 mm |

Above ~0.88 m the reach collapses. **Stepping without lowering the pelvis
slightly is geometrically impossible**, which is why humans bend their knees to
walk. The controller therefore caps pelvis height by the reach actually required
by the current foot anchors, with a floor (`A3_HEIGHT_FLOOR`, 0.84 m) so it
never looks like a squat.

For the *swing* foot, lowering the pelvis does not help — its target is
expressed in the pelvis frame. What it needs is a bent knee, obtained by
projecting the target onto the reach sphere instead of letting the IK saturate.

### Travel: the strong signal instead of the weak one

The first version derived travel purely from foot anchoring. It produced
−0.908 m while the person walked net forward.

**The diagnosis, camera-independent:** in real walking the foot moves forward
during swing and backward during stance, so the correlation between foot height
and sagittal foot velocity must be positive. Measured: **−0.087** (left) and
**−0.116** (right) — essentially zero. The lifter's sagittal foot signal carries
no gait phase at all. The travel was integrated noise, which is why merely
flipping a sign turned −0.908 m into +1.078 m without improving anything.

**Decision:** take travel from the image, not from the legs. Distance is
measured **absolutely** rather than by integrating a gated velocity, so it
cannot drift. Two independent cues are fused:

| Cue | Basis | Standing spread | S/N |
|---|---|---|---|
| Body measures (median of 5) | apparent size, d ∝ 1/px | 159 mm | 1.57 |
| Ground plane | ankle image-y with a fixed camera | 153 mm | 1.73 |
| **Fusion** | plane regressed against size | **140 mm** | **1.79** |

Camera height is estimated from the data itself (0.916 m, R² = 0.79) and focal
length from the image geometry. The two cues agree to 0.885, which also shows
that most of the apparent "standing noise" is **real** weight shifting rather
than measurement error.

On the test recording this yields −0.45 → +0.65 → −0.41 → +0.58 m, matching the
described sequence of forward, backward, forward toward the camera.

### Combining travel with ground contact

The pelvis follows the stance foot in the short term and the camera signal in
the long term (complementary filter, `A3_TRAVEL_TAU` 0.35 s). Because the IK
correction forces the planted foot onto its anchor regardless of where the
pelvis is, ground contact survives. Three limits keep it physical:

- **Reach room instead of a fixed drift cap**: the pelvis may only lead as far
  as the stance leg can carry at an admissible height.
- **Height ceiling with a floor**, as above.
- **Swing target projected onto the reach sphere.**

### Final state

| Quantity | Before | After |
|---|---|---|
| Forward travel | −0.908 m (wrong direction) | **+0.407 m** (target +0.580 m) |
| Travel tracking error, median | — | 19.3 mm |
| Penetration | not honestly measured | −37.96 mm in 4 of 3375 frames |
| Frames deeper than 10 mm | — | 0.12 % |
| Airborne | — | 1.29 % |
| Slide p95 | — | 0.156 / 0.133 mm |
| Joint tracking lag p95 | 5.80° | **1.43°** |
| Upper-body error, median | 0.855° | 0.937° |
| Neck joint `NeckAy` | −10.3 … +5.3° | **−26.7 … +18.5°** |

The larger leg correction (median 5.4° → 9.3°) is the price of travelling at
all: the stance leg carries the difference between the 1:1 pose and the camera
path.

### The neck

The lifter resolves head pitch poorly — 14° raw range, sd 2.3° — because the
head-to-neck segment is short. `head_from_2d` instead measures the nose against
the ear midpoint, normalised by head width: a direct image measurement needing
no depth, with a 32.6° range. The lifter remains the fallback when the ears are
occluded.

---

## 13. Live operation

Until this point the controller read a **precomputed** angle file. The live path
replaces it with a stream:

```
camera --> Pose2D --> FullBodyRetargeter --> UDP :8768 --> a3_puppet
              |            + LocomotionTracker
              +--> monitoring window with the 2D skeleton
```

New components: `transport/angles.py` (packet format `a3-angles`, 440 bytes for
24 joints), `tools/live_puppet.py` (driver and monitoring), `tools/run_live.py`
(launcher). `transport/udp.py` now takes a codec module, so the older upper-body
stream keeps working unchanged.

### Measured performance

| Stage | Median |
|---|---|
| 2D pose (wholebody, lightweight, DirectML) | 25.6 ms |
| Retargeting (lifter + leg IK + arms) | 8.1 ms |
| **Total** | **34.0 ms, about 29 fps** |

In operation: 30 fps with no person in frame, 16–22 fps under full processing.

Live camera session (30 s, person in frame throughout): 476 frames, 476 packets
sent, 450 received in Webots, travel +0.063 m over a 0.779 m range, 15 plant
events, airborne 1.33 %, upper-body error 0.223°.

### The camera died after 200 frames

ffmpeg writes 1080×1920 raw frames at 30 fps into a pipe — **187 MB/s**. The
consumer reads more slowly, the buffer fills, and ffmpeg reports
`real-time buffer [Brio 100] too full ... frame dropped!` until the stream
breaks.

**Decision:** `FFmpegCameraSource` runs a **reader thread** that continuously
drains the pipe and keeps only the newest frame (`drain=True`, `frames_dropped`
counts the discards). The source then runs indefinitely, and latency stays low
because a stale frame is never processed. Discarding old frames is correct for
live imitation, where currency matters more than completeness.

### Calibration without a person in frame

The locomotion tracker initially calibrated on any detection, including a
low-confidence one with nobody in view. The datum landed at −1.20 m and the
airborne fraction rose to 21 %. Distance is now only updated when the
retargeter has produced a **valid** pose. Airborne fell to 1.33 %.

---

## 14. Errors in my own measurements

These cost more time than the engineering, and they share one shape: a metric
calibrated for one configuration, applied to another, and not re-checked.

1. **Slide measured against my own set-points.** The metric compared foot
   position against the very equation that produced it — a tautology reporting
   0.17 mm. Replaced by reading the simulation's own foot node through
   `getFromProtoDef`.
2. **Lag compared against the wrong time step.** Sensors are read *before*
   `setPosition`, so comparing them to the *new* command reported 8.3° of lag
   where there was none. Against the *previous* command the true median lag is
   0.0°.
3. **Foot node compared against a pelvis position not yet applied.** The world
   file places the pelvis at z = 1.0; 1.0 − 0.846 = 0.154 is exactly the value
   measured. The "72.8 mm offset" and the conclusion that the robot was floating
   were both wrong.
4. **Three foot-contact metrics were tautologies** in an earlier iteration;
   honest measurement then showed 1355/3249 mm of real sliding.
5. **"All support switches go to R"** — I had printed every fourth entry of an
   alternating sequence.
6. **A "180° ArmEly jump"** — `.get(name, 0.0)` for frames where the joint does
   not exist.
7. **A fall detector with an absolute threshold** reported low-CoM profiles as
   fallen while they stood still.
8. **An unverified assumption declared a fact, then wrongly refuted.**
   `PLAN.md` assumed the segment masses were in the sub-PROTOs. That is true —
   the sum is exactly 89.000 kg — but it was never verified; in M6 I first
   declared it false. One fetch of `UtorsoSolid.proto` would have settled it
   either way.

### Reproducibility is a property of the measurement setup

M3 and M4 were measured in real time. A repeat run gave 10.4° instead of 5.0°
and fell where it previously had not. **All single-run statements in those
chapters do not hold.** From M6 onwards Webots runs in fast mode without
rendering against a recording, three runs per configuration.

Imitation error is then reproducible to about 0.1°. The fall decision is not:
UDP timing is not tick-synchronous, and near the stability boundary the outcome
flips between runs. Fall counts are therefore reported over several runs rather
than as yes/no — a profile that falls in one run of three is exactly on the
boundary.

---

## 15. Decision log

| # | Decision | Problem it solves | Evidence |
|---|---|---|---|
| 1 | Neutralise `DEFAULT_PHYSICS` in the PROTO | Webots merges the placeholder inertia additively; ankles 287× too sluggish | M0 probe, 286.6× ratio validated to 1.9 % |
| 2 | Add 28 `PositionSensor`s | Stock model has zero sensors and zero IMUs | M0 device audit |
| 3 | DirectML through ONNX Runtime | No CUDA on RX 6800 XT; CPU is 8× too slow | 4.87 vs 38.53 ms |
| 4 | Install rtmlib with `--no-deps` | Plain `onnxruntime` silently replaces the DirectML build | M2 packaging trap |
| 5 | Two loop rates, camera outside the balance loop | Tipping time constant 0.318 s vs 80–150 ms camera latency | M1 |
| 6 | UDP transport, keep only the newest packet | A late pose is worthless; the producer must never stall | architecture |
| 7 | Rotate immediately after capture | 51.6° vertical FOV, 1.8× more pixels on the subject | M2 |
| 8 | Reject RTMW3D depth | Foreshortening correlation −0.138 / +0.339, one sign wrong | M3 |
| 9 | Segment foreshortening for arms | Directly measurable, physically sound | M3 |
| 10 | Adopt the 3D lifter for legs, torso, head | Foreshortening is unsigned; caused 34 cm ankle offset, 50° torso lean, no downward head pitch | M8 |
| 11 | Fold 5-D attention axes in the lifter export | DirectML has no 5-D MatMul | M2 |
| 12 | Reject the whole-body QP | 15.6° vs 5.0° joint error — 3× worse at imitation | M4 correction |
| 13 | Abandon the stepping state machine | A position-level kinematic QP cannot generate angular momentum | M5 |
| 14 | Redistribute mass, profile `core` | Arms are 27.8 % of mass vs 10 % in humans; 22.6 % of poses statically unstable | M6 |
| 15 | Do not reduce total mass | Halving all masses is numerically identical; only distribution acts | M6 |
| 16 | DCM planning with a constant 4-step preview | Shrinking horizon gives 49 mm jump vs 0.028 mm | M7 |
| 17 | Express sole orientation relative to heading | World-frame orientation demanded 1.57 rad against a 1.14 rad limit | M7, 654 mm → 0.017 mm |
| 18 | Drop physics entirely for the puppet | Authority spent on balance is taken from imitation | M8 |
| 19 | Plant/lift decided on the uncorrected pose | Locked feet can otherwise never rise; zero lift events | M8 |
| 20 | Travel from the image, not the legs | Sagittal leg signal carries no gait phase (correlation −0.09 / −0.12) | M8 |
| 21 | Absolute distance, not integrated velocity | Integration drifts; the velocity gate discarded over half the travel | M8 |
| 22 | Fuse size and ground-plane cues | Independent cues, correlation 0.885, S/N 1.57 → 1.79 | M8 |
| 23 | Cap pelvis height by required reach, with a floor | Reach collapses from 372 mm to 119 mm between 0.85 m and 0.92 m pelvis height | M8 |
| 24 | Project the swing target onto the reach sphere | Prevents IK saturation; produces the bent swing knee naturally | M8 |
| 25 | Limit the command rate to what the joints can follow | Commanded 61 rad/s against a much lower achievable rate; lag p95 5.8° → 1.43° | M8 |
| 26 | Neck pitch from a 2D nose-to-ear measure | Lifter resolves it poorly: 14° range vs 32.6° | M8 |
| 27 | Draining reader thread on the camera | 187 MB/s pipe backpressure killed the stream after ~200 frames | live |
| 28 | Update the distance datum only on a valid pose | Ghost detections put the datum at −1.20 m, airborne 21 % → 1.33 % | live |
| 29 | Measure against the simulation's own nodes | My own metrics were tautologies | §14 |
| 30 | Three runs per configuration, fast mode, no rendering | Real-time single runs are not reproducible (5.0° vs 10.4°) | M6 |

---

## 16. What was not achieved

- **Genuine stepping under full physics with imitation at the same time.** The
  DCM walker walks, and the puppet imitates, but no configuration does both at
  full quality simultaneously.
- **50 Ns disturbance rejection** (M1 acceptance criterion). 40 Ns is reached,
  which is essentially the capture-point bound of 42.3 Ns.
- **Wrist joints** `ArmUwy` and `ArmMwx` are unmapped; hand normals are
  transmitted but not used.
- **Root rotation is fixed** — the robot does not turn its body to follow the
  person's heading.
- **Camera calibration for metric intrinsics.** Focal length is derived from the
  field of view, not from a calibration target. This scales absolute travel
  proportionally.
- **Torso yaw sign** is unverified (magnitude corrected).
- **Squat to 0.25 m and single-leg stance** are untested — the recording
  contains neither.
- **Travel magnitude** reaches 70 % of the camera-measured distance
  (+0.407 m of +0.580 m). The remainder is the price paid for keeping ground
  contact and leg-pose fidelity.

---

## 17. Reproduction

All measurement controllers write JSON to `a3/results/`, because Webots does not
forward controller stdout on Windows.

| Check | Command |
|---|---|
| GPU path | `python tools/check_directml.py` |
| PROTO integrity, mass sum | `python tools/check_proto.py` |
| Leg IK | `python tools/check_legik.py` |
| DCM generator | `python tools/check_lipm.py` |
| Receding horizon | `python tools/check_receding.py` |
| Gait phase in a recording | `python tools/check_gait_phase.py` |
| Rebuild the PROTOs | `python tools/build_atlas_proto.py --masses core [--kinematic]` |
| Angles from a recording | `python tools/make_angles.py` |
| Puppet against a recording | `python tools/run_puppet.py --env A3_ANGLES=<file>` |
| Live session | `python tools/run_live.py` |

See `OPERATIONS.md` for day-to-day use and `PORTING_LAB.md` for moving to the
laboratory hardware.
