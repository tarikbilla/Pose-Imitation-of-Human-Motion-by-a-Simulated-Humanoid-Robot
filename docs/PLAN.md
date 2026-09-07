# A3 — Pose Imitation with Atlas: Rebuild

Status: draft for approval · Branch `A3` · As of 2026-09-03

Complete rebuild. No code from `src/`, `main/` or `scripts/` is carried over, no
bug inherited from it, nothing in it repaired. The existing repository was read
solely to identify and avoid the failure causes of A1 (MediaPipe/NAO) and A2
(MeTRAbs/NAO).

Objective: **maximum motion fidelity with guaranteed standing safety, in real
time.** These three quantities compete. This plan makes the conflict explicit
rather than hiding it.

---

## 1. Constraints

| | |
|---|---|
| GPU | **AMD Radeon RX 6800 XT, 16 GB** — no CUDA |
| CPU | i7-11700KF — **thermally limited, must be spared** |
| RAM | 32 GB · OS Windows 11 · Webots R2025a |
| Camera now | **Logitech Brio 100** — 1080p30, 58° dFOV, fixed focus, 2 MP |
| Camera later | Sony (specification open) |
| Team | one person |

Two of these rows determine the architecture more than anything else: **no
CUDA** and **a tight CPU budget**. Everything heavy belongs on the GPU, and the
GPU path under Windows + AMD is ONNX Runtime with the DirectML provider.

---

## 2. The central diagnosis

Falling over is **not a depth-estimation problem**. It is a control-architecture
problem, and a better 3D pose would not have saved A1/A2.

Standing, Atlas has a CoM height of about 0.95 m. As an inverted pendulum:

```
omega = sqrt(g / z_c) = sqrt(9.81 / 0.95) = 3.21 rad/s
tau_tip = 1/omega = 0.31 s
```

0.31 s is the window within which a disturbance must be corrected. The Brio 100
delivers 30 fps — with inference, filtering and transport, realistically
80–150 ms of total latency, that is **9 measurements across the whole tipping
window, with a third to a half of it pure dead time**.

**A balance controller closed through the camera is therefore structurally
unstable** — independent of pose accuracy. That is exactly what A1 and A2 built:
landmarks went straight into leg joint angles. The roughly 1,700 lines of
heuristics in `balance.py`, `lower_body.py` and `gait.py` are the retrospective
attempt to damp that instability with clamps and gates. It does not converge,
because the cause lies in the loop structure.

**Consequence: the camera comes out of the control loop.** It supplies a
*reference*, never a *control action*.

---

## 3. Choice of control approach

Three approaches are candidates. The choice determines the potential for
success, so the trade-off is laid out openly.

### A — heuristic gates and clamps

What A1/A2 did. A collection of rules instead of a model. **Rejected**: not
convergent, not justifiable, not extensible.

### B — reinforcement-learning motion imitation

The state of the art. [H2O](https://arxiv.org/abs/2403.04436),
[OmniH2O](https://arxiv.org/abs/2406.08858) and
[ExBody2](https://arxiv.org/pdf/2412.13196) demonstrate exactly our goal:
real-time whole-body teleoperation of a man-sized humanoid **from a single RGB
camera**, including walking, stepping and turning, without falling. The policy
implicitly learns which motion it can afford.

The highest ceiling — but unreachable on this hardware:

- Training needs massively parallel simulation. MJX/JAX on AMD runs exclusively
  under Linux with ROCm ([AMD tutorial 03/2026](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html)
  explicitly excludes Windows). Isaac Gym is CUDA-only.
- The target simulator is Webots. Training would happen in MuJoCo/Isaac → an
  additional sim-to-sim gap on top.
- It would require an AMASS dataset retargeted onto Atlas plus reward design
  plus domain randomisation.

That is a multi-month project of its own for a team, not a module inside this
one. **Rejected for the main path, documented as an outlook (§10).**

### C — model-based whole-body control with a QP ← **chosen**

Classical humanoid control, and documented in the literature for exactly our
task: [*A Whole-Body Motion Imitation Framework from Human Data for Full-Size
Humanoid Robot*](https://arxiv.org/html/2508.00362) describes the same chain —
geometric retargeting as a QP, model-based control with dynamic constraints, ZMP
inside the support polygon.

The decisive point against approach A: **the task is formulated as an
optimisation problem, not as a collection of rules.** Per control step:

```
minimise    ||q_ddot - q_ddot_desired||^2_W    (motion fidelity: as close as possible)
subject to  ZMP in support polygon             (standing safety: hard constraint)
            joint angle and rate limits
            torque limits
            contact forces in the friction cone
```

That is precisely the semantics wanted: *imitate as exactly as possible, and do
not fall over while doing it*. Motion fidelity is the objective, standing safety
is a constraint — not the other way round and not mixed together. If the wish is
unsatisfiable, the QP automatically supplies the next-best satisfiable solution
instead of guessing it heuristically.

In addition: no training, deterministic, debuggable, runs directly in Webots,
and at about 30 variables the solver is below 1 ms — negligible in the CPU
budget.

**Honesty caveat.** "1:1" is not fully achievable physically for the lower body.
A human walks over a toe joint and rolls the foot; Atlas has a 0.26 × 0.125 m
flat plate and 89 kg. For arms, head and trunk, 1:1 is realistic. For the legs
the QP delivers the demonstrable optimum within the physical limits — that is
the honest maximum, and approach B promises no more for the lower body without
training.

---

## 4. Pipeline

### 4.1 Split by compute unit

Guiding principle: **everything neural on the GPU, the CPU only does physics and
algebra.**

| Stage | Where | Load |
|---|---|---|
| Camera capture, colour conversion | CPU | low, YUY2 instead of MJPEG |
| Person detection (RTMDet-nano) | **GPU** (DirectML) | ~2 ms |
| 2D keypoints (RTMPose-m, Halpe-26) | **GPU** (DirectML) | ~4 ms |
| 2D→3D lifting (MotionBERT, causal) | **GPU** (DirectML) | ~5 ms |
| Filtering, reference extraction | CPU | µs |
| Webots physics (ODE) | CPU | **main load, unavoidable** |
| Retargeting QP + whole-body QP | CPU | < 1 ms |

The reference figure for model selection: **RTMPose-m reaches 90+ FPS on an
i7-11700 with ONNX Runtime alone** ([RTMPose paper](https://arxiv.org/abs/2303.07399))
— exactly this CPU. On the GPU the headroom is correspondingly large, and more
importantly: the CPU stays free for Webots and cool.

Perception and Webots run as **separate processes**, connected over UDP. One
stage therefore never blocks the other, and perception is testable without
Webots.

### 4.2 Data flow

```
+- 30 Hz - reference path - mostly GPU ---------------------+
|                                                           |
|  Brio 100 --> capture --> RTMDet --> RTMPose --> lifter   |
|   (portrait)    CPU        GPU        GPU         GPU     |
|                          person      2D x26     3D x26    |
|                                                    |      |
|                          One-Euro filter <---------+      |
|                                 |                         |
|                                 v                         |
|                    retargeting QP (skeleton -> Atlas)     |
|                    - arm angles, trunk, head: direct      |
|                    - legs: CoM target, stance width,      |
|                      hip height, foot lift events         |
|                                 |                         |
+---------------------------------+-------------------------+
                          UDP (versioned schema)
+- 125 Hz - control path - CPU, camera-independent ---------+
|                                 v                         |
|   support state machine (DS / SSL / SSR)                  |
|            |  foot lift only after CoM shift              |
|            v                                              |
|   centroidal planner (LIPM, ZMP reference)                |
|            v                                              |
|   +-----------------------------------------------+       |
|   |  WHOLE-BODY QP                                |       |
|   |  min ||q_ddot - q_ddot_ref||^2_W              |       |
|   |  s.t. ZMP in support polygon                  |       |
|   |       joint, rate and torque limits           |       |
|   +-----------------------------------------------+       |
|            v                                              |
|   28 motors  <-- IMU - foot force sensors - encoders      |
+-----------------------------------------------------------+
```

The lower block runs at `basicTimeStep 8 ms` and sees **only
simulation-internal sensors**. If the camera fails, the reference freezes and
the robot keeps standing stably. That is the structural difference to A1/A2.

### 4.3 Why this perception choice

The two-stage proposal was right — 2D first, then 3D lifting is the correct
structure. In detail:

- **RTMPose instead of MediaPipe:** MediaPipe has no GPU delegate under Windows
  Python; it would run on the CPU. RTMPose exports cleanly to ONNX (MMDeploy)
  and therefore runs on DirectML. It is also more accurate (75.8 % AP on COCO
  for the m variant).
- **Halpe-26 instead of COCO-17:** contains feet (toes, heels). For foot lift
  events and foot orientation that is the difference between guessing and
  measuring.
- **Lifting instead of direct 3D:** separates a well-solved problem (2D
  keypoints, very robust) from an ill-posed one (depth). The 2D result stays
  usable even when lifting struggles on unusual poses.
- **Causal lifting window:** MotionBERT originally uses 243 frames — that would
  be 8 s of latency. We run a short causal window (27 frames, past only, no
  lookahead). It costs some accuracy but keeps the real-time condition.

**No absolute depth.** We do not need a metric world position but joint angles
and task-space quantities normalised to leg length. Both are considerably more
robust against depth noise. That is the real lesson from A1: `pose_world_
landmarks` was never used there (A1 read `pose_landmarks`, the dimensionless
pseudo-depth), which produced the misdiagnosis "monocular depth is the problem"
— and from that the switch to MeTRAbs, which for lack of CUDA was never runnable
on this machine anyway.

---

## 5. Camera

### 5.1 Brio 100: the FOV is the problem, not the resolution

58° diagonal gives, at 16:9, **51.6° horizontal and 30.4° vertical**. For a
1.8 m person with a margin in frame:

| Mounting | Usable vertical angle | Required distance | Body height in pixels |
|---|---|---|---|
| Landscape (normal) | 30.4° | **3.50 m** | ~1020 px |
| **Portrait (rotated 90°)** | 51.6° | **1.97 m** | **~1820 px** |

Mounting the camera portrait halves the required distance and delivers **1.8×
more pixels on the body**. At a shorter distance the automatic exposure also
needs less gain and a shorter exposure time, which reduces motion blur — at
30 fps with fixed focus the limiting factor for fast motion. The detectors work
on a person crop anyway, so the sensor's aspect ratio is irrelevant to them.

**Recommendation: portrait mounting.** It costs nothing and is the single
largest gain in accuracy across the entire perception chain.

### 5.2 Preparing the switch to the Sony

The code never addresses a camera directly but a `CameraProfile`: resolution,
FPS, intrinsics (fx, fy, cx, cy), distortion coefficients, mounting orientation,
pixel format. The Brio 100 is one such profile, the Sony will be a second.
Calibration by checkerboard, the result stored as YAML next to the profile.

The camera change is then a configuration entry, not a code change — and the
recordings from the Brio phase stay reproducibly playable with their profile.

---

## 6. Robot model

### 6.1 Atlas is the right choice

For humanoids, Webots R2025a ships only NAO (58 cm), Darwin-OP (45 cm), Bioloid
and KHR-2HV/3HV — all small robots with substantially different body
proportions. **Atlas is the only man-sized humanoid** and therefore the only
sensible candidate for human motion imitation.

### 6.2 But the stock model is unusable as it is

The PROTO is loaded from GitHub through `EXTERNPROTO` (699 lines, extracted from
a 2013 DRCSim URDF).

**Finding A — zero sensing.** 28 RotationalMotors, but **0** PositionSensor,
**0** InertialUnit, **0** Gyro, **0** Accelerometer, **0** TouchSensor. Without
encoders the controller does not know the actual angles, without an IMU not its
orientation, without foot force sensors not the ZMP. The QP from §3 would have
no inputs. The only extension point (`pelvisSlot`) is not enough for foot
sensing.

→ Fork to `a3/webots/protos/AtlasA3.proto`. The 28 sub-PROTOs stay referenced
unchanged by URL; only the top-level file is touched. Thanks to the supervisor,
however, the amount of retrofitting turns out to be small — see §6.3.

**Finding B — suspected falsified inertias.** The PROTO defines *one* physics
node and uses it 28 times:

```
physics DEF DEFAULT_PHYSICS Physics {
  mass 0.001
  inertiaMatrix [ 1 1 1 0 0 0]
}
```

The real segment data (sum **89.00 kg**) sit in the sub-PROTOs, which hang as
solid children without a joint and are correctly summed by Webots' *implicit
solid merging*. But the merge also adds the inertia of the outer solid —
1 kg·m² per axis, 28 times:

| Assembly | real [kg·m²] | parasitic | Factor |
|---|---|---|---|
| foot | ~0.004 | +1.0 | **~250×** |
| shank | ~0.077 | +1.0 | **~13×** |
| thigh | ~0.090 | +1.0 | **~12×** |

For comparison: the total inertia about the ankle axis is about
`89 · 0.5² ≈ 22 kg·m²`. If the finding were real, the robot would be more than
twice as sluggish as it should be, and the feet — decisive for a capture step —
by two orders of magnitude.

**This is a hypothesis, not a measurement.** Webots requires a solid with
`physics` to also have a `boundingObject`; these solids have none, so Webots
might silently discard the node. **M0 measures this** before anything is built.

**Finding C — the actuation is sufficient.** 89 kg · 9.81 = 873 N. Ankle pitch
needs **155 Nm** of 220 available for the full CoP travel to the toe (0.178 m),
ankle roll **55 Nm** of 90 to the foot edge. The ankles can place the CoP across
the entire foot area — if Atlas falls, it is not because of torque.

**Finding D — two retargeting traps.** `ArmUsy` has a rotation axis tilted by
60° (`0 0.5 ±0.866025`), not an axis-parallel shoulder pitch — naive angle
mapping is guaranteed wrong there. And `LLegUhz` has 110 Nm against 260 Nm for
`RLegUhz`; this asymmetry must go into the limit model.

Complete joint, mass and geometry data:
[`a3/configs/atlas_model.yaml`](../configs/atlas_model.yaml).

### 6.3 Sensing: what we actually build — and what the supervisor replaces

A Webots **supervisor** is a controller with privileged access to the scene
graph. It reads and writes the simulation directly instead of observing it
through sensors: exact positions, orientations, velocities, centres of mass and
contact points of every node — free of noise and delay. It is enabled through
the robot node's `supervisor TRUE` field.

Since transferability to real hardware is not a criterion (§11.4), the
supervisor replaces most of the planned sensing — and delivers it more precisely
than retrofitted sensors could:

| Required quantity | Sensor route | Supervisor route | Decision |
|---|---|---|---|
| Actual joint angles | `PositionSensor` | not directly available | **build the sensor in** |
| Trunk orientation | `InertialUnit` | `getOrientation()`, exact | supervisor |
| Angular velocity | `Gyro` | `getVelocity()`, exact | supervisor |
| Total CoM | FK from 28 segment masses | `getCenterOfMass()`, **exact** | supervisor |
| Support polygon | foot force sensors | `getContactPoints()`, exact | supervisor |
| Standing safety test | compute ourselves | `getStaticBalance()`, built in | supervisor (monitor) |
| Contact forces / CoP | `TouchSensor type "force-3d"` | contact points carry **no** forces | **optional, see below** |

**Result: the PROTO needs only 28 `PositionSensor`s — plus the physics fix.**
Everything else falls away. That is not a minor side effect: it removes the
entire forward-kinematics CoM model from the project, that is exactly the part
that is most error-prone across 28 segments and that made up most of the
heuristics in A1/A2 (`balance.py`).

`getCenterOfMass()` returns the centre of mass of a solid including all
descendants in the world frame. `getStaticBalance()` forms the convex hull of
the contact points in the plane perpendicular to gravity and checks whether the
CoM projection lies inside it — exactly our stability criterion, already
implemented. From the same contact points we additionally compute the *margin*
(distance to the polygon edge), because the QP needs a continuous value, not a
boolean.

The **ZMP** is obtained analytically from CoM and CoM acceleration
(Newton-Euler) instead of from force sensors. With an exact CoM that is even
more accurate than the sensory route.

**Force sensors remain optional.** Should M4 show that the differentiated CoP is
too noisy, the pattern is known: Webots' own NAO uses foot FSRs which, as
`TouchSensor` with `type "force-3d"`, a `boundingObject` and their own
`physics`, take over the sole geometry. For Atlas that would mean additionally
forking `LFootSolid`/`RFootSolid`. We build it only when it is needed.

**Syntactic note for the fork:** Atlas uses the SFNode form
`device RotationalMotor { … }`. For two devices per joint it must be converted
to the MFNode form — the pattern can be read off the NAO:

```
device [
  RotationalMotor { name "LLegKny" … }
  PositionSensor  { name "LLegKnyS" }
]
```

---

## 7. Repository layout

Completely separate from the old code base. `src/`, `main/` and `scripts/`
remain untouched — as a comparison baseline, not as a source of code.

```
a3/
├── docs/PLAN.md
├── configs/
│   ├── atlas_model.yaml             extracted model data
│   ├── cameras/brio100.yaml         CameraProfile + intrinsics
│   └── a3.yaml
├── webots/
│   ├── protos/AtlasA3.proto         fork: + sensing, possibly inertia fix
│   └── worlds/a3.wbt
├── controllers/a3_controller/
│   ├── a3_controller.py             Webots entry point, 125 Hz loop
│   ├── devices.py                   motor and sensor handles
│   ├── kinematics.py                FK, Jacobian
│   ├── ground_truth.py              supervisor: CoM, contact points, margin
│   ├── support.py                   support polygon, ZMP, state machine
│   ├── centroidal.py                LIPM planner
│   ├── wbc.py                       whole-body QP
│   └── receiver.py
├── perception/
│   ├── capture.py                   CameraProfile, portrait handling
│   ├── runtime.py                   ONNX Runtime + DirectML session setup
│   ├── detect.py                    RTMDet-nano
│   ├── keypoints2d.py               RTMPose-m, Halpe-26
│   ├── lift3d.py                    MotionBERT, causal window
│   ├── filters.py                   One-Euro
│   ├── retarget.py                  retargeting QP
│   └── overlay.py
├── transport/{schema.py,udp.py}
├── tools/
│   ├── measure_inertia.py           M0: verify finding B
│   ├── audit_devices.py
│   ├── calibrate_camera.py
│   ├── export_models.py             ONNX export/download + verification
│   └── replay.py
├── tests/
└── README.md
```

One principle A1/A2 lacked: **`perception/` and `controllers/` share only
`transport/schema.py`.** No shared utilities, no imported constants. The control
path therefore stays fully testable without a camera existing — and that was
exactly why nothing in A1/A2 could be debugged in isolation.

---

## 8. Milestones

Every milestone has a measurable acceptance criterion.

**M0 — foundations and verification (½ day).** Python 3.11 (venv; only the store
alias is currently installed on the machine), verify `onnxruntime-directml`
against the 6800 XT, Webots integration. `tools/measure_inertia.py`: a defined
torque on a leg joint, measure the angular acceleration, back out the effective
inertia.
→ *Acceptance:* DirectML reports the 6800 XT as the active device. A numeric
value for the foot inertia exists, **finding B is confirmed or refuted**.

**M1 — `AtlasA3.proto`, standing robot (¾ day).** Fork with 28 `PositionSensor`s
(MFNode conversion) and `supervisor TRUE`. If M0 confirms finding B: neutralise
`DEFAULT_PHYSICS`. Supervisor integration for CoM, contact points and support
polygon margin. Static posture control through ankle admittance against the ZMP
computed from the CoM.
→ *Acceptance:* 60 s standing without input, ZMP drift < 2 cm, a lateral impulse
of 50 Ns absorbed without lifting a foot. `getStaticBalance()` stays `true`
throughout.

**M2 — GPU perception (1.5 days).** Obtain/export the ONNX models, DirectML
sessions, RTMDet → RTMPose → lifter, One-Euro filter, portrait capture,
calibration, recording + replay.
→ *Acceptance:* ≥ 28 fps sustained, end-to-end latency < 100 ms (p95), **CPU
load of the perception process < 25 %**, GPU demonstrably carries the inference.

**M3 — upper body 1:1 · MVP (1 day).** ◄ **first complete project goal.** Arm
chains by IK with the correct slanted `ArmUsy` axis, trunk, head, rate limiting.
The legs stay position-controlled in the standing pose; only the ankle
admittance from M1 works.  A full QP is not yet needed here.
→ *Acceptance:* the continuous chain camera → GPU → retargeting → Webots runs
live. Both arms, trunk and head follow visibly and without jitter; mean angle
error against the reference < 5°; the robot stays standing during fast arm
motion.

**M4 — whole-body QP and lower body (2 days).** Support polygon and CoM from the
supervisor, LIPM planner, QP with the ZMP constraint, state machine DS ↔
SSL/SSR.
→ *Acceptance:* squat to 0.25 m of hip drop, stance width to ±0.15 m,
single-leg stance ≥ 3 s — without a fall. The robot visibly stays in DS when the
weight shift is insufficient, instead of tipping.

**M5 — steps and locomotion (2–3 days).** A requirement, but downstream. The
step planner generates footholds from cadence and direction — not a foot copy.
*5a:* stepping in place. *5b:* locomotion.
→ *Acceptance 5a:* 10 steps without falling. *5b:* 2 m of forward motion with a
changed world position.

**M6 — evaluation (1 day).** Supervisor metrics: fall rate, ZMP margin,
per-joint tracking error, latency distribution. Comparison run against A2 on the
same recordings.

---

## 9. Risks

| Risk | Likelihood | Countermeasure |
|---|---|---|
| DirectML provider breaks on a model layer (falls back to CPU) | **medium** | M0 verifies early; if necessary replace an operator or switch model variant |
| Causal lifting window too imprecise for foot events | medium-high | Detect lift primarily from 2D image height + Halpe foot points, not from z |
| Webots ODE cannot hold 8 ms under CPU throttling | medium | Limit contact points, throttle rendering, if necessary 16 ms with documented consequences |
| Finding B wrong, model sluggish anyway | medium | M0 measures first; an inertia override stays possible |
| Locomotion (M5b) unreachable | **high** | M0–M4 are independently acceptable; M5a (stepping in place) is the dependable intermediate result if 5b fails within the time frame |
| Motion blur at 30 fps / fixed focus | medium | Portrait mounting, lighting, fix the exposure time manually |

One risk is deliberately **not** listed: "an inaccurate 3D pose makes the robot
fall over". Per §2 and §4.2 a poor reference can lower imitation quality but can
no longer endanger standing safety. That is the purpose of the architecture.

---

## 10. Outlook: the RL path

Should the project be continued and an NVIDIA GPU or a Linux system with ROCm
become available, approach B (§3) is the route to true 1:1 including walking.
The infrastructure built here stays fully usable: the GPU perception, the
`AtlasA3` model, the transport schema and the evaluation metrics are
approach-independent. Only the block between reference and motor command would
be exchanged — the QP would give way to a trained policy. The repository layout
is deliberately cut so that this exchange affects one file.

---

## 11. Decisions taken

1. **The supervisor is permitted for control.** This removes the need for
   rebuilt sensing (§6.3) and the entire CoM model.
2. **Locomotion is a requirement**, but downstream. First priority is the
   standing pipeline with trunk, arm and head mirroring (M3 as the MVP).
3. **The physics may be corrected.** The inertia finding thereby becomes a
   result of the case study in its own right.
4. **The objective is functionality in simulation**, not transferability to real
   hardware. Sim-to-real purity is explicitly not a criterion.

---

## 12. Next step

M0: Python environment, verify DirectML against the 6800 XT,
`tools/measure_inertia.py`. The result is a number that decides what
`AtlasA3.proto` looks like.

---

## Retrospective note

This plan was written before any measurement. What it got right and what it got
wrong is recorded in [`FINDINGS.md`](FINDINGS.md). In short: finding B was
confirmed at 287× rather than 250×; the tipping time constant of 0.31 s was
exact; approach C (the whole-body QP) was built and then **rejected**, because it
turned out to be three times worse at imitation than the direct control of M3;
and the assumption in §6.2 that the segment masses sit in the sub-PROTOs was
correct but unverified, which later caused a false finding of its own.
