# A3 — Final Report

Branch `A3` · September 2026 · rebuild of the pose imitation with Boston
Dynamics Atlas in Webots R2025a

For the full record with every decision and its rationale, see
[`FINDINGS.md`](FINDINGS.md).

---

## What was achieved

A continuous chain from camera to moving robot, running live:

**camera → GPU pose → 3D lifting → retargeting → UDP → direct control → 28 motors**

| Milestone | Result |
|---|---|
| M0 foundations and verification | **complete** |
| M1 model, sensing, posture control | **complete** |
| M2 GPU perception including the 3D lifter | **complete** |
| M3 upper-body imitation | **complete** |
| M4 whole-body QP | **rejected** — a regression against M3 |
| M5 stepping | **superseded by M7** |
| M6 mass distribution | **complete** |
| M7 walking (DCM, with physics) | **complete** — 2.6 m without falling |
| M8 puppet (no physics) | **complete** — live camera, foot contact, travel |

The dependable state is **M3 plus M6 plus M7** for the physics track: upper-body
imitation at full amplitude without safety damping, on a robot with corrected
mass distribution, and walking through the divergent component of motion — 14
steps, 2.6 m, no fall. For imitation quality the dependable state is **M8**,
which is what the live demonstration runs on.

---

## The findings that shaped the project

### 1. The Atlas mass distribution does not suit pose imitation

One Atlas arm weighs **12.35 kg** — **27.8 %** of the 89 kg total sits in the
two arms, against 10 % in humans. Pose imitation moves exactly those arms.
Consequence: **22.6 % of all reachable upper-body poses are statically
unstable**, meaning the centre of mass lies outside the support polygon. This is
not a control problem — in such a pose no controller can prevent the fall.

Nor is it a simulation artefact: it is the real Atlas, whose hydraulics and
structure sit in the arm. Since only the simulation matters for this case study
it is changeable, and `tools/build_atlas_proto.py --masses` changes it.

### 2. Removing weight does not help; only redistributing does

Halving every mass uniformly is **numerically identical** to the original state.
The tipping condition is "centre of mass over the support polygon"; mass cancels
out. Measured, only the distribution acts — arm sway, meaning how far pure arm
motion drags the centre of mass, falls from 73.7 mm to 10.6 mm.

### 3. The PROTO's placeholder inertia tensor

The Atlas PROTO defines a `Physics` node with `inertiaMatrix [1 1 1 0 0 0]` and
attaches it by `USE` to all 29 links. Webots merges it **in addition** into each
link. For a 0.817 kg foot the placeholder dominates completely: measured, the
ankles were 83.7× too sluggish (M0/M1) — precisely the joints a humanoid
balances over.

### 4. The camera must not sit inside the control loop

Measured tipping time constant: **0.318 s** (M1). The camera chain costs
80–150 ms. A balance controller closed through the camera is therefore
structurally unstable, independent of pose accuracy. The architecture separates
them: reference path 30 Hz, control path 125 Hz on simulation-internal
quantities.

### 5. Absolute pixels confuse distance with rotation

Across the test recording, shoulder width varies by **98 %** and body height by
**66 %** — purely from changes in distance. One frame with a frontal posture was
read as a 52° rotation. Fixed by normalising against a smoothed body measure
and, for torso yaw, by the ratio of shoulder width to hip width.

### 6. Measuring beats guessing — including when measuring itself

Every controller only worked once its plant had been measured. In M1 the guessed
admittance controller knocked the robot over with no disturbance (sign fed back
positively, hip roll 1.9× stronger laterally than the ankle); in M3 a missing
minus sign coupled both axes.

The more expensive mistakes were not in the controllers but in the **metrics and
assumptions**:

- **M4 was accepted on the wrong quantity.** What was measured was `fell`, CoM
  error and QP status — that is standing safety, while the claim was imitation.
  Measured afterwards on joint error: M4 **15.6°** against M3 **5.0°**, three
  times worse.
- **M5 declared a controller limit to be a physical one.** The 89 mm of lateral
  shift it demanded needs 0.125 rad of hip roll against a 0.436 rad limit. The
  robot can do three times that; the QP never commanded it.
- **An unverified assumption was declared fact and then wrongly refuted.**
  `PLAN.md` assumed the segment masses were in the sub-PROTOs. That is true (the
  sum is exactly 89.000 kg) but was never verified; in M6 I first declared it
  false. A single fetch of `UtorsoSolid.proto` would have settled it either way.
  See [`M6_RESULTS.md`](M6_RESULTS.md) §0.
- **A fall detector with an absolute threshold** reported low-CoM profiles as
  fallen although they were standing still.

Common denominator: a metric calibrated for one configuration, applied to
another, and not re-checked.

### 7. Reproducibility is a property of the measurement setup

M3 and M4 were measured in real time; a repeat run gave 10.4° instead of 5.0°
and fell where it previously had not. All single-run statements in those
chapters do not hold. From M6 onward Webots runs in fast mode without rendering
against a recording, three runs per configuration.

Imitation error is then reproducible to about 0.1°. The fall decision is not:
UDP timing is not tick-synchronous, and near the stability boundary the outcome
flips between runs. Fall counts are therefore reported across several runs
rather than as yes/no — a profile that falls in one run of three sits exactly on
the boundary.

---

## Key figures

**Perception** (RX 6800 XT via DirectML, 1080×1920 portrait)

| | |
|---|---|
| 2D chain (RTMDet + RTMW wholebody) | 26 ms |
| 3D lifter (MotionBERT, window 27, half rate) | 11.5 ms |
| **total** | **37.5 ms → 27 Hz** |
| Detection rate over 1602 frames | **100 %** |
| Jitter after the One-Euro filter | 2.83 → 1.46 px |

Without the GPU the chain would not run: DirectML is **8.0× faster** than the
CPU at identical output.

**Control and model**

| | |
|---|---|
| Arm IK error | **1.4°** (criterion < 5°) |
| IK compute time | 2.1 ms |
| Lateral disturbance rejection | 40 Ns (capture-point prediction: 42.3 Ns) |
| Kinematic model, total mass | 89.000 kg against 89.000 kg from the sub-PROTOs |
| CoM Jacobian vs numeric derivative | 3.3 · 10⁻¹² |

---

## What was not achieved, and why

**Genuine stepping under the whole-body QP.** For single support the centre of
mass must move 89 mm laterally over the stance foot; the controller reached
26 mm. The cause is **not** missing authority — there is three times enough —
but the position-level kinematic QP, which cannot generate angular momentum.
Humans swing torso and arms when marching to generate exactly that. It needs
centroidal dynamics, LIPM preview across several steps and force-controlled
contacts. M7 supplies the preview and walks; M8 sidesteps the question entirely
by removing physics.

**Further open items**

- Hand normals are transmitted but not mapped to `ArmUwy` / `ArmMwx`
- Torso yaw sign unverified (magnitude corrected)
- Camera calibration for metric intrinsics
- Squat to 0.25 m and single-leg stance untested — the recording contains neither

---

## Where to start

| Document | Contents |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **consolidated record, all decisions and rationale** |
| [`OPERATIONS.md`](OPERATIONS.md) | **how to run it** |
| [`PORTING_LAB.md`](PORTING_LAB.md) | **moving to the lab hardware** |
| [`PLAN.md`](PLAN.md) | architecture and rationale as originally planned |
| [`M0_RESULTS.md`](M0_RESULTS.md) | inertia finding, DirectML evidence |
| [`M1_RESULTS.md`](M1_RESULTS.md) | `AtlasA3.proto`, plant identification, disturbance rejection |
| [`M2_RESULTS.md`](M2_RESULTS.md) | GPU perception, camera, 3D lifter |
| [`M3_RESULTS.md`](M3_RESULTS.md) | arm IK, depth from foreshortening |
| [`M4_RESULTS.md`](M4_RESULTS.md) | whole-body QP — **with correction, rejected** |
| [`M5_RESULTS.md`](M5_RESULTS.md) | stepping state machine — **with correction** |
| [`M6_RESULTS.md`](M6_RESULTS.md) | mass distribution, profiles, measurement method |
| [`M7_RESULTS.md`](M7_RESULTS.md) | walking: leg IK, DCM, results |
| [`M8_RESULTS.md`](M8_RESULTS.md) | puppet: imitation without physics, live operation |

Getting started: `README.md` one directory up.
