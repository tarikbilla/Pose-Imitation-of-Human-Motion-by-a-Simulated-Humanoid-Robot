# M4 — Whole-Body QP and Lower Body

Date 2026-09-04 · Branch `A3` · Raw data in `a3/results/m4_*.json`

> **Correction of 2026-09-05. This chapter was accepted on the wrong grounds.**
>
> The original status read "achieved". It rested on `fallen: False`, the CoM
> error and the QP status — that is, on not a single quantity that measures
> **whether the robot copies the human**. Measured afterwards, using joint error
> against the target angle as the metric:
>
> | | Median | Maximum |
> |---|---|---|
> | M3 (direct control) | **5.0°** | 21° |
> | M4 (whole-body QP) | **15.6°** | 49° |
>
> M4 is a **regression by a factor of 3** against M3. Five repair attempts
> (CoM dead band, `W_ARM` from 1 to 25, arms taken out of the QP, `ARM_SCALE`
> reintroduced, integration on the command instead of on the measurement)
> brought none below 18°; most of them fell. The fault was not the QP but the
> **choice of target quantity**: I measured standing safety and claimed
> imitation.
>
> The kinematic model itself is fine. An interim suspicion that it computes with
> masses the simulation does not have was not confirmed: the 28 sub-PROTOs of
> the Webots Atlas sum to exactly 89.000 kg with precisely the segment masses
> from `configs/atlas_model.yaml`. The fault lay solely in the target quantity
> and in the weighting of the QP. See [`M6_RESULTS.md`](M6_RESULTS.md) section 0.
>
> Furthermore the runs are **not reproducible**: a repeat M3 run gave 10.4°
> instead of 5.0° and fell where it previously had not. All "no fall" statements
> in this chapter rest on single runs and do not hold.
>
> **M4 is considered rejected.** The direct control from M3 is the better path
> and is continued in M6 with corrected physics.

---

## 1. The difference to M3

M3 had two separate controllers fighting over the same joints: one for the arm
reference, one for balance. They could only yield in sequence, and the emergency
brake was a blanket amplitude damping.

M4 solves **one** optimisation problem per tick:

```
minimise   w_com ‖J_com Δq − v_com‖²        balance
         + w_leg ‖Δq_leg − Δq_leg,ref‖²     leg reference
         + w_torso ‖…‖² + w_arm ‖…‖²        upper body
         + λ‖Δq‖²                           regularisation
subject to joint limits, rate limits
           CoM displacement inside the support polygon
```

The solver distributes the conflict across all 28 joints by itself instead of
resolving it by a priority rule. `w_com = 260` against `w_arm = 1` says: when
imitation and balance collide, balance wins — but only as far as necessary.

Solved with **OSQP**: 28 variables, 30 constraints, **0.044 ms** per tick.
Negligible within the 8 ms time step.

---

## 2. The kinematic model

The QP needs the Jacobian of the centre of mass. `atlaskin.py` builds it
analytically from the model data extracted in M0 — chain, axes, anchors, segment
masses. Verification against independent references:

| Quantity | Model | Reference | Source |
|---|---|---|---|
| Total mass | **89.000 kg** | 89.00 kg | M0 extraction |
| Hip to sole | **0.9221 m** | 0.922 m | M0 geometry |
| Stance width | **0.1780 m** | 0.178 m | M0 geometry |
| CoM Jacobian | — | numeric derivative | **deviation 3.3 · 10⁻¹²** |

Compute time for FK, CoM and Jacobian together: 1.59 ms.

> **Addendum 2026-09-05.** This table verifies the model against the data it was
> built from — it is circular. That the absolute CoM position deviates only
> 24 mm from the supervisor value was read as confirmation but proves little: in
> the symmetric neutral pose the centre of mass lies near the midline for **any**
> symmetric mass distribution. The deviation only becomes large in extreme poses
> — exactly where pose imitation happens, and exactly where it was never checked.

The absolute CoM position deviates 24 mm from the supervisor value because the
translations of the sub-solids were not extracted. That is inconsequential: the
**position** comes from the supervisor, the model supplies only the
**derivative**.

---

## 3. Three failures on the way

**The hard CoM constraint was infeasible.** First version: "the CoM must lie
inside the support polygon". The QP reported `primal infeasible`. The reason is
arithmetic: at a 1.8 rad/s rate limit and an 8 ms tick each joint may move
0.014 rad — the reachable CoM displacement is in the millimetre range. As soon
as the CoM lies even slightly outside, the demand "back inside immediately" is
mathematically impossible. Reformulated to "do not move the CoM further out, at
most by `step_cap`"; the weighted term, which is always solvable, does the
recovery.

**The CoM term is what carries it.** Measured directly:

| | Fell | Packets | max \|err_x\| |
|---|---|---|---|
| without the CoM term | yes, after 18.5 s | 584 | 0.721 |
| **with the CoM term** | **no** | **1476** | **0.125** |

**The leg references knocked the robot over.** With them it fell reproducibly
after 3.5 s; without them it ran through. Cause: the lifter needs 27 frames of
lead-in, so its first value arrives about a second after the stream starts — and
lands as a step function. Two countermeasures: a low-pass on the reference
(τ = 0.45 s) and a fade-in over 2.5 s on first arrival. Plus halved gains
(`SQUAT_GAIN` 1.6 → 0.9, `STANCE_GAIN` 0.55 → 0.30).

---

## 4. Result

Against `recordings/testvideo_wb.jsonl`, 53 s of real recording, real time:

| | M3 | **M4** |
|---|---|---|
| Arm amplitude | 75 % (damped) | **100 %** |
| Legs | rigid in the standing pose | **follow (squat, stance width)** |
| Fell | no | **no** |
| \|CoM error x\| median / max | 6 / 40 mm | 80 / 125 mm |
| \|CoM error y\| median / max | 4 / 27 mm | 37 / 78 mm |
| Min contact points | 6 of 8 | **6 of 8** |
| Knee deflection | constant 0.30 | **0.30 … 0.48 rad** |

The CoM error is larger than in M3 — that is expected and correct: the legs now
move, so the centre of mass travels. What matters is that it stays inside the
support polygon (sagittal margin 148 mm, lateral 151 mm from M1) and that the
contact count never drops below 6.

The hard constraint was never active in any tick (`saturated = 0`) — the
weighted term keeps the CoM far enough inside on its own. It stays enabled
anyway, because it costs nothing and serves as a safety net for more aggressive
motion.

---

## 5. Acceptance

| Criterion (plan §8) | Result |
|---|---|
| No fall across the recording | **passed** |
| Squat follows the reference | **passed** — knee 0.30 → 0.48 rad |
| Stance width follows | **passed** |
| `ARM_SCALE` no longer needed | **passed** |
| Squat to 0.25 m hip drop | **not tested** — the recording contains only a shallow knee bend (`hip_height` max 0.16) |
| Single-leg stance ≥ 3 s | **not tested** — not present in the recording |

The last two points need a recording with a deep knee bend and a single-leg
stance. Without such material one can only confirm that the chain transfers the
motions that are present.

**These criteria are superseded by the correction at the top of this chapter:
they measure standing safety, not imitation, which is exactly the error that
made M4 look successful.**

---

## 6. Reproduction

```bat
rem terminal 1
set A3_FINISH_IDLE=3
"%WEBOTS%\webots.exe" --batch --mode=realtime webots\worlds\m4_wholebody.wbt

rem terminal 2
C:\venvs\a3-pose\Scripts\python.exe tools\drive_robot.py ^
    --recording recordings\testvideo_wb.jsonl --wait-ready --preview
```

Ablation switches: `A3_W_COM` (CoM weight, 0 disables it), `A3_USE_LEGS`,
`A3_COM_CONSTRAINT`, `A3_SQUAT_GAIN`, `A3_STANCE_GAIN`.

Check the kinematics separately: `webots\worlds\m4_verify_kin.wbt`.
