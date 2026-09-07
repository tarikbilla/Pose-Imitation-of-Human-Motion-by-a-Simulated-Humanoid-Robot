# M5 — Stepping

Date 2026-09-04 · Branch `A3` · Raw data in `a3/results/m5_march.json`

Status: **partial success.** The stepping state machine runs and is safe; the
robot does not fall. But it reaches **no genuine single-leg stance** — the foot
never fully leaves the ground. M5a is therefore only half met, and M5b
(locomotion) was not started.

---

## 1. Setup

The camera says **whether** and **how fast** to step, never **how**. The step
planner is a state machine inside the controller:

```
DS ──(march request)──► SHIFT ──(weight shifted?)──► LIFT ──► PLACE ──► DS
                          │                                            (side alternates)
                          └──(not shifted, 1.5 s)──► back to DS
```

The decisive transition is `SHIFT → LIFT`. It is released **only** when the
centre of mass has actually arrived over the stance foot:

```python
reached = abs(com_error_y - com_target) < SHIFT_TOLERANCE
```

That ordering is the entire point. A direct joint-angle copy from the human
destroys it: in a human the weight shift is *implicit* in the pose, whereas in
the robot it must happen *explicitly and first*.

The planner gives the QP from M4 only two things: a shifted CoM target and joint
offsets for the swing leg. The QP decides how to realise them.

---

## 2. Result

Forced march request (0.5) from second 8, across the 53 s recording:

| | |
|---|---|
| Fell | **no** |
| Steps | **3** in 49 s |
| States | DS 254 · **SHIFT 920** · LIFT 30 · PLACE 26 |
| Contact points during LIFT | **6–8** of 8 |
| Max lateral shift | 0.055 m |
| CoM error x / y (median) | 0.074 / 0.075 m |

**The foot does not lift.** With 6 of 8 contact points, part of the sole is
still down. And the state distribution tells the rest: 920 ticks in `SHIFT`
against 30 in `LIFT` — the robot spends nearly all its time trying in vain to
shift its weight.

---

## 3. Why, with numbers

For genuine single-leg stance the centre of mass must move over the stance foot,
that is **89 mm** laterally (half the hip width, M0). Measured, the controller
reaches **26 mm** — barely a third.

Trying to force it by loosening the release condition ends predictably:

| Configuration | Steps | Fell |
|---|---|---|
| target 55 mm, tolerance 20 mm | 3 | **no** |
| target 45 mm, tolerance 30 mm | 1 | **yes** |
| target 35 mm, tolerance 35 mm | 1 | **yes** |

As soon as the release becomes more generous, the robot lifts the foot before
the weight has really shifted — and tips over the now much narrower support
area. That is not a malfunction of the state machine but proof that its
condition is necessary.

> **Correction of 2026-09-05.** The original text here claimed that the
> *physical* authority for the weight shift was insufficient. That is measurably
> false. `tools/mass_compare.py` determines the hip roll angle that shifts the
> centre of mass by the required 89 mm laterally: **0.123 rad against a
> 0.436 rad joint limit, that is 28 %.** The kinematics can do three times what
> is needed. The 26 mm were a limit of the **controller**, not of the robot —
> the QP never commanded the shift, because its CoM term fought against the
> imitation terms instead of with them. The cause therefore lies in M4, not in
> the mechanics.
>
> The second paragraph of this correction initially claimed that the robot had
> no mass distribution at all. That too was wrong and is set right in
> [`M6_RESULTS.md`](M6_RESULTS.md) section 0: the 28 sub-PROTOs carry the real
> segment masses and sum to exactly 89.000 kg. The centre-of-mass computation in
> this chapter therefore rests on the correct masses. What remains is the
> statement above: the authority was sufficient, the controller did not use it.

## 4. What would be missing

A kinematic QP at position level cannot generate angular momentum. Genuine
stepping and walking would need:

- **Centroidal dynamics** instead of pure kinematics — linear and angular
  momentum as state variables, not just positions
- **LIPM preview** across several steps, so that the shift begins *before* it is
  needed rather than reactively
- **Force-controlled contacts** — this would require the `TouchSensor`s with
  `force-3d` deferred as optional in M1, to measure the centre of pressure
  instead of estimating it from the CoM

That is the scope described in plan §3 as approach B (RL) or as full centroidal
control — both deliberately outside this project. M7 supplies the preview part
and does walk.

---

## 5. Acceptance

| Criterion (plan §8) | Result |
|---|---|
| State machine with weight shift before foot lift | **passed** |
| No fall during the march attempt | **passed** |
| 10 steps in place | **failed** — 3 attempts, no genuine lift |
| 2 m of locomotion (M5b) | **not started** |

M5a therefore counts as **half achieved**: the mechanism is built, verified and
safe, but no genuine step occurs. Per the correction in §3, the cause is the
controller and not the available authority.

The plan had listed M5 as a risk milestone and described M0–M4 as independently
acceptable. That assessment was confirmed.

---

## 6. Reproduction

```bat
set A3_FORCE_MARCH=0.5
set A3_MARCH_START_S=8
set A3_FINISH_IDLE=3
"%WEBOTS%\webots.exe" --batch --mode=realtime webots\worlds\m4_wholebody.wbt

C:\venvs\a3-pose\Scripts\python.exe tools\drive_robot.py ^
    --recording recordings\testvideo_wb.jsonl --wait-ready
```

Adjustments: `A3_SHIFT_Y` (shift target), `A3_SHIFT_TOL` (release),
`A3_SHIFT_S`, `A3_LIFT_KNEE`, `A3_USE_STEPPER`.
