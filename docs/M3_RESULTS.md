# M3 — Upper-Body Imitation (MVP)

Date 2026-09-04 · Branch `A3` · Raw data in `a3/results/m3_run.json`

Status: **achieved, with a quantified limitation.** The chain runs end to end,
the robot follows arms, torso and head and stays standing — at 55 % arm
amplitude. Full amplitude requires the whole-body QP from M4 (§5).

---

## 1. Depth from 2D: why not RTMW3D

`rtmlib` offers direct 3D keypoints through `Wholebody3d` (27.8 fps). For our
purpose the z values are **unusable**, and that is measurable:

A segment of length *L*, inclined by θ out of the image plane, appears as
*L·cos θ* and has a depth difference of *L·sin θ*. Foreshortening and |Δz| must
therefore correlate strongly. Measured over 401 samples:

| | correlation(2D foreshortening, \|Δz\|) |
|---|---|
| left upper arm | **−0.138** |
| right upper arm | **+0.339** |

Inconsistent in sign and far too weak. On top of that, Δz reaches at most 23 px
while the 2D arm length varies between 105 and 200 px — an arm pointing at the
camera would need a depth component on the order of its own length. The second
output array (`keypoints_simcc`) gives the same correlations, merely scaled by a
factor of about 88.

**Instead: segment foreshortening.** It is directly measurable, physically
unambiguous and requires no trust in a model:

```
cos θ = L_measured / L_reference        θ = inclination out of the image plane
```

`L_reference` is tracked self-calibrating as a decaying maximum
(`SegmentReference`, half-life ~46 s). The measured range from 105 to 200 px
corresponds to up to 58° of inclination — a strong signal.

**Known limitation:** `acos` returns no sign. Whether a segment lies in front of
or behind the image plane stays ambiguous; we assume "forward", which is almost
always right for arms. The same applies to `torso_pitch`. A 3D lifter (plan
§3.4, stage 2) would resolve it.

The same method incidentally yields the **torso yaw**: when turning, shoulder
width shortens in the image. The sign comes from the visibility of the ears.

---

## 2. Architecture

The split follows plan §3.2: perception sends **task-space references**, not
joint angles. The controller owns the robot model and solves the IK.

```
perception/                          transport/            controllers/
  skeleton.py   model -> 29 points    schema.py  contract    a3_upper_body.py
  filters.py    One-Euro              udp.py     transport   armkin.py
  frames.py     body frame + depth         │
  retarget.py   direction vectors  ───────┴──► IK ──► 28 motors
```

Six unit vectors are sent in the body frame (x forward, y left, z up) — upper
arm, forearm and hand normal per side — plus torso yaw/pitch/roll and head
angles. **Validation across the whole recording:** 99.1 % of frames produce
output, unit-norm error at most 2.2 · 10⁻¹⁶.

---

## 3. Arm IK with the slanted shoulder axis

`ArmUsy` rotates about `(0, 0.5, ±0.866)` — not an axis-parallel direction
(plan, finding D). It is solved numerically in two stages: `usy`/`shx` for the
upper-arm direction, then `ely`/`elx` for the forearm.

Two errors on the way, both found by measurement:

**Singularity.** With the elbow extended (`elx ≈ 0`), `ely` has no influence on
the forearm direction. A gradient descent gets stuck there — the forearm was
initially at a median of 13° with p90 above 100°.

**Wrong assumption.** Computing the elbow angle analytically as the angle
between upper-arm and forearm directions fails: `REST_UPPER` and `REST_FORE` are
already inclined 11° against each other at rest.

Solved by multi-start (upper arm) and a 2D grid search with refinement
(forearm), plus a warm start from the previous solution.

| | before | after |
|---|---|---|
| Upper arm within 5° | 86 % | **100 %** (max 2.4°) |
| Forearm within 5° | 46 % | **100 %** (max 3.7°) |
| Median compute time | 13.5 ms | **2.1 ms** |

In the run against the real recording: **mean IK error 0.025 ≈ 1.4°.** The
acceptance criterion (< 5°) is met.

---

## 4. Result of the full run

`webots/worlds/m3_upper_body.wbt`, fed from `recordings/testvideo_wb.jsonl`
(53 s of real recording, streamed in real time).

| | ARM_SCALE 0.55 | ARM_SCALE 1.0 |
|---|---|---|
| Fell | **no** | yes, after 32 s |
| Packets received | **1588 / 1588** | 1014 |
| Active imitation | **52.9 s** | 31.9 s |
| \|CoM error x\| median / max | **6 / 40 mm** | 12 / 756 mm |
| \|CoM error y\| median / max | **4 / 27 mm** | 9 / 79 mm |
| Min contact points | **6 of 8** | 2 |
| IK error | 1.4° | 1.6° |

---

## 5. Why 55 % and not 100 %

The arms weigh **24.7 kg of 89 kg — 28 % of total mass.** Extended forward or
asymmetrically they shift the centre of mass further than the lower body can
pull it back:

```
hip roll:      0.570 m/rad · 0.45 rad = 0.257 m of compensation
measured error at full amplitude:       0.769 m
```

A factor of 3 beyond the available authority. This is not a controller weakness
but a balance limit of double support.

The path there was instructive and is documented because it substantiates the
diagnosis:

1. **Pure P controller** → steady-state error that grows under a persistent
   disturbance. Fall after 36 s.
2. **PI controller** (integral with anti-windup) → control quality jumped to a
   10 mm lateral median. Fall only after 59 s.
3. **Authority limiting on \|error\|** → engaged too late; at 0.115 m of
   deflection the robot is already tipping, and the tipping time constant is
   0.318 s.
4. **Authority limiting on the capture point** `cp = e + ė/ω` → saved the
   lateral axis (max 0.769 → 0.081 m), but the sagittal axis became the
   bottleneck.
5. **Amplitude limiting** to 55 % → stable across the whole recording.

Point 4 is the most interesting finding: the predictive criterion removes the
problem on the axis where it acts but moves it to the other one. Serving both
axes at once requires coordination across the whole body — that is, exactly the
whole-body QP from M4, which solves arm reference and standing safety in **one**
optimisation problem rather than in two separate controllers.

`ARM_SCALE` is adjustable through `A3_ARM_SCALE` and was expected to disappear
in M4.

---

## 6. Reproduction

```bat
rem terminal 1: Webots
set A3_FINISH_IDLE=3
"%WEBOTS%\webots.exe" --batch --mode=realtime webots\worlds\m3_upper_body.wbt

rem terminal 2: stream the references
C:\venvs\a3-pose\Scripts\python.exe tools\drive_robot.py ^
    --recording recordings\testvideo_wb.jsonl

rem live instead of from a recording:
C:\venvs\a3-pose\Scripts\python.exe tools\drive_robot.py
```

Check the retargeting without Webots:

```bat
C:\venvs\a3-pose\Scripts\python.exe tools\check_retarget.py recordings\testvideo_wb.jsonl
```

---

## 7. Open

- Whole-body QP (M4) — lifts the amplitude limit
- Sign of the depth (3D lifter or a plausibility heuristic)
- Hand normal is sent but not yet mapped onto `ArmUwy` / `ArmMwx`
- Live run with the Brio instead of from the recording
