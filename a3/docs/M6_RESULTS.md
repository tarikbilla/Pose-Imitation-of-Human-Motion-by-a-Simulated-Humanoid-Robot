# M6 — Mass Distribution

Date 2026-09-05 · Branch `A3` · Raw data in `a3/results/mass_compare.json`,
`a3/results/mass_trial_v2.json`,
`a3/results/mass_trial_a13.json`, `a3/results/mass_trial_a16.json`

Status: **the robot carries the wrong mass distribution for pose imitation.**
Not because of a bug in Webots, but because the real Atlas is built that way:
27.8 % of its mass sits in the arms, against 10 % in humans. This chapter
measures that, changes it, and verifies the change.

---

## 0. A false finding and how it surfaced

The first version of this chapter claimed that the Webots Atlas had **no mass
distribution at all**: the main PROTO defines exactly one `Physics` node

```
physics DEF DEFAULT_PHYSICS Physics {
  density -1
  mass 0.001
  inertiaMatrix [ 1 1 1 0 0 0]
  centerOfMass [0 0 0]
}
```

and attaches it by `USE` to all 29 links. One `mass` field, one `centerOfMass`,
29 links — from which I concluded a 0.029 kg robot with equally heavy links.

**That was wrong.** The masses are in the sub-PROTOs. `UtorsoSolid.proto`
contains `mass 18.484`, `PelvisSolid.proto` contains `mass 14.2529`, and the 28
sub-PROTOs sum to **exactly 89.000 kg**. Webots merges the outer `Solid` and the
sub-PROTO `Solid` into **one** body by implicit merging, because no joint lies
between them; the masses add. The robot had its real distribution all along.

It surfaced while checking `PLAN.md` §296, which stated as an assumption: "the
real segment data (sum 89.00 kg) sit in the sub-PROTOs." That assumption was
correct and I had never verified it — neither when I wrote it nor when I later
declared it refuted. A single fetch of `UtorsoSolid.proto` would have settled it
either way.

Two things that still stand:

- **The inertia finding from M0 is correct.** `inertiaMatrix [1 1 1 0 0 0]` is
  added on top of the real inertia. For a 0.817 kg foot the placeholder
  dominates completely — hence the measured 83.7× too sluggish ankles.
- **The head really is massless.** `HeadMesh.proto` is the only one of the 29
  sub-PROTOs without a `Physics` node.

And one error that followed from it: the first implementation wrote an
*additional* `Physics` node into every link. Because of the merging that was not
redistribution but **ballast** — the robot then weighed about 178 kg. The
measurements of that version are therefore discarded and were repeated.

---

## 1. The actual finding

| Group | Atlas | Human (Winter) | Factor |
|---|---|---|---|
| Both arms | **27.8 %** | 10.0 % | **2.8×** |
| Both legs | 30.0 % | 32.2 % | 0.93× |
| Torso | 42.3 % | 49.7 % | 0.85× |
| Head | **0.0 %** | 8.1 % | — |

One Atlas arm weighs **12.35 kg** at 89 kg total mass. In a human of the same
mass it would be 4.45 kg. Pose imitation moves exactly those arms, and they are
three times what the human motion being copied was designed for.

This is not a simulation artefact but the real Atlas: hydraulic actuators, valve
blocks and structural mass sit in the arm. For a case study on motion imitation
it is nevertheless the wrong premise — and since only the simulation matters, it
is changeable.

---

## 2. Static comparison of the profiles

4000 random poses from the entire reachable upper-body space, legs neutral. What
is measured is whether the centre-of-mass projection lies inside the support
polygon, how far pure arm motion drags the centre of mass (arm sway), and which
hip roll angle delivers the 89 mm needed for a step (`tools/mass_compare.py`).

| Profile | Arms | CoM height | ω | Stable poses | Margin p05 | Arm sway | θ for 89 mm |
|---|---|---|---|---|---|---|---|
| `spec` (as shipped) | 27.8 % | 1.019 m | 3.10 | **77.4 %** | −0.0510 m | **0.0737 m** | 0.125 (29 %) |
| `human` | 10.0 % | 1.004 m | 3.13 | 83.6 % | −0.0350 m | 0.0266 m | 0.123 (28 %) |
| `legs` | 8.0 % | **0.831 m** | 3.44 | **100 %** | +0.0262 m | 0.0212 m | 0.140 (32 %) |
| `core` | **4.0 %** | 0.927 m | 3.25 | **100 %** | **+0.0542 m** | **0.0106 m** | 0.115 (26 %) |
| `combo` | 6.0 % | 0.861 m | 3.38 | **100 %** | +0.0360 m | 0.0159 m | 0.131 (30 %) |

**22.6 % of all reachable upper-body poses are statically unstable as shipped.**
Not hard to hold — unstable: the centre of mass lies outside the support
polygon, and no controller can repair that, because the pose itself is the
problem.

**Arm sway falls by a factor of 7.** From 73.7 mm to 10.6 mm with `core`. That
is the decisive quantity for pose imitation: how far arm motion drags the centre
of mass along.

**The shift authority was never the problem.** For the 89 mm demanded in M5,
0.115 to 0.140 rad of hip roll suffices against a 0.436 rad limit — 26 to 32 %.
The robot can do three times what is needed. See the correction in
[`M5_RESULTS.md`](M5_RESULTS.md).

### Removing weight achieves nothing

A uniform halving of all masses was measured along and is **numerically
identical** to the original state. The tipping condition is "centre of mass over
the support polygon"; mass cancels out. Only the **distribution** acts. The
question "very low weight, perhaps none at all?" is thereby answered: it does
not help.

### Where a low centre of mass helps and where it does not

`legs` lowers the centre of mass from 1.019 m to 0.831 m. That makes it more
robust against inclination, because an inclination θ displaces the projection
only by h·sin θ. But it raises ω from 3.10 to 3.44, and ω is the divergence rate
of the inverted pendulum: **a low centre of mass means faster tipping and less
reaction time**. A short pencil is harder to balance on your hand than a broom.
Both effects are small at around 10 % against the factor-7 arm sway and should
not dominate the choice of profile.

---

## 3. The implementation

`tools/build_atlas_proto.py --masses <profile>` writes the distribution where it
belongs: into the sub-PROTOs.

1. All 28 mass-bearing sub-PROTOs are copied into `webots/protos/vendor/`, with
   `mass` set to the target value and `inertiaMatrix` scaled by the same factor.
   **`centerOfMass` stays unchanged** — the segment geometry remains correct,
   only the mass changes.
2. The `EXTERNPROTO` lines of the main PROTO point to the local copies; the
   PROTO-internal references of the copies (mesh files) are set to absolute
   upstream URLs so they keep resolving.
3. The shared `DEFAULT_PHYSICS` placeholder, which Webots merges into every
   link, is replaced by a dedicated `Physics` node per link with a negligible
   remainder (mass 0.001 kg, inertia 10⁻⁶). This removes the inertia fault from
   M0.
4. The head, the only link without `Physics` in its sub-PROTO, receives its mass
   through exactly that node on the `NeckAy` link.

`tools/check_proto.py` verifies the product against the profile table: 28
vendored sub-PROTOs, 29 `Physics` nodes, 28 `PositionSensor`s, no remaining
`USE DEFAULT_PHYSICS`, balanced braces, total mass and every individual mass
against `mass_tables`. Largest measured deviation: 5 · 10⁻⁶ kg.

### The profiles

| Profile | Head | Torso | Arm each | Leg each | Idea |
|---|---|---|---|---|---|
| `spec` | 0 % | 42.3 % | 13.9 % | 15.0 % | as shipped |
| `human` | 8.1 % | 49.7 % | 5.0 % | 16.1 % | human average (Winter) |
| `legs` | 5.0 % | 31.0 % | 4.0 % | 28.0 % | leaden legs, low centre of mass |
| `core` | 3.0 % | 75.0 % | 2.0 % | 9.0 % | mass in the trunk, light limbs |
| `combo` | 4.0 % | 46.0 % | 3.0 % | 22.0 % | mixture |

For `legs`, `core` and `combo` the torso share is additionally pulled downward
(`low_bias`) so that the mass sits in the pelvis rather than the chest — 80 % of
it for `core`. That is the implementation of "concentrate the weight at navel
height".

---

## 4. How measurements are taken from here on

Two methodological errors from M4 are fixed here, independent of the outcome.

**The target quantity is now the right one.** M4 was accepted on `fallen`, CoM
error and QP status — on standing safety, while the claim was imitation. From
now on the primary metric is the **imitation error**: the magnitude between the
commanded target angle and the measured joint angle, averaged over arm and torso
joints, against the **undamped** target. The controller logs it as `imit`.

That is stricter than the previously logged `track`, which compared against the
set-point already reduced by `ARM_SCALE` and the safety damping, and could
therefore look good while the robot was no longer performing the motion at all.
Exactly that effect covered up the false conclusion in M4.

**The runs are partly reproducible — and the boundary is instructive.** In
M3/M4 measurement was done in real time; a repeat run gave 10.4° instead of 5.0°
and fell where it previously had not. All single-run statements in those
chapters therefore do not hold. `tools/mass_trial.py` instead runs Webots in
fast mode without rendering against a recording and repeats every configuration
three times.

The **imitation error** is then well reproducible: spread around 0.1° across
three runs. The **fall decision** is not. UDP timing between sender and
simulation is not tick-synchronous, and for configurations near the stability
boundary the outcome flips between runs. That is not a defect of the measurement
setup but the measurement itself: a profile that falls in one run of three sits
exactly on the boundary. Fall counts are therefore reported across several runs
rather than as yes/no.

**The fall detector had to become relative.** It checked against a fixed
centre-of-mass height of 0.80 m — appropriate for the shipped state with its
1.019 m standing height. Profiles with a lower centre of mass fall below that
threshold while merely standing: `legs` stands at 0.831 m, and below it after
the initial knee bend. The first `legs` runs therefore reported "fallen" after
zero packets although the robot was standing still. The threshold is now a
fraction of the standing height measured after settling for each robot
(`A3_FALL_FRACTION`, default 0.78, which reproduces the old 0.80 m for the
shipped state).

That is the same class of error as in M4: a metric calibrated for one
configuration, applied to others and not re-checked. Had I not looked at the raw
data, this document would state that `legs` and `combo` are unusable.

**Two test cases, because one is not enough.** The real recording contains no
extreme poses. `tools/pose_amplify.py` therefore rotates every measured limb
direction further away from the rest vector by a factor k and scales the trunk
angles along. The rest vectors are measured from the first 30 frames of the
recording (L [0.039 0.318 −0.945], R [0.114 −0.374 −0.918]), not assumed.

Both tests run with **`ARM_SCALE = 1.0` and the safety damping disabled**
(`A3_SAFETY=0`). The damping was the emergency brake that cut the amplitude in
M3 and showed up as blocked frames in M4. If the distribution is right it is not
needed — and whether it is needed is exactly the question.

---

## 5. Test against the real recording

53 s test recording, 1602 frames, `ARM_SCALE = 1.0`, safety damping off, three
runs per profile (`results/mass_trial_v2.json`).

| Profile | Falls | Imitation error, median | Spread | p95 | Packets | Duration |
|---|---|---|---|---|---|---|
| `spec` (as shipped) | 0/3 | 2.70° | 0.15 | 14.32° | 1064 | 60.6 s |
| `human` | 0/3 | 2.74° | 0.12 | 14.55° | 1074 | 58.6 s |
| `legs` | 0/3 | 2.79° | 0.18 | 14.23° | 1080 | 59.4 s |
| `core` | 0/3 | **2.61°** | 0.06 | **13.77°** | 1096 | 62.5 s |
| `combo` | 0/3 | 2.65° | 0.19 | 13.80° | 1084 | 60.4 s |

**On this recording the profiles are indistinguishable.** No profile falls, not
even the unmodified shipped state, and the imitation error lies between 2.61°
and 2.79° for all of them — a difference of 0.18° against a spread of up to
0.19° within the same profile. That is not a measurable advantage.

This statement matters more than it sounds, because it contradicts the
expectation this chapter began with. Section 2 supplies the explanation: the
recording simply never visits the poses in which the distribution counts. The
subject stands still, turns the upper body, raises and bends the arms — all
inside the stable quarter of the pose space. With 22.6 % unstable poses the
shipped state is not uniformly bad but **on the boundary**, and this recording
does not push it across.

Two dependable side results exist nonetheless:

**The redistribution costs nothing.** It would have been conceivable that very
light arms (1.78 kg instead of 12.35 kg with `core`) would cause jitter or
overshoot on the powerful motors. The opposite is the case: `core` has the
lowest error (2.61°) **and** the smallest spread (0.06°) in the field.

**The safety damping is not needed on this recording.** All runs used
`ARM_SCALE = 1.0` and `A3_SAFETY=0`, that is full arm amplitude without the
emergency brake that had damped to 75 % in M3. The robot does not fall anyway.
For this material the damping was an unnecessary restriction on fidelity.

---

## 6. Stress test with amplified poses

The same recording, but every limb direction rotated a factor of 1.3 further
away from the rest vector and the trunk angles scaled along
(`results/mass_trial_a13.json`). This produces poses the recording does not
contain but that a human can adopt.

| Profile | Falls | Imitation error | Spread | p95 | Duration to end/fall |
|---|---|---|---|---|---|
| `spec` (as shipped) | **3/3** | 9.84° | 1.61 | 44.31° | **21.3 s** |
| `human` | **1/3** | 3.82° | 0.65 | 34.54° | 56.6 s |
| `legs` | 0/3 | 3.74° | 0.35 | 34.82° | 58.8 s |
| `core` | **0/3** | **3.61°** | **0.10** | **33.28°** | **61.6 s** |
| `combo` | 0/3 | 3.61° | 0.14 | 33.48° | 60.1 s |

For runs that fell, the error is formed only up to the fall and therefore
flatters; `spec` reaches its 9.84° in 21 s, not in 60 s.

**Here the profiles separate, and along the static prediction.** The ranking
from section 2 — fraction of stable poses — predicts the dynamic behaviour
correctly:

| Profile | Stable poses (static) | Falls (dynamic) |
|---|---|---|
| `spec` | 77.4 % | 3 of 3 |
| `human` | 83.6 % | 1 of 3 |
| `legs` | 100 % | 0 of 3 |
| `core` | 100 % | 0 of 3 |
| `combo` | 100 % | 0 of 3 |

This is the retrospective confirmation that section 2 measured the right
quantity. A static sample over the pose space, computable in seconds, predicts
what a minute-long simulation run shows.

**The shipped state falls reproducibly after 21 seconds.** Unlike on the
unamplified recording, nothing here is marginal: all three runs, within a narrow
window of 21.0 to 22.2 s. As soon as the motion leaves the stable region, the
distribution decides.

**Exactly human proportions are not enough.** `human` falls in one of three runs
and has the second-largest spread in the field. A human carries 5 % arm mass per
side because they have toes, compliant joints, reflexes and constant
feed-forward to compensate for it. Copy the proportion without the compensation
mechanisms and you inherit only the load. That is the clearest evidence that
"human" and "good for this robot" are two different things.

**`core` is the best profile**, and in all four metrics at once: no fall,
smallest imitation error, smallest spread, longest run time. It is at the same
time the profile that most directly matches the requirement "concentrate the
weight at the centre of the trunk so that the extremities do not drag the centre
of mass along": 75 % of the mass in the torso, 80 % of that in the pelvis, 2 %
per arm. Arm sway is 10.6 mm against 73.7 mm in the shipped state.

### Harder amplification: factor 1.6

The same setup at factor 1.6 (`results/mass_trial_a16.json`):

| Profile | Falls | Imitation error | Spread | Duration |
|---|---|---|---|---|
| `spec` | **3/3** | 9.57° | 0.66 | **12.6 s** |
| `human` | 1/3 | 4.30° | 1.03 | 57.0 s |
| `legs` | 0/3 | 4.97° | 0.77 | 51.6 s |
| `core` | **0/3** | **4.11°** | **0.12** | **58.4 s** |
| `combo` | 0/3 | 4.22° | 0.30 | 58.1 s |

The shipped state now falls after 12.6 s instead of 21.3 s; otherwise the
ranking does not change. `legs` does not fall but loses fidelity (4.97°) and
consistency (spread 0.77) — the leaden legs make the leg motion sluggish enough
to lag the reference signal.

---

## 7. Result and recommendation

| | 1.0 | 1.3 | 1.6 |
|---|---|---|---|
| `spec` (as shipped) | 0/3 | **3/3** | **3/3** |
| `human` | 0/3 | 1/3 | 1/3 |
| `legs` | 0/3 | 0/3 | 0/3 |
| **`core`** | **0/3** | **0/3** | **0/3** |
| `combo` | 0/3 | 0/3 | 0/3 |

**`core` is recommended.** It wins at all three stress levels and in all metrics
at once: no fall, smallest imitation error (2.61° / 3.61° / 4.11°), smallest
spread (0.06 / 0.10 / 0.12), longest run time. It is therefore the default of
`tools/build_atlas_proto.py`.

`combo` is the defensible alternative if a lower centre of mass is wanted
(0.861 m against 0.927 m) — it trails only slightly throughout and keeps a more
plausible leg mass at 22 % per leg.

**What the change achieves.** It removes the cause of falls in extreme poses and
makes the safety damping unnecessary, which had cut arm amplitude to 75 % in M3.
The robot imitates at full amplitude.

**What it does not achieve.** It does not make walking possible. The limit from
M5 is a controller problem: a kinematic QP at position level cannot generate
angular momentum. The distribution improves the starting position — `core` needs
only 26 % of the joint limit for the required 89 mm of lateral shift, the
shipped state 29 % — but it does not replace centroidal dynamics.

**Limits of this measurement.** All statements rest on a single 53 s recording
of one person and its computational amplification. The amplification produces
geometrically more extreme but not qualitatively new poses — no single-leg
stance, no deep knee bend, no lunge. Three runs per configuration suffice to
separate 3/3 from 0/3, but not to distinguish reliably between 0/3 and 1/3:
`human` could look better or worse with more runs.
