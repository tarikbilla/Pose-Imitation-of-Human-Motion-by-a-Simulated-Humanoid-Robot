# M8 — The Puppet: Imitation Without Physics

Date 2026-09-05 · Branch `A3` · Raw data in `a3/results/puppet_run.json`

Status: **works.** The robot takes over the human's joint angles 1:1, stands
with the support foot exactly on the floor, walks forward and backward, and
cannot fall over or be flung away because there are no dynamics.

---

## 1. Why this path

The physics path (M6/M7) achieves walking, but with limits: about 16 steps or
2.9 m, after which the robot tips over due to accumulated lateral drift, and
imitation fidelity suffers from the balance requirement.

The two references of the proposal span the design space:

**[1] TWIST (Ze et al., CoRL 2025)** trains in **IsaacGym** and evaluates in
**MuJoCo** with full dynamics: the reward function contains terms for foot
contact and foot slippage, and domain randomisation varies gravity. The
retargeted motion is **not played back** — an RL+BC policy tracks it and issues
joint targets to a PD controller at 50 Hz. Balance emerges from the tracking
rewards, not from a designed controller.

**[2] Tao et al. (CCRIS 2021)** does almost exactly what M3 does: 2D keypoints,
3D lifting, **inverse kinematics**, plus "a balance strategy based on ankle
joint adjustment". Locomotion is not mentioned; an ankle correction is
quasi-static and therefore the same class on which M5 failed.

There is nothing in between. A hand-designed controller that runs physically
correct from camera data appears in neither reference, and IsaacGym is out on an
AMD card.

**What this approach costs must be stated honestly.** The proposal names as its
scientific interest that "bipedal locomotion requires intelligent steering to
maintain balance". M8 **does not answer that question, it sidesteps it.** M6/M7
are therefore retained; only the comparison quantifies what physics costs in
imitation fidelity.

---

## 2. The method

Three building blocks, all measurable:

**Physics-free model.** `tools/build_atlas_proto.py --kinematic` produces
`AtlasA3Kin.proto`: all 29 `Physics` nodes are removed, in the main PROTO and in
all 29 vendored sub-PROTOs (`webots/protos/vendor_kin/`). Webots therefore does
not simulate the robot in ODE. There is no gravity, no contact force and no
ground reaction — the robot cannot be flung away by a step, because a step
generates no force. Verified: 0 `Physics`, 28 `PositionSensor`, balanced braces.

**Root placement instead of dynamics.** The supervisor sets `translation` and
`rotation` of the root node each tick so that the sole of the support foot lies
exactly on the ground plane:

```
pelvis_world = anchor - R * sole_local(support foot)
```

`anchor` is the world point the support foot sticks to. As long as the same foot
carries, the pelvis travels relative to it — that is exactly how locomotion
arises.

**Support foot switching.** The foot whose sole is lower carries (with 4 mm of
hysteresis). On a switch the anchor is set to the current world position of the
new support foot. Since both soles are at the same height at the moment of the
switch, the transition is continuous. If the human lifts a foot, the robot lifts
it, the other takes over, and the pelvis follows forward.

### An error that made the method unusable at first

The first version placed the root using the **commanded** joint angles. But the
motors lag at 12 rad/s; on a 0.35 rad set-point step the pelvis teleported by
**262 mm** in a single tick. Using the **measured** angles from the
`PositionSensor`s, the largest jump falls to **13.8 mm**, and the support foot
sits correctly by construction.

---

## 3. Results with a parametric gait

14 s, step period 1.2 s (`results/puppet_run.json`):

| Direction | Forward travel | Lateral drift | Switches | Max jump | Sole above floor |
|---|---|---|---|---|---|
| forward | **+4.92 m** | **0.000 m** | 22 | 13.8 mm | 0.00 mm |
| backward | **-2.70 m** | 0.000 m | 23 | 16.8 mm | -2.34 mm |

And with an alternating command — forward, stop, backward, stop, forward over
26 s:

| t | Support switches | Position |
|---|---|---|
| 1.1 -> 8.2 s | forward | y 0.17 -> **2.87 m** |
| 8 -> 12 s | standing | y stays 2.87 |
| 12 -> 19.3 s | backward | y 2.76 -> **1.38 m** |
| 21.6 -> 25.1 s | forward | y 1.48 -> **2.83 m** |

Lateral drift across the whole 26 s: **0.000 m**. Largest jump 16.8 mm, median
8.1 mm. The support foot never leaves the ground upward
(`support_foot_z_max = 0.00 mm`).

For comparison: the physics path had **0.54 m** of lateral drift over 2.6 m of
travel and fell after about 16 steps.

---

## 4. Whole-body retargeting

`perception/fullbody.py` computes all joint angles from the keypoints: torso,
head and arms as in M3, the legs through the 3D lifter.

### Why the foreshortening method is unsuitable for legs

The first version derived leg directions from perspective foreshortening, as for
the arms. That produced a **posture in which the torso hung 34 cm behind the
feet**. Two lines in `frames.py` explain it:

- `depth_component` returns `ref * sqrt(1 - ratio^2)` — the depth is **always
  positive**, hence always forward. The sign is fundamentally unobtainable from
  foreshortening.
- `SegmentReference.update` takes the decaying **maximum** as the reference
  length. Almost every frame therefore counts as foreshortened and receives a
  depth component.

For arms this is tolerable, because they do not carry the body. For legs the
error accumulates into a constant forward offset of the foot, and since the root
is placed via the support foot, the trunk moves backward by exactly that amount.

### The 3D lifter, plus bias correction

MotionBERT provides signed depth. The axes were checked against the data (left
hip x +0.16, right -0.16; ankle y +0.91 below the hip) and read
`body = [-z, x, -y]`. A constant model bias remains, which a high-pass with an
8 s time constant removes without touching the 0.7 s step motion.

| Source | Ankle in front of the hip, median |
|---|---|
| foreshortening | +0.339 / +0.353 m |
| 3D lifter | -0.142 / -0.168 m |
| **lifter + bias correction** | **+0.011 / -0.014 m** |

Sole centre relative to the pelvis: **+0.047 m** against +0.35 m before.

### The trunk: a clamp from M3 that has no business here

While re-checking the posture, a second independent cause surfaced. The trunk
angles in `fullbody.py` were clamped to values inherited from M3, where a tilted
trunk endangers **balance**:

| Joint | Joint limit | my clamp | Factor |
|---|---|---|---|
| `BackMby` (pitch) | -1.200 … +1.280 | ±0.10 | **12x too tight** |
| `BackUbx` (roll) | ±0.791 | ±0.30 | 2.6x |
| `BackLbz` (yaw) | ±0.611 | ±0.55 | 1.1x |

**66.4 % of all frames sat at the pitch clamp.** For two thirds of the time the
trunk therefore followed the clamp, not the human. In the puppet there is no
balance to protect; the clamp was pure loss of fidelity.

Removing it exposed a third error of the same class as the leg depth:
`BodyFrame.torso_tilt` computes `pitch = acos(ratio)` — **always positive**.
Unclamped, the trunk therefore leaned permanently forward by up to 50°. Here too
the 3D lifter provides the solution; pitch, roll and yaw now come from the spine
and shoulder vectors with a sign, and the pitch is additionally bias-corrected:

| Joint | before | **now** | Fraction negative |
|---|---|---|---|
| `BackMby` | +0.000 … +0.874 | **-0.083 … +0.094** | **41 %** |
| `BackUbx` | ±0.300 (clamped) | -0.405 … +0.391 | 22 % |
| `BackLbz` | ±0.550 (clamped) | -0.558 … +0.481 | 35 % |

Symmetric about zero instead of one-sided — which matches a standing human
moving their arms. The clamps of the physics path in `a3_upper_body.py` are
untouched; there they are justified.

### Foot orientation and workspace

The sole is prescribed **flat and body-aligned**. The first version derived it
from the heel-to-toe direction, which over-constrains a leg whose ankle has only
pitch and roll:

| Foot specification | Residual, median | p99 |
|---|---|---|
| heel-to-toe direction | 20.0 / 27.8 mm | 293 / 354 mm |
| **flat sole** | **0.019 / 0.015 mm** | 12.2 / 44.5 mm |

Remaining outliers came from **hip and ankle roll hitting their limits at the
same time** — the reconstruction placed the foot too far out laterally. Limiting
the lateral component to 0.26 m lowers p99 from 129 to **50.7 mm** and the
fraction above 50 mm from 2.62 % to 1.12 %.

---

## 5. Locomotion: the strong signal instead of the weak one

With a correct posture the robot nevertheless travelled only 0.003 m. The
measurement shows why, and it is **not a programming weakness but a data
limit**:

| Metric in the walking segment | Measured | Expected |
|---|---|---|
| `corr(foot height, sagittal position)` left | **+0.313** | strongly negative |
| the same, right | **-0.508** | strongly negative |
| swing leg relative to stance leg | **-0.033 against +0.049** | swing leg **in front of** the stance leg |
| sagittal swing | ±0.06 m | ±0.3 m |

The two legs do not even share the same cycle sense, and the swing leg lies on
average *behind* the stance leg. The reason is the frontal camera: walking
toward the camera moves the legs almost entirely along the **depth axis**, the
worst observable one. The lifter returns about 20 % of the true amplitude there.

**The solution separates pose from locomotion.** The angles stay 1:1 from the
lifter; the travel comes from the signal the camera measures well:

- **forward** from the apparent body size. Torso length 303 -> 538 -> 313 ->
  525 px across the walking segment, noise 0.0027 per frame. Metrically through
  `travel = f * L * delta(1/p)`; `A3_FOCAL_PX` (default 1000 px) is the only
  assumption a camera calibration would replace.
- **sideways** from the image x axis of the hip centre. Here the focal length
  cancels: `lateral = delta_u * L / p`. No assumption needed.

Height and ground contact continue to come from the support-foot anchoring.

---

## 6. Second iteration: from the sliding to the stepping puppet

The first version placed the root over the support foot **and** then overwrote
the horizontal position with the distance signal. The anchoring was thereby
void: the feet slid by exactly the difference. The robot imitated the pose but
glided over the floor instead of stepping on it.

The rework separates the responsibilities cleanly:

| Quantity | Source |
|---|---|
| Trunk, head, arms | angles 1:1 from the 3D lifter |
| Legs | **inverse kinematics** onto foot targets, not copied angles |
| Support foot | pinned in the world frame |
| Swing foot | trajectory with 50 mm of ground clearance |
| Forward, sideways | from perception, see below |

Copying thereby becomes **retargeting under constraints**: the ground-contact
constraint takes precedence, and style follows. That matches the criterion
"adapt the motion to the robot as well as possible".

### The workspace was measured, not estimated

Reachability of the left leg with a flat sole, criterion below 1 mm of residual
error:

| Stance height | forward | backward | sideways |
|---|---|---|---|
| 0.76 m | — | — | — (already impossible when standing) |
| 0.80 m | 0.44 m | **0.04 m** | 0.34 m |
| **0.83 m** | **0.44 m** | **0.28 m** | **0.36 m** |
| 0.89 m | 0.33 m | 0.14 m | 0.22 m |

The space is strongly **asymmetric**: backward the leg reaches only half as far,
because with a flat sole the ankle offers only ±0.698 rad. At 0.80 m the
backward reach collapses to 4 cm — walking backward would be impossible. 0.83 m
is the optimum.

A deep squat below 0.76 m is **kinematically impossible** with a flat sole, not
merely difficult. The solver reported 31 mm of residual error there and burned
50–76 ms on restarts that could not help.

### Parameter study

Offline over the pelvis trajectory of the recording, without Webots
(`tools/sweep_puppet.py`):

| Stance height | Step trigger | IK error p95 | IK max |
|---|---|---|---|
| 0.83 m | 0.13 m | 50.5 mm | 91.3 mm |
| 0.83 m | 0.09 m | 0.194 mm | 39.6 mm |
| **0.83 m** | **0.06 m** | **0.173 mm** | **9.50 mm** |
| 0.85 m | 0.10 m | 15.7 mm | 72.4 mm |

The step trigger dominates everything: from 0.13 to 0.06 m the error falls by a
factor of 260. Swing duration, by contrast, is almost without effect.

### The time step is a control parameter, not a detail

In physics-free mode `setVelocity` has **no effect** — measured 0.913 rad/s at a
commanded 12 as well as at 60 rad/s. The achievable joint rate scales with the
simulation time step instead:

| Time step | Imitation error, median | p95 | Slip p95 | Foot height |
|---|---|---|---|---|
| 8 ms | 1.746° | 13.64° | 3.10 mm | -14.5 … +10.7 mm |
| **16 ms** | **0.883°** | **7.73°** | 5.28 mm | **-2.5 … +2.6 mm** |
| 32 ms | 1.022° | 3.93° | 10.37 mm | -4.9 … +4.7 mm |

16 ms halves the imitation error **and** lowers the foot-height error by a
factor of five. Without physics the larger time step costs nothing.

---

## 7. Locomotion from the strong signal

Torso length as a distance measure was unusable: it also shortens when bending
forward. Across the standing phase the estimated position drifted by
**0.559 m**, and the robot shuffled 24 times while the human just stood there.

Five measures were compared against each other (noise while standing against
signal swing while walking):

| Measure | Signal-to-noise ratio |
|---|---|
| **body height (nose to ankle)** | **8.3** |
| torso length | 6.4 |
| hip width | 5.9 |
| head width | 3.8 |
| shoulder width | 3.5 |

Plus a **velocity gate**: locomotion produces a sustained velocity, posture
noise does not. While standing, velocity is at p99 = 0.186 m/s; while walking it
is 0.166 m/s at the median and 0.260 m/s at p90. A gate at 0.22 m/s with 0.6
hysteresis separates the two completely:

| | before | **now** |
|---|---|---|
| Drift while standing | 0.559 m | **0.0000 m** |
| Travel while walking | 0.789 m | 0.455 m |
| Steps while standing (38 s) | 24 | **10** (weight shifts) |

The remaining 10 steps while standing are genuine lateral weight shifts; the
lateral threshold of 0.16 m was measured for this (0.07 m gave 28 steps, 0.20 m
only 6, but with clamping).

Sideways needs no focal-length assumption: `lateral = delta_u * d / f` with
`d = f*L/p` cancels to `delta_u * L / p`.

### Two measurement artefacts in my own analysis

Both nearly led to false conclusions and belong in the record:

- The report "all support-foot switches go to `R`" arose because I printed
  **every fourth** entry of an alternating sequence. The real sequence is
  `RLRLRL…`.
- A "180° jump" in `LArmEly` was a `.get(name, 0.0)` in my analysis: the joint
  simply does not exist in the first 14 frames. Across frames where it does
  exist, the largest jump is exactly 15.28° — the rate limit.

### Smoothing the arms

`ArmEly` (elbow yaw) jumped by up to 180°, because it is undetermined with the
arm extended (|ArmElx| at jumps: median 7.7° against 33° otherwise). Two
measures: tracking weighted by the elbow flexion, and a rate limit of 8 rad/s
for arms, 4 rad/s for trunk and head. Result: p99 from 55.0° to 15.3°, no more
solution switches.

---

## 8. State after the iteration

53.4 s recording, time step 16 ms (`results/puppet_run.json`):

| | first version | **now** |
|---|---|---|
| Imitation error, upper body, median | — | **0.883°** |
| p95 | — | 7.73° |
| Support-foot slip, median | (sliding) | **0.668 mm** |
| Slip p95 / max | — | 5.28 / 7.93 mm |
| Foot above/below floor | -92 … +24 mm | **-2.5 … +2.6 mm** |
| In double support | — | **exactly 0.00 mm** |
| Leg IK, median / max | — | 0.002 / 11.8 mm |
| Workspace clamping | 28.0 % | **0.0 %** |
| Steps, of which while standing | 45 / 24 | **24 / 10** |
| Forward travel / range | -0.955 m | +0.288 m / 0.404 m |
| Compute time | — | 6.2 s for 53.4 s |

---

## 9. What remained open at this point

**Focal-length assumption.** The absolute scale rests on `A3_FOCAL_PX = 1000`
and `A3_BODY_M = 1.70`. The *shape* of the trajectory is independent of it, the
*magnitude* is not.

**The velocity gate costs travel.** Of 0.789 m of measured signal, 0.455 m
remains; the rest falls below the threshold. A clean replacement would be a step
detection that ties locomotion to actual touchdown events rather than to a
velocity threshold.

**Four wrist joints are missing** (`ArmUwy`, `ArmMwx`), and the **root rotation
stays fixed** — a body turn does not turn the robot.

**The legs are retargeting, not a copy.** The human's leg angles no longer go
directly onto the robot; they only determine *when* it steps. That is the price
of the foot actually standing.

**The monitor shows the input data**, not the output, and the controller reads a
precomputed angle file. For the live camera the driver still has to be switched
to `fullbody.py` and `locomotion.py`.

---

## Iteration 3: locomotion from the camera image, honestly measured ground contact

### Why the travel previously went in the wrong direction

The travel arose solely from the foot anchoring: the pelvis followed the support
foot, and how far the robot got followed from the sagittal position of the feet
in the lifted skeleton. `tools/check_gait_phase.py` shows that this signal
carries no gait phase. In real walking the foot moves forward during swing and
backward during stance, so the correlation between foot height and sagittal
velocity must be positive. Measured: **-0.087** (left) and **-0.116** (right),
that is practically zero. The travel was integrated noise, which is why a mere
sign flip could turn it from -0.908 m into +1.078 m without anything improving.

### Distance from the image instead of from the legs

`perception/locomotion.py` now estimates distance **absolutely** instead of
integrating a gated velocity; that removes all drift. Two independent cues are
fused:

| Cue | Basis | Standing spread | S/N |
|---|---|---|---|
| Body measures (median of 5) | apparent size, d proportional to 1/px | 159 mm | 1.57 |
| Ground plane | ankle y with a fixed camera | 153 mm | 1.73 |
| **Fusion** | plane regressed against size | **140 mm** | **1.79** |

The camera height is estimated from the data itself (0.916 m, R^2 = 0.79) and
the focal length from the image geometry (0.72 x 1920 = 1382 px). The two
independent methods agree to 0.885: most of the apparent standing spread is
**real** weight shifting, not noise.

Result for the recording: -0.45 -> +0.65 -> -0.41 -> +0.58 m, that is exactly
the described sequence of forward, backward, forward toward the camera.

### How the travel reaches the robot without losing ground contact

The pelvis continues to follow the support foot in the short term, while the
**slow** component follows the camera signal (complementary filter,
`A3_TRAVEL_TAU` 0.35 s). Because the foot IK forces the anchored foot onto its
anchor anyway, ground contact survives. Three limits keep this physically
honest:

* **Reach room** instead of a fixed drift cap: the pelvis may only lead as far
  as the stance leg can carry at an admissible height (`drift_room`).
* **Height ceiling with a floor**: the robot bends its knees only as far as the
  step demands, never below `A3_HEIGHT_FLOOR` = 0.84 m. At 0.92 m of pelvis
  height only 119 mm of horizontal foot room remains; at 0.85 m it is 372 mm —
  which is why stepping without a slight drop is geometrically impossible.
* **Swing foot projected onto the reach sphere** instead of letting the IK
  saturate; this produces the bent swing knee by itself.

### Three measurement errors in my own metrics

1. **Slide measured against set-points.** The old metric compared foot position
   against the very equation that produced it. Now the simulation's foot node is
   read through `getFromProtoDef`.
2. **Lag measured against the wrong time step.** Sensors are read before
   `setPosition`; the comparison ran against the *new* command instead of the
   previous one and reported 8.3° of lag where there was none.
3. **Foot node compared against a pelvis position not yet applied.** The world
   file places the pelvis at z = 1.0; 1.0 - 0.846 = 0.154 is exactly the value
   measured. The derived "72.8 mm offset" and the conclusion that the robot was
   floating were both wrong.

`controlPID` in the PROTO (generator: `A3_CONTROL_GAIN`) remains ineffective for
the physics-free build — Webots does not control without `Physics`. What is
effective is the **command rate limit** `A3_COMMAND_RATE` = 3 rad/s: it lowers
the true lag from p95 5.8° to 1.43°.

### Final state (54 s, `results/puppet_run.json`)

| Quantity | before | now |
|---|---|---|
| Forward travel | -0.908 m (wrong direction) | **+0.407 m** (target +0.580 m) |
| Travel range | 1.171 m | 1.113 m |
| Tracking error, median | — | 19.3 mm |
| Penetration (simulation) | not honestly measured | -37.96 mm in 4 of 3375 frames |
| Frames deeper than 10 mm | — | **0.12 %** |
| Airborne | — | 1.29 % |
| Slide p95 (simulation) | — | 0.156 / 0.133 mm |
| Joint lag p95 | 5.80° | **1.43°** |
| Upper-body error, median | 0.855° | 0.937° |
| Pelvis height | 0.853 – 0.924 m | 0.848 – 0.922 m |
| Leg correction, median / p95 | 5.42 / 12.81° | 9.34 / 21.65° |
| Steps (lifts) | 15 | 26 |
| Neck joint `NeckAy` | -10.3 … +5.3° | **-26.7 … +18.5°** |

The larger leg correction is the price of the robot travelling at all: the
stance leg carries the difference between the 1:1 pose and the camera path.

### The neck joint

The lifter resolves head pitch poorly (raw range 14°, sd 2.3°) because the
head-to-neck segment is too short. `head_from_2d` instead measures the position
of the nose against the ear midpoint, normalised by head width — a direct image
measurement without depth, range 32.6°. The lifter remains the fallback when the
ears are occluded.

---

## Live operation: camera -> retargeting -> Webots

Until this point the controller read a precomputed angle file. The live path
replaces it with a stream:

```
camera --> Pose2D --> FullBodyRetargeter --> UDP :8768 --> a3_puppet
              |            + LocomotionTracker
              +--> monitoring window with the 2D skeleton
```

Start with one command:

    python tools/run_live.py

The launcher brings up Webots in real-time mode with `A3_LIVE=1`, waits until
the receiver has opened the port, and then starts the driver with the monitoring
window. `q` ends the session, `m` toggles mirroring, `n` resets the distance
datum, `s` saves a snapshot. If the sender stays quiet for longer than
`A3_LIVE_TIMEOUT` (10 s), the simulation ends by itself.

New components: `transport/angles.py` (packet format `a3-angles`, 440 bytes for
24 joints), `tools/live_puppet.py` (driver and monitoring), `tools/run_live.py`
(launcher). `transport/udp.py` now accepts a codec module, so the older
upper-body stream keeps working unchanged.

### Focal length from the camera profile

`configs/cameras/brio100.yaml` states a 58° diagonal. At 1080x1920 that yields
f = 1987 px, against 1382 px from the generic estimate — a 44 % difference that
goes directly into the distance. The driver therefore computes the focal length
from the profile and the actual frame size.

### The camera aborted after 200 frames

ffmpeg writes 1080x1920 raw frames at 30 fps into the pipe, that is 187 MB/s.
Processing collects them more slowly, the buffer fills, and ffmpeg reports
`real-time buffer [Brio 100] too full ... frame dropped!` until the stream
breaks. `FFmpegCameraSource` now has a **reader thread** that continuously
drains the pipe and always keeps only the newest frame (`drain=True`,
`frames_dropped` counts the discards). The source then runs indefinitely, and
latency stays low because a stale frame is never processed.

### Measured performance

| Stage | Median |
|---|---|
| 2D pose (wholebody, lightweight, DirectML) | 25.6 ms |
| Retargeting (lifter + leg IK + arms) | 8.1 ms |
| **Total** | **34.0 ms, about 29 fps** |

In operation: 30 fps with no person in frame, 16–22 fps under full processing.

Session over the live camera (30 s, person in frame throughout):

| Quantity | Value |
|---|---|
| Frames / packets sent / received in Webots | 476 / 476 / 450 |
| Forward travel / range | +0.063 m / 0.779 m |
| Plant events | 15 |
| Airborne | 1.33 % |
| Penetration | -53.7 mm |
| Upper-body error, median | 0.223° |

### Pitfall: calibration without a person in frame

At first `LocomotionTracker` calibrated on any detection, including an uncertain
one with nobody in frame. The datum then landed at -1.20 m and the airborne
fraction rose to 21 %. Distance is now only advanced when the retargeter has
delivered a **valid** pose (trunk points visible). That brought the value down
to 1.33 %.
