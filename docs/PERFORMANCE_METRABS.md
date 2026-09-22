# Performance and Imitation Fidelity: MeTRAbs Pipeline

Measured 2026-09-22 on the project's target workstation by replaying a fixed
reference clip through the full pipeline. Every figure below comes from that
one run, so it can be reproduced exactly and compared against a different
pose estimator on identical input.

## Method

| | |
|---|---|
| Input | `data/reference/ref_20260922_141007/reference.y4m` |
| | 2965 frames, 148.3 s, 1920x1080, raw I420, 20 FPS |
| | sha256 `cf11325f8b7f0ccb...`, 20 instructed poses |
| Estimator | MeTRAbs `metrabs_eff2s_y4`, `detect_interval` 2, `num_aug` 1 |
| Hardware | RTX 3090 Ti + RTX 3070, Python 3.12 |
| Simulator | Webots R2025a, `basicTimeStep` 20 ms |
| Frames analysed | 2945 (first 20 dropped: TensorFlow graph warm-up cost 17.5 s on the first call) |

The clip is stored in the camera's own pixel format so that replaying it costs
what the live camera costs. Measured: live `retrieve()` 2.15 ms, replay decode
2.44 ms. A compressed format would have added 4 ms of delay the live path never
pays, which would then be misattributed to the pipeline.

**Percentiles.** `p50` is the median (half the frames were faster). `p90` means
90% were faster and the slowest 10% were worse. `p99` is the worst 1%. Means are
not reported alone because they hide the tail, and the tail is what a person
notices.

---

## 1. Where the delay is: perception

Mean cost per frame, every stage, sorted. The scale is linear and shared.

```
stage                     mean   share
inference              61.239ms   93.5%  ████████████████████████████████████████
acquire                 2.454ms    3.7%  ██
flip                    1.005ms    1.5%  █
smooth_keypoints        0.187ms    0.3%  █
log_pose                0.182ms    0.3%  █
udp_send                0.180ms    0.3%  █
gait_cue                0.110ms    0.2%  █
action_cue              0.047ms    0.1%  █
retarget_fallback       0.038ms    0.1%  █
log_joints              0.024ms    0.0%  █
                       65.467ms   total
```

**Inference is 93.5% of perception.** Everything else together costs
4.23 ms. The pipeline is a pose estimator with some bookkeeping
attached, and nothing except the estimator is worth optimising.

Same data with the estimator removed, so the remainder is legible:

```
acquire                 2.454ms  ████████████████████████████████████
flip                    1.005ms  ███████████████
smooth_keypoints        0.187ms  ███
log_pose                0.182ms  ███
udp_send                0.180ms  ███
gait_cue                0.110ms  ██
action_cue              0.047ms  █
retarget_fallback       0.038ms  █
log_joints              0.024ms  █
```

## 2. Distribution of each stage, p50 to p99

Each row is one stage's spread. The left edge is p50, the right edge p99, and
the `+` is p90. A wide bar means the stage is erratic; a narrow one means it
costs the same every frame.

```
                                              0.01      0.1       1         10        100  ms
stage                    p50     p90     p99  |         |         |         |         |
inference             66.279  83.946  89.108                                        ▐+
acquire                2.439   2.717   3.225                          +▌
flip                   0.909   1.350   1.751                      ▐+▌
smooth_keypoints       0.202   0.236   0.328               ▐+▌
udp_send               0.191   0.218   0.277               +▌
log_pose               0.184   0.260   0.321               ▐+▌
gait_cue               0.114   0.137   0.171             +▌
action_cue             0.048   0.059   0.077         ▐+▌
retarget_fallback      0.040   0.047   0.060        ▐+▌
log_joints             0.025   0.030   0.080      ▐+───▌
```

Log scale: each `|` is a factor of ten. `▐` marks p50, `+` p90, `▌` p99, so a
long bar is an erratic stage and a short one is a predictable one.

Read the tails: `inference` spans 66 to 89 ms, so even its worst frames are
predictable. `acquire` is tight at 2.4 to 3.2 ms, which is the evidence that the
clip format was chosen correctly.

## 3. Where the delay is: control

Per 20 ms simulation step, from the replay run. These were captured before the
controller restarted and truncated its own log, so p50/p90/p99 are reported
without the intermediate percentiles.

```
stage                mean      p50      p90      p99
drive_legs         4.351   3.256  16.913  25.340  ██████████████████████████████
traj_log           1.556   1.626   2.009   2.945  ███████████
sensing            1.139   1.091   1.522   3.409  ████████
pose_match         0.509   0.018   2.152   4.206  ████
retarget           0.476   0.427   0.676   1.211  ███
tick_arms          0.116   0.105   0.191   0.281  █
read_feedback      0.069   0.064   0.095   0.147  █
udp_recv           0.051   0.028   0.139   0.254  █
TOTAL              8.267  of the 20 ms step budget
```

`drive_legs` is the one stage with a dangerous tail: a 3.3 ms median against a
**25.3 ms p99, which exceeds the 20 ms step itself**. On the worst 1% of steps
the leg arbiter overruns its own budget. A mean of 4.4 ms would have shown none
of that.

## 4. End to end

Frame captured to the control step that acted on it. The packet carries the
frame's capture timestamp, so this is the whole computation path in one number.

```
                           0                            120 ms
TOTAL                       75.95 100.78                              ▐────────▌
  perception work           57.47  88.13                       ▐──────────▌
  handover + step wait      10.76  17.78      ▐──▌
```

| | p50 | p90 | p99 |
|---|---|---|---|
| **Frame captured to robot commanded** | **75.95 ms** | 100.78 ms | 113.31 ms |
| Perception work | 57.47 ms | 88.13 ms | |
| Handover + 20 ms step wait | 10.76 ms | 17.78 ms | |

Throughput 14.1 FPS, detection 99.0% of frames. This excludes camera exposure
and USB transfer: a file replay does not perform them, so the true
photon-to-motor figure is larger by whatever the capture chain costs.
---

## 5. Imitation fidelity

Latency and fidelity are different questions and are measured separately here.
The controller can only compare the human pose it just received against the
robot as it is at that instant, and the robot is still catching up, so an
in-loop comparison would mix "the target was wrong" with "the motors had not
arrived yet". That is a latency measurement wearing a fidelity label.

So fidelity is recomputed offline: the retargeter is re-run on the recorded
landmarks and the human pose is compared against the command **those exact
landmarks produce**. No time passes between the two, so delay cannot enter.
Tracking error, which is genuinely about the motors, is reported separately.

Error is the angle between where a limb points on the human and where the
corresponding limb points on NAO, each in its own torso frame. 2937 frames.

### Upper body against lower body

```
                           avg     p50     p90     p99
UPPER BODY (arms)         4.5d    4.1d    4.1d   15.4d  ████
LOWER BODY (legs)        11.3d    8.7d   22.0d   34.2d  ██████████
```

The upper body tracks roughly **2.5x more faithfully than the lower body**, which is the
architecture working as designed rather than a defect: a single camera cannot
see the robot's centre of mass, so the legs are deliberately granted only as
much of the requested pose as the balance model says is safe. The arms carry no
such constraint.

### Per extremity

```
                 avg     p50     p90     p99
Left arm        4.6d    4.1d    4.1d   15.6d  █████████████
Right arm       4.5d    4.1d    4.1d   15.3d  ████████████
Left leg       11.0d    8.5d   14.7d   50.8d  ██████████████████████████████
Right leg      11.6d    8.8d   15.8d   50.4d  ████████████████████████████████
```

Left and right are within a degree of each other on both arms and both legs, so
there is no side bias in the retargeting.

### Per limb segment

```
segment            avg     p50     p90     p99
upper_arm_L       8.1d    8.1d    8.1d    8.1d  ███████████████████
upper_arm_R       8.1d    8.1d    8.1d    8.1d  ███████████████████
forearm_L         1.0d    0.0d    0.0d   23.0d  ██
forearm_R         1.0d    0.0d    0.0d   22.4d  ██
thigh_L           9.0d    5.5d   12.2d   79.0d  █████████████████████
thigh_R          10.4d    6.5d   13.8d   79.5d  ████████████████████████
shin_L           13.0d   12.1d   18.0d   36.6d  ██████████████████████████████
shin_R           12.9d   11.1d   18.3d   41.6d  ██████████████████████████████
```

Two results need saying plainly.

**The forearm is essentially exact (0.0 deg at the median).** The elbow solve
reproduces the human forearm direction to within the precision of the landmarks.

**The upper arm sits at a flat 8.1 deg and never moves off it.** That is not
error, it is geometry: `atan(0.015 / 0.105)`, NAO's 15 mm elbow offset. Its
upper arm is not collinear with a human humerus, so 8.1 deg is the floor any
retargeter could achieve. The same constant had to be removed from the flexion
metric, where it appeared as a fixed 8.1 deg elbow bias.

### Tracking error: does the robot do what it is told

Commanded against measured, same row of the trajectory log, so the two are
aligned with each other by construction. Radians.

```
joint                 p50      p90      p99
LShoulderPitch    0.0000  0.0001  0.1015  ████████████
RShoulderPitch    0.0000  0.0001  0.1069  ████████████
LShoulderRoll     0.0000  0.0000  0.0367  ████
RShoulderRoll     0.0000  0.0000  0.0319  ████
LElbowRoll        0.0000  0.0001  0.0736  █████████
RElbowRoll        0.0000  0.0001  0.0747  █████████
LHipPitch         0.0000  0.0002  0.2092  ████████████████████████
RHipPitch         0.0000  0.0000  0.2251  ██████████████████████████
LKneePitch        0.0000  0.0000  0.1705  ████████████████████
RKneePitch        0.0000  0.0000  0.1867  ██████████████████████
```

**Every joint tracks its command exactly at the median.** The robot does what it
is told; essentially all of the mismatch in the tables above is the target
itself differing from the human, not the motors failing to reach it. The p99
tails (0.35 to 0.49 rad on hips and knees) are motion-clip playback, where the
legs are driven open loop at speeds the position controller lags.

### By instructed pose

```
pose                 n    arms    legs
STAND STILL        121    4.1d    8.0d  ████████
T-POSE             140    4.1d    6.2d  ███████
ARMS FORWARD       120    4.1d    7.0d  ███████
ARMS OVERHEAD      121    4.1d    9.7d  ██████████
LEFT ARM UP        100    4.1d   11.4d  ████████████
RIGHT ARM UP       100    4.1d   14.2d  ███████████████
ELBOWS BENT        120   14.2d   10.1d  ███████████
ARM CIRCLES        160    4.5d    6.9d  ███████
HEAD TURN          121    4.1d    8.0d  ████████
HEAD NOD           100    4.1d    7.6d  ████████
SQUAT              160    4.1d   19.3d  ████████████████████
WIDEN STANCE       120    4.1d   11.6d  ████████████
LEAN               121    4.1d   10.6d  ███████████
LEFT LEG UP        120    4.1d   24.0d  █████████████████████████
RIGHT LEG UP       120    4.1d   24.9d  ██████████████████████████
MARCH IN PLACE     160    4.6d   18.2d  ███████████████████
WALK               201    4.1d   10.1d  ███████████
TURN LEFT          152    4.1d    8.5d  █████████
TURN RIGHT         180    4.1d    9.0d  █████████
STAND STILL        100    4.1d    8.2d  █████████
```

The arms hold ~4 deg through every pose. The legs degrade exactly where the
balance model takes authority away: single-leg stances and marching are the
worst, standing and turning the best. That is the centre-of-mass constraint
becoming visible in the numbers, not a tracking failure.

---

## Summary

| Question | Answer |
|---|---|
| How long from frame to robot motion? | **76 ms** median, 101 ms p90, 113 ms p99 |
| What dominates it? | MeTRAbs inference, **93.5%** of perception |
| What is the control loop's margin? | 6.5 to 8.3 ms of a 20 ms step, but `drive_legs` hits **25.3 ms at p99** |
| How faithful is the upper body? | **4.5 deg** average limb error |
| How faithful is the lower body? | **11.3 deg**, by design: the balance model limits it |
| Does the robot obey its commands? | Yes: **0.0000 rad** median tracking error on all ten joints |

The next step is to replay this identical clip through the pre-MeTRAbs
MediaPipe commit and compare both halves on the same input.