# Performance and Imitation Fidelity: MediaPipe Baseline

The same measurement applied to the last commit before the MeTRAbs migration,
on the same input, so the two can be read side by side. The comparison itself
is in `PERFORMANCE_COMPARISON.md`; this file is the baseline's own numbers.

## Method

| | |
|---|---|
| Commit | `36bece7`, 2026-08-20, last before MeTRAbs |
| Input | `data/reference/ref_20260922_141007/reference.y4m` |
| | 2965 frames, 148.3 s, 1920x1080, raw I420, 20 FPS |
| | sha256 `cf11325f8b7f0ccb...`, 20 instructed poses |
| Estimator | MediaPipe 0.10.13, 33 landmarks, normalised + relative depth |
| Hardware | RTX 3090 Ti + RTX 3070, Python 3.12 |
| Frames analysed | 2945 (first 20 dropped for warm-up parity with the MeTRAbs run) |
| Frames graded for fidelity | 2965 |

Measurement points are in the same slots as the MeTRAbs build and use the same
`perf.py` and `pose_match.py`. Two stages present there are absent here because
the code did not exist yet: `action_cue`, and `smooth_keypoints` (the
speed-adaptive landmark filter). Their absence is a property of the system, not
a gap in the measurement.

**Percentiles.** `p50` is the median. `p90` means 90% of frames were faster and
the slowest 10% worse. `p99` is the worst 1%.

---

## 1. Where the delay is: perception

```
stage                     mean   share
inference              39.163ms   91.0%  ████████████████████████████████████████
acquire                 2.660ms    6.2%  ███
flip                    0.881ms    2.0%  █
log_pose                0.224ms    0.5%  █
gait_cue                0.053ms    0.1%  █
retarget_fallback       0.041ms    0.1%  █
log_joints              0.023ms    0.1%  █
                       43.044ms   total
```

**Inference is 91.0% of perception.** Everything else together costs
3.88 ms. The same shape as the MeTRAbs build: a pose estimator with
bookkeeping attached.

With the estimator removed, so the remainder is legible:

```
acquire                 2.660ms  ████████████████████████████████████
flip                    0.881ms  ████████████
log_pose                0.224ms  ███
gait_cue                0.053ms  █
retarget_fallback       0.041ms  █
log_joints              0.023ms  █
```

## 2. Distribution of each stage, p50 to p99

```
                                              0.01      0.1       1         10        100  ms
stage                    p50     p90     p99  |         |         |         |         |
inference             35.581  52.134  58.925                                      ▐+▌
acquire                2.635   2.909   3.404                          ▐+
flip                   0.847   1.081   1.395                     ▐+▌
log_pose               0.209   0.312   0.420               ▐─+▌
gait_cue               0.052   0.066   0.084         ▐+▌
retarget_fallback      0.039   0.050   0.066        ▐+▌
log_joints             0.021   0.029   0.067     ▐─+──▌
```

Log scale: each `|` is a factor of ten. `▐` marks p50, `+` p90, `▌` p99.

`inference` spans 35.6 to 58.9 ms against the MeTRAbs build's 66.3 to 89.1. The
absolute spread is almost identical (23.3 ms against 22.8), but relative to its
own median MediaPipe is the less consistent of the two: p99 is 1.66x its p50,
where MeTRAbs is 1.34x.

## 3. Throughput

```
frame period       p50  39.58 ms   p90  56.22 ms   -> 25.3 FPS
detection rate     100.0%  (2945 of 2945 frames)
```

MediaPipe reports a pose on every frame whether or not it is confident, so a
100% detection rate is not directly comparable with an estimator that declines.
Where that difference actually shows is the fidelity below.

## 4. Control loop and end to end

Not measured for this commit. Control-side stage timings, transport delay and
motor tracking error all require Webots running this commit's world, which was
out of scope for this pass. The current system's figures are in
`PERFORMANCE_METRABS.md`; the perception and fidelity halves above are complete
and are what the comparison rests on.

## 5. Imitation fidelity

Angle between where a limb points on the human and where the corresponding limb
points on NAO, each in its own torso frame. Recomputed offline: the retargeter
is re-run on the recorded landmarks and the human pose compared against the
command those exact landmarks produce, so no time passes between the two and
latency cannot contaminate the number.

### Upper body against lower body

```
                           avg     p50     p90     p99
UPPER BODY (arms)        39.0d   39.5d   48.6d   69.3d  ████████████████████████████
LOWER BODY (legs)        21.4d   18.0d   36.9d   60.9d  ███████████████
```

Unusually, the **upper body is worse than the lower body here**, which is the
reverse of the current system. The arms are the part this commit could not
place: without a torso-local basis there is no stable frame to express a limb
direction in, so shoulder and elbow targets drift with the subject's heading.

### Per extremity

```
                 avg     p50     p90     p99
Left arm       42.4d   43.0d   52.3d   74.7d  ████████████████████████████
Right arm      36.6d   36.0d   48.9d   66.5d  ████████████████████████
Left leg       21.3d   17.6d   31.8d   65.1d  ██████████████
Right leg      21.6d   17.6d   33.0d   67.5d  ██████████████
```

Left and right arms differ by 5.8 deg, a larger asymmetry than the
current system's sub-degree agreement, and a sign the arm solve was not
side-symmetric at this commit.

### Per limb segment

```
segment            avg     p50     p90     p99
upper_arm_L      18.7d   13.0d   35.3d   77.2d  ████████
upper_arm_R      17.7d   13.9d   33.5d   64.6d  ███████
forearm_L        66.1d   72.2d   80.6d   95.8d  ████████████████████████████
forearm_R        55.5d   60.0d   69.1d   83.2d  ███████████████████████
thigh_L          17.2d   11.1d   32.0d  117.5d  ███████
thigh_R          17.9d   11.7d   32.5d  118.3d  ████████
shin_L           25.3d   24.2d   38.6d   53.3d  ███████████
shin_R           25.4d   23.4d   39.8d   54.3d  ███████████
```

**The forearm dominates the error at 55 to 66 deg.** Sixty degrees means the
robot's forearm was routinely pointing somewhere the human's was not. The upper
arm at 18 deg sits 2.3x above the 8.1 deg geometric floor set by NAO's elbow
offset, so roughly 10 deg of it was recoverable and was later recovered.

### By instructed pose

```
pose                 n    arms    legs
STAND STILL        121   39.7d   14.2d  ██████████████
T-POSE             140   39.5d   11.8d  ██████████████
ARMS FORWARD       120   68.4d   18.7d  ████████████████████████
ARMS OVERHEAD      121   23.5d   22.1d  ████████
LEFT ARM UP        100   35.3d   20.2d  ████████████
RIGHT ARM UP       100   17.0d   30.0d  ██████
ELBOWS BENT        120   42.5d   16.4d  ███████████████
ARM CIRCLES        160   43.2d   15.8d  ███████████████
HEAD TURN          121   39.3d   16.0d  ██████████████
HEAD NOD           100   40.5d   16.2d  ██████████████
SQUAT              160   44.3d   26.1d  ████████████████
WIDEN STANCE       120   40.0d   18.4d  ██████████████
LEAN               121   42.7d   21.0d  ███████████████
LEFT LEG UP        120   40.3d   38.9d  ██████████████
RIGHT LEG UP       120   37.0d   36.8d  █████████████
MARCH IN PLACE     160   43.7d   26.1d  ███████████████
WALK               201   40.2d   18.5d  ██████████████
TURN LEFT          180   30.9d   24.1d  ███████████
TURN RIGHT         180   30.2d   29.8d  ███████████
STAND STILL        100   38.2d   16.4d  █████████████
```

---

## Summary

| Question | Answer |
|---|---|
| Throughput | **25.3 FPS** (39.6 ms per frame) |
| What dominates it? | MediaPipe inference, **91.0%** of perception |
| Upper-body fidelity | **39.0 deg** average limb error |
| Lower-body fidelity | **21.4 deg** |
| Worst segment | forearm, 55 to 66 deg |
| Control loop, end to end, tracking | not measured at this commit |

Side by side with the current system: `PERFORMANCE_COMPARISON.md`.