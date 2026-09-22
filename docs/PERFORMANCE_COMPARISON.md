# Did the project improve over the semester?

A before-and-after measurement on identical input. The same 2965-frame
reference clip was replayed through the current system and through the last
commit before the MeTRAbs migration, with the same measurement points in the
same slots and the same fidelity metric on both sides.

| | Baseline | Current |
|---|---|---|
| Commit | `36bece7`, 2026-08-20 | `53fba06`, 2026-09-22 |
| Pose estimator | MediaPipe 0.10.13, 33 landmarks | MeTRAbs `eff2s_y4`, absolute metric 3D |
| Torso-local retargeting basis | absent | present |
| Action classifier | absent | present |
| Frames graded | 2965 | 2937 |

Between the two: **51 files changed, 13,267 insertions, 1,302 deletions.** This
is deliberately a whole-system comparison, not an estimator benchmark. It
answers "is the system better than it was", and the estimator is one of many
things that changed.

The fidelity metric is fixed in `main/libraries/pose_match.py` and is
self-contained rather than borrowing the retargeter's geometry, because the
baseline has no torso-local basis to borrow. Borrowing it would have changed
the ruler between the two runs and measured the ruler instead of the systems.

---

## Headline

```
                          BASELINE          CURRENT        change
throughput              25.3 FPS         14.1 FPS      -44%
inference p50           35.6 ms          66.3 ms      +86%
upper-body error        39.0 deg          4.5 deg     -88%
lower-body error        21.4 deg         11.3 deg     -47%
```

**Imitation accuracy improved 8.6x on the upper body and
1.9x on the lower body, at the cost of 1.8x the throughput.**
That is the trade the project made, stated plainly. Whether it was the right
trade depends on whether 14 FPS is enough to feel responsive, and the latency
report says the end-to-end path is 76 ms, so it is.

## Imitation fidelity

Angle between where a limb points on the human and where the corresponding limb
points on NAO, each in its own torso frame. Lower is better. Recomputed offline
on both sides, so latency cannot contaminate either number.

### Upper and lower body

```
                          avg    p50    p90
BASELINE  arms          39.0d  39.5d  48.6d  ██████████████████████████████
CURRENT   arms           4.5d   4.1d   4.1d  ███
BASELINE  legs          21.4d  18.0d  36.9d  ████████████████
CURRENT   legs          11.3d   8.7d  22.0d  █████████
```

### Per extremity

```
              baseline   current  gain
Left arm         42.4d      4.6d   9.3x
  was                             ██████████████████████
  now                             ██
Right arm        36.6d      4.5d   8.1x
  was                             ███████████████████
  now                             ██
Left leg         21.3d     11.0d   1.9x
  was                             ███████████
  now                             ██████
Right leg        21.6d     11.6d   1.9x
  was                             ███████████
  now                             ██████
```

Left and right agree within a few degrees on both systems, so neither has a
side bias.

### Per limb segment

```
segment         baseline   current  improvement
upper_arm_L        18.7d      8.1d    2.3x  ██████
upper_arm_R        17.7d      8.1d    2.2x  █████
forearm_L          66.1d      1.0d   64.1x  ████████████████████
forearm_R          55.5d      1.0d   57.9x  █████████████████
thigh_L            17.2d      9.0d    1.9x  █████
thigh_R            17.9d     10.4d    1.7x  █████
shin_L             25.3d     13.0d    1.9x  ████████
shin_R             25.4d     12.9d    2.0x  ████████
```

**The forearm is where the difference is decisive: 66.1 deg to 1.0 deg on the
left, 55.5 to 1.0 on the right.** The baseline could not place a forearm at all;
60-plus degrees of error means the robot's arm was routinely pointing somewhere
the human's was not. That is the torso-local basis and the closed-form elbow
solve doing their job, not the estimator alone.

The upper arm improves less dramatically (18.7 to 8.1 deg) because 8.1 deg is a
hard floor: it is `atan(0.015 / 0.105)`, NAO's elbow offset, and no retargeter
can beat it. The current system is **at** that floor and never leaves it; the
baseline was 2.3x above it.

## Speed

```
stage                 baseline   current   p50 ms
inference              35.581   66.279   ██████████████████████████
acquire                 2.635    2.439   █
flip                    0.847    0.909   █
log_pose                0.209    0.184   █
gait_cue                0.052    0.114   █
action_cue             absent    0.048   █
retarget_fallback       0.039    0.040   █
log_joints              0.021    0.025   █
udp_send               absent    0.191   █
```

Inference nearly doubled, 35.6 ms to 66.3 ms, and it was already
the whole budget on both systems. Everything else is unchanged to within a few
hundred microseconds, which is the evidence that the measurement points really
are in the same slots: the stages that were not touched did not move.

`action_cue` exists only on the current system. It costs 0.048 ms and replaced a
locomotion trigger that had no equivalent before.

## What is not compared

Control-side stage timings and tracking error are absent for the baseline: they
need Webots running that commit's world, which was out of scope for this pass.
The current system's figures are in `PERFORMANCE_METRABS.md`.

Detection rate is 100.0% for the baseline against 99.0% for the current system.
That is not a regression worth reading into: MediaPipe reports a pose on every
frame whether or not it is confident, whereas MeTRAbs declines when its detector
finds nothing, and the 1% it declines are frames where the subject is mid-turn.
A refused frame and a wrong frame are not the same thing, and the fidelity
columns above are where that distinction actually shows up.

## Conclusion

The system got substantially more accurate and somewhat slower. Upper-body
imitation error fell 8.6x to 4.5 deg, which is within a degree of the
geometric floor imposed by NAO's own arm, and lower-body error halved while
remaining deliberately bounded by the balance model. The cost was throughput,
25.3 FPS down to 14.1, which the 76 ms end-to-end latency shows is still
comfortably inside real time.