# M2 — GPU Perception

Date 2026-09-03 · Branch `A3` · Raw data in `a3/results/m2_*.json`

Status: **software complete and verified on the GPU.** The camera run with the
Brio 100 was still pending at the time of writing — §5 is its start guide.

---

## 1. Model stack

Instead of the heavy MMDeploy toolchain,
[`rtmlib`](https://github.com/Tau-J/rtmlib) is used: RTMPose without `mmcv`,
`mmpose` or `mmdet`, directly on ONNX Runtime. It downloads the models itself on
first run into `~/.cache/rtmlib`.

`BodyWithFeet` is used — the **Halpe-26** variant. Compared with COCO-17 its 26
points additionally contain head, neck, hip centre and, **per foot, big toe,
small toe and heel**. Those six foot points are the reason for the choice: foot
lift events and foot orientation are then measured rather than estimated.

| Mode | Detector | Pose | Resolution |
|---|---|---|---|
| `lightweight` | YOLOX-tiny | RTMPose-s | 256×192 |
| `balanced` (default) | YOLOX-m | RTMPose-m | 256×192 |
| `performance` | YOLOX-x | RTMPose-x | 384×288 |

---

## 2. DirectML: rtmlib does not know it — patch

`RTMLIB_SETTINGS` knows only `cpu`, `cuda`, `rocm`, `mps`. Since it is an
ordinary dict, a runtime addition suffices (`perception/runtime.py`):

```python
RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
```

After that every rtmlib class accepts `device="dml"`.
`runtime.resolve_device()` falls back to CPU cleanly and reports the reason if
the provider is missing — rather than silently becoming slow.

### The packaging trap

`pip install rtmlib` pulls in **`onnxruntime` (CPU)** as a dependency. That
package **replaces `onnxruntime-directml`**, and afterwards
`DmlExecutionProvider` has vanished without trace — the code keeps running, just
8× slower. That is exactly what happened here.

Countermeasure, hard-wired into `tools/setup_env.py`:

- `rtmlib` is installed with `--no-deps`;
- after every setup `verify_directml()` checks that the provider is still there
  and prints the repair instruction otherwise.

---

## 3. Measurements

Test image: COCO `val2017/000000000785`, one person. 15 runs, median, after 3
warm-up passes.

**640×425 (original image), mode `balanced`:**

| Provider | Median | FPS |
|---|---|---|
| **DirectML** | **33.9 ms** | **29.5** |
| CPU | 271.8 ms | 3.7 |

**Speedup 8.0×** — nearly identical to the synthetic MatMul benchmark from M0
(7.9×). Both providers deliver the same 26 keypoints at confidence 0.844; the
provider changes the speed, not the result.

**1080×1920 (Brio in portrait, letterboxed):**

| Mode | Median | FPS |
|---|---|---|
| `lightweight` | 18.8 ms | **53.2** |
| `balanced` | 32.4 ms | **30.9** |

Both are above the 30 fps limit of the Brio 100. `balanced` stays the default;
`lightweight` is the fallback when retargeting and transport need more headroom.
The CPU stays free for Webots in both cases — that was the goal from §4.1 of the
plan.

---

## 4. Camera rotation — why, where and which way round

### Why rotate at all

The Brio 100 has a 58° diagonal field of view, so only 30.4° vertically in
landscape. To fit entirely into frame you would have to stand 3.50 m away.
Mounted portrait this becomes 51.6° vertically: **1.97 m distance and 1.8× more
pixels on the body.** Details in §5.1 of the plan.

### Where in the data flow

The sensor stays in landscape — with the camera physically rotated, the person
therefore appears **lying down** in the raw image. RTMDet and RTMPose are
trained on upright humans and detect a lying person poorly or not at all.

**The frame is therefore rotated immediately after capture, before any
inference.** `Camera.read()` does this itself — every frame the class hands out
is already rotated:

```
sensor 1920×1080 (person lying down)
        │
        ▼  cv2.rotate()  in Camera.read()
frame 1080×1920 (person upright)
        │
        ├──► RTMDet + RTMPose      keypoints in the rotated image
        ├──► display / overlay     the same image
        └──► retargeting           the same coordinates
```

From this point **everything works in the rotated image**. There is no inverse
transform and no pair of coordinate systems that can drift apart. That is the
reason to rotate at the very front rather than somewhere in the middle.

The **camera intrinsics** must be rotated with it — at 90° `fx`↔`fy` and
`cx`↔`cy` swap. `Intrinsics.rotated()` handles this; it becomes relevant as soon
as 3D computations with a camera model are added.

### Which way round — the tool finds out by itself

Whether 90° or 270° is correct depends on which way you turn the camera. You do
not have to know: `tools/calibrate_rotation.py` captures eight frames, tries all
four orientations and scores each by

```
score = detection rate × confidence × uprightness
```

"Uprightness" measures whether the head is above the ankles — that reliably
separates 90° from 270° and 0° from 180°, which confidence alone cannot do. The
result is stored in `configs/cameras/brio100.yaml`.

### Mirroring: deliberately for display only

A mirrored image feels more natural but **swaps left and right**. If inference
ran on the mirrored image, `left_wrist` would in truth be your right hand — and
the robot would raise the wrong arm.

Therefore: **inference always on the unmirrored image**, mirroring only for the
display, with the keypoints mirrored along (`mirror_for_display()`). Side
assignment then stays correct across the whole chain.

---

## 5. Start guide for the camera run

Mount the camera **portrait** (either direction), about 2 m away, full body in
frame, even lighting.

```bat
cd a3

rem 1) check the environment (also repairs the onnxruntime trap)
python tools\setup_env.py

rem 2) determine the rotation automatically and write it into the profile
rem    Stand fully in frame, upright, arms slightly out.
C:\venvs\a3-pose\Scripts\python.exe tools\calibrate_rotation.py --preview

rem 3) live view with skeleton
C:\venvs\a3-pose\Scripts\python.exe tools\live_pose.py
```

Keys in the live window: `q` quit · `m` toggle mirroring ·
`r` advance rotation by 90° · `s` snapshot and save profile.

If step 2 finds no person: check the lighting, stand further away, and step
through manually with `live_pose.py --no-mirror` and `r` — the HUD shows
detection and confidence live.

**What to look for:**

| | expected |
|---|---|
| Frame size in the HUD | 1080×1920 (not 1920×1080) |
| FPS | ~30 (limited by the camera) |
| Inference time | 30–35 ms in `balanced` |
| Confidence | > 0.7 with good lighting |
| Foot points | heels and toes visibly drawn |

The Brio 100 has a **fixed focus** and 30 fps. In low light it lengthens the
exposure, which produces motion blur on fast movements — the limiting factor for
fast gestures. Bright, even lighting is therefore the most effective lever on
quality.

---

## 6. Open

- Camera run with the Brio 100 (§5) — confirm rotation, measure FPS and latency
- Camera calibration (checkerboard) for metric intrinsics
- One-Euro filter on the keypoint time series
- Recording and replay, so M3/M4 stay testable without a camera
- Task-space references (`references.py`) from §3.3 of the plan

---

## 7. Addendum: 3D lifter (2026-09-04)

The plan foresaw a 3D lifter after the 2D layer. I had initially skipped this
stage because `Wholebody3d` — a direct 3D estimator — was rejected in §4. That
was a false inference: what had been rejected was *one model*, not the
*approach*. A lifter works differently: it lifts a **sequence** of 2D keypoints
into 3D and exploits temporal context.

### Two technical obstacles

**DirectML cannot execute MotionBERT.** The temporal attention builds operands
of shape `(B, H, N, T, C)` — five-dimensional. DirectML has no MatMul kernel for
that; the run aborts with a `FusedMatMulActivation` error, independent of the
optimisation level. Solved by folding the leading axes into `(B·H·N, T, C)` —
mathematically identical but three-dimensional. The patch sits in
`tools/build_lifter.py` and is applied automatically during the build.

| | Window 27 | 81 | 243 |
|---|---|---|---|
| **DirectML** | **23 ms** | 108 ms | 135 ms |
| CPU (lite) | 50 ms | 163 ms | 625 ms |
| CPU (full) | 119 ms | 376 ms | 1301 ms |

On the GPU `full` is as fast as `lite` (24 against 23 ms) — at these model sizes
the call overhead dominates.

**The input normalisation was wrong.** I normalised to the image; MotionBERT
expects `crop_scale`, that is the person bounding box mapped to [−1, 1]. Before
the correction all measurements were worthless.

### What the lifter delivers

Measured as the spread of limb length (`std/mean`) — with real 3D it must be
clearly below the 2D value, because a bone is rigid:

| Segment | 2D | 3D | Gain |
|---|---|---|---|
| forearm L | 0.324 | 0.220 | **32 %** |
| thigh L | 0.067 | 0.049 | **27 %** |
| thigh R | 0.075 | 0.058 | 23 % |
| forearm R | 0.342 | 0.277 | 19 % |
| upper arm L | 0.166 | 0.143 | 14 % |
| shank L | 0.127 | 0.132 | **−4 %** |
| shank R | 0.129 | 0.129 | 0 % |

Mean: **legs +11 %, arms +19 %.** That is honestly modest — a good lifter should
push the spread below 5 %, and we stay at 13–14 %. A benchmark with window 81
showed 36–40 % for the thighs, but 108 ms of inference is too expensive for real
time.

**Adopted nonetheless — for the sign.** The segment foreshortening from §4
yields `cos θ = L/L_ref` and therefore only the *magnitude* of the inclination,
never the direction. Whether a leg is in front of or behind the body plane is
fundamentally undecidable that way. The lifter provides real coordinates with a
sign. For M5 that is the decisive quantity — and precisely the gap on which
A1/A2 failed when walking.

The axes were determined empirically (`tools/check_lifter.py`): **x** correlates
0.96 with the image x axis, **y** 0.98 with the image y axis, **z** is the
remaining depth.

### Integration

The lifter runs with **window 27, causal** (no lookahead, therefore no
additional latency) and on **every second frame** — leg poses change more slowly
than arm gestures. Effective cost: 11.5 ms per frame instead of 23.

| | |
|---|---|
| 2D chain | 26 ms |
| Lifter (half rate) | 11.5 ms |
| **total** | **37.5 ms → 27 Hz** |

The arms stay on segment foreshortening, which is equivalent there. The 3D
skeleton supplies the **lower-body references** (`perception/lower_body.py`), all
normalised to the subject's leg length:

| Field | Range in the test recording |
|---|---|
| `hip_height` (squat) | 0.00 … 0.16 |
| `stance_width` | 0.12 … 0.68 |
| `left_foot_lift` | 0.00 … 0.56 |
| `com_offset_x/y` | −0.37 … +0.29 |

98.4 % of frames deliver valid lower-body references. Packet size 599 bytes.

The model is not in the repository (62 MB). Build it once with:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
C:\venvs\a3-pose\Scripts\python.exe tools\build_lifter.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_lifter.py
```

**Open:** `com_offset_x` is consistently negative (median −0.26). That is an
anatomical offset between hip and ankles, not an error — for control the rest
value must be calibrated and subtracted. That belongs in M4.
