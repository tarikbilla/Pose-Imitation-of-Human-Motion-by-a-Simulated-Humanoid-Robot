# Porting to the Laboratory Rig

Everything measured in [`FINDINGS.md`](FINDINGS.md) was taken on the home-office
rig. This document lists what must change for the on-site laboratory setup, why
each item matters, and how to verify it.

The repository supports both rigs side by side through **site profiles**, so
neither setup overwrites the other.

---

## 1. The two rigs

| | Home office (`home`) | Laboratory (`lab`) |
|---|---|---|
| Camera | Logitech Brio 100, USB webcam | **Sony A7 III** on a tripod, via HDMI-to-USB (Elgato Cam Link 4K) as a UVC device |
| Orientation | portrait, `rotation: 90` | landscape, `rotation: 0` |
| Resolution | 1920×1080 sensor | 1920×1080 |
| Frame rate | 30 fps | 25–100 fps, adaptive per the PRD |
| Field of view | 58° diagonal, known | **depends on the lens — must be determined** |
| GPU | AMD Radeon RX 6800 XT | **NVIDIA, CUDA** |
| Inference provider | `DmlExecutionProvider` | `CUDAExecutionProvider` |
| ONNX Runtime package | `onnxruntime-directml` | `onnxruntime-gpu` |
| Capture backend | `dshow` (Windows) | `v4l2` (Linux) or `avfoundation` (macOS) |
| OS | Windows 11 | macOS or Linux per the PRD |
| Webots | R2025a | R2023b or newer per the PRD |

Source for the laboratory column: [`../../docs/PRD.md`](../../docs/PRD.md)
§5.1 (capture hardware), §7 NFR-7 (portability), §8 (tooling), §11 (the CUDA
requirement).

---

## 2. Select the site

```bash
python tools/run_live.py --site lab
# or
export A3_SITE=lab
```

`configs/sites/lab.yaml` bundles camera profile, inference device, capture
backend and subject height. Nothing else needs to be edited to switch rigs.

```yaml
name: lab
camera: sony_a7iii
device: cuda
device_fallback: cpu
capture_backend: auto      # resolves to v4l2 on Linux, avfoundation on macOS
capture_device_hint: Cam Link
pose_mode: lightweight
body_height_m: 1.75
```

`device_fallback: cpu` means the pipeline still runs if CUDA is missing, but at
roughly one eighth of the speed and with a loud warning. Do not leave it there.

---

## 3. Camera: what must be measured on site

### 3.1 Field of view is unknown and matters

The Brio has a fixed 58° diagonal field of view, so its focal length follows
from the image geometry. **The A7 III is an interchangeable-lens camera: its
field of view depends entirely on the mounted lens**, and the focal length in
pixels is what converts apparent body size into metres.

`configs/cameras/sony_a7iii.yaml` therefore ships with `diagonal_fov_deg: 0.0`,
which makes the pipeline fall back to a generic estimate and print:

```
WARNING: no camera calibration for this profile; absolute travel distance
will be wrong. See docs/PORTING_LAB.md.
```

Getting this wrong scales all travel distances proportionally. On the home rig
the difference between the generic estimate (1382 px) and the true value
(1987 px) is **44 %**.

The pipeline resolves focal length in this order, first match wins:

1. `intrinsics.fx` when `intrinsics.calibrated: true` — best
2. `focal_px` — a measured value in pixels
3. `diagonal_fov_deg` — derived from geometry
4. generic estimate — flagged as `UNCALIBRATED`

### 3.2 Fastest acceptable method: one tape measure

Place a person of known height at a measured distance `d` from the sensor plane,
facing the camera, standing straight. Read the nose-to-ankle pixel distance from
the monitoring window, then:

```
focal_px = d [m] * nose_ankle_px / (0.897 * body_height [m])
```

0.897 is the nose-to-ankle fraction of stature. Write the result into
`configs/cameras/sony_a7iii.yaml` as `focal_px`. Repeat at two distances; the
two results should agree within a few percent.

### 3.3 Better: a proper calibration

A checkerboard calibration with OpenCV gives `fx, fy, cx, cy`. Put them into the
`intrinsics` block and set `calibrated: true`. This also removes lens distortion
error, which the focal-length shortcut ignores.

**If the lens is a zoom, lock it** and mark the setting. Any change invalidates
the calibration silently — the pipeline cannot detect it.

### 3.4 Orientation

The home rig mounts the camera portrait because it nearly doubles the usable
vertical angle (30.4° → 51.6°) and puts 1.8× more pixels on the body. A
tripod-mounted A7 III in the lab is presumably landscape, so `rotation: 0`.

If the subject does not fill the frame vertically, consider rotating the camera
90° and setting `rotation: 90`. Everything downstream works in the rotated
image, so this is a one-line change — but re-derive the focal length, because
`focal_for` uses the rotated frame size.

### 3.5 Verify the capture chain

```bash
python tools/live_puppet.py --site lab --headless --seconds 30
```

Check the printed source description:

```
source    {'kind': 'ffmpeg-camera', 'device': ..., 'backend': 'v4l2',
           'sensor': '1920x1080', 'drain': True, ...}
frame     1920x1080   focal 1987 px (calibrated intrinsics)
```

- `kind` must be `ffmpeg-camera`. If it says `camera`, the ffmpeg path failed
  and OpenCV took over; the reason is printed as a warning. Fix it rather than
  accepting the fallback — the OpenCV path has no draining reader.
- `drain: True` must be present. Without it the capture dies after roughly 200
  frames (see §6).
- The focal-length origin must not say `UNCALIBRATED`.

If the device is not found, list what ffmpeg sees:

```bash
ffmpeg -f v4l2 -list_formats all -i /dev/video0        # Linux
ffmpeg -f avfoundation -list_devices true -i ""        # macOS
```

Then set `capture_device_hint` in `configs/sites/lab.yaml` to a substring of the
device name, or `source` in the camera profile to the exact device path.

### 3.6 Capture chain latency

The PRD flags the HDMI-to-USB chain as a latency risk. Measure it: point the
camera at a screen showing a millisecond timer and photograph screen and
monitoring window together. The pipeline itself costs 34 ms; anything the
capture device adds is on top of that and shows up as the robot lagging the
person.

The draining reader keeps this from accumulating — it always processes the
newest frame — but it cannot remove latency the capture device has already
introduced.

---

## 4. GPU: AMD/DirectML to NVIDIA/CUDA

### 4.1 Packages

`requirements.txt` pins `onnxruntime-directml`. For the lab, install
`onnxruntime-gpu` built against the installed CUDA and cuDNN **instead** — never
both, they conflict over the same module name.

```bash
pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
pip install onnxruntime-gpu
```

> **The same trap as on the home rig applies here, and it is the single most
> likely way to lose a day.** Installing `rtmlib` normally pulls in plain
> `onnxruntime`, which shadows `onnxruntime-gpu` and silently removes
> `CUDAExecutionProvider`. The pipeline then runs on CPU with no error, only an
> 8× slowdown. Install rtmlib with `--no-deps`, exactly as `tools/setup_env.py`
> already does, and verify afterwards.

Match the ONNX Runtime build to the CUDA version actually installed. A mismatch
usually surfaces as `CUDAExecutionProvider` being absent from
`ort.get_available_providers()` with no further explanation.

### 4.2 Code changes: none required

`perception/runtime.py` already knows CUDA:

```python
PROVIDER_BY_DEVICE = {"dml": "DmlExecutionProvider",
                      "cuda": "CUDAExecutionProvider",
                      "cpu": "CPUExecutionProvider"}
```

`patch_rtmlib()` registers both device names in rtmlib's backend table, and
`resolve_device()` falls back with an explanatory message when the requested
provider is missing. Setting `device: cuda` in the site profile is enough.

### 4.3 Verify

```bash
python tools/check_directml.py
```

Despite the name it reports whatever providers ONNX Runtime offers. Expect
`CUDAExecutionProvider` in the list and a speedup well above 2× against CPU.
The home rig reaches 7.9× with DirectML; a modern NVIDIA card should exceed it.

Then the end-to-end perception benchmark:

```bash
python tools/check_perception.py --cpu-compare
```

Budget: the whole chain must stay under about 34 ms to hold 30 fps
(2D pose 25.6 ms + retargeting 8.1 ms on the home rig). If CUDA is faster,
consider raising `pose_mode` from `lightweight` to `balanced` in
`configs/sites/lab.yaml` — better accuracy for roughly 14 ms more.

### 4.4 The 3D lifter export

`tools/build_lifter.py` folds the 5-D attention operands into 4-D because
**DirectML has no 5-D MatMul**. CUDA does not need this, but the folded graph is
mathematically identical, so the existing export works unchanged. There is no
reason to re-export unless you want the marginal speed of the original layout.

If you do re-export on the lab machine, re-run `tools/check_lifter.py`: it
verifies the axis convention (x = image right, y = image down, z = depth away
from the camera), which the whole retargeting depends on.

---

## 5. Operating system

The home rig is Windows; the PRD specifies macOS or Linux for the lab.

| Concern | Status |
|---|---|
| Capture backend | handled: `perception/ffmpeg_source.py` builds `dshow`, `v4l2` and `avfoundation` command lines and picks by platform |
| Path length limit | Windows-only; the venv may live anywhere on Linux/macOS |
| `runtime.ini` absolute paths | still required, `tools/setup_env.py` generates them |
| Controller stdout | Webots suppresses it on Windows; on Linux it usually appears. The JSON files in `results/` remain the reliable channel either way |
| Webots version | R2025a here, R2023b+ per the PRD. The PROTOs declare `#VRML_SIM R2025a utf8`; check they load, and regenerate with `tools/build_atlas_proto.py` if the header must change |
| `taskkill` in the launchers | Windows-specific; `tools/run_live.py` and `tools/run_puppet.py` need a `kill`/`pkill` branch on Linux and macOS |

The last item is the one concrete code change the port needs.

---

## 6. Things that will bite, in order of likelihood

1. **`onnxruntime` shadowing `onnxruntime-gpu`.** Silent 8× slowdown. Verify the
   provider list, never assume.
2. **Uncalibrated focal length.** Travel distances scale wrong. The pipeline
   warns; do not ignore it.
3. **ffmpeg falling back to OpenCV.** Without the draining reader the capture
   dies after ~200 frames with `real-time buffer too full`. Check `drain: True`
   and `kind: ffmpeg-camera`.
4. **Zoom lens moved after calibration.** Undetectable. Lock and mark it.
5. **Distance datum taken with nobody in frame.** Puts the origin metres away
   and raises the airborne fraction to over 20 %. Guarded now — the datum only
   advances on a valid pose — but press `n` once standing in position.
6. **`taskkill` on a non-Windows host.** The launcher cannot stop Webots.
7. **Adaptive frame rate.** The PRD asks for 25–100 fps. The retargeter is
   timestamp-driven and handles a varying rate, but `A3_COMMAND_RATE` (rad/s)
   and the plant/lift thresholds were tuned at 30 fps and a 16 ms simulation
   step. Re-check `results/puppet_run.json` for `real_airborne_percent` and
   `deep_count` after changing the rate.

---

## 7. Acceptance checklist for the lab rig

Run in order; each line has a command and an expected value.

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | GPU provider present | `check_directml.py` | `CUDAExecutionProvider` listed, speedup > 2× |
| 2 | Perception budget | `check_perception.py --cpu-compare` | end to end < 34 ms |
| 3 | Lifter axes | `check_lifter.py` | axis convention confirmed |
| 4 | PROTO integrity | `check_proto.py` | mass sum 89.000 kg, 28 sensors |
| 5 | Leg IK | `check_legik.py` | p99 < 0.5 mm |
| 6 | Camera opens through ffmpeg | `live_puppet.py --site lab --headless --seconds 30` | `kind: ffmpeg-camera`, `drain: True` |
| 7 | Focal calibrated | same output line | not `UNCALIBRATED` |
| 8 | Sustained capture | same run, 30 s | no `source exhausted`, ≥ 20 fps |
| 9 | Full live session | `run_live.py --site lab` | `live_packets` > 0, `real_airborne_percent` < 2 %, `deep_count` < 10 |
| 10 | Travel plausible | walk 1 m toward the camera | `travel_forward` within ~30 % of 1 m |

Check 10 is the end-to-end test of the calibration: if the reported travel is
systematically off by a constant factor, the focal length or `body_height_m` is
wrong, and both are single values in the configuration.

---

## 8. Files touched by a port

| File | What changes |
|---|---|
| `configs/sites/lab.yaml` | device, backend, device hint, subject height |
| `configs/cameras/sony_a7iii.yaml` | resolution, fps, rotation, **focal_px or intrinsics** |
| `requirements.txt` | `onnxruntime-directml` → `onnxruntime-gpu` |
| `tools/run_live.py`, `tools/run_puppet.py` | `taskkill` → platform-appropriate process termination |
| `tools/setup_env.py` | venv location and interpreter path on non-Windows |

No changes are required in `perception/runtime.py`, `perception/ffmpeg_source.py`
or any controller: the device, backend and focal-length paths are already
parameterised.
