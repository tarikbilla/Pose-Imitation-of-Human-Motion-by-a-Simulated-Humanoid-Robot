# M0 — Measurement Results

Date 2026-09-03 · Branch `A3` · Raw data in `a3/results/*.json`

All three M0 questions are answered. Two findings change the implementation, one
confirms it.

---

## E1 — Inertia finding: **confirmed, factor 287**

`a3/webots/worlds/m0_inertia_probe.wbt` places two hinge joints side by side,
both carrying the same real foot mass and inertia as payload:

- **pattern** — reproduces the Atlas nesting pattern exactly: an outer solid
  with `mass 0.001` and `inertiaMatrix [1 1 1 0 0 0]` and no `boundingObject`,
  containing a child solid with the real foot values.
- **reference** — only the child solid, without an outer physics node.

Both are accelerated at 0.1 Nm under zero gravity (`gravity 0`); angular
acceleration follows from a least-squares fit to θ(t) = ½·α·t².

| Probe | α [rad/s²] | I_eff [kg·m²] | Expected |
|---|---|---|---|
| reference | 29.122 | **0.003434** | 0.0035 (deviation 1.9 %) |
| pattern | 0.1016 | **0.984126** | 0.0035 if correct |
| | | **ratio 286.6×** | |

The reference measurement hits the known expected value to within 2 %, which
validates the method. The Atlas pattern sits **286.6× above it**.

**Webots adds the inertia of the outer physics node during implicit solid
merging.** The suspicion from the plan (~250× for the foot) was correct and was
in fact slightly too low.

→ `AtlasA3.proto` **must** neutralise `DEFAULT_PHYSICS`. Without it, every
balance controller is built on a model whose feet are 287× too sluggish.

---

## E2 — Atlas inventory: **audit confirmed**

`a3/webots/worlds/m0_atlas_probe.wbt`, supervisor controller, 4 s observation.

**Device inventory — exactly as read from the PROTO:**

```
ROTATIONAL_MOTOR         28
POSITION_SENSOR           0
INERTIAL_UNIT             0
TOUCH_SENSOR              0
GYRO                      0
ACCELEROMETER             0
```

28 devices in total, motors only. On the stock model the controller is
completely blind.

**Mass distribution — merging of the segment masses works.** The supervisor
reports the centre of mass 0.073 m above the pelvis origin. If only the chain of
28 × 1 g links existed, the CoM would be an unweighted mean of the joint
positions and would sit considerably lower. The masses from the sub-PROTOs (sum
89.00 kg) are therefore effective — unlike the inertias, the mass distribution
is correct.

**Rest position.** The robot settles by 7.7 cm during the first 0.26 s and then
stands motionless on 8 contact points (4 box corners per foot).
`getStaticBalance()` stays `true` throughout. The stock Atlas does *not* fall
over on its own from standing — the extended legs carry it through position
control.

**CoM height when standing: 0.995 m** above the floor. The tipping time constant
from the plan is now measured rather than estimated:

```
ω = sqrt(9.81 / 0.995) = 3.140 rad/s
τ_tip = 0.318 s
```

The planning assumption of 0.31 s was exact. The argument against a
camera-closed control loop (§2 of the plan) now rests on measured ground.

---

## E3 — GPU path: **confirmed, 7.9× over CPU**

`a3/tools/check_directml.py` builds eight chained 1024×1024 matrix
multiplications as an ONNX model and measures the median of 20 runs after 3
warm-up passes.

| Execution provider | Median | Throughput |
|---|---|---|
| **DmlExecutionProvider** | **4.87 ms** | **3527 GFLOP/s** |
| CPUExecutionProvider | 38.53 ms | 446 GFLOP/s |

**Speedup 7.9×.** ONNX Runtime 1.24.4 reports both providers; DirectML performs
real GPU work on the RX 6800 XT. The GPU path described in §4.1 of the plan is
viable, and the CPU stays free for Webots' physics.

---

## Side findings with consequences for the setup

**1. Windows path limit.** Installing `onnx` into a venv *inside* the repository
fails with `WinError 206` — the repository name is long, the 260-character limit
is reached, and `LongPathsEnabled` is `0`.
→ The venv now lives at `C:\venvs\a3-pose`, outside the repository.
`tools/setup_env.py` enforces and checks this.

**2. `runtime.ini` needs absolute paths.** With a relative interpreter path
Webots crashes on load with no error message (SIGSEGV, exit 139). With an
absolute path it runs cleanly (exit 0).
→ `tools/setup_env.py` generates the `runtime.ini` files; they are in
`.gitignore` because they are machine-specific.

**3. Webots does not forward controller `stdout` to the calling console on
Windows**, not even with `--stdout --stderr`.
→ All measurement controllers write their results as JSON to `a3/results/`.
That is the better basis for the evaluation in M6 anyway.

---

## Reproduction

```bat
python tools\setup_env.py
C:\venvs\a3-pose\Scripts\python.exe tools\check_directml.py

set WEBOTS_HOME=C:\Users\<user>\AppData\Local\Programs\Webots
"%WEBOTS_HOME%\msys64\mingw64\bin\webots.exe" --batch --mode=fast --minimize webots\worlds\m0_inertia_probe.wbt
"%WEBOTS_HOME%\msys64\mingw64\bin\webots.exe" --batch --mode=fast --minimize webots\worlds\m0_atlas_probe.wbt
```

---

## Clearance for M1

| Question | Answer | Consequence |
|---|---|---|
| Inertia falsified? | **yes, 287×** | the physics fix in `AtlasA3.proto` is mandatory |
| Sensing present? | **no, 0** | retrofit 28 `PositionSensor`s |
| Supervisor usable? | **yes** | CoM, contact points, balance test are usable |
| DirectML usable? | **yes, 7.9×** | GPU perception as planned |

M0 is complete.
