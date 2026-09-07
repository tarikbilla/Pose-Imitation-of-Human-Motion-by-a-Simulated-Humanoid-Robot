import json
import os
import sys

from controller import Supervisor

APPLIED_TORQUE = 0.1
SETTLE_STEPS = 5
MEASURE_SECONDS = 0.5

REAL_IYY = 0.0035
PARASITIC_IYY = 1.0

PROBES = ("pattern", "reference")


def fit_angular_acceleration(times, angles):
    numerator = sum(a * t * t for t, a in zip(times, angles))
    denominator = sum(t ** 4 for t in times)
    if denominator == 0.0:
        return 0.0
    return 2.0 * numerator / denominator


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    motors = {}
    sensors = {}
    for name in PROBES:
        motor = robot.getDevice(name)
        sensor = robot.getDevice(name + "S")
        if motor is None or sensor is None:
            print(f"FATAL: device '{name}' not found", file=sys.stderr)
            return 1
        sensor.enable(timestep)
        motors[name] = motor
        sensors[name] = sensor

    for _ in range(SETTLE_STEPS):
        if robot.step(timestep) == -1:
            return 1

    origin = {name: sensors[name].getValue() for name in PROBES}
    for name in PROBES:
        motors[name].setTorque(APPLIED_TORQUE)

    times = []
    samples = {name: [] for name in PROBES}
    elapsed = 0.0
    while elapsed < MEASURE_SECONDS:
        if robot.step(timestep) == -1:
            return 1
        elapsed += dt
        times.append(elapsed)
        for name in PROBES:
            samples[name].append(sensors[name].getValue() - origin[name])

    for name in PROBES:
        motors[name].setTorque(0.0)

    result = {
        "applied_torque_nm": APPLIED_TORQUE,
        "measure_seconds": MEASURE_SECONDS,
        "timestep_ms": timestep,
        "expected_real_iyy": REAL_IYY,
        "expected_if_parasitic_added": REAL_IYY + PARASITIC_IYY,
        "probes": {},
    }

    for name in PROBES:
        alpha = fit_angular_acceleration(times, samples[name])
        inertia = APPLIED_TORQUE / alpha if alpha != 0.0 else float("inf")
        result["probes"][name] = {
            "angular_acceleration_rad_s2": alpha,
            "effective_inertia_kg_m2": inertia,
            "final_angle_rad": samples[name][-1],
        }

    pattern = result["probes"]["pattern"]["effective_inertia_kg_m2"]
    reference = result["probes"]["reference"]["effective_inertia_kg_m2"]
    ratio = pattern / reference if reference else float("inf")
    result["pattern_over_reference"] = ratio
    result["parasitic_inertia_confirmed"] = ratio > 10.0

    print("=" * 62)
    print("M0 / E1  Atlas DEFAULT_PHYSICS nesting pattern vs reference body")
    print("=" * 62)
    print(f"applied torque            {APPLIED_TORQUE:.4f} Nm")
    print(f"reference effective Iyy   {reference:.6f} kg m^2   (expected {REAL_IYY})")
    print(f"pattern   effective Iyy   {pattern:.6f} kg m^2")
    print(f"ratio pattern/reference   {ratio:.1f}x")
    print("-" * 62)
    if result["parasitic_inertia_confirmed"]:
        print("VERDICT: parasitic inertia IS added. The outer Physics node")
        print("         contributes its inertiaMatrix to the merged body.")
        print("         -> AtlasA3.proto MUST neutralise DEFAULT_PHYSICS.")
    else:
        print("VERDICT: parasitic inertia is NOT added. Webots discards the")
        print("         outer Physics node (no boundingObject).")
        print("         -> AtlasA3.proto needs sensors only, no inertia fix.")
    print("=" * 62)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m0_e1_inertia.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"written: {out_path}")
    sys.stdout.flush()
    robot.simulationQuit(0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
