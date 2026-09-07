import json
import os
import sys

from controller import Supervisor

APPLIED_TORQUE = 0.05
MEASURE_SECONDS = 0.2
SETTLE_SECONDS = 0.4

JOINTS = ("LLegLax", "LLegUay", "LLegKny", "LArmElx")

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)


def log(message):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "_m1_joint_inertia.log"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


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
    model = robot.getName()
    log(f"start model={model} timestep={timestep}")

    motors = {}
    sensors = {}
    for joint in JOINTS:
        motor = robot.getDevice(joint)
        sensor = robot.getDevice(joint + "S")
        if motor is None or sensor is None:
            log(f"missing device for {joint}")
            continue
        sensor.enable(timestep)
        motors[joint] = motor
        sensors[joint] = sensor
    log(f"devices resolved: {sorted(motors)}")

    if robot.step(timestep) == -1:
        return 1
    log("first step done")

    measurements = {}
    for joint in sorted(motors):
        motor = motors[joint]
        sensor = sensors[joint]

        motor.setPosition(0.0)
        for _ in range(int(SETTLE_SECONDS / dt)):
            if robot.step(timestep) == -1:
                return 1
        log(f"{joint}: settled at {sensor.getValue():.6f}")

        origin = sensor.getValue()
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motor.setTorque(APPLIED_TORQUE)
        log(f"{joint}: torque applied")

        times = []
        angles = []
        elapsed = 0.0
        while elapsed < MEASURE_SECONDS:
            if robot.step(timestep) == -1:
                return 1
            elapsed += dt
            times.append(elapsed)
            angles.append(sensor.getValue() - origin)

        alpha = fit_angular_acceleration(times, angles)
        inertia = APPLIED_TORQUE / alpha if alpha else float("inf")
        measurements[joint] = {
            "angular_acceleration_rad_s2": alpha,
            "effective_inertia_kg_m2": inertia,
            "final_angle_rad": angles[-1] if angles else 0.0,
        }
        log(f"{joint}: alpha={alpha:.5f} I={inertia:.6f}")

        motor.setTorque(0.0)
        motor.setPosition(0.0)

    result = {
        "model": model,
        "applied_torque_nm": APPLIED_TORQUE,
        "measure_seconds": MEASURE_SECONDS,
        "joints": measurements,
    }

    print("=" * 62)
    print(f"M1  joint inertia  [{model}]")
    print("=" * 62)
    for joint, data in measurements.items():
        print(
            f"{joint:<12}{data['angular_acceleration_rad_s2']:>16.4f}"
            f"{data['effective_inertia_kg_m2']:>16.6f}"
        )
    print("=" * 62)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"m1_joint_inertia_{model}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"written {out_path}")

    sys.stdout.flush()
    robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
