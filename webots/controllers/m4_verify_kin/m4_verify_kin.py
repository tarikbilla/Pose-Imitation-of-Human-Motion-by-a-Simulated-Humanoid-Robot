import json
import os
import random
import sys

import numpy as np
from controller import Node, Supervisor

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "a3_wholebody")))

from atlaskin import AtlasModel

SETTLE_STEPS = 90
TRIALS = 12
AMPLITUDE = 0.35

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    model = AtlasModel()

    motors = {}
    sensors = {}
    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        kind = device.getNodeType()
        if kind == Node.ROTATIONAL_MOTOR:
            device.setVelocity(2.0)
            motors[device.getName()] = device
        elif kind == Node.POSITION_SENSOR:
            device.enable(timestep)
            sensors[device.getName()] = device

    self_node = robot.getSelf()
    print(f"model mass {model.total_mass:.3f} kg over {len(model.names)} joints")
    sys.stdout.flush()

    random.seed(3)
    rows = []

    for trial in range(TRIALS):
        targets = {}
        for name in model.names:
            if trial == 0:
                targets[name] = 0.0
            else:
                low, high = model.limits[name]
                span = min(high - low, 2 * AMPLITUDE)
                centre = max(low, min(high, 0.0))
                targets[name] = model.clamp(
                    name, centre + random.uniform(-span / 2, span / 2))
            if name in motors:
                motors[name].setPosition(targets[name])

        for _ in range(SETTLE_STEPS):
            if robot.step(timestep) == -1:
                return 1

        measured = {n: sensors[n + "S"].getValue() for n in model.names
                    if n + "S" in sensors}

        poses = model.frames(measured)
        com_local = model.com(measured, poses)

        rotation = np.asarray(self_node.getOrientation()).reshape(3, 3)
        root = np.asarray(self_node.getPosition())
        com_predicted = root + rotation @ com_local
        com_actual = np.asarray(self_node.getCenterOfMass())

        error = float(np.linalg.norm(com_predicted - com_actual))
        rows.append({
            "trial": trial,
            "predicted": [round(v, 5) for v in com_predicted],
            "actual": [round(v, 5) for v in com_actual],
            "error_m": round(error, 5),
        })
        print(f"trial {trial:2d}  error {error * 1000:7.2f} mm   "
              f"pred {com_predicted.round(4)}  actual {com_actual.round(4)}")
        sys.stdout.flush()

    errors = [r["error_m"] for r in rows]
    print("-" * 60)
    print(f"CoM error  mean {np.mean(errors)*1000:.2f} mm   max {max(errors)*1000:.2f} mm")

    probe = [n for n in ("LLegKny", "LLegLhy", "LArmShx", "BackMby") if n in model.names]
    analytic = model.com_jacobian(measured, joints=probe)
    numeric = np.zeros_like(analytic)
    delta = 1e-4
    for column, name in enumerate(probe):
        plus = dict(measured); plus[name] += delta
        minus = dict(measured); minus[name] -= delta
        numeric[:, column] = (model.com(plus) - model.com(minus)) / (2 * delta)
    jac_error = float(np.max(np.abs(analytic - numeric)))
    print(f"Jacobian max deviation (analytic vs numeric): {jac_error:.2e}")
    for column, name in enumerate(probe):
        print(f"  {name:<10} analytic {analytic[:, column].round(5)}  "
              f"numeric {numeric[:, column].round(5)}")

    summary = {
        "total_mass": model.total_mass,
        "com_error_mean_mm": float(np.mean(errors) * 1000),
        "com_error_max_mm": float(max(errors) * 1000),
        "jacobian_max_deviation": jac_error,
        "trials": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m4_kinematics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sys.stdout.flush()
    robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
