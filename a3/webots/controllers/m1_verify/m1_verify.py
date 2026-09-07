import json
import os
import sys

from controller import Node, Supervisor

OBSERVE_SECONDS = 6.0
SAMPLE_EVERY = 8

EXPECTED_MOTORS = 28
EXPECTED_SENSORS = 28

TYPE_NAMES = {
    value: name
    for name, value in vars(Node).items()
    if name.isupper() and isinstance(value, int)
}


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    inventory = {}
    motors = {}
    sensors = {}
    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        type_name = TYPE_NAMES.get(device.getNodeType(), str(device.getNodeType()))
        inventory[type_name] = inventory.get(type_name, 0) + 1
        name = device.getName()
        if type_name == "ROTATIONAL_MOTOR":
            motors[name] = device
        elif type_name == "POSITION_SENSOR":
            sensors[name] = device
            device.enable(timestep)

    unmatched = [name for name in motors if name + "S" not in sensors]

    self_node = robot.getSelf()
    self_node.enableContactPointsTracking(timestep, True)

    trace = []
    steps = int(OBSERVE_SECONDS / dt)
    for step in range(steps):
        if robot.step(timestep) == -1:
            break
        if step % SAMPLE_EVERY:
            continue
        trace.append(
            {
                "t": round(step * dt, 4),
                "com": [round(c, 5) for c in self_node.getCenterOfMass()],
                "root": [round(p, 5) for p in self_node.getPosition()],
                "contact_count": len(self_node.getContactPoints(True)),
                "static_balance": bool(self_node.getStaticBalance()),
            }
        )

    joint_angles = {name: sensors[name].getValue() for name in sorted(sensors)}
    max_drift = max((abs(v) for v in joint_angles.values()), default=0.0)

    settled = trace[len(trace) // 2 :]
    com_z = [s["com"][2] for s in settled]
    com_xy_span = max(
        max(s["com"][i] for s in settled) - min(s["com"][i] for s in settled)
        for i in (0, 1)
    )

    result = {
        "device_inventory": inventory,
        "motors": len(motors),
        "position_sensors": len(sensors),
        "unmatched_motors": unmatched,
        "joint_angles_rad": joint_angles,
        "max_joint_drift_rad": max_drift,
        "com_z_settled_m": sum(com_z) / len(com_z) if com_z else None,
        "com_horizontal_span_m": com_xy_span,
        "final_contact_count": trace[-1]["contact_count"] if trace else 0,
        "static_balance_always": all(s["static_balance"] for s in trace),
        "trace": trace,
    }

    ok = (
        inventory.get("ROTATIONAL_MOTOR") == EXPECTED_MOTORS
        and inventory.get("POSITION_SENSOR") == EXPECTED_SENSORS
        and not unmatched
    )
    result["inventory_ok"] = ok

    print("=" * 62)
    print("M1  AtlasA3 verification")
    print("=" * 62)
    for type_name in sorted(inventory):
        print(f"    {type_name:<24} {inventory[type_name]}")
    print(f"motor/sensor pairing      {'OK' if not unmatched else unmatched}")
    print("-" * 62)
    if trace:
        print(f"CoM height settled        {result['com_z_settled_m']:.4f} m")
        print(f"CoM horizontal span       {com_xy_span * 1000:.1f} mm")
        print(f"contacts at end           {result['final_contact_count']}")
        print(f"static balance always     {result['static_balance_always']}")
        print(f"max joint drift           {max_drift:.4f} rad")
    print("=" * 62)
    print("VERDICT:", "PASS" if ok else "FAIL")
    print("=" * 62)

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "m1_verify.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    sys.stdout.flush()
    robot.simulationQuit(0 if ok else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
