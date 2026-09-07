import json
import os
import sys

from controller import Node, Supervisor

OBSERVE_SECONDS = 4.0
SAMPLE_EVERY = 8

TYPE_NAMES = {
    value: name
    for name, value in vars(Node).items()
    if name.isupper() and isinstance(value, int)
}


def device_inventory(robot):
    inventory = {}
    names = []
    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        type_name = TYPE_NAMES.get(device.getNodeType(), str(device.getNodeType()))
        inventory[type_name] = inventory.get(type_name, 0) + 1
        names.append({"name": device.getName(), "type": type_name})
    return inventory, names


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    inventory, names = device_inventory(robot)

    self_node = robot.getSelf()
    if self_node is None:
        print("FATAL: supervisor could not resolve its own node", file=sys.stderr)
        return 1

    self_node.enableContactPointsTracking(timestep, True)

    trace = []
    steps = int(OBSERVE_SECONDS / dt)
    for step in range(steps):
        if robot.step(timestep) == -1:
            break
        if step % SAMPLE_EVERY:
            continue
        com = self_node.getCenterOfMass()
        position = self_node.getPosition()
        contacts = self_node.getContactPoints(True)
        trace.append(
            {
                "t": round(step * dt, 4),
                "com": [round(c, 5) for c in com],
                "root": [round(p, 5) for p in position],
                "contact_count": len(contacts),
                "static_balance": bool(self_node.getStaticBalance()),
            }
        )

    first = trace[0] if trace else None
    last = trace[-1] if trace else None

    result = {
        "device_inventory": inventory,
        "device_names": names,
        "observe_seconds": OBSERVE_SECONDS,
        "timestep_ms": timestep,
        "trace": trace,
    }

    print("=" * 62)
    print("M0 / E2  Atlas stock model probe")
    print("=" * 62)
    print("device inventory:")
    for type_name in sorted(inventory):
        print(f"    {type_name:<24} {inventory[type_name]}")
    for expected in ("POSITION_SENSOR", "INERTIAL_UNIT", "TOUCH_SENSOR", "GYRO", "ACCELEROMETER"):
        if expected not in inventory:
            print(f"    {expected:<24} 0   <- absent")
    print("-" * 62)
    if first and last:
        print(f"CoM  t=0.00  {first['com']}   root {first['root']}")
        print(f"CoM  t={last['t']:.2f}  {last['com']}   root {last['root']}")
        drop = first["root"][2] - last["root"][2]
        print(f"root z drop over {OBSERVE_SECONDS:.1f}s   {drop:+.4f} m")
        print(f"com  relative to root z    {first['com'][2] - first['root'][2]:+.4f} m")
        print(f"contacts at end            {last['contact_count']}")
        print(f"static balance at end      {last['static_balance']}")
        result["root_z_drop_m"] = drop
        result["com_minus_root_z_m"] = first["com"][2] - first["root"][2]
    print("=" * 62)

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m0_e2_atlas.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"written: {out_path}")
    sys.stdout.flush()
    robot.simulationQuit(0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
