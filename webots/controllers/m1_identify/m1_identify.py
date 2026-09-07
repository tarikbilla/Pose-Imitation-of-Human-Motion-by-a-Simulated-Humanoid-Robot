import json
import os
import sys

from controller import Node, Supervisor

CROUCH_RAD = 0.15
RAMP_SECONDS = 1.5
SETTLE_SECONDS = 1.5
HOLD_SECONDS = 1.5
PROBE_RAD = 0.03
PROBE_RAMP_SECONDS = 0.6
JOINT_VELOCITY = 1.5
FALL_COM_HEIGHT_M = 0.80

ANKLE_PITCH = ("LLegUay", "RLegUay")
ANKLE_ROLL = ("LLegLax", "RLegLax")
HIP_ROLL = ("LLegMhx", "RLegMhx")

PROBE_PLAN = (
    ("ankle_pitch_plus", ANKLE_PITCH, (+1.0, +1.0)),
    ("ankle_pitch_minus", ANKLE_PITCH, (-1.0, -1.0)),
    ("ankle_roll_plus", ANKLE_ROLL, (+1.0, +1.0)),
    ("ankle_roll_minus", ANKLE_ROLL, (-1.0, -1.0)),
    ("hip_roll_sym_plus", HIP_ROLL, (+1.0, +1.0)),
    ("hip_roll_sym_minus", HIP_ROLL, (-1.0, -1.0)),
    ("hip_roll_anti_plus", HIP_ROLL, (+1.0, -1.0)),
    ("hip_roll_anti_minus", HIP_ROLL, (-1.0, +1.0)),
)

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)


def stance_pose():
    pose = {}
    for side in "LR":
        pose[f"{side}LegLhy"] = -CROUCH_RAD
        pose[f"{side}LegKny"] = 2.0 * CROUCH_RAD
        pose[f"{side}LegUay"] = -CROUCH_RAD
        pose[f"{side}LegMhx"] = 0.0
        pose[f"{side}LegLax"] = 0.0
        pose[f"{side}LegUhz"] = 0.0
    return pose


def world_to_body(rotation, vector):
    return [
        rotation[0] * vector[0] + rotation[3] * vector[1] + rotation[6] * vector[2],
        rotation[1] * vector[0] + rotation[4] * vector[1] + rotation[7] * vector[2],
        rotation[2] * vector[0] + rotation[5] * vector[1] + rotation[8] * vector[2],
    ]


def hold(robot, timestep, seconds):
    for _ in range(int(seconds / (timestep / 1000.0))):
        if robot.step(timestep) == -1:
            return False
    return True


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    motors = {}
    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        if device.getNodeType() == Node.ROTATIONAL_MOTOR:
            device.setVelocity(JOINT_VELOCITY)
            motors[device.getName()] = device

    self_node = robot.getSelf()
    self_node.enableContactPointsTracking(timestep, True)

    pose = stance_pose()
    robot.step(timestep)

    ramp_steps = int(RAMP_SECONDS / dt)
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        for name, target in pose.items():
            if name in motors:
                motors[name].setPosition(target * alpha)
        if robot.step(timestep) == -1:
            return 1
    if not hold(robot, timestep, SETTLE_SECONDS):
        return 1

    rotation = self_node.getOrientation()
    root = self_node.getPosition()
    com = self_node.getCenterOfMass()
    contacts = [c.point for c in self_node.getContactPoints(True)]

    def com_in_body():
        current = self_node.getCenterOfMass()
        delta = [current[i] - root[i] for i in range(3)]
        return world_to_body(rotation, delta)

    reference = com_in_body()
    contacts_body = [
        world_to_body(rotation, [p[i] - root[i] for i in range(3)]) for p in contacts
    ]

    def ramp_to(joints, signs, magnitude, seconds):
        steps = max(1, int(seconds / dt))
        lowest = 99
        for step in range(steps):
            alpha = (step + 1) / steps
            for name, sign in zip(joints, signs):
                if name in motors:
                    motors[name].setPosition(pose[name] + sign * magnitude * alpha)
            if robot.step(timestep) == -1:
                return False, lowest
            lowest = min(lowest, len(self_node.getContactPoints(True)))
            if self_node.getCenterOfMass()[2] < FALL_COM_HEIGHT_M:
                return False, lowest
        return True, lowest

    probes = {}
    aborted = None
    for label, joints, signs in PROBE_PLAN:
        ok, lowest = ramp_to(joints, signs, PROBE_RAD, PROBE_RAMP_SECONDS)
        if not ok or not hold(robot, timestep, HOLD_SECONDS):
            aborted = label
            break
        if self_node.getCenterOfMass()[2] < FALL_COM_HEIGHT_M:
            aborted = label
            break

        displaced = com_in_body()
        probes[label] = {
            "applied_rad": PROBE_RAD,
            "signs": list(signs),
            "com_body": [round(v, 5) for v in displaced],
            "delta_body": [round(displaced[i] - reference[i], 5) for i in range(3)],
            "contacts": len(self_node.getContactPoints(True)),
            "min_contacts_during_ramp": lowest,
        }

        ok, _ = ramp_to(joints, signs, 0.0, PROBE_RAMP_SECONDS)
        if not ok or not hold(robot, timestep, SETTLE_SECONDS):
            aborted = label + "_return"
            break

    def gain(plus, minus, axis):
        if plus not in probes or minus not in probes:
            return None
        return (probes[plus]["delta_body"][axis] - probes[minus]["delta_body"][axis]) / (
            2.0 * PROBE_RAD
        )

    result = {
        "aborted_at": aborted,
        "orientation_row_major": [round(v, 5) for v in rotation],
        "root_world": [round(v, 5) for v in root],
        "com_world": [round(v, 5) for v in com],
        "com_body_reference": [round(v, 5) for v in reference],
        "contacts_body": [[round(v, 4) for v in p] for p in contacts_body],
        "probes": probes,
        "gain_pitch_to_body_x": gain("ankle_pitch_plus", "ankle_pitch_minus", 0),
        "gain_roll_to_body_y": gain("ankle_roll_plus", "ankle_roll_minus", 1),
        "gain_hip_sym_to_body_y": gain("hip_roll_sym_plus", "hip_roll_sym_minus", 1),
        "gain_hip_anti_to_body_y": gain("hip_roll_anti_plus", "hip_roll_anti_minus", 1),
    }

    print("=" * 66)
    print("M1  plant identification: ankle angle -> CoM displacement (body frame)")
    print("=" * 66)
    print(f"CoM in body frame     x={reference[0]:+.5f}  y={reference[1]:+.5f}  z={reference[2]:+.5f}")
    if contacts_body:
        xs = [p[0] for p in contacts_body]
        ys = [p[1] for p in contacts_body]
        print(f"support x range       {min(xs):+.4f} .. {max(xs):+.4f}  (span {max(xs)-min(xs):.4f})")
        print(f"support y range       {min(ys):+.4f} .. {max(ys):+.4f}  (span {max(ys)-min(ys):.4f})")
    print("-" * 66)
    for label, data in probes.items():
        d = data["delta_body"]
        print(
            f"{label:<22} dx={d[0]:+.5f} dy={d[1]:+.5f}"
            f"  contacts={data['contacts']} (min {data['min_contacts_during_ramp']})"
        )
    print("-" * 66)
    for key in (
        "gain_pitch_to_body_x",
        "gain_roll_to_body_y",
        "gain_hip_sym_to_body_y",
        "gain_hip_anti_to_body_y",
    ):
        value = result[key]
        print(f"{key:<28} {value:+.5f} m/rad" if value is not None else f"{key:<28} n/a")
    if aborted:
        print(f"ABORTED at            {aborted}  (robot fell)")
    print("=" * 66)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m1_identify.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    sys.stdout.flush()
    robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
