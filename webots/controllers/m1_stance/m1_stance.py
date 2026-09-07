import json
import os
import sys

from controller import Node, Supervisor

CROUCH_RAD = 0.15
RAMP_SECONDS = 1.5
SETTLE_SECONDS = 1.5
OBSERVE_SECONDS = 5.0

JOINT_VELOCITY = 1.5
IMPULSE_SECONDS = 0.1
FALL_COM_HEIGHT_M = 0.80

GAIN_PITCH_TO_X = -1.2812
GAIN_HIP_ROLL_TO_Y = -0.5698

AUTHORITY = 0.45
KD_SAGITTAL = 0.12
KD_LATERAL = 0.15
DERIVATIVE_TAU = 0.04

ANKLE_LIMIT_PITCH = 0.55
HIP_LIMIT_ROLL = 0.40

ANKLE_PITCH = ("LLegUay", "RLegUay")
HIP_ROLL = ("LLegMhx", "RLegMhx")

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


def body_axis_in_world(rotation, axis):
    return [rotation[axis], rotation[3 + axis], rotation[6 + axis]]


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    impulse_ns = float(os.environ.get("A3_IMPULSE", "0"))
    admittance = os.environ.get("A3_ADMITTANCE", "1") == "1"
    tag = os.environ.get("A3_TAG", "run")
    observe_seconds = float(os.environ.get("A3_OBSERVE", str(OBSERVE_SECONDS)))

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
    for _ in range(int(SETTLE_SECONDS / dt)):
        if robot.step(timestep) == -1:
            return 1

    rotation = self_node.getOrientation()
    root = self_node.getPosition()

    def com_body():
        current = self_node.getCenterOfMass()
        return world_to_body(rotation, [current[i] - root[i] for i in range(3)])

    setpoint = com_body()
    lateral_axis = body_axis_in_world(rotation, 1)

    filtered = [0.0, 0.0]
    previous = [0.0, 0.0]
    alpha_d = dt / (DERIVATIVE_TAU + dt)

    impulse_steps = int(IMPULSE_SECONDS / dt) if impulse_ns > 0 else 0
    impulse_force = impulse_ns / (impulse_steps * dt) if impulse_steps else 0.0

    trace = []
    fallen = False
    for step in range(int(observe_seconds / dt)):
        if step < impulse_steps:
            self_node.addForce([axis * impulse_force for axis in lateral_axis], False)

        current = com_body()
        error_x = current[0] - setpoint[0]
        error_y = current[1] - setpoint[1]

        for index, error in enumerate((error_x, error_y)):
            derivative = (error - previous[index]) / dt
            filtered[index] += alpha_d * (derivative - filtered[index])
            previous[index] = error

        if admittance:
            pitch = clamp(
                -(AUTHORITY * error_x + KD_SAGITTAL * filtered[0]) / GAIN_PITCH_TO_X,
                ANKLE_LIMIT_PITCH,
            )
            roll = clamp(
                -(AUTHORITY * error_y + KD_LATERAL * filtered[1]) / GAIN_HIP_ROLL_TO_Y,
                HIP_LIMIT_ROLL,
            )
            for name in ANKLE_PITCH:
                if name in motors:
                    motors[name].setPosition(clamp(pose[name] + pitch, ANKLE_LIMIT_PITCH))
            for name in HIP_ROLL:
                if name in motors:
                    motors[name].setPosition(clamp(roll, HIP_LIMIT_ROLL))

        if step % 4 == 0:
            trace.append(
                {
                    "t": round(step * dt, 4),
                    "err": [round(error_x, 5), round(error_y, 5)],
                    "com_z": round(self_node.getCenterOfMass()[2], 5),
                    "contacts": len(self_node.getContactPoints(True)),
                    "balance": bool(self_node.getStaticBalance()),
                }
            )

        if self_node.getCenterOfMass()[2] < FALL_COM_HEIGHT_M:
            fallen = True
            break
        if robot.step(timestep) == -1:
            return 1

    tail = trace[-int(len(trace) / 3) :] if trace else []
    result = {
        "tag": tag,
        "impulse_ns": impulse_ns,
        "observe_seconds": observe_seconds,
        "admittance": admittance,
        "authority": AUTHORITY,
        "fallen": fallen,
        "setpoint_body": [round(v, 5) for v in setpoint],
        "peak_lateral_error_m": max((abs(s["err"][1]) for s in trace), default=0.0),
        "peak_sagittal_error_m": max((abs(s["err"][0]) for s in trace), default=0.0),
        "residual_lateral_error_m": max((abs(s["err"][1]) for s in tail), default=0.0),
        "min_contacts": min((s["contacts"] for s in trace), default=0),
        "balance_always": all(s["balance"] for s in trace) if trace else False,
        "trace": trace,
    }

    print("=" * 66)
    print(f"M1 stance  tag={tag}  impulse={impulse_ns} Ns  admittance={admittance}")
    print("=" * 66)
    print(f"peak lateral error     {result['peak_lateral_error_m'] * 1000:7.1f} mm")
    print(f"residual lateral       {result['residual_lateral_error_m'] * 1000:7.1f} mm")
    print(f"min contacts           {result['min_contacts']}")
    print(f"fallen                 {fallen}")
    print("=" * 66)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"m1_stance_{tag}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    sys.stdout.flush()
    robot.simulationQuit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
