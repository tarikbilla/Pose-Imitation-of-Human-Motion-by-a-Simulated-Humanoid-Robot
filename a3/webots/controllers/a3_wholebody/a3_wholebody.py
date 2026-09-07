import json
import math
import os
import sys

import numpy as np
from controller import Node, Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "a3_upper_body"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import armkin
from atlaskin import AtlasModel
from transport import udp
from stepper import Stepper
from wbc import WholeBodyQP

CROUCH_RAD = 0.15
RAMP_SECONDS = 1.2
SETTLE_SECONDS = 0.8
JOINT_VELOCITY = 2.5
RATE_LIMIT = 1.8
FALL_COM_HEIGHT_M = 0.80
TIMEOUT_SECONDS = 0.5

SQUAT_GAIN = float(os.environ.get("A3_SQUAT_GAIN", "0.9"))
STANCE_GAIN = float(os.environ.get("A3_STANCE_GAIN", "0.30"))
LEG_REF_TAU = 0.45          # smooth the lifted references
LEG_BLEND_SECONDS = 2.5     # fade them in once they first arrive
DIRECTION_DEADBAND = 0.02

TORSO_PITCH_LIMIT = 0.22
TORSO_ROLL_LIMIT = 0.35
TORSO_YAW_LIMIT = 0.55

ARM_SCALE = float(os.environ.get("A3_ARM_SCALE", "0.75"))
SCALED_JOINTS = frozenset(("LArmUsy", "LArmShx", "RArmUsy", "RArmShx"))

DIRECT_JOINTS = ("LArmUsy", "LArmShx", "LArmEly", "LArmElx",
                 "RArmUsy", "RArmShx", "RArmEly", "RArmElx", "NeckAy")

ARM_MOTORS = {
    "L": {"usy": "LArmUsy", "shx": "LArmShx", "ely": "LArmEly", "elx": "LArmElx"},
    "R": {"usy": "RArmUsy", "shx": "RArmShx", "ely": "RArmEly", "elx": "RArmElx"},
}

OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "results"))
FINISH_AFTER_IDLE = float(os.environ.get("A3_FINISH_IDLE", "0"))
USE_LEGS = os.environ.get("A3_USE_LEGS", "1") == "1"
USE_STEPPER = os.environ.get("A3_USE_STEPPER", "1") == "1"
FORCE_MARCH = float(os.environ.get("A3_FORCE_MARCH", "0"))
MARCH_START_S = float(os.environ.get("A3_MARCH_START_S", "6"))
LOG_EVERY = 400


def clamp(value, low, high):
    return max(low, min(high, value))


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
    return np.array([
        rotation[0] * vector[0] + rotation[3] * vector[1] + rotation[6] * vector[2],
        rotation[1] * vector[0] + rotation[4] * vector[1] + rotation[7] * vector[2],
        rotation[2] * vector[0] + rotation[5] * vector[1] + rotation[8] * vector[2],
    ])


def direction_changed(previous, current, threshold=DIRECTION_DEADBAND):
    if previous is None:
        return True
    return sum((a - b) ** 2 for a, b in zip(previous, current)) > threshold * threshold


class LegReference:
    """Turns lifted lower-body wishes into joint targets.

    The lifter needs a 27-frame window, so its first value arrives about a
    second late and would otherwise land as a step input. Two guards: a
    low-pass on the reference itself and a fade-in on first arrival.
    """

    def __init__(self, dt):
        self.dt = dt
        self.squat = 0.0
        self.stance = 0.0
        self.blend = 0.0
        self.seen = False

    def update(self, packet, base):
        targets = dict(base)
        if not USE_LEGS or not packet or not packet.get("lower_body_valid"):
            return targets

        raw_squat = clamp(float(packet.get("hip_height", 0.0)), 0.0, 0.45) * SQUAT_GAIN
        raw_stance = clamp(float(packet.get("stance_width", 0.0)) - 0.35,
                           -0.25, 0.35) * STANCE_GAIN

        if not self.seen:
            self.seen = True
            self.squat, self.stance = 0.0, 0.0

        alpha = self.dt / (LEG_REF_TAU + self.dt)
        self.squat += alpha * (raw_squat - self.squat)
        self.stance += alpha * (raw_stance - self.stance)
        self.blend = min(1.0, self.blend + self.dt / LEG_BLEND_SECONDS)

        squat = self.squat * self.blend
        stance = self.stance * self.blend
        for side in "LR":
            sign = 1.0 if side == "L" else -1.0
            targets[f"{side}LegLhy"] = base[f"{side}LegLhy"] - squat
            targets[f"{side}LegKny"] = base[f"{side}LegKny"] + 2.0 * squat
            targets[f"{side}LegUay"] = base[f"{side}LegUay"] - squat
            targets[f"{side}LegMhx"] = base[f"{side}LegMhx"] + sign * stance
        return targets


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    model = AtlasModel()
    qp = WholeBodyQP(model, dt, rate_limit=RATE_LIMIT)
    legs = LegReference(dt)
    stepper = Stepper()

    motors = {}
    sensors = {}
    for i in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(i)
        kind = device.getNodeType()
        if kind == Node.ROTATIONAL_MOTOR:
            device.setVelocity(JOINT_VELOCITY)
            motors[device.getName()] = device
        elif kind == Node.POSITION_SENSOR:
            device.enable(timestep)
            sensors[device.getName()] = device

    self_node = robot.getSelf()
    self_node.enableContactPointsTracking(timestep, True)

    base = stance_pose()
    commanded = {name: 0.0 for name in model.names}
    robot.step(timestep)

    ramp_steps = int(RAMP_SECONDS / dt)
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        for name, value in base.items():
            if name in motors:
                commanded[name] = value * alpha
                motors[name].setPosition(commanded[name])
        if robot.step(timestep) == -1:
            return 1
    for _ in range(int(SETTLE_SECONDS / dt)):
        if robot.step(timestep) == -1:
            return 1

    rotation = self_node.getOrientation()
    root = np.asarray(self_node.getPosition())
    com_setpoint = world_to_body(rotation, np.asarray(self_node.getCenterOfMass()) - root)

    receiver = udp.Receiver()
    targets = dict(base)
    seeds = {"L": {}, "R": {}}
    last_directions = {}
    last_packet_time = None
    last_packet = None
    elapsed = 0.0

    packets = 0
    fallen = False
    trace = []
    saturated = 0
    steps = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m4_ready"), "w", encoding="utf-8") as handle:
        handle.write("ready")
    print(f"a3_wholebody ready on {udp.DEFAULT_HOST}:{udp.DEFAULT_PORT}")
    sys.stdout.flush()

    while robot.step(timestep) != -1:
        elapsed += dt
        steps += 1
        packet = receiver.poll()

        if packet is not None and packet.get("valid"):
            last_packet = packet
            packets += 1
            last_packet_time = elapsed
            for side, prefix in (("L", "left"), ("R", "right")):
                upper = packet.get(f"{prefix}_upper_arm")
                fore = packet.get(f"{prefix}_fore_arm")
                if not upper or all(abs(c) < 1e-6 for c in upper):
                    continue
                if fore and all(abs(c) < 1e-6 for c in fore):
                    fore = None
                if not (direction_changed(last_directions.get(side + "u"), upper)
                        or (fore is not None
                            and direction_changed(last_directions.get(side + "f"), fore))):
                    continue
                last_directions[side + "u"] = upper
                if fore is not None:
                    last_directions[side + "f"] = fore
                angles_solved, _, _ = armkin.solve_arm(side, upper, fore, seed=seeds[side])
                seeds[side] = {
                    "upper": (angles_solved["usy"], angles_solved["shx"]),
                    "fore": (angles_solved.get("ely", 0.0), angles_solved.get("elx", 0.0)),
                }
                for key, value in angles_solved.items():
                    name = ARM_MOTORS[side][key]
                    factor = ARM_SCALE if name in SCALED_JOINTS else 1.0
                    targets[name] = factor * value

            targets["BackLbz"] = clamp(packet.get("torso_yaw", 0.0),
                                       -TORSO_YAW_LIMIT, TORSO_YAW_LIMIT)
            targets["BackMby"] = clamp(packet.get("torso_pitch", 0.0),
                                       -TORSO_PITCH_LIMIT, TORSO_PITCH_LIMIT)
            targets["BackUbx"] = clamp(-packet.get("torso_roll", 0.0),
                                       -TORSO_ROLL_LIMIT, TORSO_ROLL_LIMIT)
            targets["NeckAy"] = clamp(packet.get("head_pitch", 0.0), -0.61, 1.13)
            targets.update(legs.update(packet, base))

        stale = last_packet_time is None or (elapsed - last_packet_time) > TIMEOUT_SECONDS
        active = dict(base) if stale else targets

        march = 0.0
        if FORCE_MARCH > 0.0 and elapsed > MARCH_START_S:
            march = FORCE_MARCH
        elif last_packet is not None and last_packet.get("lower_body_valid"):
            march = max(float(last_packet.get("left_foot_lift", 0.0)),
                        float(last_packet.get("right_foot_lift", 0.0)))

        measured = {n: sensors[n + "S"].getValue() for n in model.names
                    if n + "S" in sensors}
        com_body = world_to_body(
            rotation, np.asarray(self_node.getCenterOfMass()) - root)
        com_shift = 0.0
        swing = None
        lift = 0.0
        if USE_STEPPER:
            com_shift, swing, lift = stepper.update(
                dt, march, float(com_body[1] - com_setpoint[1]))
        com_error = com_body - com_setpoint
        com_error[1] -= com_shift

        contacts = self_node.getContactPoints(True)
        if contacts:
            points = np.asarray([world_to_body(rotation, np.asarray(c.point) - root)
                                 for c in contacts])
            polygon = (points[:, 0].min(), points[:, 0].max(),
                       points[:, 1].min(), points[:, 1].max())
        else:
            polygon = (com_setpoint[0] - 0.05, com_setpoint[0] + 0.05,
                       com_setpoint[1] - 0.05, com_setpoint[1] + 0.05)

        poses = model.frames(measured)
        jacobian = model.com_jacobian(measured, poses)

        if swing is not None and lift > 0.0:
            active = dict(active)
            for name, offset in stepper.leg_offsets(swing, lift).items():
                active[name] = active.get(name, base.get(name, 0.0)) + offset

        qp.com_saturated = False
        delta = qp.solve(commanded, com_error, com_setpoint, polygon,
                         active, base, jacobian,
                         free_joints=DIRECT_JOINTS)
        if qp.com_saturated:
            saturated += 1

        # Integrate on the previous command, not on the measurement: the servo
        # always lags, and feeding that lag back in eats the motion and drifts.
        for i, name in enumerate(model.names):
            if name in motors:
                commanded[name] = model.clamp(name, commanded[name] + delta[i])
                motors[name].setPosition(commanded[name])

        if steps % 5 == 0:
            track = {}
            for group, joints in (("arm", ("LArmShx", "RArmShx", "LArmElx", "RArmElx",
                                           "LArmUsy", "RArmUsy", "LArmEly", "RArmEly")),
                                  ("torso", ("BackLbz", "BackMby", "BackUbx")),
                                  ("leg", ("LLegKny", "RLegKny", "LLegLhy", "RLegLhy"))):
                errs = [abs(active.get(j, 0.0) - measured.get(j, 0.0))
                        for j in joints if j in active]
                track[group] = round(sum(errs) / len(errs), 4) if errs else 0.0
            want = {j: round(active.get(j, 0.0), 3)
                    for j in ("LArmShx", "RArmShx", "LArmElx")}
            got = {j: round(measured.get(j, 0.0), 3)
                   for j in ("LArmShx", "RArmShx", "LArmElx")}
            trace.append({
                "track": track, "want": want, "got": got,
                "t": round(elapsed, 3),
                "err": [round(float(com_error[0]), 4), round(float(com_error[1]), 4)],
                "contacts": len(contacts),
                "status": qp.status,
                "knee": round(measured.get("LLegKny", 0.0), 3),
                "shx": round(measured.get("LArmShx", 0.0), 3),
                "state": stepper.state,
                "swing": swing or "-",
                "shift": round(com_shift, 4),
            })

        if self_node.getCenterOfMass()[2] < FALL_COM_HEIGHT_M:
            fallen = True
            print("FALLEN")
            break
        if (FINISH_AFTER_IDLE > 0.0 and packets > 0 and last_packet_time is not None
                and (elapsed - last_packet_time) > FINISH_AFTER_IDLE):
            print("stream finished")
            break
        if packets and packets % LOG_EVERY == 0 and packet is not None:
            print(f"t={elapsed:6.1f} packets={packets} qp={qp.status} "
                  f"com=({com_error[0]*1000:+.0f},{com_error[1]*1000:+.0f})mm")
            sys.stdout.flush()

    summary = {
        "steps_taken": stepper.steps,
        "packets": packets,
        "fallen": fallen,
        "duration_s": elapsed,
        "com_saturated_steps": saturated,
        "steps": steps,
        "trace": trace,
    }
    with open(os.path.join(OUT_DIR, "m4_run.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    try:
        os.remove(os.path.join(OUT_DIR, "m4_ready"))
    except OSError:
        pass

    receiver.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
