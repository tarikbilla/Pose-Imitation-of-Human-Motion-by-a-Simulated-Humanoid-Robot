import json
import os
import sys

from controller import Node, Supervisor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import armkin
from transport import udp

CROUCH_RAD = 0.15
RAMP_SECONDS = 1.2
SETTLE_SECONDS = 0.8
JOINT_VELOCITY = 1.5
FALL_FRACTION = float(os.environ.get("A3_FALL_FRACTION", "0.805"))

GAIN_PITCH_TO_X = -1.2812
GAIN_HIP_ROLL_TO_Y = -0.5698
AUTHORITY = 0.65
KD_SAGITTAL = 0.12
KD_LATERAL = 0.15
DERIVATIVE_TAU = 0.04
KI_SAGITTAL = 0.9
KI_LATERAL = 1.2
INTEGRAL_LIMIT = 0.08
SAFETY_ENTER = 0.030
SAFETY_FULL = 0.080
SAFETY_RATE = 4.0
OMEGA = 3.1395
ARM_SCALE = float(os.environ.get("A3_ARM_SCALE", "0.75"))
SAFETY_ENABLED = os.environ.get("A3_SAFETY", "1") == "1"
SCALED_JOINTS = frozenset(("LArmUsy", "LArmShx", "RArmUsy", "RArmShx"))
FINISH_AFTER_IDLE = float(os.environ.get("A3_FINISH_IDLE", "0")) 
ANKLE_LIMIT_PITCH = 0.55
HIP_LIMIT_ROLL = 0.45

ANKLE_PITCH = ("LLegUay", "RLegUay")
HIP_ROLL = ("LLegMhx", "RLegMhx")

ARM_MOTORS = {
    "L": {"usy": "LArmUsy", "shx": "LArmShx", "ely": "LArmEly", "elx": "LArmElx"},
    "R": {"usy": "RArmUsy", "shx": "RArmShx", "ely": "RArmEly", "elx": "RArmElx"},
}

TORSO_LIMITS = {
    "BackLbz": (-0.610865, 0.610865),
    "BackMby": (-1.2, 1.28),
    "BackUbx": (-0.790809, 0.790809),
}
NECK_LIMITS = (-0.610865238, 1.13446401)

MAX_JOINT_RATE = 1.8
TIMEOUT_SECONDS = 0.5
LOG_EVERY = 250

DIRECTION_DEADBAND = 0.02
TORSO_PITCH_LIMIT = 0.10
TORSO_ROLL_LIMIT = 0.30
TORSO_YAW_LIMIT = 0.55


def direction_changed(previous, current, threshold=DIRECTION_DEADBAND):
    if previous is None:
        return True
    return sum((a - b) ** 2 for a, b in zip(previous, current)) > threshold * threshold

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)


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
    return [
        rotation[0] * vector[0] + rotation[3] * vector[1] + rotation[6] * vector[2],
        rotation[1] * vector[0] + rotation[4] * vector[1] + rotation[7] * vector[2],
        rotation[2] * vector[0] + rotation[5] * vector[1] + rotation[8] * vector[2],
    ]


class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.values = {}

    def step(self, name, target, dt):
        current = self.values.get(name)
        if current is None:
            self.values[name] = target
            return target
        maximum = self.rate * dt
        delta = clamp(target - current, -maximum, maximum)
        self.values[name] = current + delta
        return self.values[name]


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    motors = {}
    sensors = {}
    for index in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(index)
        if device.getNodeType() == Node.ROTATIONAL_MOTOR:
            device.setVelocity(JOINT_VELOCITY)
            motors[device.getName()] = device
        elif device.getNodeType() == Node.POSITION_SENSOR:
            device.enable(timestep)
            sensors[device.getName()] = device

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

    stand_height = float(self_node.getCenterOfMass()[2])
    fall_height = FALL_FRACTION * stand_height
    print(f"stand_height={stand_height:.3f} fall_below={fall_height:.3f}")
    sys.stdout.flush()

    rotation = self_node.getOrientation()
    root = self_node.getPosition()

    def com_body():
        current = self_node.getCenterOfMass()
        return world_to_body(rotation, [current[i] - root[i] for i in range(3)])

    setpoint = com_body()
    error_x_prev = 0.0
    error_y_prev = 0.0
    filtered = [0.0, 0.0]
    previous = [0.0, 0.0]
    integral = [0.0, 0.0]
    alpha_d = dt / (DERIVATIVE_TAU + dt)

    receiver = udp.Receiver()
    limiter = RateLimiter(MAX_JOINT_RATE)
    seeds = {"L": {}, "R": {}}
    last_directions = {}
    targets = {}
    neutral = {}
    authority_scale = 1.0
    last_packet_time = None
    elapsed = 0.0

    packets = 0
    solved = 0
    ik_errors = []
    fallen = False
    trace = []
    step_count = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    ready_marker = os.path.join(OUT_DIR, "m3_ready")
    with open(ready_marker, "w", encoding="utf-8") as handle:
        handle.write(str(elapsed))
    print(f"a3_upper_body ready, listening on {udp.DEFAULT_HOST}:{udp.DEFAULT_PORT}")
    sys.stdout.flush()

    while robot.step(timestep) != -1:
        elapsed += dt
        packet = receiver.poll()

        if packet is not None and packet.get("valid"):
            packets += 1
            last_packet_time = elapsed
            for side, prefix in (("L", "left"), ("R", "right")):
                upper = packet.get(f"{prefix}_upper_arm")
                fore = packet.get(f"{prefix}_fore_arm")
                if not upper or all(abs(c) < 1e-6 for c in upper):
                    continue
                if fore and all(abs(c) < 1e-6 for c in fore):
                    fore = None

                moved = direction_changed(last_directions.get(side + "u"), upper)
                if fore is not None:
                    moved = moved or direction_changed(
                        last_directions.get(side + "f"), fore
                    )
                if not moved:
                    continue
                last_directions[side + "u"] = upper
                if fore is not None:
                    last_directions[side + "f"] = fore

                angles, error_upper, error_fore = armkin.solve_arm(
                    side, upper, fore, seed=seeds[side]
                )
                seeds[side] = {
                    "upper": (angles["usy"], angles["shx"]),
                    "fore": (angles.get("ely", 0.0), angles.get("elx", 0.0)),
                }
                for key, value in angles.items():
                    targets[ARM_MOTORS[side][key]] = value
                ik_errors.append(max(error_upper, error_fore))
                solved += 1

            targets["BackLbz"] = clamp(
                packet.get("torso_yaw", 0.0), -TORSO_YAW_LIMIT, TORSO_YAW_LIMIT)
            targets["BackMby"] = clamp(
                packet.get("torso_pitch", 0.0), -TORSO_PITCH_LIMIT, TORSO_PITCH_LIMIT)
            targets["BackUbx"] = clamp(
                -packet.get("torso_roll", 0.0), -TORSO_ROLL_LIMIT, TORSO_ROLL_LIMIT)
            targets["NeckAy"] = clamp(packet.get("head_pitch", 0.0), *NECK_LIMITS)

        capture_x = error_x_prev + filtered[0] / OMEGA
        capture_y = error_y_prev + filtered[1] / OMEGA
        margin = max(abs(capture_x), abs(capture_y))
        if margin <= SAFETY_ENTER:
            desired_scale = 1.0
        elif margin >= SAFETY_FULL:
            desired_scale = 0.0
        else:
            desired_scale = 1.0 - (margin - SAFETY_ENTER) / (SAFETY_FULL - SAFETY_ENTER)
        if not SAFETY_ENABLED:
            desired_scale = 1.0
        step_limit = SAFETY_RATE * dt
        authority_scale += clamp(desired_scale - authority_scale, -step_limit, step_limit)

        stale = last_packet_time is None or (elapsed - last_packet_time) > TIMEOUT_SECONDS
        if not stale:
            for name, value in targets.items():
                if name in motors:
                    rest = neutral.get(name, 0.0)
                    factor = ARM_SCALE if name in SCALED_JOINTS else 1.0
                    scaled = rest + factor * authority_scale * (value - rest)
                    motors[name].setPosition(limiter.step(name, scaled, dt))

        current = com_body()
        error_x = current[0] - setpoint[0]
        error_y = current[1] - setpoint[1]
        error_x_prev, error_y_prev = error_x, error_y
        for index, error in enumerate((error_x, error_y)):
            derivative = (error - previous[index]) / dt
            filtered[index] += alpha_d * (derivative - filtered[index])
            previous[index] = error
            integral[index] = clamp(integral[index] + error * dt,
                                    -INTEGRAL_LIMIT, INTEGRAL_LIMIT)

        pitch = clamp(
            -(AUTHORITY * error_x + KI_SAGITTAL * integral[0]
              + KD_SAGITTAL * filtered[0]) / GAIN_PITCH_TO_X,
            -ANKLE_LIMIT_PITCH, ANKLE_LIMIT_PITCH,
        )
        roll = clamp(
            -(AUTHORITY * error_y + KI_LATERAL * integral[1]
              + KD_LATERAL * filtered[1]) / GAIN_HIP_ROLL_TO_Y,
            -HIP_LIMIT_ROLL, HIP_LIMIT_ROLL,
        )
        for name in ANKLE_PITCH:
            if name in motors:
                motors[name].setPosition(
                    clamp(pose[name] + pitch, -ANKLE_LIMIT_PITCH, ANKLE_LIMIT_PITCH)
                )
        for name in HIP_ROLL:
            if name in motors:
                motors[name].setPosition(clamp(roll, -HIP_LIMIT_ROLL, HIP_LIMIT_ROLL))

        step_count += 1
        if step_count % 5 == 0:
            com = self_node.getCenterOfMass()
            measured_now = {n: sensors[n].getValue() for n in sensors}
            track = {}
            imitation = []
            for group, joints in (("arm", ("LArmShx", "RArmShx", "LArmElx", "RArmElx",
                                           "LArmUsy", "RArmUsy", "LArmEly", "RArmEly")),
                                  ("torso", ("BackLbz", "BackMby", "BackUbx"))):
                errs = []
                for j in joints:
                    if j in targets and (j + "S") in measured_now:
                        rest = neutral.get(j, 0.0)
                        want = rest + ARM_SCALE * authority_scale * (targets[j] - rest)                             if j in SCALED_JOINTS else targets[j]
                        errs.append(abs(want - measured_now[j + "S"]))
                        imitation.append(abs(targets[j] - measured_now[j + "S"]))
                track[group] = round(sum(errs) / len(errs), 4) if errs else 0.0
            trace.append({
                "track": track,
                "imit": round(sum(imitation) / len(imitation), 4) if imitation else 0.0,
                "t": round(elapsed, 3),
                "com": [round(v, 4) for v in com],
                "err": [round(error_x, 4), round(error_y, 4)],
                "contacts": len(self_node.getContactPoints(True)),
                "pitch": round(pitch, 4),
                "roll": round(roll, 4),
                "torso": round(targets.get("BackMby", 0.0), 4),
                "larm": round(targets.get("LArmShx", 0.0), 4),
                "rarm": round(targets.get("RArmShx", 0.0), 4),
                "scale": round(authority_scale, 3),
                "cp": [round(capture_x, 4), round(capture_y, 4)],
            })

        if self_node.getCenterOfMass()[2] < fall_height:
            fallen = True
            print("FALLEN")
            break

        if (FINISH_AFTER_IDLE > 0.0 and packets > 0 and last_packet_time is not None
                and (elapsed - last_packet_time) > FINISH_AFTER_IDLE):
            print("stream finished")
            break

        if packets and packets % LOG_EVERY == 0 and packet is not None:
            mean_error = sum(ik_errors[-LOG_EVERY:]) / min(len(ik_errors), LOG_EVERY)
            print(f"t={elapsed:6.1f}s packets={packets} dropped={receiver.dropped} "
                  f"ik_err={mean_error:.4f} com_err=({error_x*1000:+.0f},{error_y*1000:+.0f})mm")
            sys.stdout.flush()

    summary = {
        "packets": packets,
        "solved_arms": solved,
        "dropped": receiver.dropped,
        "mean_ik_error": sum(ik_errors) / len(ik_errors) if ik_errors else None,
        "max_ik_error": max(ik_errors) if ik_errors else None,
        "fallen": fallen,
        "stand_height": round(stand_height, 4),
        "duration_s": elapsed,
        "trace": trace,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "m3_run.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    receiver.close()
    try:
        os.remove(os.path.join(OUT_DIR, 'm3_ready'))
    except OSError:
        pass
    print(json.dumps(summary))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
