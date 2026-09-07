import json
import math
import os
import sys
import time

import numpy as np
from controller import Supervisor

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "..", ".."))
sys.path.insert(0, A3_ROOT)

from kinematics.atlaskin import AtlasModel
from kinematics.legik import LegIK
from transport import angles as angle_codec
from transport import udp

OUT_DIR = os.path.join(A3_ROOT, "results")

RUN_SECONDS = float(os.environ.get("A3_RUN_SECONDS", "60.0"))
GROUND_Z = float(os.environ.get("A3_GROUND_Z", "0.0"))
LOG_EVERY = int(os.environ.get("A3_LOG_EVERY", "10"))

ANGLES_FILE = os.environ.get("A3_ANGLES", "")
ANGLES_START = float(os.environ.get("A3_ANGLES_START", "0.0"))
WALL_CLOCK = os.environ.get("A3_WALL_CLOCK", "1") == "1"
WAIT_DRIVER = os.environ.get("A3_WAIT_DRIVER", "1") == "1"

PLANT_ON = float(os.environ.get("A3_PLANT_ON", "0.008"))
PLANT_OFF = float(os.environ.get("A3_PLANT_OFF", "0.020"))
LOCK_LEGS = os.environ.get("A3_LOCK_LEGS", "1") == "1"
SETTLE_SECONDS = float(os.environ.get("A3_SETTLE", "1.5"))
MOTOR_VELOCITY = float(os.environ.get("A3_MOTOR_VELOCITY", "12.0"))
COMMAND_RATE = float(os.environ.get("A3_COMMAND_RATE", "3.0"))
MOVIE = os.environ.get("A3_MOVIE", "")
LIVE = os.environ.get("A3_LIVE", "0") == "1"
LIVE_PORT = int(os.environ.get("A3_LIVE_PORT", "8768"))
LIVE_TIMEOUT = float(os.environ.get("A3_LIVE_TIMEOUT", "10.0"))
TRAVEL_MAX_SPEED = float(os.environ.get("A3_TRAVEL_MAX_SPEED", "2.0"))
BASE_TAU = float(os.environ.get("A3_BASE_TAU", "0.15"))
TRAVEL_TAU = float(os.environ.get("A3_TRAVEL_TAU", "0.35"))
TRAVEL_GAIN = float(os.environ.get("A3_TRAVEL_GAIN", "1.0"))
TRAVEL_SLEW = float(os.environ.get("A3_TRAVEL_SLEW", "2.0"))
DRIFT_LIMIT = float(os.environ.get("A3_DRIFT_LIMIT", "0.20"))
RELEASE_ERROR = float(os.environ.get("A3_RELEASE_ERROR", "0.015"))
MIN_STANCE = float(os.environ.get("A3_MIN_STANCE", "0.25"))
STRAIN_HOLD = float(os.environ.get("A3_STRAIN_HOLD", "0.10"))
REACH_MARGIN = float(os.environ.get("A3_REACH_MARGIN", "0.98"))
BASE_SERVO = os.environ.get("A3_BASE_SERVO", "1") == "1"
STEP_GAIN = float(os.environ.get("A3_STEP_GAIN", "1.0"))
STEP_LIMIT = float(os.environ.get("A3_STEP_LIMIT", "0.35"))
STEP_TAU = float(os.environ.get("A3_STEP_TAU", "0.20"))
HEIGHT_FLOOR = float(os.environ.get("A3_HEIGHT_FLOOR", "0.84"))
HEIGHT_TAU = float(os.environ.get("A3_HEIGHT_TAU", "0.10"))
HEIGHT_SLEW = float(os.environ.get("A3_HEIGHT_SLEW", "0.6"))

UPPER_JOINTS = ("BackLbz", "BackMby", "BackUbx", "NeckAy",
                "LArmUsy", "LArmShx", "LArmEly", "LArmElx",
                "RArmUsy", "RArmShx", "RArmEly", "RArmElx")
LEG_SUFFIX = ("LegUhz", "LegMhx", "LegLhy", "LegKny", "LegUay", "LegLax")
SOLE_BOTTOM = 0.005
FOOT_CORNERS = np.array([[0.13, 0.0624, -0.005], [0.13, -0.0624, -0.005],
                         [-0.13, 0.0624, -0.005], [-0.13, -0.0624, -0.005]])


def load_angles(path):
    frames = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "angles" in entry:
                frames.append(entry)
    return frames


def locate(frames, value):
    if value <= frames[0]["t"]:
        return frames[0], frames[0], 0.0
    if value >= frames[-1]["t"]:
        return frames[-1], frames[-1], 0.0
    low, high = 0, len(frames) - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if frames[middle]["t"] <= value:
            low = middle
        else:
            high = middle
    span = max(frames[high]["t"] - frames[low]["t"], 1e-6)
    return frames[low], frames[high], (value - frames[low]["t"]) / span


def sample_angles(frames, value):
    first, second, blend = locate(frames, value)
    merged = dict(first["angles"])
    for name, item in second["angles"].items():
        start = first["angles"].get(name, item)
        merged[name] = start + blend * (item - start)
    return merged


def sample_travel(frames, value):
    first, second, blend = locate(frames, value)
    out = []
    for name in ("forward", "lateral"):
        start = float(first.get(name, 0.0))
        out.append(start + blend * (float(second.get(name, 0.0)) - start))
    return np.array(out)


def smoothstep(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def axis_angle_from(rotation):
    trace = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(trace)
    if angle < 1e-9:
        return [0.0, 0.0, 1.0, 0.0]
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(angle))
    return [float(axis[0]), float(axis[1]), float(axis[2]), float(angle)]


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0
    self_node = robot.getSelf()
    translation_field = self_node.getField("translation")
    rotation_field = self_node.getField("rotation")

    model = AtlasModel()
    ik = {"L": LegIK(model, "L"), "R": LegIK(model, "R")}
    frames = load_angles(ANGLES_FILE) if ANGLES_FILE else []
    receiver = None
    if LIVE:
        receiver = udp.Receiver(port=LIVE_PORT, codec=angle_codec)
        print(f"a3_puppet live, listening on {udp.DEFAULT_HOST}:{LIVE_PORT}",
              flush=True)
    elif not frames:
        print("FATAL: no angle file", file=sys.stderr)
        return 1

    motors = {}
    sensors = {}
    for name in model.names:
        motor = robot.getDevice(name)
        if motor is None:
            continue
        motor.setVelocity(MOTOR_VELOCITY)
        motors[name] = motor
        sensor = robot.getDevice(name + "S")
        if sensor is not None:
            sensor.enable(timestep)
            sensors[name] = sensor

    start = np.array(translation_field.getSFVec3f(), dtype=np.float64)
    yaw = float(rotation_field.getSFRotation()[3])
    body = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                     [math.sin(yaw), math.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])
    heading = body[:2, :2]
    rotation_field.setSFRotation(axis_angle_from(body))

    def leg_values(targets, side):
        return np.array([targets.get(side + suffix, 0.0) for suffix in LEG_SUFFIX])

    def sole_world_offset(targets, side):
        return body @ ik[side].pose(leg_values(targets, side))[0]

    def sole_lowest(targets, side):
        sole, rotation = ik[side].pose(leg_values(targets, side))
        corners = sole + (rotation @ FOOT_CORNERS.T).T
        return float((corners @ body.T)[:, 2].min())

    hip_local = {}
    leg_reach = {}
    for side in ("L", "R"):
        origins, _ = ik[side].chain(np.zeros(6))
        sole = ik[side].pose(np.zeros(6))[0]
        hip_local[side] = np.asarray(origins[0], dtype=np.float64)
        leg_reach[side] = float(np.linalg.norm(sole - hip_local[side]))

    def drift_room(horizontal, active):
        rooms = []
        for side in active:
            hip = body @ hip_local[side]
            span = REACH_MARGIN * leg_reach[side]
            lift = HEIGHT_FLOOR + float(hip[2])
            reach = math.sqrt(max(span * span - lift * lift, 0.0))
            gap = float(np.linalg.norm(locks[side] - (horizontal + hip[:2])))
            rooms.append(reach - gap)
        return min(rooms) if rooms else DRIFT_LIMIT

    def height_ceiling(horizontal, active):
        limits = []
        for side in active:
            hip = body @ hip_local[side]
            gap = float(np.linalg.norm(locks[side] - (horizontal + hip[:2])))
            span = REACH_MARGIN * leg_reach[side]
            drop = math.sqrt(max(span * span - gap * gap, 0.0))
            limits.append(drop - float(hip[2]))
        return min(limits) if limits else None

    live = {"angles": {}, "forward": 0.0, "lateral": 0.0, "raw": None,
            "seq": -1, "seen": 0, "quiet": 0.0, "clipped": 0}
    if receiver is not None:
        deadline = time.time() + 120.0
        while time.time() < deadline and not live["angles"]:
            packet = receiver.poll()
            if packet is not None and packet.get("valid"):
                live["angles"] = dict(packet["a"])
                live["seq"] = packet["seq"]
                live["seen"] += 1
            if robot.step(timestep) == -1:
                return 1
        if not live["angles"]:
            print("FATAL: no live packet received", file=sys.stderr)
            return 1
        print(f"first packet received, {len(live['angles'])} joints",
              flush=True)

    first = dict(live["angles"]) if receiver is not None         else sample_angles(frames, ANGLES_START)
    ramp = max(1, int(SETTLE_SECONDS / dt))
    for index in range(ramp):
        blend = smoothstep((index + 1) / ramp)
        for name, value in first.items():
            if name in motors:
                motors[name].setPosition(float(value) * blend)
        if robot.step(timestep) == -1:
            return 1

    offsets = {side: sole_world_offset(first, side) for side in ("L", "R")}
    lowest = min(sole_lowest(first, "L"), sole_lowest(first, "R"))
    base = np.array([start[0], start[1], GROUND_Z - lowest])
    locks = {side: (base + offsets[side])[:2].copy() for side in ("L", "R")}
    planted = {"L": True, "R": True}
    seeds = {side: leg_values(first, side) for side in ("L", "R")}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "puppet_ready"), "w", encoding="utf-8") as handle:
        handle.write("1")
    if WAIT_DRIVER:
        marker = os.path.join(OUT_DIR, "driver_started")
        deadline = time.time() + 120.0
        while time.time() < deadline and not os.path.exists(marker):
            if robot.step(timestep) == -1:
                return 1

    view_field = None
    view_origin = None
    frame_index = 0
    if MOVIE:
        os.makedirs(MOVIE, exist_ok=True)
        view = robot.getFromDef("VIEW")
        if view is not None:
            view_field = view.getField("position")
            view_origin = np.array(view_field.getSFVec3f(), dtype=np.float64)

    wall_start = time.time()
    elapsed = 0.0
    trace = []
    events = []
    slide = {"L": [], "R": []}
    penetration = []
    airborne = 0
    upper_error = []
    previous_command = None
    leg_error = []
    lagging = []
    leg_rate = []
    previous_leg = None
    lift_log = []
    correction = []
    previous_world = None
    previous_measured = None
    foot_nodes = {}
    for side in ("L", "R"):
        joint = self_node.getFromProtoDef(side + "LegLax")
        if joint is None:
            continue
        field = joint.getField("endPoint")
        if field is None:
            continue
        wrapper = field.getSFNode()
        children = wrapper.getField("children")
        picked = None
        if children is not None:
            for index in range(children.getCount()):
                child = children.getMFNode(index)
                if child.getTypeName() == side + "FootSolid":
                    picked = child
                    break
        foot_nodes[side] = picked if picked is not None else wrapper
    sole_datum = {}
    real_slide = {"L": [], "R": []}
    real_height = []
    deep = []
    settled = 0
    previous_low = {"L": 0.0, "R": 0.0}
    previous_base = None
    jumps = []
    smoothed_base = None
    smoothed_height = None
    drift = None
    step_bias = np.zeros(2)
    last_planar = np.zeros(2)
    stance_since = {"L": 0.0, "R": 0.0}
    strain_since = {"L": None, "R": None}
    travel_error = []
    travel_log = []

    while robot.step(timestep) != -1:
        elapsed += dt
        clock = (time.time() - wall_start) if WALL_CLOCK else elapsed
        if clock > RUN_SECONDS:
            break
        if receiver is None and clock > frames[-1]["t"]:
            break

        if receiver is not None:
            packet = receiver.poll()
            if packet is not None and packet.get("valid"):
                live["angles"].update(packet["a"])
                wanted = np.array([float(packet.get("forward", 0.0)),
                                   float(packet.get("lateral", 0.0))])
                if live["raw"] is None:
                    live["forward"], live["lateral"] = wanted
                else:
                    span = TRAVEL_MAX_SPEED * max(dt, live["quiet"] + dt)
                    move = wanted - np.array([live["forward"],
                                              live["lateral"]])
                    reach = float(np.linalg.norm(move))
                    if reach > span:
                        move = move * (span / reach)
                        live["clipped"] += 1
                    live["forward"] += float(move[0])
                    live["lateral"] += float(move[1])
                live["raw"] = wanted
                live["seq"] = packet["seq"]
                live["seen"] += 1
                live["quiet"] = 0.0
            else:
                live["quiet"] += dt
                if live["quiet"] > LIVE_TIMEOUT:
                    print("sender went quiet, stopping", flush=True)
                    break
            targets = dict(live["angles"])
        else:
            targets = sample_angles(frames, ANGLES_START + clock)
        offsets = {side: sole_world_offset(targets, side) for side in ("L", "R")}

        wanted = []
        for side in ("L", "R"):
            if planted[side]:
                wanted.append(np.array([locks[side][0] - offsets[side][0],
                                        locks[side][1] - offsets[side][1]]))
        if wanted:
            horizontal = np.mean(np.asarray(wanted), axis=0)
        else:
            horizontal = base[:2]

        if receiver is not None:
            stream = np.array([live["forward"], live["lateral"]])
        else:
            stream = sample_travel(frames, ANGLES_START + clock)
        travel = TRAVEL_GAIN * stream
        if BASE_SERVO:
            desired = start[:2] + heading @ travel
            if drift is None:
                drift = desired - horizontal
            else:
                alpha = min(1.0, dt / max(TRAVEL_TAU, dt))
                drift = drift + alpha * ((desired - horizontal) - drift)
            room = drift_room(horizontal,
                              [s for s in ("L", "R") if planted[s]])
            allow = float(np.clip(room, 0.0, DRIFT_LIMIT))
            span = float(np.linalg.norm(drift))
            if span > allow:
                drift = drift * (allow / max(span, 1e-9))
            horizontal = horizontal + drift
        travel_log.append(travel[0])

        if smoothed_base is None:
            smoothed_base = horizontal.copy()
        else:
            alpha = min(1.0, dt / max(BASE_TAU, dt))
            change = alpha * (horizontal - smoothed_base)
            span = TRAVEL_SLEW * dt
            smoothed_base = smoothed_base + np.clip(change, -span, span)
        horizontal = smoothed_base

        wish = np.clip(STEP_GAIN * (travel - last_planar),
                       -STEP_LIMIT, STEP_LIMIT)
        step_bias += min(1.0, dt / max(STEP_TAU, dt)) * (wish - step_bias)

        lowest = min(sole_lowest(targets, "L"), sole_lowest(targets, "R"))
        height = GROUND_Z - lowest
        ceiling = height_ceiling(horizontal,
                                 [s for s in ("L", "R") if planted[s]])
        if ceiling is not None:
            height = min(height, ceiling)
        height = max(height, HEIGHT_FLOOR)
        if smoothed_height is None:
            smoothed_height = height
        else:
            change = min(1.0, dt / max(HEIGHT_TAU, dt)) * (height - smoothed_height)
            limit = HEIGHT_SLEW * dt
            smoothed_height += max(-limit, min(limit, change))
        base = np.array([horizontal[0], horizontal[1], smoothed_height])
        raw_height = {side: float((base + offsets[side])[2])
                      for side in ("L", "R")}

        worst = 0.0
        residual = {}
        for side in ("L", "R"):
            if planted[side] and LOCK_LEGS:
                goal = np.array([locks[side][0], locks[side][1],
                                 GROUND_Z + SOLE_BOTTOM])
                relative = body.T @ (goal - base)
                values, error, _ = ik[side].solve(relative, body.T @ body,
                                                  seed=seeds[side])
                seeds[side] = values
                worst = max(worst, error)
                residual[side] = error
                reference = leg_values(targets, side)
                correction.append(float(np.abs(values - reference).max()))
                for name, item in zip(ik[side].names, values):
                    targets[name] = float(item)
                offsets[side] = sole_world_offset(targets, side)

        for side in ("L", "R"):
            if planted[side]:
                continue
            local = ik[side].pose(leg_values(targets, side))[0]
            goal = local + np.array([step_bias[0], step_bias[1], 0.0])
            arm = goal - hip_local[side]
            span = REACH_MARGIN * leg_reach[side]
            length = float(np.linalg.norm(arm))
            if length > span:
                goal = hip_local[side] + arm * (span / length)
            for _ in range(2):
                values, _, _ = ik[side].solve(goal, np.eye(3), seed=seeds[side])
                seeds[side] = values
                for name, item in zip(ik[side].names, values):
                    targets[name] = float(item)
                deficit = GROUND_Z - (base[2] + sole_lowest(targets, side))
                if deficit <= 1e-4:
                    break
                goal = goal + np.array([0.0, 0.0, deficit])
            offsets[side] = sole_world_offset(targets, side)

        if previous_command is not None:
            span = COMMAND_RATE * dt
            for name, value in list(targets.items()):
                before = previous_command.get(name)
                if before is not None:
                    targets[name] = before + max(-span,
                                                 min(span, value - before))
            for side in ("L", "R"):
                offsets[side] = sole_world_offset(targets, side)
        previous_command = {name: float(value)
                            for name, value in targets.items()}

        sunk = min(base[2] + sole_lowest(targets, side) for side in ("L", "R"))
        if sunk < GROUND_Z:
            base = base + np.array([0.0, 0.0, GROUND_Z - sunk])
            smoothed_height = float(base[2])
            for side in ("L", "R"):
                if planted[side] and LOCK_LEGS:
                    goal = np.array([locks[side][0], locks[side][1],
                                     GROUND_Z + SOLE_BOTTOM])
                    values, _, _ = ik[side].solve(body.T @ (goal - base),
                                                  body.T @ body,
                                                  seed=seeds[side])
                    seeds[side] = values
                    for name, item in zip(ik[side].names, values):
                        targets[name] = float(item)
                    offsets[side] = sole_world_offset(targets, side)

        strained = []
        for side in ("L", "R"):
            error = residual.get(side)
            if error is None or error <= RELEASE_ERROR:
                strain_since[side] = None
                continue
            if strain_since[side] is None:
                strain_since[side] = clock
            if (clock - strain_since[side] >= STRAIN_HOLD
                    and clock - stance_since[side] >= MIN_STANCE):
                strained.append(side)
        if strained:
            if len(strained) > 1:
                strained = [max(strained, key=lambda s: raw_height[s])]
            for side in strained:
                planted[side] = False
                strain_since[side] = None
                events.append({"t": round(clock, 2), "foot": side,
                               "action": "release",
                               "mm": round(residual[side] * 1000, 1)})

        for side in ("L", "R"):
            if planted[side]:
                if raw_height[side] > PLANT_OFF:
                    planted[side] = False
                    events.append({"t": round(clock, 2), "foot": side,
                                   "action": "lift"})
            elif raw_height[side] < PLANT_ON:
                planted[side] = True
                stance_since[side] = clock
                strain_since[side] = None
                world = base + offsets[side]
                locks[side] = world[:2].copy()
                events.append({"t": round(clock, 2), "foot": side,
                               "action": "plant",
                               "x": round(float(world[0]), 4),
                               "y": round(float(world[1]), 4)})

        commanded = np.array([targets.get(side + suffix, 0.0)
                              for side in ("L", "R") for suffix in LEG_SUFFIX])
        actual = np.array([sensors[side + suffix].getValue()
                           if side + suffix in sensors else 0.0
                           for side in ("L", "R") for suffix in LEG_SUFFIX])
        if previous_leg is not None:
            gap = float(np.abs(previous_leg - actual).max())
            leg_error.append(gap)
            if gap > math.radians(5.0):
                lagging.append({"t": round(clock, 2),
                                "deg": round(math.degrees(gap), 1),
                                "joint": LEG_SUFFIX[int(np.argmax(np.abs(
                                    previous_leg - actual))) % 6]})
            leg_rate.append(float(np.abs(commanded - previous_leg).max()) / dt)
        previous_leg = commanded

        deviation = 0.0
        count = 0
        for name, value in targets.items():
            if name in motors:
                motors[name].setPosition(float(value))
            if name in UPPER_JOINTS and name in sensors:
                deviation += abs(float(value) - sensors[name].getValue())
                count += 1
        if count:
            upper_error.append(deviation / count)

        translation_field.setSFVec3f([float(base[0]), float(base[1]),
                                      float(base[2])])
        if view_field is not None:
            shift = view_origin + np.array([base[0] - start[0],
                                            base[1] - start[1], 0.0])
            view_field.setSFVec3f([float(shift[0]), float(shift[1]),
                                   float(shift[2])])
        if MOVIE and frame_index * 2 <= elapsed / dt:
            robot.exportImage(os.path.join(
                MOVIE, f"f{frame_index:05d}.jpg"), 88)
            frame_index += 1
        if previous_base is not None:
            jumps.append(float(np.linalg.norm(base - previous_base)))
        previous_base = base.copy()

        last_planar = heading.T @ (base[:2] - start[:2])

        if foot_nodes:
            spot = {side: np.array(node.getPosition(), dtype=np.float64)
                    for side, node in foot_nodes.items()}
            level = {}
            for side, node in foot_nodes.items():
                frame = np.array(node.getOrientation(),
                                 dtype=np.float64).reshape(3, 3)
                corners = spot[side] + (frame @ (np.array([0.048, 0.0, -0.076119])
                                                 + FOOT_CORNERS).T).T
                level[side] = float(corners[:, 2].min())
            settled += 1
            if settled < 4:
                previous_measured = spot
                real_height.append(0.0)
                continue
            worst_side = min(level, key=level.get)
            real_height.append(level[worst_side])
            if level[worst_side] < -0.010:
                deep.append({"t": round(clock, 2), "foot": worst_side,
                             "mm": round(level[worst_side] * 1000, 1),
                             "planted": bool(planted[worst_side]),
                             "soll_mm": round(float(previous_low[worst_side])
                                              * 1000, 1)})
            if previous_measured is not None:
                for side in spot:
                    if planted[side] and side in previous_measured:
                        real_slide[side].append(float(np.linalg.norm(
                            spot[side][:2] - previous_measured[side][:2])))
            previous_measured = spot

        previous_low = {side: float(base[2] + sole_lowest(targets, side))
                        for side in ("L", "R")}
        world_now = {side: base + offsets[side] for side in ("L", "R")}
        heights = [world_now[side][2] for side in ("L", "R")]
        penetration.append(min(heights))
        if min(heights) > 0.005:
            airborne += 1
        lift_log.append(abs(heights[0] - heights[1]))
        if previous_world is not None:
            for side in ("L", "R"):
                if world_now[side][2] < 0.010:
                    slide[side].append(float(np.linalg.norm(
                        world_now[side][:2] - previous_world[side][:2])))
        previous_world = world_now

        if len(trace) * LOG_EVERY <= elapsed / dt:
            planar = heading.T @ (base[:2] - start[:2])
            travel_error.append(abs(float(planar[0] - travel_log[-1])))
            trace.append({
                "want": round(float(travel_log[-1]), 4),
                "t": round(clock, 3),
                "forward": round(float(planar[0]), 4),
                "lateral": round(float(planar[1]), 4),
                "height": round(float(base[2]), 4),
                "planted": "".join(s for s in ("L", "R") if planted[s]) or "-",
                "foot_z": [round(float(world_now["L"][2]) * 1000, 2),
                           round(float(world_now["R"][2]) * 1000, 2)],
                "lift_mm": round(lift_log[-1] * 1000, 1),
                "ik_mm": round(worst * 1000, 3),
            })

    final = heading.T @ (previous_base[:2] - start[:2]) if previous_base is not None \
        else np.zeros(2)
    forwards = [s["forward"] for s in trace] or [0.0]
    heights = [s["height"] for s in trace] or [0.0]

    def total(values):
        return round(float(np.sum(values)) * 1000, 1) if values else 0.0

    def p95(values):
        return round(float(np.percentile(values, 95)) * 1000, 3) if values else 0.0

    summary = {
        "duration_s": round(elapsed, 2),
        "wall_s": round(time.time() - wall_start, 2),
        "plant_events": len(events),
        "travel_forward": round(float(final[0]), 4),
        "travel_lateral": round(float(final[1]), 4),
        "travel_range_m": round(float(max(forwards) - min(forwards)), 4),
        "travel_target_m": round(float(travel_log[-1]), 4) if travel_log else 0.0,
        "travel_error_median_mm": round(float(np.median(travel_error)) * 1000, 1)
            if travel_error else 0.0,
        "travel_error_p95_mm": round(float(np.percentile(travel_error, 95)) * 1000, 1)
            if travel_error else 0.0,
        "pelvis_height_min": round(float(min(heights)), 4),
        "pelvis_height_max": round(float(max(heights)), 4),
        "slide_total_L_mm": total(slide["L"]),
        "slide_total_R_mm": total(slide["R"]),
        "slide_p95_L_mm": p95(slide["L"]),
        "slide_p95_R_mm": p95(slide["R"]),
        "penetration_min_mm": round(float(min(penetration)) * 1000, 2)
            if penetration else 0.0,
        "penetration_below_5mm_percent": round(100.0 * float(np.mean(
            np.asarray(penetration) < -0.005)), 2) if penetration else 0.0,
        "airborne_percent": round(100.0 * airborne / max(len(penetration), 1), 2),
        "foot_lift_p90_mm": round(float(np.percentile(lift_log, 90)) * 1000, 1)
            if lift_log else 0.0,
        "foot_lift_max_mm": round(float(max(lift_log)) * 1000, 1) if lift_log else 0.0,
        "leg_correction_median_deg": round(float(np.degrees(
            np.median(correction))), 3) if correction else 0.0,
        "leg_correction_p95_deg": round(float(np.degrees(
            np.percentile(correction, 95))), 3) if correction else 0.0,
        "upper_error_median_deg": round(float(np.degrees(np.median(upper_error))), 4)
            if upper_error else 0.0,
        "upper_error_p95_deg": round(float(np.degrees(
            np.percentile(upper_error, 95))), 4) if upper_error else 0.0,
        "real_slide_L_mm": total(real_slide["L"]),
        "real_slide_R_mm": total(real_slide["R"]),
        "real_slide_p95_L_mm": p95(real_slide["L"]),
        "real_slide_p95_R_mm": p95(real_slide["R"]),
        "real_ground_min_mm": round(float(min(real_height)) * 1000, 2)
            if real_height else 0.0,
        "real_ground_max_mm": round(float(max(real_height)) * 1000, 2)
            if real_height else 0.0,
        "real_airborne_percent": round(100.0 * float(np.mean(
            np.asarray(real_height) > 0.005)), 2) if real_height else 0.0,
        "foot_nodes": len(foot_nodes),
        "lag_count": len(lagging),
        "lag_worst": sorted(lagging, key=lambda e: -e["deg"])[:10],
        "lag_after_2s": sum(1 for e in lagging if e["t"] > 2.0),
        "leg_error_median_deg": round(float(np.degrees(np.median(leg_error))), 3)
            if leg_error else 0.0,
        "leg_error_p95_deg": round(float(np.degrees(
            np.percentile(leg_error, 95))), 3) if leg_error else 0.0,
        "leg_error_max_deg": round(float(np.degrees(max(leg_error))), 3)
            if leg_error else 0.0,
        "leg_rate_p95_rad_s": round(float(np.percentile(leg_rate, 95)), 3)
            if leg_rate else 0.0,
        "leg_rate_max_rad_s": round(float(max(leg_rate)), 3) if leg_rate else 0.0,
        "deep_count": len(deep),
        "deep_planted_percent": round(100.0 * float(np.mean(
            [d["planted"] for d in deep])), 1) if deep else 0.0,
        "deep_worst": sorted(deep, key=lambda d: d["mm"])[:12],
        "base_jump_median_mm": round(float(np.median(jumps)) * 1000, 3)
            if jumps else 0.0,
        "base_jump_max_mm": round(float(max(jumps)) * 1000, 3) if jumps else 0.0,
        "events": events,
        "trace": trace,
    }
    if receiver is not None:
        summary["live_packets"] = live["seen"]
        summary["live_travel_clipped"] = live["clipped"]
        receiver.close()
    with open(os.path.join(OUT_DIR, "puppet_run.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for path in ("puppet_ready", "driver_started"):
        try:
            os.remove(os.path.join(OUT_DIR, path))
        except OSError:
            pass
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("trace", "events")}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "puppet_error.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
