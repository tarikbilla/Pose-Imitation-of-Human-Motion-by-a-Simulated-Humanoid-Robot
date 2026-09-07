import json
import math
import os
import sys

import numpy as np
from controller import Supervisor

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "..", ".."))
sys.path.insert(0, A3_ROOT)
sys.path.insert(0, os.path.join(A3_ROOT, "tools"))

import mass_tables
from kinematics.atlaskin import SEGMENT_OF, AtlasModel
from kinematics.gait import plan_walk, swing_pose
from kinematics.legik import LegIK
from kinematics.lipm import Phase, WalkPattern

OUT_DIR = os.path.join(A3_ROOT, "results")

CROUCH = float(os.environ.get("A3_WALK_CROUCH", "0.45"))
SETTLE_SECONDS = float(os.environ.get("A3_WALK_SETTLE", "1.5"))
RAMP_SECONDS = 1.5
STEP_WIDTH = float(os.environ.get("A3_STEP_WIDTH", "0.178"))
SINGLE_SUPPORT = float(os.environ.get("A3_SINGLE_S", "0.62"))
DOUBLE_SUPPORT = float(os.environ.get("A3_DOUBLE_S", "0.20"))
CLEARANCE = float(os.environ.get("A3_CLEARANCE", "0.045"))
PREVIEW = int(os.environ.get("A3_PREVIEW", "4"))

DCM_GAIN = float(os.environ.get("A3_DCM_GAIN", "0.3"))
DCM_SIGN = float(os.environ.get("A3_DCM_SIGN", "1.0"))
DCM_LIMIT = float(os.environ.get("A3_DCM_LIMIT", "0.06"))
STEP_LIMIT = float(os.environ.get("A3_STEP_LIMIT", "0.30"))
ADAPT_GAIN = float(os.environ.get("A3_ADAPT_GAIN", "0.0"))
ADAPT_LIMIT = float(os.environ.get("A3_ADAPT_LIMIT", "0.05"))

GAIT_SCRIPT = os.environ.get("A3_GAIT_SCRIPT", "0:0.20")
RUN_SECONDS = float(os.environ.get("A3_RUN_SECONDS", "20.0"))

VELOCITY_TAU = 0.05
JOINT_VELOCITY = 4.0
FALL_FRACTION = float(os.environ.get("A3_FALL_FRACTION", "0.60"))
LOG_EVERY = int(os.environ.get("A3_LOG_EVERY", "25"))
MASS_PROFILE = os.environ.get("A3_MASS_PROFILE", "core")


def parse_script(text):
    entries = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        when, _, value = item.partition(":")
        entries.append((float(when), float(value)))
    entries.sort()
    return entries or [(0.0, 0.0)]


def script_value(entries, time):
    value = entries[0][1]
    for when, item in entries:
        if time >= when:
            value = item
        else:
            break
    return value


def apply_profile(model, variant):
    table = mass_tables.build(variant)
    for joint in model.names:
        segment = SEGMENT_OF.get(joint)
        entry = table.get(segment) if segment else None
        model.mass[joint] = entry["mass"] if entry else 0.0
        model.com_local[joint] = (np.asarray(entry["com"], dtype=np.float64)
                                  if entry else np.zeros(3))
    head = table[mass_tables.HEAD_SEGMENT]
    model.mass["NeckAy"] = head["mass"]
    model.com_local["NeckAy"] = np.asarray(head["com"], dtype=np.float64)
    pelvis = table["PelvisSolid"]
    model.root_mass = pelvis["mass"]
    model.root_com = np.asarray(pelvis["com"], dtype=np.float64)
    model.total_mass = model.root_mass + sum(model.mass.values())
    return model


def matrix(values):
    return np.array(values, dtype=np.float64).reshape(3, 3)


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0
    self_node = robot.getSelf()

    model = apply_profile(AtlasModel(), MASS_PROFILE)
    ik = {"L": LegIK(model, "L"), "R": LegIK(model, "R")}
    script = parse_script(GAIT_SCRIPT)

    motors = {}
    for name in model.names:
        motor = robot.getDevice(name)
        if motor is None:
            continue
        motor.setVelocity(JOINT_VELOCITY)
        motors[name] = motor
        sensor = robot.getDevice(name + "S")
        if sensor is not None:
            sensor.enable(timestep)

    stance = np.array([0.0, 0.0, -CROUCH, 2.0 * CROUCH, -CROUCH, 0.0])
    hold = {}
    for side in ("L", "R"):
        for name, value in zip(ik[side].names, stance):
            hold[name] = float(value)

    ramp = max(1, int(RAMP_SECONDS / dt))
    for step in range(ramp):
        blend = (step + 1) / ramp
        for name, value in hold.items():
            motors[name].setPosition(value * blend)
        if robot.step(timestep) == -1:
            return 1
    for _ in range(max(1, int(SETTLE_SECONDS / dt))):
        if robot.step(timestep) == -1:
            return 1

    angles = {name: 0.0 for name in model.names}
    angles.update(hold)
    pelvis = np.array(self_node.getPosition(), dtype=np.float64)
    rotation = matrix(self_node.getOrientation())

    feet_world = {}
    for side in ("L", "R"):
        offset, _ = ik[side].pose(np.array([angles[n] for n in ik[side].names]))
        feet_world[side] = pelvis + rotation @ offset

    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    heading = np.array([[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]])
    heading_t = heading.T
    foot_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                              [math.sin(yaw), math.cos(yaw), 0.0],
                              [0.0, 0.0, 1.0]])
    origin = 0.5 * (feet_world["L"][:2] + feet_world["R"][:2])

    def to_local(point):
        return heading_t @ (np.asarray(point[:2], dtype=np.float64) - origin)

    def to_world(point):
        return origin + heading @ np.asarray(point[:2], dtype=np.float64)

    feet = {}
    for side in ("L", "R"):
        planar = to_local(feet_world[side])
        feet[side] = np.array([planar[0], planar[1], feet_world[side][2]])

    com0 = np.array(self_node.getCenterOfMass(), dtype=np.float64)
    ground = min(feet_world["L"][2], feet_world["R"][2])
    com_height = float(com0[2] - ground)
    fall_height = FALL_FRACTION * float(com0[2])
    omega = math.sqrt(9.81 / com_height)

    def make_plan(anchor, length, first):
        phases, landings = plan_walk(
            {"L": anchor["L"][:2], "R": anchor["R"][:2]},
            length, STEP_WIDTH, PREVIEW,
            single=SINGLE_SUPPORT, double=DOUBLE_SUPPORT,
            first=first, settle=DOUBLE_SUPPORT, taper_first=False)
        built = WalkPattern(com_height, omega).plan(phases)
        index = {}
        for order, (swing, target, phase) in enumerate(landings):
            index[id(phase)] = (swing, target, order)
        return built, index

    def make_stand(anchor):
        centre = 0.5 * (np.asarray(anchor["L"][:2], dtype=np.float64)
                        + np.asarray(anchor["R"][:2], dtype=np.float64))
        built = WalkPattern(com_height, omega).plan(
            [Phase(centre, 600.0, None, "LR", False)])
        return built, {}

    print(f"profile={MASS_PROFILE} com_in_pelvis={np.round(model.com(angles), 4)} "
          f"com_height={com_height:.4f} omega={omega:.4f} preview={PREVIEW} "
          f"script={script}")
    sys.stdout.flush()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "walk_ready"), "w", encoding="utf-8") as handle:
        handle.write("1")

    swing_side = "R"
    commanded = float(np.clip(script_value(script, 0.0), -STEP_LIMIT, STEP_LIMIT))
    walking = abs(commanded) > 1e-6
    pattern, plan_index = (make_plan(feet, commanded, swing_side) if walking
                           else make_stand(feet))

    seeds = {side: np.array([angles[n] for n in ik[side].names])
             for side in ("L", "R")}
    swing_origin = {}
    swing_target = {}
    previous_com = com0.copy()
    com_velocity = np.zeros(3)
    com_state = None
    elapsed = 0.0
    plan_time = 0.0
    fallen = False
    trace = []
    steps_done = 0
    replans = 0
    step_marks = []

    while robot.step(timestep) != -1:
        elapsed += dt
        plan_time += dt
        if elapsed > RUN_SECONDS:
            break

        com = np.array(self_node.getCenterOfMass(), dtype=np.float64)
        raw = (com - previous_com) / dt
        previous_com = com
        blend = dt / max(VELOCITY_TAU, dt)
        com_velocity += blend * (raw - com_velocity)

        pelvis = np.array(self_node.getPosition(), dtype=np.float64)
        rotation = matrix(self_node.getOrientation())

        com_planar = to_local(com)
        velocity_planar = heading_t @ com_velocity[:2]
        dcm_measured = com_planar + velocity_planar / omega
        dcm_reference = pattern.dcm(plan_time)
        dcm_error = dcm_measured - dcm_reference

        phase, local = pattern.phase_at(plan_time)
        if com_state is None:
            com_state = com_planar.copy()
        com_state = com_state + omega * (dcm_reference - com_state) * dt

        correction = np.clip(DCM_SIGN * DCM_GAIN * dcm_error,
                             -DCM_LIMIT, DCM_LIMIT)
        com_target = np.array([com_state[0] - correction[0],
                               com_state[1] - correction[1],
                               ground + com_height])

        entry = plan_index.get(id(phase))
        if entry is not None and phase.lift:
            swing, target, _ = entry
            if swing not in swing_origin:
                swing_origin[swing] = feet[swing].copy()
                shift = np.clip(ADAPT_GAIN * dcm_error,
                                -ADAPT_LIMIT, ADAPT_LIMIT)
                swing_target[swing] = np.array([target[0] + shift[0],
                                                target[1] + shift[1], ground])
            fraction = local / phase.duration
            feet[swing] = swing_pose(swing_origin[swing], swing_target[swing],
                                     fraction, CLEARANCE)
        elif swing_origin:
            landed = list(swing_origin)[0]
            feet[landed] = swing_target[landed].copy()
            feet[landed][2] = ground
            swing_origin.clear()
            swing_target.clear()
            steps_done += 1
            step_marks.append({"t": round(elapsed, 2), "side": landed,
                               "x": round(float(feet[landed][0]), 4),
                               "length": round(commanded, 3)})
            swing_side = "L" if landed == "R" else "R"
            commanded = float(np.clip(script_value(script, elapsed),
                                      -STEP_LIMIT, STEP_LIMIT))
            if abs(commanded) > 1e-6:
                walking = True
                pattern, plan_index = make_plan(feet, commanded, swing_side)
            else:
                walking = False
                pattern, plan_index = make_stand(feet)
            plan_time = 0.0
            replans += 1
        elif not walking:
            wanted = script_value(script, elapsed)
            if abs(wanted) > 1e-6:
                commanded = float(np.clip(wanted, -STEP_LIMIT, STEP_LIMIT))
                walking = True
                pattern, plan_index = make_plan(feet, commanded, swing_side)
                plan_time = 0.0
                replans += 1

        planar_target = to_world(com_target)
        pelvis_target = np.array([planar_target[0], planar_target[1],
                                  com_target[2]]) - rotation @ model.com(angles)
        transpose = rotation.T
        worst = 0.0
        for side in ("L", "R"):
            planar_foot = to_world(feet[side])
            foot_world = np.array([planar_foot[0], planar_foot[1], feet[side][2]])
            relative = transpose @ (foot_world - pelvis_target)
            values, position_error, _ = ik[side].solve(
                relative, transpose @ foot_rotation, seed=seeds[side])
            seeds[side] = values
            worst = max(worst, position_error)
            for name, value in zip(ik[side].names, values):
                angles[name] = float(value)
                motors[name].setPosition(float(value))

        if len(trace) * LOG_EVERY <= elapsed / dt:
            trace.append({
                "t": round(elapsed, 3),
                "com": [round(float(v), 4) for v in com_planar],
                "com_ref": [round(float(v), 4) for v in com_state],
                "dcm_err": round(float(np.linalg.norm(dcm_error)), 4),
                "com_err": round(float(np.linalg.norm(com_planar - com_state)), 4),
                "cmd": round(commanded, 3),
                "phase": phase.support if phase else "-",
                "swing": phase.swing if phase else None,
                "feet": [round(float(feet["L"][2]), 4),
                         round(float(feet["R"][2]), 4)],
                "contacts": len(self_node.getContactPoints(True)),
                "ik_mm": round(worst * 1000, 3),
            })

        if com[2] < fall_height:
            fallen = True
            print("FALLEN")
            break

    final = np.array(self_node.getCenterOfMass(), dtype=np.float64)
    summary = {
        "fallen": fallen,
        "duration_s": round(elapsed, 2),
        "profile": MASS_PROFILE,
        "preview": PREVIEW,
        "adapt_gain": ADAPT_GAIN,
        "script": GAIT_SCRIPT,
        "steps_done": steps_done,
        "replans": replans,
        "com_height": round(com_height, 4),
        "omega": round(omega, 4),
        "travel_forward": round(float(to_local(final)[0] - to_local(com0)[0]), 4),
        "travel_lateral": round(float(to_local(final)[1] - to_local(com0)[1]), 4),
        "com_error_median": round(float(np.median([s["com_err"] for s in trace]))
                                  if trace else 0.0, 4),
        "dcm_error_median": round(float(np.median([s["dcm_err"] for s in trace]))
                                  if trace else 0.0, 4),
        "dcm_error_max": round(float(max([s["dcm_err"] for s in trace]))
                               if trace else 0.0, 4),
        "ik_error_max_mm": round(max([s["ik_mm"] for s in trace]) if trace else 0.0, 3),
        "steps": step_marks,
        "trace": trace,
    }
    with open(os.path.join(OUT_DIR, "walk_run.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    try:
        os.remove(os.path.join(OUT_DIR, "walk_ready"))
    except OSError:
        pass
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("trace", "steps")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
