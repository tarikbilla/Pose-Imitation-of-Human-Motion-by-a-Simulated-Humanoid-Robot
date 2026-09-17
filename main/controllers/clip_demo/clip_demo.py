"""Play the whole certified clip library back to back. No camera, no pipeline.

This is the live counterpart to ``scripts/build_motion_clips.py``. That script
certifies each clip against NAO's mass model offline and refuses to write one
that fails; this controller runs the survivors on the real simulated robot, in
order, and reports what actually happened. The certificate is a static and
quasi-static argument -- it can prove a clip WILL fall, never that it will not --
so this is the half that the maths cannot supply.

Deliberately standalone. It opens no socket, imports nothing from the imitation
controller and needs no human in front of a camera, so a failure here is a
property of the clip and of the robot, with nothing else in the frame to blame.

Why it does not use Webots' ``Motion`` class
--------------------------------------------
``.motion`` playback through ``Motion`` has a history of quiet failures on this
project: ``Motion.play()`` returns ``None`` in the R2025a Python binding (so any
``if not motion.play()`` reads as a refusal while the clip is in fact running),
and ``Motion.isValid()`` compares two freshly-allocated pointers and is therefore
always ``True``. Both are Webots-binding quirks rather than anything about the
clips, and neither belongs in a test of the clips. So the keyframes are read
straight out of the file and interpolated onto the simulation step here. What
runs is then exactly the data the certifier certified.

Reading the output
------------------
Each clip prints its own line: how long it took, how far the robot travelled,
how much it turned, and the lowest its torso got. Torso height is the fall
detector -- deliberately the crudest one available, because it needs no model, no
sensor calibration and no assumption about which way is down. A standing NAO's
torso origin sits about 0.33 m up; one on the floor is nearer 0.10 m.
"""
import os
import sys

LIB = os.path.join(os.path.dirname(__file__), "..", "..", "libraries")
sys.path.insert(0, os.path.abspath(LIB))

from controller import Supervisor  # noqa: E402  (Webots supplies this)

from pose_control_utils import get_default_motor_configs  # noqa: E402
from walk_motion import motion_poses  # noqa: E402

MOTIONS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "pose_imitation_controller", "motions"))

# The running order. Quasi-static clips first: they are the ones this project
# generated from scratch and the ones with the strongest guarantee, so if
# anything is wrong with the rig itself it shows up before the dynamic gaits get
# a chance to muddy it.
PROGRAMME = [
    ("Squat", "generated, quasi-static"),
    ("RaiseLegLeft", "generated, quasi-static"),
    ("RaiseLegRight", "generated, quasi-static"),
    ("Forwards", "refined, one-shot walk"),
    ("Forwards50", "refined, continuous gait"),
    ("Backwards", "refined; freed 10 upper-body joints"),
    ("TurnLeft40", "refined"),
    ("TurnRight40", "refined"),
    ("TurnLeft60", "refined"),
    ("TurnRight60", "refined"),
    ("TurnLeft180", "refined; freed 8 upper-body joints"),
    ("TurnRight180", "MIRRORED -- Webots ships no right about-face"),
    ("SideStepLeft", "refined; freed 8 upper-body joints"),
    ("SideStepRight", "refined; freed 8 upper-body joints"),
]

SETTLE_S = 1.5          # pause between clips, so each starts from rest
FALLEN_TORSO_M = 0.20   # a standing NAO's torso origin is ~0.33 m up


def interpolate(poses, t):
    """Joint angles at time ``t``, linearly between the surrounding keyframes.

    The clips are 40 ms apart and the simulation steps at 20 ms, so half the
    steps fall between keyframes. Holding the previous keyframe instead would
    turn every clip into a staircase and hand the motors a velocity step twice
    per keyframe -- which is exactly the thing the certifier's velocity check
    exists to keep out.
    """
    if t <= poses[0][0]:
        return poses[0][1]
    if t >= poses[-1][0]:
        return poses[-1][1]
    lo = 0
    hi = len(poses) - 1
    while hi - lo > 1:                      # binary search: clips run to 280 kf
        mid = (lo + hi) // 2
        if poses[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, a), (t1, b) = poses[lo], poses[hi]
    span = t1 - t0
    s = 0.0 if span <= 1e-9 else (t - t0) / span
    return {j: a[j] + (b.get(j, a[j]) - a[j]) * s for j in a}


class Demo:
    def __init__(self):
        # Supervisor rather than Robot: the demo needs getSelf() to measure how
        # far the robot actually travelled and turned, which is the only honest
        # way to tell a walk clip from a clip that shuffles on the spot. The
        # world already grants it.
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.configs = get_default_motor_configs()

        self.motors = {}
        for name, cfg in self.configs.items():
            device = self.robot.getDevice(name)
            if device is None:
                continue
            self.motors[name] = device
            # Full rated speed. The imitation controller throttles the legs on
            # purpose (a jolt there moves the mass off the feet), but a clip's
            # balance was computed FROM its keyframe timing -- throttled motors
            # arrive late and the robot is then somewhere the clip never
            # intended. Every clip here is certified to stay inside 90% of these
            # ceilings, so handing over the full figure is safe by construction.
            #
            # Taken from the DEVICE, not from our table: the proto's own figure
            # is 8.26797 rad/s where the table rounds to 8.27, and Webots
            # rejects the request with a warning and keeps the previous value --
            # so trusting the table would leave the motor at whatever it had.
            try:
                ceiling = float(device.getMaxVelocity())
            except Exception:  # noqa: BLE001
                ceiling = cfg.max_velocity
            device.setVelocity(min(cfg.max_velocity, ceiling))

        self.node = self.robot.getSelf()
        print(f"[demo] {len(self.motors)}/{len(self.configs)} motors, "
              f"timestep {self.timestep} ms")
        print(f"[demo] clips from {MOTIONS}")

    # -- robot state -------------------------------------------------------
    def torso_height(self):
        """Metres from the floor to the torso origin, or None without supervisor.

        The fall detector, and deliberately the crudest possible one: it needs no
        model, no sensor calibration and no assumption about which way is down. A
        NAO standing has its torso origin about 0.33 m up; one on the floor has
        it around 0.10 m. Anything in between is on its way to one or the other.
        """
        if self.node is None:
            return None
        return float(self.node.getPosition()[2])

    def ground_pose(self):
        """``(x, y, yaw)`` of the robot in the world, or ``None``."""
        if self.node is None:
            return None
        import math
        position = self.node.getPosition()
        m = self.node.getOrientation()
        return (float(position[0]), float(position[1]),
                math.atan2(float(m[3]), float(m[0])))

    # -- driving -----------------------------------------------------------
    def hold(self, pose, seconds):
        """Command ``pose`` and let the simulation run for ``seconds``."""
        end = self.robot.getTime() + seconds
        lowest = 1e9
        while self.robot.getTime() < end:
            for joint, angle in pose.items():
                motor = self.motors.get(joint)
                if motor is not None:
                    motor.setPosition(angle)
            if self.robot.step(self.timestep) == -1:
                return False, lowest
            height = self.torso_height()
            if height is not None:
                lowest = min(lowest, height)
        return True, lowest

    def rest_pose(self):
        """Every joint at its neutral. The arms are parked here for the whole
        demo: with no imitation running they would otherwise hang wherever the
        last command left them, and a clip that is meant to be legs-only is
        easier to judge against arms that are visibly not moving."""
        return {name: cfg.rest_angle for name, cfg in self.configs.items()}

    def play(self, name):
        """Run one clip. Returns a dict of what happened."""
        path = os.path.join(MOTIONS, f"{name}.motion")
        if not os.path.isfile(path):
            return {"name": name, "status": "MISSING"}
        poses = motion_poses(path)
        if len(poses) < 2:
            return {"name": name, "status": "UNREADABLE"}

        before = self.ground_pose()
        duration = poses[-1][0]
        start = self.robot.getTime()
        lowest = 1e9
        while True:
            t = self.robot.getTime() - start
            if t > duration:
                break
            for joint, angle in interpolate(poses, t).items():
                motor = self.motors.get(joint)
                if motor is not None:
                    motor.setPosition(angle)
            if self.robot.step(self.timestep) == -1:
                return {"name": name, "status": "ABORTED"}
            height = self.torso_height()
            if height is not None:
                lowest = min(lowest, height)

        after = self.ground_pose()
        result = {
            "name": name,
            "status": "fell" if lowest < FALLEN_TORSO_M else "ok",
            "took_s": self.robot.getTime() - start,
            "min_torso_m": lowest,
        }
        if before and after:
            import math
            result["travel_m"] = math.dist(before[:2], after[:2])
            turned = math.degrees(after[2] - before[2])
            result["turned_deg"] = (turned + 180.0) % 360.0 - 180.0
        return result

    # -- the programme -----------------------------------------------------
    def run(self):
        print("[demo] standing up ...")
        rest = self.rest_pose()
        alive, _ = self.hold(rest, 2.0)
        if not alive:
            return

        results = []
        for index, (name, note) in enumerate(PROGRAMME, start=1):
            print(f"\n[demo] {index}/{len(PROGRAMME)}  {name}  ({note})")
            result = self.play(name)
            results.append(result)
            if result["status"] in ("MISSING", "UNREADABLE"):
                print(f"[demo]    {result['status']}: build it with "
                      "scripts/build_motion_clips.py")
                continue
            if result["status"] == "ABORTED":
                print("[demo]    simulation stopped")
                return
            print(f"[demo]    {result['status']}  {result['took_s']:5.2f}s"
                  + (f"  travelled {result.get('travel_m', 0)*100:5.1f} cm"
                     f"  turned {result.get('turned_deg', 0):+6.1f} deg"
                     if "travel_m" in result else "")
                  + f"  min torso {result['min_torso_m']:.3f} m")

            # Back to rest between clips. Every clip in this library already
            # ENDS at the standing pose, so this is a pause rather than a
            # recovery -- and if the robot does not actually settle here, that
            # is itself the finding.
            alive, _ = self.hold(rest, SETTLE_S)
            if not alive:
                return

        print("\n" + "=" * 78)
        print(f"{'clip':<16s} {'result':<7s} {'time':>7s} {'travel':>9s} "
              f"{'turn':>9s} {'min torso':>9s}")
        print("-" * 78)
        for result in results:
            if "took_s" not in result:
                print(f"{result['name']:<16s} {result['status']}")
                continue
            print(f"{result['name']:<16s} {result['status']:<7s} "
                  f"{result['took_s']:6.2f}s "
                  f"{result.get('travel_m', 0)*100:8.1f}cm "
                  f"{result.get('turned_deg', 0):+8.1f}d "
                  f"{result['min_torso_m']:8.3f}m")
        print("=" * 78)
        fell = [r["name"] for r in results if r.get("status") == "fell"]
        print(f"{len(results) - len(fell)}/{len(results)} clips completed upright"
              + (f"; FELL during: {', '.join(fell)}" if fell else ""))
        print("[demo] finished -- the robot will now stand still.")
        while self.robot.step(self.timestep) != -1:
            for joint, angle in rest.items():
                motor = self.motors.get(joint)
                if motor is not None:
                    motor.setPosition(angle)


if __name__ == "__main__":
    Demo().run()
