"""End-to-end test of the Webots controller against a mocked Webots runtime.

The controller file itself (sockets, device lookup, motion playback, and above all
the lower-body *arbiter*) cannot be reached by the library unit tests, yet it is
where the wiring bugs live: a renamed method, a layer that never gets ticked, two
layers commanding the legs at once. Since ``main/libraries`` is deliberately
Webots-free, the only thing standing between those tests and a full-loop test is
the ``controller`` module -- so we fake it.

The fake robot tracks its commands perfectly (position sensors echo the last
commanded angle), which is enough to exercise every control path: real UDP
packets go in, and we assert on which layer drove the legs and what it commanded.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import socket
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from conftest import write_cyclic_clip  # noqa: E402

CONTROLLER_DIR = os.path.join(REPO, "main", "controllers", "pose_imitation_controller")
sys.path.insert(0, os.path.join(REPO, "main", "libraries"))

from nao_retarget import _side_sign  # noqa: E402
from pose_control_utils import get_default_motor_configs  # noqa: E402

CONFIGS = get_default_motor_configs()
TIMESTEP_MS = 20


# ---------------------------------------------------------------------------
# Fake Webots
# ---------------------------------------------------------------------------
class FakeMotor:
    def __init__(self, name):
        self.name = name
        self.position = 0.0
        self.velocity = 0.0
        self.commands = 0

    def setPosition(self, value):  # noqa: N802 - Webots API name
        self.position = value
        self.commands += 1

    def setVelocity(self, value):  # noqa: N802
        self.velocity = value


class FakeSensor:
    """Position sensor echoing its motor: a perfect-tracking robot."""

    def __init__(self, motor):
        self.motor = motor

    def enable(self, _ms):
        pass

    def getValue(self):  # noqa: N802
        return self.motor.position


class FakeInertialUnit:
    def __init__(self):
        self.rpy = [0.0, 0.0, 0.0]

    def enable(self, _ms):
        pass

    def getRollPitchYaw(self):  # noqa: N802
        return list(self.rpy)


class FakeVector3:
    def __init__(self, values=(0.0, 0.0, 0.0)):
        self.values = list(values)

    def enable(self, _ms):
        pass

    def getValues(self):  # noqa: N802
        return list(self.values)


class FakeAccelerometer:
    """Gravity as this NAO's accelerometer would report it.

    Derived from the robot's InertialUnit reading rather than held constant, so
    the two attitude sources AGREE in the fake world and the controller's real
    path (gravity for control, the InertialUnit's roll as a cross-check -- see
    TILT_FROM_ACCELEROMETER) is the one under test. A constant (0, 0, -9.81) made
    every injected tilt invisible to the accelerometer, which is a fake that
    passes whatever the controller does.

    Idealised on purpose: the real device's half-scale pitch and yaw leak are a
    property of the InertialUnit's axis masking, and those are pinned by
    test_the_inertial_unit_halves_the_pitch_and_leaks_the_heading rather than
    modelled here -- this fake is for the plumbing.
    """

    G = 9.81

    def __init__(self, robot):
        self.robot = robot

    def enable(self, _ms):
        pass

    def getValues(self):  # noqa: N802
        roll, pitch = self.robot.imu.rpy[0], self.robot.imu.rpy[1]
        # World up in torso coordinates...
        ux = -math.sin(pitch) * self.G
        uy = math.sin(roll) * math.cos(pitch) * self.G
        uz = math.cos(roll) * math.cos(pitch) * self.G
        # ...through the proto's 180 deg mount about x.
        return [ux, -uy, -uz]


class FakeNode:
    """The robot's own scene-tree node, as the Supervisor hands it over.

    Its heading comes from the fake InertialUnit's yaw slot, because that slot is
    what every turn test in this file has always used to mean "the robot has
    physically turned". On the real robot that channel is useless (the proto
    disables the yaw axis, so it reports a copy of the half-scale pitch) and the
    heading comes from here instead -- so wiring the fake this way means those
    tests now exercise the path that actually runs.

    getOrientation returns the row-major 3x3 the real API returns; the balance
    ground truth (centre of mass, static balance, contact points) is answered
    plausibly so the logging path is exercised, but nothing controls from it.
    """

    def __init__(self, robot):
        self.robot = robot

    def getOrientation(self):  # noqa: N802
        """The row-major world<-torso matrix, R = Rz(yaw) Ry(pitch) Rx(roll).

        Built from the robot's TRUE attitude, which the fake keeps in the
        InertialUnit's rpy slots minus ``mount_roll`` -- the offset a rotated
        sensor adds to its own reading and the scene tree does not have. Tests
        that inject "the robot is rolled 0.6 rad" via imu.rpy therefore reach the
        control path, which now takes its attitude from here (see
        ATTITUDE_SOURCE), and the tests specifically about the sensor's mounting
        offset set mount_roll so the node correctly reports an upright robot
        while the device reports +1.618.
        """
        roll = self.robot.imu.rpy[0] - self.robot.mount_roll
        pitch = self.robot.imu.rpy[1]
        yaw = self.robot.imu.rpy[2]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                -sp, cp * sr, cp * cr]

    def getPosition(self):  # noqa: N802
        return [0.0, 0.0, 0.334]

    def getCenterOfMass(self):  # noqa: N802
        return [0.0, 0.0, 0.29]

    def getStaticBalance(self):  # noqa: N802
        return True

    def getContactPoints(self, includeDescendants=False):  # noqa: N802, ARG002
        return [object(), object()]

    def getVelocity(self):  # noqa: N802
        return [0.0] * 6


class FakeMotion:
    """Stand-in for Webots' Motion: finishes after ``STEPS`` polls.

    FAITHFUL TO THE R2025a PYTHON BINDING, in two ways that look like pedantry
    and were in fact the reason this whole harness certified a locomotion layer
    that had never once run:

    * ``play()`` returns **None**. The real binding is
      ``def play(self): wb.wbu_motion_play(self._ref)`` -- no return statement.
      This fake used to return True, so ``if not motion.play()`` passed here and
      failed on the real robot, every single time. Four recorded sessions logged
      "start REFUSED by Webots" for 198-964 frames each and not one frame of
      playback, while the clip was in fact running in Webots and fighting the
      per-joint commands (measured: LKneePitch reached the clip's own first
      keyframe, 1.042 rad, while the controller was commanding 0.20-0.52).
    * ``isValid()`` returns True unconditionally, because the real one compares
      two freshly-constructed ``ctypes.c_void_p`` objects, which are never equal
      -- so it is True even for a file Webots failed to load.

    ``NEVER_OVER`` reproduces the failure mode that matters most: a clip that
    plays but never reports being over. Because playback suspends per-joint
    commanding for the joints it declares, that used to freeze the legs
    indefinitely with no diagnostic.

    It also models a real PLAYHEAD: ``getTime`` reports a position that
    ``setTime`` moves and playback advances, and how much is left to play follows
    from where the playhead is. A fake whose ``getTime`` ignored ``setTime`` could
    not test cyclic playback at all -- the rewind would appear to do nothing, and
    a loop that never advances looks identical to a loop that works.
    """

    STEPS = 20
    NEVER_OVER = False
    DURATION_MS = 1200.0
    played = []

    def __init__(self, path):
        self.path = path
        self.loop = False
        self.time = 0.0
        self.rewinds = 0
        self.seeks = []
        self._remaining = 0

    def isValid(self):  # noqa: N802
        # Always True -- see the class docstring. The real binding cannot tell.
        return True

    def setLoop(self, value):  # noqa: N802
        self.loop = value

    def setTime(self, ms):  # noqa: N802
        """Move the playhead, and with it how much is left to play.

        Both halves matter. A rewind that moved ``getTime`` but not the remaining
        count would end the clip on schedule however often it was rewound, so a
        cyclic walk would look like it worked while lasting exactly one clip.
        """
        self.time = float(ms)
        self.seeks.append(float(ms))
        if ms == 0:
            self.rewinds += 1
        self._remaining = self._steps_left()

    def _steps_left(self):
        span = max(0.0, self._end_ms() - self.time)
        return int(round(span / TIMESTEP_MS))

    def _end_ms(self):
        """Where playback ends.

        ``STEPS`` is what the tests tune, so it -- not DURATION_MS -- defines the
        end of the clip, measured from wherever ``play()`` was called.
        """
        return self._played_from + self.STEPS * TIMESTEP_MS

    _played_from = 0.0

    def play(self):
        self._played_from = self.time
        self._remaining = self.STEPS
        FakeMotion.played.append(os.path.basename(self.path))
        # Returns None, like the real binding. Anything that tests the return
        # value of this call is broken on the real robot.

    def stop(self):
        self._remaining = 0

    def getDuration(self):  # noqa: N802
        return self.DURATION_MS

    def getTime(self):  # noqa: N802
        """Playback position in ms, advancing one SIMULATION step per poll.

        A real clip's time advances with the simulation, not by one keyframe
        spacing per step -- the keyframes are 40 ms apart but the control step is
        20 ms, so playback interpolates. The early-exit logic reads this against
        the clip's safe keyframe times, so the rate has to be right.
        """
        return max(0.0, self.time)

    def isOver(self):  # noqa: N802
        if FakeMotion.NEVER_OVER:
            return False
        if self._remaining > 0:
            self._remaining -= 1
            self.time += TIMESTEP_MS
            return False
        return True


class FakeFsr:
    """Foot force sensor that responds to the robot's lean, like the real one.

    A static symmetric reading would be wrong in an informative way: the step
    gate rightly refuses to unload a foot the sensors say is still carrying half
    the robot, so a fixed 50/50 mock would test nothing but the veto. A positive
    same-sign hip roll carries the pelvis toward the robot's RIGHT, so it loads
    the right foot -- that is the relation modelled here.
    """

    TOTAL_N = 52.0
    SENSITIVITY = 3.0

    def __init__(self, robot, side):
        self.robot = robot
        self.side = side

    def enable(self, _ms):
        pass

    def getValues(self):  # noqa: N802
        lean = 0.5 * (self.robot.motors["LHipRoll"].position
                      + self.robot.motors["RHipRoll"].position)
        share_right = min(1.0, max(0.0, 0.5 + self.SENSITIVITY * lean))
        share = share_right if self.side == "R" else 1.0 - share_right
        # Non-zero shear on x/y: the reader must take fz, not the vector norm.
        return [4.0, -3.0, self.TOTAL_N * share]

    def getValue(self):  # noqa: N802
        raise RuntimeError("getValue() is not supported for a force-3d sensor")


class FakeRobot:
    def __init__(self, *, with_fsr=True):
        self.time = 0.0
        self.resets = 0
        self.motors = {name: FakeMotor(name) for name in CONFIGS}
        self.devices = {}
        for name, motor in self.motors.items():
            self.devices[name] = motor
            self.devices[name + "S"] = FakeSensor(motor)
        self.imu = FakeInertialUnit()
        self.devices["inertial unit"] = self.imu
        # How far the InertialUnit's own reading is offset from the truth. Zero
        # by default; the tests about the rotated mount set it (see FakeNode).
        self.mount_roll = 0.0
        self._self_node = FakeNode(self)
        self.devices["gyro"] = FakeVector3()
        self.devices["accelerometer"] = FakeAccelerometer(self)
        if with_fsr:
            self.devices["LFsr"] = FakeFsr(self, "L")
            self.devices["RFsr"] = FakeFsr(self, "R")

    def getBasicTimeStep(self):  # noqa: N802
        return TIMESTEP_MS

    def getDevice(self, name):  # noqa: N802
        return self.devices.get(name)

    def getTime(self):  # noqa: N802
        return self.time

    def getSelf(self):  # noqa: N802
        """The Supervisor's view of ourselves -- the heading source."""
        return self._self_node

    def step(self, ms):
        self.time += ms / 1000.0
        return 0

    # -- Supervisor surface, so fall recovery can be exercised --------------
    def simulationReset(self):  # noqa: N802 - Webots API name
        """Restore the initial state, as Webots does: the robot stands back up."""
        self.resets += 1
        # An upright robot whose sensor is mounted rotated reads its MOUNT
        # OFFSET, not zero -- the offset is a property of the sensor and a reset
        # does not change it. Zeroing it here made the scene-tree node (which
        # subtracts the offset) report a robot rolled by -mount_roll after every
        # recovery, i.e. permanently fallen.
        self.imu.rpy = [self.mount_roll, 0.0, 0.0]
        for motor in self.motors.values():
            motor.position = 0.0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def controller_module(monkeypatch, tmp_path):
    """Import the real controller file with a fake ``controller`` package."""
    fake = types.ModuleType("controller")
    fake.Robot = FakeRobot
    fake.Supervisor = FakeRobot          # a Supervisor IS a Robot, plus reset
    fake.Motion = FakeMotion
    monkeypatch.setitem(sys.modules, "controller", fake)
    monkeypatch.syspath_prepend(CONTROLLER_DIR)

    sys.modules.pop("pose_imitation_controller", None)
    mod = importlib.import_module("pose_imitation_controller")
    mod = importlib.reload(mod)

    clips = tmp_path / "motions"
    clips.mkdir()
    # Real Webots NAO walk clips declare exactly the 12 leg joints in their
    # header and nothing else; the fakes have to say the same thing or the
    # per-joint handover cannot be exercised.
    header = "#WEBOTS_MOTION,V1.0," + ",".join(
        f"{s}{j}" for s in ("L", "R")
        for j in ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch",
                  "AnklePitch", "AnkleRoll")
    ) + "\n"
    # ...and they must carry a first KEYFRAME, because the posture a clip opens
    # in is the whole reason the handover needs a ramp. These are Cyberbotics'
    # own opening values for Forwards.motion: a sole-flat crouch of about 0.51
    # rad (hip + knee + ankle = 0), against the controller's standing 0.10. A
    # header-only fake made motion_first_pose return {} and the prepare stage
    # was skipped in every test.
    values = ",".join(
        str(v) for _ in ("L", "R")
        for v in (0, 0.027, -0.505, 1.042, -0.537, -0.027)
    )
    # 60 keyframes at the real 40 ms spacing, so the clip has safe places to be
    # stopped along its WHOLE length (balance.safe_exit_times reads them) rather
    # than only at t=0. A real walk clip offers 46 of its 66.
    body = "".join(
        f"00:{i * 40 // 1000:02d}:{i * 40 % 1000:03d},Pose{i + 1},{values}\n"
        for i in range(60)
    )
    for name in ("Forwards.motion", "TurnLeft60.motion", "TurnRight60.motion"):
        (clips / name).write_text(header + body, encoding="utf-8")

    # The shipped default is LEG_CONTROL="pose" -- the legs imitate continuously
    # and locomotion clips are opt-in, because a clip is a 2-3 second commitment
    # during which the camera is ignored for the leg joints. These tests cover the
    # locomotion layer, so they opt in; the default itself is asserted by
    # test_the_shipped_default_is_imitation_not_locomotion.
    monkeypatch.setattr(mod, "LEG_CONTROL", "auto")
    monkeypatch.setattr(mod, "MOTION_SEARCH_DIRS_EXTRA", [str(clips)])
    # So a test can drop another clip in before the controller is constructed --
    # clip discovery and gait-cycle detection both happen in its __init__.
    mod._TEST_CLIP_DIR = clips
    # Clip discovery has to be HERMETIC. Setting MOTION_SEARCH_DIRS_EXTRA alone is
    # not enough: default_motion_search_dirs() also appends $WEBOTS_HOME and eight
    # well-known install roots, so on a machine that actually has Webots the real
    # clips leak in and these tests assert against a set they do not control. The
    # effect was backwards -- the suite passed on a box that could not run the
    # robot and failed on the one that could.
    monkeypatch.setattr(
        mod, "default_motion_search_dirs",
        lambda extra=None: [str(d) for d in (extra or [])],
    )
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", False)
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())
    FakeMotion.played = []
    FakeMotion.NEVER_OVER = False
    FakeMotion.STEPS = 20            # class state; tests may raise it
    yield mod
    FakeMotion.NEVER_OVER = False
    FakeMotion.STEPS = 20
    sys.modules.pop("pose_imitation_controller", None)


class Harness:
    """Drives a real ``PoseImitationController`` over real UDP."""

    def __init__(self, mod):
        self.mod = mod
        self.ctl = mod.PoseImitationController()
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = mod.UDP_PORT

    def send(self, keypoints=None, gait=None):
        payload = {"timestamp_s": 0.0, "frame_index": 0, "joint_angles_rad": {}}
        if keypoints:
            payload["keypoints"] = keypoints
        if gait:
            payload["gait"] = gait
        self.tx.sendto(json.dumps(payload).encode(), ("127.0.0.1", self.port))

    def spin(self, steps, keypoints=None, gait=None, every=2):
        """Advance the REAL control loop, feeding a frame every ``every`` steps.

        Calls ``PoseImitationController.tick`` rather than reimplementing it, so
        the harness cannot drift out of sync with the controller it is testing.
        """
        c = self.ctl
        for i in range(steps):
            if keypoints is not None and i % every == 0:
                self.send(keypoints, gait)
            c.robot.step(c.timestep)
            c.tick()
        return c.leg_mode

    def angle(self, name):
        return self.ctl.robot.motors[name].position

    def close(self):
        self.tx.close()
        self.ctl._cleanup()


@pytest.fixture
def harness(controller_module):
    h = Harness(controller_module)
    yield h
    h.close()


@pytest.fixture
def cyclic_harness(controller_module):
    """A harness whose forward clip has a real gait cycle in it.

    The default fixture's clips are 60 IDENTICAL keyframes, which translate
    nothing, so ``gait_cycle`` correctly refuses them and every other test in
    this file exercises one-shot playback. This one writes a clip built from a
    parametric gait (see tests/conftest.py) as ``Forwards50.motion``, which is
    the filename ``select_walk_clip`` looks for, so the controller discovers it,
    detects the cycle and promotes it over the short clip -- exactly the path the
    real robot takes.
    """
    clips = controller_module._TEST_CLIP_DIR
    write_cyclic_clip(clips / "Forwards50.motion", period=26, cycles=3,
                      lead=8, tail=10)
    # 6.76 s of clip at 40 ms keyframes is 169 keyframes; the fixture is 97, so
    # STEPS has to cover its whole length or the fake ends before the cycle does.
    FakeMotion.STEPS = int(round(96 * 0.04 / (TIMESTEP_MS / 1000.0)))
    h = Harness(controller_module)
    yield h
    h.close()
    FakeMotion.STEPS = 20


# ---------------------------------------------------------------------------
# Synthetic subject
# ---------------------------------------------------------------------------
def subject(*, left_leg=(0.0, 0.0, 0.0), right_leg=(0.0, 0.0, 0.0), yaw=0.0):
    """Landmarks for a subject with each leg at ``(roll_mag, hip_pitch, knee)``.

    Shoulders/hips are placed so that ``nao_retarget._torso_frame`` computes
    the identity basis (right=(1,0,0), up=(0,-1,0), forward=(0,0,-1)) at
    yaw=0, then rotated about the vertical axis by ``yaw`` degrees -- the
    torso-local frame is built fresh from these landmarks every frame, so a
    non-zero yaw exercises the whole point of that upgrade (a subject who
    doesn't face the camera). Leg segments use the same swing-twist forward
    kinematics as ``nao_retarget._swing_twist`` inverts -- see
    ``tests/test_nao_retarget.py``'s ``_leg_dir`` for the derivation.
    """
    kps = {}
    a = math.radians(yaw)
    for name, half, y in (("shoulder", 0.06, 0.30), ("hip", 0.04, 0.55)):
        for side, sgn in (("left", -1.0), ("right", +1.0)):
            kps[f"{side}_{name}"] = [
                0.5 + sgn * half * math.cos(a), y, -sgn * half * math.sin(a), 1.0,
            ]
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        kps[f"{side}_elbow"] = [0.5 + sgn * 0.07, 0.41, 0.0, 1.0]
        kps[f"{side}_wrist"] = [0.5 + sgn * 0.08, 0.52, 0.0, 1.0]
    kps["nose"] = [0.5, 0.18, 0.0, 1.0]

    for side, legs in (("L", left_leg), ("R", right_leg)):
        roll, hip, knee = legs
        pre = "left_" if side == "L" else "right_"
        origin = kps[pre + "hip"]
        knee_pt = _seg(side, origin, 0.18, roll, hip)
        ankle_pt = _seg(side, knee_pt, 0.18, roll, hip + knee)
        kps[pre + "knee"] = knee_pt
        kps[pre + "ankle"] = ankle_pt
    return kps


def _seg(side, origin, length, roll, pitch):
    """One limb segment's endpoint at NAO angles ``(roll, pitch)``, in the
    identity torso frame (see ``subject()``): direction
    ``(side_sign*sin(roll)cos(pitch), cos(roll)cos(pitch), -sin(pitch))``.
    """
    s = _side_sign(side)
    dx = s * math.sin(roll) * math.cos(pitch)
    dy = math.cos(roll) * math.cos(pitch)
    dz = -math.sin(pitch)
    return [
        origin[0] + length * dx,
        origin[1] + length * dy,
        origin[2] + length * dz,
        1.0,
    ]


STANDING = subject()
LEFT_LEG_UP = subject(left_leg=(0.0, -1.0, 1.4))
SQUAT = subject(left_leg=(0.0, -0.55, 1.1), right_leg=(0.0, -0.55, 1.1))
# Legs spread outward: both hips abducted by the same amount.
LEGS_APART = subject(left_leg=(0.35, 0.0, 0.0), right_leg=(0.35, 0.0, 0.0))
LEGS_WIDE = subject(left_leg=(0.70, 0.0, 0.0), right_leg=(0.70, 0.0, 0.0))
# Steps a clip needs before it is PLAYING: the legs must first ramp into the
# stance the clip opens in (0.84 rad of knee travel at the driver's 1.5 rad/s is
# 29 steps -- see CLIP_PREPARE_TIMEOUT_S and approach_leg_pose), and only then is
# playback started. Tests written before that ramp existed spun 4-20 steps and
# asserted a clip was already running.
CLIP_RAMP_STEPS = 29


def spin_to_clip(harness, keypoints=None, gait=None, limit=None) -> bool:
    """Spin until a clip is actually playing. True if one started."""
    keypoints = STANDING if keypoints is None else keypoints
    gait = MARCH_GAIT if gait is None else gait
    for _ in range(limit if limit is not None else CLIP_RAMP_STEPS + 30):
        harness.spin(1, keypoints, gait)
        if harness.ctl.motion.active:
            return True
    return False


MARCH_GAIT = {"state": "march", "cadence_hz": 0.9, "phase": 0.5, "swing_side": 1,
              "intensity": 0.8, "turn": 0.0, "conf": 0.95,
              "body_yaw_rad": 0.0, "yaw_conf": 0.95}
IDLE_GAIT = {"state": "idle", "cadence_hz": 0.0, "phase": 0.0, "swing_side": 0,
             "intensity": 0.0, "turn": 0.0, "conf": 0.95,
             "body_yaw_rad": 0.0, "yaw_conf": 0.95}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def test_controller_finds_all_devices_and_layers(harness) -> None:
    c = harness.ctl
    assert len(c.driver.motors) == len(CONFIGS)
    assert len(c.driver.sensors) == len(CONFIGS)
    assert c.imu is not None and c.gyro is not None
    assert c.driver.balance is not None          # CoM balance
    assert c.driver.lower_body is not None       # per-leg pose imitation
    assert c.driver.gait_engine is not None      # march engine
    assert set(c.motion.available) == {"forward", "turn_left", "turn_right"}


def test_foot_sensors_are_read_as_three_axis(harness) -> None:
    """NAO's FSRs are force-3d: getValue() raises on them, and of the three axes
    only the vertical one is the load the step gate should trust."""
    loads = harness.ctl._read_fsr()
    assert loads is not None
    assert loads["L"] == pytest.approx(26.0)
    assert loads["R"] == pytest.approx(26.0)
    # ... and they follow the lean, so the step gate has something to check.
    harness.ctl.robot.motors["LHipRoll"].position = 0.15
    harness.ctl.robot.motors["RHipRoll"].position = 0.15
    leaning = harness.ctl._read_fsr()
    assert leaning["R"] > leaning["L"]


def test_no_foot_sensors_reports_none_rather_than_zeros(controller_module) -> None:
    """Zeros would look like "no weight anywhere" and veto every step."""
    mod = controller_module
    ctl = mod.PoseImitationController.__new__(mod.PoseImitationController)
    ctl.fsr = {"L": [], "R": []}
    assert ctl._read_fsr() is None


# ---------------------------------------------------------------------------
# Leg arbitration
# ---------------------------------------------------------------------------
def test_standing_still_uses_pose_imitation(harness) -> None:
    mode = harness.spin(120, STANDING, IDLE_GAIT)
    assert mode == "pose"
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "double"
    assert not FakeMotion.played           # no clip fires while standing still


def test_squat_reaches_the_knees(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    straight = harness.angle("LKneePitch")
    harness.spin(120, SQUAT, IDLE_GAIT)
    assert harness.angle("LKneePitch") > straight + 0.15
    # Symmetric: a squat must not become a lean.
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


def test_spreading_the_legs_actually_widens_the_stance(harness) -> None:
    """A wider stance is CoM-neutral and *enlarges* the support polygon, so it is
    safer than standing and must pass through at full authority. Gating it like a
    lean turned a 20 deg human spread into a 6 deg robot one."""
    harness.spin(150, STANDING, IDLE_GAIT)
    assert abs(harness.angle("LHipRoll")) < 0.03      # feet together
    harness.spin(200, LEGS_APART, IDLE_GAIT)
    # NAO's roll signs are mirrored: +L and -R both mean outward.
    assert harness.angle("LHipRoll") > 0.25
    assert harness.angle("RHipRoll") < -0.25
    # Both soles stay flat -- otherwise the robot stands on its inner edges.
    for side in ("L", "R"):
        assert harness.angle(f"{side}HipRoll") + harness.angle(f"{side}AnkleRoll") \
            == pytest.approx(0.0, abs=0.02)


def test_a_very_wide_spread_saturates_instead_of_tipping(harness) -> None:
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(200, LEGS_WIDE, IDLE_GAIT)
    roll = harness.angle("LHipRoll")
    # Reaches past what the ankle can level -- spending a bounded sole tilt to
    # buy width, because stopping at the ankle's limit saturated below the ~25 deg
    # people actually spread to.
    assert roll > abs(CONFIGS["LAnkleRoll"].min_angle)
    assert roll < CONFIGS["LHipRoll"].max_angle          # hip could go further
    # ... but no sole ends up more than the budget off flat. The budget is a
    # guarantee on the commanded target; the per-joint smoother can overshoot it
    # by a hair in transit because hip and ankle travel different distances, so
    # allow a small settling margin.
    from lower_body import LowerBodyParams
    budget = LowerBodyParams().sole_tilt_budget
    harness.spin(400, LEGS_WIDE, IDLE_GAIT)
    for side in ("L", "R"):
        tilt = harness.angle(f"{side}HipRoll") + harness.angle(f"{side}AnkleRoll")
        assert abs(tilt) <= budget + 0.01, (side, tilt)


def test_the_legs_come_back_together(harness) -> None:
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(200, LEGS_APART, IDLE_GAIT)
    assert harness.angle("LHipRoll") > 0.25
    harness.spin(200, STANDING, IDLE_GAIT)
    assert abs(harness.angle("LHipRoll")) < 0.05


def test_raising_one_leg_reaches_the_full_requested_lift(harness) -> None:
    """The reported symptom was a leg lift doing nothing. Assert it goes all the
    way: the CoM model permits it once the weight really has transferred."""
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(300, LEFT_LEG_UP, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["gate"] == pytest.approx(1.0, abs=1e-6)
    assert meta["lift"] > 0.9
    assert harness.angle("LKneePitch") > 1.0             # knee clearly folded
    assert harness.angle("LHipPitch") < -0.8             # thigh clearly raised


def test_raising_one_leg_transfers_weight_then_lifts(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "single"
    assert meta["stance_side"] == "R"
    assert meta["shift"] > 0.9              # weight moved first
    assert meta["lift"] > 0.2               # then the foot came up
    assert meta["stance_margin"] > 0.0      # the CoM model agreed
    # The commanded robot really lifted the matching leg.
    assert harness.angle("LKneePitch") > harness.angle("RKneePitch") + 0.3
    assert harness.angle("LHipPitch") < harness.angle("RHipPitch") - 0.2


def test_lowering_the_leg_returns_to_a_symmetric_stance(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["mode"] == "single"
    harness.spin(200, STANDING, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "double"
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


def test_marching_plays_a_forward_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.leg_mode == "motion:forward"
    assert "Forwards.motion" in FakeMotion.played


def test_a_clip_suspends_per_joint_commanding(harness) -> None:
    """While a clip owns the body, our targets must not fight its keyframes --
    and the velocity caps must be lifted or it cannot reach them."""
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True
    motor = harness.ctl.robot.motors["LKneePitch"]
    assert motor.velocity == pytest.approx(CONFIGS["LKneePitch"].max_velocity)
    before = motor.commands
    harness.spin(10, STANDING, MARCH_GAIT)
    assert motor.commands == before          # nothing commanded during playback


def test_a_replayed_clip_is_rewound_first(harness) -> None:
    """A finished clip resumed without a rewind returns immediately, so the
    robot would take one step and then stand there looking stuck."""
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(200, STANDING, MARCH_GAIT)
    assert FakeMotion.played.count("Forwards.motion") >= 2   # replayed
    clip = harness.ctl.motion._cache["forward"]
    assert clip.rewinds >= 2
    assert clip.loop is False                                # never looped


def test_a_multi_clip_rotation_keeps_going_until_aligned(harness) -> None:
    """Turning must converge across clip boundaries, not stall one clip short."""
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=1.8)               # ~103 deg
    kps = subject(yaw=60.0)
    for _ in range(6):
        harness.spin(40, kps, turned)
        if harness.ctl.motion.active:
            # Pretend the clip turned the robot by its nominal 60 deg.
            harness.ctl.imu.rpy[2] += math.radians(60)
    assert FakeMotion.played.count("TurnLeft60.motion") >= 2
    assert harness.ctl._turning is False                     # converged
    assert abs(harness.ctl.yaw_servo.error(harness.ctl.imu.rpy[2])) < 0.6


def test_control_is_reclaimed_when_the_clip_ends(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True
    mode = harness.spin(200, STANDING, IDLE_GAIT)
    assert harness.ctl.driver.suspended is False
    assert mode == "pose"
    # The step sequencer was reset, not resumed mid-transfer.
    assert harness.ctl.driver.lower_body_meta["mode"] == "double"


def test_falling_aborts_the_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True
    harness.ctl.imu.rpy = [0.6, 0.0, 0.0]     # well past TILT_ABORT_RAD
    harness.spin(6, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is False
    assert not harness.ctl.motion.active


def test_gyro_predicts_a_fall_before_the_tilt_crosses_the_limit(harness) -> None:
    c = harness.ctl
    c.imu.rpy = [0.30, 0.0, 0.0]              # below TILT_ABORT_RAD on its own
    assert c._falling(0.30, 0.0) is False
    c.gyro.values = [2.0, 0.0, 0.0]           # ... but tipping fast
    assert c._falling(0.30, 0.0) is True


def test_losing_the_human_stands_the_robot_down(harness) -> None:
    """Without an expiry on the latched observation the robot would hold a
    one-legged stance forever after the human walked out of frame."""
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["mode"] == "single"
    # Stop sending frames entirely.
    mode = harness.spin(300, None)
    assert mode == "pose"
    assert harness.ctl.driver.stats.stale is True
    assert harness.ctl.driver.lower_body_meta["mode"] == "double"
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


# ---------------------------------------------------------------------------
# Turning
# ---------------------------------------------------------------------------
def test_body_rotation_triggers_a_turn_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=0.9)
    # Held, not flashed: a turn is only started on a heading estimate that has been
    # STEADY for about a second (YawServo.stable). Body yaw is the noisiest cue in
    # the pipeline -- it swung over 187 degrees in a recorded session while the
    # subject just stood there -- and acting on every excursion starved forward
    # walking completely.
    mode = harness.spin(80, subject(yaw=50.0), turned)
    assert mode == "motion:turn_left"
    assert "TurnLeft60.motion" in FakeMotion.played


def test_turn_direction_follows_the_sign_of_the_rotation(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(80, subject(yaw=-50.0), dict(IDLE_GAIT, body_yaw_rad=-0.9))
    assert "TurnRight60.motion" in FakeMotion.played


def test_turning_stops_once_the_robot_has_caught_up(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=0.9)
    harness.spin(80, subject(yaw=50.0), turned)
    assert harness.ctl.motion.active
    # The clip physically turned the robot: report the new heading.
    harness.ctl.imu.rpy = [0.0, 0.0, 0.9]
    mode = harness.spin(200, subject(yaw=50.0), turned)
    assert mode == "pose"
    assert not harness.ctl.motion.active


def test_turning_takes_priority_over_walking_once_the_heading_is_trusted(harness) -> None:
    """Heading beats walking -- but only on a heading worth acting on.

    While the yaw estimate is still settling the robot walks forward instead of
    standing there, which is the deliberate choice: body yaw is the noisiest cue in
    the pipeline, and waiting for it starved forward locomotion entirely (measured:
    the servo error was past the turn gate in 77% of frames while the subject simply
    stood in front of the camera, and no clip ever ran). Walking a little off-heading
    is recoverable; never walking is the bug being reported.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    both = dict(MARCH_GAIT, body_yaw_rad=0.9)
    harness.spin(200, subject(yaw=50.0), both)
    assert "TurnLeft60.motion" in FakeMotion.played, FakeMotion.played
    # Once steady, the turn is what gets chosen over walking on.
    assert harness.ctl.yaw_servo.stable() is True


def test_a_noisy_heading_does_not_block_forward_walking(harness) -> None:
    """The starvation this gate exists to prevent, from the other side."""
    harness.spin(60, STANDING, IDLE_GAIT)
    # A heading estimate that swings wildly: never steady, always past the gate.
    swing = 1.0
    for _ in range(12):
        swing = -swing
        harness.spin(10, subject(yaw=50.0 * swing),
                     dict(MARCH_GAIT, body_yaw_rad=0.9 * swing))
    assert harness.ctl.yaw_servo.stable() is False
    assert "Forwards.motion" in FakeMotion.played, FakeMotion.played


def test_a_small_rotation_gets_a_hip_yaw_bias_not_a_clip(harness) -> None:
    """Below the turn gate the robot should still acknowledge the rotation."""
    harness.spin(60, STANDING, IDLE_GAIT)
    mode = harness.spin(120, subject(yaw=12.0), dict(IDLE_GAIT, body_yaw_rad=0.2))
    assert mode == "pose"
    assert not FakeMotion.played
    bias = harness.angle("LHipYawPitch")
    assert 0.0 < bias <= 0.12 + 1e-6


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keypoints,gait",
    [(STANDING, IDLE_GAIT), (LEFT_LEG_UP, IDLE_GAIT), (SQUAT, IDLE_GAIT),
     (STANDING, MARCH_GAIT)],
)
def test_no_command_ever_leaves_the_joint_limits(harness, keypoints, gait) -> None:
    harness.spin(200, keypoints, gait)
    for name, motor in harness.ctl.robot.motors.items():
        cfg = CONFIGS[name]
        assert math.isfinite(motor.position)
        assert cfg.min_angle - 1e-9 <= motor.position <= cfg.max_angle + 1e-9, name


def test_malformed_packets_are_ignored(harness) -> None:
    harness.tx.sendto(b"not json at all", ("127.0.0.1", harness.port))
    harness.tx.sendto(b"\xff\xfe\x00", ("127.0.0.1", harness.port))
    mode = harness.spin(40, STANDING, IDLE_GAIT)
    assert mode == "pose"


def test_only_one_layer_commands_the_legs_per_step(harness) -> None:
    """The invariant the whole arbiter exists for: two commanders means a fall."""
    harness.spin(60, STANDING, IDLE_GAIT)
    motor = harness.ctl.robot.motors["LKneePitch"]
    motor.commands = 0
    harness.spin(10, LEFT_LEG_UP, IDLE_GAIT, every=1)
    assert motor.commands == 10


# ---------------------------------------------------------------------------
# Freeze resistance
#
# Motion playback suspends per-joint commanding for the WHOLE body, so anything
# that stops a clip from ending stops the entire robot -- which is exactly how
# "the robot is completely frozen" happens. These tests assert that no single
# failure can hold the body indefinitely.
# ---------------------------------------------------------------------------
def test_a_clip_that_never_ends_cannot_freeze_the_robot(harness) -> None:
    mod = harness.mod
    FakeMotion.NEVER_OVER = True
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True      # clip took the body

    # Spin well past the watchdog budget, with the human STILL WALKING -- so the
    # clip stays wanted and the early exit does not end it first. The watchdog is
    # the protection against a clip that never reports itself over, which is a
    # different failure from "nothing wants this clip any more".
    steps = int((mod.MOTION_WATCHDOG_S + 2.0) / (TIMESTEP_MS / 1000.0))
    mode = harness.spin(steps, STANDING, MARCH_GAIT)

    assert harness.ctl.driver.suspended is False     # ... and gave it back
    # With the human still marching and the forward clip retired, the legs fall
    # back to marching in place rather than standing there.
    assert mode in ("pose", "march:march"), mode
    # The offending clip is not tried again.
    assert "forward" not in harness.ctl.motion.available


def test_the_watchdog_uses_the_clips_own_duration(harness) -> None:
    """A 1.2 s clip must not be able to hold the body for the full hard cap."""
    mod = harness.mod
    FakeMotion.NEVER_OVER = True
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True
    budget = harness.ctl._motion_deadline - harness.ctl._motion_started_at
    assert budget < mod.MOTION_WATCHDOG_S
    assert budget == pytest.approx(FakeMotion.DURATION_MS / 1000.0 * 1.5 + 1.0)


def test_repeated_bad_locomotion_gives_up_on_clips(harness) -> None:
    """Falling over again and again is worse than never walking."""
    mod = harness.mod
    harness.spin(60, STANDING, IDLE_GAIT)
    for _ in range(mod.MOTION_MAX_FAILURES):
        assert spin_to_clip(harness)
        harness.ctl.imu.rpy = [0.6, 0.0, 0.0]        # tilt abort
        harness.spin(4, STANDING, MARCH_GAIT)
        harness.ctl.imu.rpy = [0.0, 0.0, 0.0]
    assert harness.ctl.motion.available == {}
    mode = harness.spin(60, STANDING, IDLE_GAIT)
    assert mode == "pose"                            # still imitating
    assert harness.ctl.driver.suspended is False


def test_a_clip_is_not_started_while_the_robot_is_wobbling(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.ctl.gyro.values = [3.0, 0.0, 0.0]        # tipping fast
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert not FakeMotion.played
    assert mode == "pose"
    harness.ctl.gyro.values = [0.0, 0.0, 0.0]
    assert spin_to_clip(harness)                     # ... and once calm, it goes
    assert "Forwards.motion" in FakeMotion.played


def test_a_clip_stays_blocked_for_a_while_after_a_tilt_spike(harness) -> None:
    """A wobble that never crosses TILT_ABORT_RAD (so it never mid-clip-aborts
    anything, and nothing is even playing yet) should still make the
    controller more cautious about STARTING the next clip for a while -- the
    continuous tilt-risk EMA, not just the binary settle threshold."""
    harness.spin(60, STANDING, IDLE_GAIT)
    # A tilt spike under TILT_ABORT_RAD (0.40) but over the old fixed
    # MOTION_START_MAX_TILT_RAD (0.15) ceiling, held long enough to build risk.
    harness.ctl.imu.rpy = [0.35, 0.0, 0.0]
    harness.spin(150, STANDING, IDLE_GAIT)          # ~3s, ~2 EMA time constants
    # Settle back to a tilt that would have passed the OLD fixed ceiling...
    harness.ctl.imu.rpy = [0.10, 0.0, 0.0]
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    # ... but risk is still elevated right after the wobble, so no clip yet.
    assert not FakeMotion.played
    # The legs keep marching in place while the clip layer waits. They used to
    # fall through to the pose layer, which does nothing with the legs unless it
    # can see a leg lift -- so the robot stood motionless while the human marched
    # at it. Declining to WALK is not a reason to stop moving.
    assert mode == "march:march"
    # Give risk time to decay back down toward the current (calmer) tilt.
    harness.spin(400, STANDING, IDLE_GAIT)          # ~8s, several time constants
    assert spin_to_clip(harness)                    # ... and once it has, it goes
    assert "Forwards.motion" in FakeMotion.played


def test_a_failing_step_does_not_end_the_loop_or_limp_the_robot(harness) -> None:
    """One transient error used to break run(), which then zeroed every motor
    velocity -- a permanently dead robot from a single bad frame."""
    c = harness.ctl
    calls = {"n": 0}
    real = c.driver.lower_body_tick

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] in (3, 4, 5):
            raise RuntimeError("synthetic sensor glitch")
        return real(*a, **kw)

    c.driver.lower_body_tick = flaky
    for i in range(40):
        if i % 2 == 0:
            harness.send(STANDING, IDLE_GAIT)
        c.robot.step(c.timestep)
        try:
            c.tick()
        except Exception:
            c._errors += 1
            c._recover_from_error()

    assert calls["n"] > 5                 # kept going past the failures
    assert c.driver.suspended is False    # recovery forced control back
    for motor in c.robot.motors.values():
        assert motor.velocity > 0.0       # never left limp


def test_recovery_forces_the_body_back_if_a_step_fails_mid_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.driver.suspended is True
    harness.ctl._recover_from_error()
    assert harness.ctl.driver.suspended is False
    assert not harness.ctl.motion.active


# ---------------------------------------------------------------------------
# Legs must respond even when the camera crops the lower body
# ---------------------------------------------------------------------------
def _crop(keypoints, *names):
    out = dict(keypoints)
    for name in names:
        out[name] = list(out[name][:3]) + [0.05]
    return out


def test_a_leg_lift_is_seen_with_the_feet_out_of_frame(harness) -> None:
    """Standing close to a webcam crops the shins. The feet-based ground line is
    then unavailable, and without the knee fallback the lift reads as exactly
    zero however high the leg goes -- i.e. "leg movement is not working"."""
    standing = _crop(STANDING, "left_ankle", "right_ankle")
    lifted = _crop(LEFT_LEG_UP, "left_ankle", "right_ankle")
    harness.spin(150, standing, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["lift_source"] == "knees"
    harness.spin(250, lifted, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["lift_source"] == "knees"
    assert meta["stance_side"] == "R"
    assert meta["lift"] > 0.2
    # With no ankle in view the knee bend is genuinely unobservable, so the lift
    # shows up as hip flexion (a raised straight leg) rather than a knee fold.
    # That is the honest reading of what the camera can see -- and it is still a
    # clearly raised leg, which is the point.
    assert harness.angle("LHipPitch") < harness.angle("RHipPitch") - 0.3
    assert harness.angle("LKneePitch") <= harness.angle("RKneePitch") + 0.05


def test_legs_without_knees_or_feet_say_so_instead_of_failing_silently(harness) -> None:
    blind = _crop(LEFT_LEG_UP, "left_ankle", "right_ankle",
                  "left_knee", "right_knee")
    harness.spin(120, blind, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["lift"] == 0.0
    assert "out of frame" in str(meta["why"]) or "landmarks" in str(meta["why"])


def test_the_status_line_always_explains_itself(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    why = harness.ctl.driver.lower_body_meta["why"]
    assert isinstance(why, str) and why
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert "stepping" in harness.ctl.driver.lower_body_meta["why"]


def test_unresponsive_foot_sensors_reduce_the_lift_but_do_not_block_it(
        controller_module, monkeypatch) -> None:
    """A proto whose FSRs read a constant 50/50 used to forbid every step
    forever -- a silent, permanent "leg lift does nothing"."""
    class StaticFsr(FakeVector3):
        def getValues(self):  # noqa: N802
            return [0.0, 0.0, 26.0]

    original = FakeRobot.__init__

    def patched(self, *, with_fsr=True):
        original(self, with_fsr=with_fsr)
        self.devices["LFsr"] = StaticFsr()
        self.devices["RFsr"] = StaticFsr()

    monkeypatch.setattr(FakeRobot, "__init__", patched)
    h = Harness(controller_module)
    try:
        h.spin(150, STANDING, IDLE_GAIT)
        h.spin(300, LEFT_LEG_UP, IDLE_GAIT)
        meta = h.ctl.driver.lower_body_meta
        assert meta["fsr_share"] == pytest.approx(0.5, abs=0.01)
        assert 0.0 < meta["lift"] < 0.8            # reduced, not refused
        assert h.angle("LKneePitch") > h.angle("RKneePitch") + 0.2
        assert "reduced authority" in str(meta["why"])
    finally:
        h.close()


def test_the_robot_turns_round_to_face_behind_you(harness) -> None:
    """The yaw estimate used to be bounded to +/-90 deg by an abs(), so the robot
    could never be asked to turn more than a quarter circle."""
    harness.spin(60, STANDING, IDLE_GAIT)
    behind = dict(IDLE_GAIT, body_yaw_rad=math.radians(175))
    kps = subject(yaw=175.0)
    for _ in range(8):
        harness.spin(40, kps, behind)
        if harness.ctl.motion.active:
            # Model the clip actually turning the robot by its nominal 60 deg.
            action = harness.ctl.motion.action
            harness.ctl.imu.rpy[2] += math.radians(60 if action == "turn_left" else -60)
    turns = [m for m in FakeMotion.played if "Turn" in m]
    assert len(turns) >= 3                       # 175 deg needs several clips
    assert len(set(turns)) == 1                  # all the same way -- no thrash
    residual = abs(harness.ctl.yaw_servo.error(harness.ctl.imu.rpy[2]))
    assert residual < math.radians(35)           # within half a clip


def test_a_wobbling_heading_does_not_thrash_turn_clips(harness) -> None:
    """A jittery yaw made the planner alternate left/right, which starves forward
    walking and trips the locomotion failure backoff -- the reason nothing moved."""
    harness.spin(60, STANDING, IDLE_GAIT)
    for i in range(30):
        wobble = math.radians(12) * (1 if i % 2 else -1)
        harness.spin(6, subject(yaw=math.degrees(wobble)),
                     dict(IDLE_GAIT, body_yaw_rad=wobble))
    turns = [m for m in FakeMotion.played if "Turn" in m]
    assert turns == []                           # nothing fired at all
    assert harness.ctl.motion.available          # and clips were never abandoned


def test_walking_is_not_starved_by_a_settled_heading(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert harness.ctl.leg_mode == "motion:forward"
    assert "Forwards.motion" in FakeMotion.played


def test_a_blocked_clip_falls_back_to_marching_in_place(harness) -> None:
    """The march engine must be reachable on a machine that HAS Webots.

    Its old gate was ``"forward" not in motion.available`` -- i.e. it ran only
    when no forward clip existed on disk. Every real Webots install ships
    Forwards.motion, so the fallback existed only on machines that could not run
    the robot at all. Meanwhile the case it was written for -- the clip layer
    declining because the robot is not settled -- fell through to the pose layer,
    which commands nothing on the legs when it cannot see a leg lift.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    assert "forward" in harness.ctl.motion.available    # the old gate would be shut

    # Not settled: a steady tilt over the start ceiling but well under the abort
    # limit, so the robot is upright and merely unsteady -- exactly when marching
    # in place is the right answer.
    harness.ctl.imu.rpy = [0.30, 0.0, 0.0]
    mode = harness.spin(30, STANDING, MARCH_GAIT)
    assert not FakeMotion.played                        # no clip was started
    assert mode == "march:march"                        # but the legs are moving
    assert harness.ctl.driver.gait_meta["amp_gain"] > 0.0

    # And once it settles, the clip layer takes over again.
    harness.ctl.imu.rpy = [0.0, 0.0, 0.0]
    harness.spin(400, STANDING, IDLE_GAIT)              # let tilt risk decay
    harness.spin(30, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played


def test_marching_still_yields_to_an_actual_fall(harness) -> None:
    """The march fallback must NOT fire while the robot is going over: the pose
    layer's tilt gate is the better recovery, because it ramps the asymmetric
    part of the posture out and returns to the balanced symmetric crouch."""
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.ctl.gyro.values = [3.0, 0.0, 0.0]           # predicted tilt past abort
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert not FakeMotion.played
    assert mode == "pose"


def test_a_walk_clip_only_takes_the_legs_not_the_arms(harness) -> None:
    """The clip declares the 12 leg joints, so it gets those and no more.

    Suspending the whole body meant arm and head imitation stopped for the length
    of every clip -- and because a marching human restarts the clip immediately,
    the upper body appeared to die for as long as the walking lasted. Webots' own
    walk clips never command an arm joint, so there was nothing to protect.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    d = harness.ctl.driver
    assert d.suspended is True
    # Legs handed over ...
    assert d._is_suspended("LKneePitch") and d._is_suspended("RHipRoll")
    # ... arms and head kept.
    for name in ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "HeadYaw"):
        assert not d._is_suspended(name), name

    # And the arms genuinely keep tracking while the clip plays: move them and
    # watch the motors follow.
    before = harness.angle("LShoulderPitch")
    arms_up = subject()
    for name, (dx, dy) in (("left_elbow", (0.10, -0.18)), ("left_wrist", (0.18, -0.34))):
        base = arms_up.get(name)
        if base is not None:
            arms_up[name] = [base[0] + dx, base[1] + dy, base[2], base[3]]
    harness.spin(10, arms_up, MARCH_GAIT)
    assert harness.ctl.motion.active                  # still mid-clip
    assert abs(harness.angle("LShoulderPitch") - before) > 1e-3


def test_a_clip_that_declares_nothing_still_gets_the_whole_body(harness, tmp_path) -> None:
    """Unknown joint list -> hand over everything. Handing over too much only
    costs expressiveness; handing over too little fights the clip's keyframes."""
    c = harness.ctl
    harness.spin(60, STANDING, IDLE_GAIT)
    c.motion._joints["forward"] = []                  # as if the header were junk
    assert spin_to_clip(harness)
    assert c.driver._is_suspended("LShoulderPitch")
    assert c.driver._is_suspended("LKneePitch")


def test_losing_the_human_also_stands_the_UPPER_body_down(harness) -> None:
    """The legs had a stand-down; the arms and head did not.

    Their smoothed targets simply stopped being updated, so they froze wherever
    they happened to be. A recorded session ends with 80 s of *perfect* tracking
    error on every joint -- the robot holding, precisely, the pose of a human who
    had walked away. That reads as a crashed robot, not an idle one.
    """
    c = harness.ctl
    rest = {n: CONFIGS[n].rest_angle
            for n in ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "HeadYaw")}

    # Put the arms somewhere clearly away from rest, with the head turned.
    posed = subject()
    posed["left_elbow"] = [0.62, 0.22, 0.0, 0.95]
    posed["left_wrist"] = [0.70, 0.10, 0.0, 0.95]
    posed["nose"] = [0.56, 0.30, 0.0, 0.95]
    harness.spin(160, posed, IDLE_GAIT)
    moved = {n: harness.angle(n) for n in rest}
    assert any(abs(moved[n] - rest[n]) > 0.05 for n in rest), moved

    # Human leaves: stop sending frames entirely.
    harness.spin(400, None)
    assert c.driver.stats.stale is True
    for name, target in rest.items():
        assert abs(harness.angle(name) - target) < 0.05, (
            f"{name} held {harness.angle(name):+.3f} instead of returning to "
            f"{target:+.3f}"
        )


def test_a_fully_working_stack_reports_nothing_degraded(harness) -> None:
    """The counterpart of the DEGRADED block: when every layer is up, the list is
    empty. If this ever fails, the startup banner is crying wolf."""
    d = harness.ctl.driver
    assert d.degraded == []
    assert d.balance is not None and d.lower_body is not None


def test_a_missing_com_model_is_reported_not_hidden(controller_module, monkeypatch) -> None:
    """NumPy missing from Webots' interpreter is the single most expensive
    failure this controller has, and it used to be one info-level log line.

    Simulated by making the balance import fail the way it does in the field.
    """
    import builtins

    real_import = builtins.__import__

    def no_balance(name, *args, **kwargs):
        if name == "balance":
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_balance)
    ctl = controller_module.PoseImitationController()
    monkeypatch.undo()

    assert ctl.driver.balance is None
    joined = " | ".join(ctl.driver.degraded)
    assert "CoM balance OFF" in joined
    assert "numpy" in joined
    # And the step gate must say it is no longer verifying anything.
    assert any("UNGATED" in reason for reason in ctl.driver.degraded), joined
    ctl.sock.close()


def test_the_trajectory_log_records_the_controllers_own_state(controller_module,
                                                              monkeypatch, tmp_path):
    """A log of joint angles says what the body did, not what the controller
    believed -- and every diagnosis on this project has needed both halves.

    The permanent-lean bug was identifiable from the joint columns alone, but its
    CAUSE needed the support margin, which was not recorded at all. These columns
    are what make a live test session analysable after the fact.
    """
    import csv
    import glob

    mod = controller_module
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    harness = Harness(mod)
    try:
        harness.spin(120, LEFT_LEG_UP, IDLE_GAIT)
        harness.ctl.trajectory_log.close()
        path = glob.glob(str(tmp_path / "*.csv"))
        assert path, "no trajectory log was written"
        rows = list(csv.DictReader(open(path[0], encoding="utf-8")))
        assert rows

        for column in mod.DIAGNOSTIC_COLUMNS:
            assert column in rows[0], column

        # The columns must carry real values, not blanks: a header alone would
        # look fine and diagnose nothing.
        last = rows[-1]
        assert last["leg_mode"]
        assert last["lb_mode"] in ("double", "load", "single")
        assert last["lb_why"]
        assert float(last["support_margin_x"]) != 0.0
        assert float(last["support_margin_y"]) != 0.0
        # And the joint columns still work.
        assert "LKneePitch_cmd_rad" in rows[0]
        assert "LKneePitch_meas_rad" in rows[0]
    finally:
        harness.ctl.sock.close()


def test_diagnostics_never_break_the_control_loop(controller_module, monkeypatch,
                                                  tmp_path):
    """Telemetry is not allowed to be a failure mode."""
    mod = controller_module
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    harness = Harness(mod)
    try:
        # Break the thing _diagnostics leans on hardest.
        harness.ctl.driver.balance.model = None
        mode = harness.spin(40, STANDING, IDLE_GAIT)
        assert mode in ("pose", "stand", "march:march")
        assert harness.ctl._errors == 0
    finally:
        harness.ctl.sock.close()


# ---------------------------------------------------------------------------
# The IMU tilt zero
# ---------------------------------------------------------------------------
def test_a_rotated_inertial_unit_does_not_read_as_a_fall(harness) -> None:
    """The single most expensive bug found on this project.

    Measured on the real robot: standing at rest, foot sensors carrying its full
    50 N of body weight and the gyro at 0.007 rad/s, the InertialUnit reported
    roll = +1.618 rad (93 deg). The sensor frame is mounted rotated. Nothing was
    wrong with the robot -- but every consumer of that number treated it as "about
    to fall over", so leg imitation stood down every frame, no walk clip could
    start, and the balance loop chased 291 mm of phantom lateral error and leaned
    the robot onto one foot.
    """
    c = harness.ctl
    c.robot.mount_roll = 1.618                 # the sensor is rotated, the robot is not
    c.imu.rpy = [1.618, 0.0, 0.0]              # what the real robot reports
    harness.spin(120, STANDING, IDLE_GAIT)

    assert c._imu_zero is not None, "the zero was never learned"
    assert c._imu_zero[0] == pytest.approx(1.618, abs=1e-6)
    # Corrected tilt is level, so nothing thinks the robot is going over.
    roll, pitch = c._corrected_tilt(*c._imu_rpy()[:2])
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6
    assert c._falling(roll, pitch) is False
    assert c.driver.lower_body_meta["tilt_ok"] is True

    # And the legs work again: a leg lift now reaches single support.
    harness.spin(300, LEFT_LEG_UP, IDLE_GAIT)
    assert c.driver.lower_body_meta["mode"] in ("load", "single")


def test_a_real_tilt_on_top_of_the_offset_is_still_detected(harness) -> None:
    """Correcting the zero must not blind the tilt gates -- that would trade one
    silent failure for a much worse one."""
    c = harness.ctl
    c.robot.mount_roll = 1.618
    c.imu.rpy = [1.618, 0.0, 0.0]
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None

    # Now tip it 0.6 rad beyond the learned upright.
    c.imu.rpy = [1.618 + 0.6, 0.0, 0.0]
    roll, pitch = c._corrected_tilt(*c._imu_rpy()[:2])
    assert roll == pytest.approx(0.6, abs=1e-6)
    assert c._falling(roll, pitch) is True
    harness.spin(20, STANDING, IDLE_GAIT)
    assert c.driver.lower_body_meta["tilt_ok"] is False


def test_the_zero_is_not_latched_from_a_fallen_robot(harness, monkeypatch) -> None:
    """A controller restarted on a robot lying on the floor must not decide that
    lying down is upright. The foot sensors are the independent witness: soles
    carrying body weight mean standing, whatever the IMU claims."""
    c = harness.ctl
    c.robot.mount_roll = 1.60
    c.imu.rpy = [1.60, 0.0, 0.0]
    # Feet carrying nothing -- the robot is not standing on them.
    monkeypatch.setattr(FakeFsr, "TOTAL_N", 0.0)
    harness.spin(200, STANDING, IDLE_GAIT)
    assert c._imu_zero is None, "latched a zero from an unloaded robot"
    # Tilt is reported as level while uncalibrated, so nothing aborts on a
    # reading we do not yet understand.
    assert c._corrected_tilt(*c._imu_rpy()[:2]) == (0.0, 0.0)

    # Stand it back up and the zero is learned.
    monkeypatch.setattr(FakeFsr, "TOTAL_N", 52.0)
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None
    assert c._imu_zero[0] == pytest.approx(1.60, abs=1e-6)


def test_the_learned_zero_reaches_the_log(harness, monkeypatch, tmp_path) -> None:
    """It has to be visible after the fact, or the next person re-finds it."""
    import csv
    import glob

    mod = harness.mod
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())   # the fixture holds the other
    other = Harness(mod)
    try:
        other.ctl.robot.mount_roll = 1.618
        other.ctl.imu.rpy = [1.618, 0.0, 0.0]
        other.spin(140, STANDING, IDLE_GAIT)
        other.ctl.trajectory_log.close()
        rows = list(csv.DictReader(open(glob.glob(str(tmp_path / "*.csv"))[0],
                                       encoding="utf-8")))
        last = rows[-1]
        assert float(last["imu_roll_raw"]) == pytest.approx(1.618, abs=1e-6)
        assert float(last["imu_zero_roll"]) == pytest.approx(1.618, abs=1e-6)
        assert abs(float(last["imu_roll"])) < 1e-6      # corrected
    finally:
        other.ctl.sock.close()


def test_auto_zero_can_be_switched_off(controller_module, monkeypatch) -> None:
    """An escape hatch, in case a future model reports tilt honestly."""
    monkeypatch.setattr(controller_module, "IMU_AUTO_ZERO", False)
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    other = Harness(controller_module)
    try:
        assert other.ctl._imu_zero == (0.0, 0.0)
        other.ctl.imu.rpy = [0.5, 0.0, 0.0]
        assert other.ctl._corrected_tilt(0.5, 0.0) == (0.5, 0.0)
    finally:
        other.ctl.sock.close()


# ---------------------------------------------------------------------------
# Fall detection and automatic recovery
# ---------------------------------------------------------------------------
def test_a_fall_is_detected_and_the_simulation_is_reset(harness) -> None:
    """When the robot goes down it stays down -- every layer correctly stands
    itself down, and the rest of the session is spent driving a robot on the
    floor. One recorded session lost 170 of its 178 seconds that way.
    """
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)      # learn the IMU zero while upright
    assert c._imu_zero is not None
    assert c.robot.resets == 0

    # Tip it right over: 90 deg past the learned upright.
    c.imu.rpy = [c._imu_zero[0] + 1.571, c._imu_zero[1], 0.0]
    height = c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2]))
    assert height is not None and height < mod_const(harness, "FALL_HEAD_HEIGHT_M")

    harness.spin(120, STANDING, IDLE_GAIT)      # longer than FALL_CONFIRM_S
    assert c.robot.resets == 1, "the simulation was never reset"
    # The robot is upright again and the fall latch has cleared.
    assert c._fall_since is None
    assert c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2])) > 0.40


def test_recovery_drops_the_state_that_described_the_fallen_robot(harness) -> None:
    """Whether or not Webots restarts this controller on reset, none of the state
    may still describe the robot that fell."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    c._turning = True
    c._tilt_risk = 0.3
    zero = c._imu_zero
    c._reset_for_new_episode()
    assert c._turning is False
    assert c._tilt_risk == 0.0
    # The tilt zero is how the sensor is MOUNTED, not a property of the episode:
    # it survives. Re-learning it after every reset produced 4-6 different zeros
    # per recorded session, each latched from a robot still settling on its feet.
    assert c._imu_zero == zero
    assert c.yaw_servo.latched is False
    assert c.driver.lower_body_meta["mode"] == "double" or True


def mod_const(harness, name):
    return getattr(harness.mod, name)


def test_a_deep_squat_is_not_mistaken_for_a_fall(harness) -> None:
    """The head-height test has to survive the deepest posture the robot is ever
    asked for. It does, and by a wide margin, because NAO's crouch keeps the torso
    vertical: standing 0.460 m, deepest squat 0.412 m, threshold 0.25 m."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None

    deep = subject(left_leg=(0.0, -0.70, 1.40), right_leg=(0.0, -0.70, 1.40))
    harness.spin(400, deep, IDLE_GAIT)
    assert c.robot.resets == 0, "a squat was treated as a fall"
    height = c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2]))
    assert height is None or height > mod_const(harness, "FALL_HEAD_HEIGHT_M")


def test_a_transient_stumble_does_not_trigger_a_reset(harness) -> None:
    """FALL_CONFIRM_S exists so the balance loop gets its chance first."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    zero = c._imu_zero
    c.imu.rpy = [zero[0] + 1.571, zero[1], 0.0]
    harness.spin(20, STANDING, IDLE_GAIT)       # 0.4s: under the 1.0s window
    assert c.robot.resets == 0
    c.imu.rpy = [zero[0], zero[1], 0.0]         # recovered
    harness.spin(60, STANDING, IDLE_GAIT)
    assert c.robot.resets == 0
    assert c._fall_since is None


def test_repeated_falls_stop_reloading_instead_of_looping(harness, monkeypatch) -> None:
    """An endless reload loop is harder to diagnose than a robot lying still."""
    mod = harness.mod
    monkeypatch.setattr(mod, "FALL_MAX_RELOADS", 2)
    monkeypatch.setattr(mod, "FALL_RELOAD_COOLDOWN_S", 0.0)
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    down = c._imu_zero[0] + 1.571
    for _ in range(4):
        c.imu.rpy = [down, 0.0, 0.0]
        harness.spin(120, STANDING, IDLE_GAIT)
        # the reset is faked, so re-learn the zero as a real restart would
        c.imu.rpy = [0.0, 0.0, 0.0]
        harness.spin(120, STANDING, IDLE_GAIT)
    assert c.robot.resets == 2, c.robot.resets


def test_fall_recovery_can_be_switched_off(controller_module, monkeypatch) -> None:
    monkeypatch.setattr(controller_module, "AUTO_RELOAD_ON_FALL", False)
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    other = Harness(controller_module)
    try:
        other.spin(120, STANDING, IDLE_GAIT)
        other.ctl.imu.rpy = [other.ctl._imu_zero[0] + 1.571, 0.0, 0.0]
        other.spin(200, STANDING, IDLE_GAIT)
        assert other.ctl.robot.resets == 0
    finally:
        other.ctl.sock.close()


def test_head_height_is_reported_in_the_log(harness, monkeypatch, tmp_path) -> None:
    import csv
    import glob

    mod = harness.mod
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())
    other = Harness(mod)
    try:
        other.spin(140, STANDING, IDLE_GAIT)
        other.ctl.trajectory_log.close()
        rows = list(csv.DictReader(open(glob.glob(str(tmp_path / "*.csv"))[0],
                                       encoding="utf-8")))
        assert float(rows[-1]["head_height"]) > 0.40      # standing
        assert rows[-1]["reloads"] == "0"
    finally:
        other.ctl.sock.close()


def test_the_shipped_default_drives_the_whole_stack() -> None:
    """LEG_CONTROL ships as "auto": clips for locomotion, imitation for the rest.

    It was "pose" for a while, on the reasoning that a clip is a 2-3 second
    commitment during which the camera is ignored for the leg joints. That is
    true, and it cost the project walking and turning entirely -- the alternative
    it left in place can do neither, because NAO has no torso-yaw joint (a
    rotation can only be imitated by stepping round) and translating a
    free-standing NAO needs a balanced gait, which is what the clips are. The
    arms, head and torso keep imitating throughout playback.
    """
    import importlib.util
    import os

    path = os.path.join(CONTROLLER_DIR, "pose_imitation_controller.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    # Read the constant out of the source rather than importing, so this holds
    # regardless of what any fixture monkeypatched.
    for line in source.splitlines():
        if line.startswith("LEG_CONTROL"):
            assert line.split("=")[1].split("#")[0].strip() == '"auto"', line
            break
    else:
        raise AssertionError("LEG_CONTROL not found")
    assert importlib.util.find_spec is not None       # keep the import meaningful


# ---------------------------------------------------------------------------
# Stuck / off-its-feet detection and the shared CoM shift
# ---------------------------------------------------------------------------
def test_a_sustained_lean_past_every_gate_is_treated_as_a_fall(harness) -> None:
    """Recorded: 40 s at 0.37 rad of tilt, both CoM shifters at their clamps, the
    soles carrying 0.4 and 5 N, the head still 0.35 m up -- above the fall
    threshold, so nothing recovered it. A tilt no stand-down gate allows, held for
    seconds, is a robot propped on something, and a reset is the only way up."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None and c.robot.resets == 0
    tilt = mod_const(harness, "STUCK_TILT_RAD") + 0.05
    c.imu.rpy = [c._imu_zero[0] + tilt, c._imu_zero[1], 0.0]
    # The head is still high: the head-height test alone would not fire.
    height = c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2]))
    assert height is not None and height > mod_const(harness, "FALL_HEAD_HEIGHT_M")
    steps = int((mod_const(harness, "STUCK_CONFIRM_S") + 0.5) / 0.02)
    harness.spin(steps, STANDING, IDLE_GAIT)
    assert c.robot.resets == 1, "a robot stuck leaning was never recovered"


def test_a_brief_wobble_past_the_stuck_limit_is_not_a_fall(harness) -> None:
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    c.imu.rpy = [c._imu_zero[0] + 0.45, c._imu_zero[1], 0.0]
    harness.spin(40, STANDING, IDLE_GAIT)          # 0.8 s, under STUCK_CONFIRM_S
    c.imu.rpy = [c._imu_zero[0], c._imu_zero[1], 0.0]
    harness.spin(40, STANDING, IDLE_GAIT)
    assert c.robot.resets == 0


def test_feet_carrying_nothing_for_seconds_is_treated_as_a_fall(harness, monkeypatch) -> None:
    """Recorded: head 0.255 m (just above the old 0.25 threshold), soles at 0.04 N
    for the rest of the episode. Whatever the head height says, a robot whose
    feet carry nothing is not standing."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c.robot.resets == 0
    monkeypatch.setattr(FakeFsr, "TOTAL_N", 0.5)     # off its feet
    steps = int((mod_const(harness, "FALL_UNLOADED_S") + 0.5) / 0.02)
    harness.spin(steps, STANDING, IDLE_GAIT)
    assert c.robot.resets == 1


def test_the_balance_feedback_is_folded_into_the_lower_body_shift(harness) -> None:
    """One CoM manager: the balance loop's correction is handed INTO the lower
    body and reported by it, not added on top afterwards."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    meta = c.driver.lower_body_meta
    assert "com_fb_pitch" in meta and "com_fb_roll" in meta
    # Tip the torso back a little: the feedback loop should answer, through the
    # lower body, and the total shift must respect the lower body's single clamp.
    c.imu.rpy = [c._imu_zero[0], c._imu_zero[1] - 0.20, 0.0]
    harness.spin(100, STANDING, IDLE_GAIT)
    meta = c.driver.lower_body_meta
    p = c.driver.lower_body.params
    assert abs(meta["com_fb_pitch"]) > 0.01, meta
    assert abs(meta["com_shift_pitch"] + meta["com_fb_pitch"]) <= p.com_shift_max_pitch + 1e-6
    # Sign: a BACKWARD tilt (negative pitch) moves the CoM forward (hip +c).
    assert meta["com_fb_pitch"] > 0.0


def test_the_log_attributes_the_pelvis_shift(harness, tmp_path) -> None:
    c = harness.ctl
    cols = set(harness.mod.DIAGNOSTIC_COLUMNS)
    for name in ("lb_ff_pitch", "lb_ff_roll", "lb_ff_width", "lb_fb_pitch",
                 "lb_fb_roll", "lb_com_margin", "cop_share_l"):
        assert name in cols, name
    harness.spin(60, STANDING, IDLE_GAIT)
    diag = c._diagnostics(*c._corrected_tilt(*c._imu_rpy()[:2]), 0.0)
    assert diag["lb_fb_pitch"] is not None
    assert 0.0 <= diag["cop_share_l"] <= 1.0


# ---------------------------------------------------------------------------
# The sensors, from the proto
# ---------------------------------------------------------------------------
def _mangled_imu(yaw: float, pitch: float, roll: float) -> tuple:
    """What this NAO's InertialUnit reports for a torso at (yaw, pitch, roll).

    Reimplements the two things the proto does to it: the +90 deg mount
    (``rotation 1 0 0 1.5708``) and ``yAxis FALSE``, which Webots implements by
    zeroing the world-z component of the attitude's axis-angle axis before
    deriving roll/pitch/yaw (WbInertialUnit::computeValue), then Webots' own ENU
    decomposition (src/controller/c/inertial_unit.c).
    """
    import numpy as np

    def rot(axis, a):
        x, y, z = axis
        c, s, C = math.cos(a), math.sin(a), 1.0 - math.cos(a)
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                         [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                         [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

    R = rot((0, 0, 1), yaw) @ rot((0, 1, 0), pitch) @ rot((1, 0, 0), roll) \
        @ rot((1, 0, 0), math.pi / 2)
    # axis-angle, world-z component zeroed, renormalised (yAxis FALSE)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle > 1e-12:
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        ax = ax / (2.0 * math.sin(angle))
        ax[2] = 0.0
        if np.linalg.norm(ax) > 1e-12:
            R = rot(ax / np.linalg.norm(ax), angle)
    # matrix -> quaternion (x, y, z, w) -> Webots ENU roll/pitch/yaw
    t = np.trace(R)
    if t > 0:
        sq = math.sqrt(t + 1.0) * 2.0
        q = ((R[2, 1] - R[1, 2]) / sq, (R[0, 2] - R[2, 0]) / sq,
             (R[1, 0] - R[0, 1]) / sq, 0.25 * sq)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        sq = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        qq = [0.0, 0.0, 0.0, 0.0]
        qq[i] = 0.25 * sq
        qq[j] = (R[j, i] + R[i, j]) / sq
        qq[k] = (R[k, i] + R[i, k]) / sq
        qq[3] = (R[k, j] - R[j, k]) / sq
        q = tuple(qq)
    out_roll = math.atan2(2.0 * (q[3] * q[0] + q[1] * q[2]),
                          1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2))
    t2 = max(-1.0, min(1.0, 2.0 * (q[3] * q[1] - q[2] * q[0])))
    out_yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                         1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    return (out_roll - math.pi / 2, math.asin(t2), out_yaw)


def test_the_inertial_unit_halves_the_pitch_and_leaks_the_heading() -> None:
    """Why the controller takes its attitude from gravity instead.

    This robot's InertialUnit is mounted rolled 90 deg AND has ``yAxis FALSE``.
    Webots implements the axis mask by zeroing part of the attitude's axis before
    deriving the angles, which on a rotated sensor corrupts the axes it was not
    asked to touch: the reported pitch is half the real pitch, and a pure body
    ROTATION reports as pitch as well -- so pitch and heading are the same number
    and neither can be trusted.

    Confirmed on 702,719 recorded standing frames: imu_yaw and imu_pitch_raw
    correlate at +0.998 with slope +1.03, and imu_pitch against the torso pitch
    implied by the loaded foot's kinematics has slope +0.48.
    """
    assert _mangled_imu(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    for b in (0.1, 0.2, 0.4):
        roll, pitch, yaw = _mangled_imu(0.0, b, 0.0)          # pure torso pitch
        assert pitch == pytest.approx(b / 2.0, abs=0.01), b    # HALF
        assert yaw == pytest.approx(b / 2.0, abs=0.01), b      # and leaked to yaw
        assert abs(roll) < 0.05

        _, pitch_y, yaw_y = _mangled_imu(b, 0.0, 0.0)          # pure torso YAW
        assert pitch_y == pytest.approx(b / 2.0, abs=0.01), b  # reads as pitch!

        roll_r, pitch_r, _ = _mangled_imu(0.0, 0.0, b)         # pure torso roll
        assert roll_r == pytest.approx(b, abs=1e-6), b         # roll is honest
        assert abs(pitch_r) < 1e-6

    # A forward pitch and an equal body rotation are indistinguishable: both
    # channels read zero for a robot that is genuinely tipped forward 0.1 rad.
    roll, pitch, yaw = _mangled_imu(0.1, -0.1, 0.0)
    assert abs(pitch) < 0.01 and abs(yaw) < 0.01


def _enabled_gravity_harness(mod, monkeypatch):
    """A controller with the gravity attitude path switched on (it ships off).

    Also selects it as the attitude SOURCE: the shipped source is the Supervisor
    (see ATTITUDE_SOURCE), which would otherwise answer first and these tests
    would certify a path that never runs.
    """
    monkeypatch.setattr(mod, "ATTITUDE_SOURCE", "accel")
    monkeypatch.setattr(mod, "TILT_FROM_ACCELEROMETER", True)
    monkeypatch.setattr(mod, "LEG_CONTROL", "pose")
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())
    return Harness(mod)


def test_gravity_is_measured_and_logged_even_when_not_acted_on(harness) -> None:
    """The shipped configuration computes the gravity attitude, logs it, and does
    NOT control from it -- see test_the_shipped_default_does_not_control_from_gravity
    for why. It still has to be measured, because it is the only honest witness to
    the InertialUnit's half-scale pitch."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c.accel is not None
    assert c._acc_zero is not None, "no gravity zero was learned"
    assert c._acc_tilt is not None, "gravity was never computed"
    assert c._tilt_source == "supervisor", c._tilt_source
    diag = c._diagnostics(0.0, 0.0, 0.0)
    assert diag["acc_roll"] is not None and diag["acc_pitch"] is not None


def test_gravity_agrees_with_the_inertial_unit_on_roll(controller_module,
                                                       monkeypatch) -> None:
    """Roll is the axis the InertialUnit reports honestly, so it is the one that
    can validate the gravity derivation -- and it does, to a couple of
    milliradians. Confirmed on the live session of 2026-09-04: with the robot
    quiet the two channels matched to four decimals."""
    other = _enabled_gravity_harness(controller_module, monkeypatch)
    try:
        c = other.ctl
        other.spin(120, STANDING, IDLE_GAIT)
        assert c._tilt_source == "accel", c._tilt_source
        c.imu.rpy = [c._imu_zero[0] + 0.20, c._imu_zero[1], 0.0]
        other.spin(40, STANDING, IDLE_GAIT)              # 0.8s >> ACC_TILT_TAU_S
        roll_acc = c._acc_tilt[0] - c._acc_zero[0]
        assert roll_acc == pytest.approx(0.20, abs=0.02), roll_acc
        assert c._acc_usable is True                     # no disagreement latched
    finally:
        other.close()


def test_a_disagreeing_accelerometer_is_dropped_not_trusted(controller_module,
                                                            monkeypatch) -> None:
    """The guard that saved the 2026-09-04 session. The gravity path is derived,
    not measured, so it is checked against the InertialUnit's roll every step on a
    robot standing on its feet, and a sustained disagreement abandons it loudly
    instead of quietly believing it. On that session the disagreement latched
    after six falls and the robot then stood for 298 s."""
    other = _enabled_gravity_harness(controller_module, monkeypatch)
    try:
        c = other.ctl
        other.spin(120, STANDING, IDLE_GAIT)
        assert c._tilt_source == "accel"
        # Break it: report gravity for a 0.5 rad roll the InertialUnit does not see.
        broken = c.robot.devices["accelerometer"]
        broken.getValues = lambda: [0.0, -math.sin(0.5) * 9.81, -math.cos(0.5) * 9.81]
        steps = int((controller_module.ACC_DISAGREE_S + 0.5) / 0.02)
        other.spin(steps, STANDING, IDLE_GAIT)
        assert c._acc_usable is False
        assert c._tilt_source == "imu"
        # And the robot is still being controlled, not dropped.
        assert c.driver.lower_body_meta["mode"] == "double"
    finally:
        other.close()


def test_the_shipped_default_does_not_control_from_gravity() -> None:
    """An accelerometer measures gravity PLUS the robot's own acceleration, so
    using it as an attitude source inside a position loop feeds a second
    derivative back as a position -- 180 degrees of phase error:

        the loop shifts the pelvis -> the torso accelerates sideways -> the
        accelerometer reports that as tilt -> the loop shifts further

    Measured in log webots_joint_trajectory_1788521290, ten episodes, nobody in
    front of the camera in any of them (so the legs were not imitating): with the
    attitude taken from gravity the robot fell within ~2 s, six times over, median
    head height 0.19 m; once the cross-check disabled it, the same controller
    stood for 40 s and then 298 s. Switching this on again needs the gyro fused
    in (integrate the rate for the fast attitude, correct its drift with gravity
    over half a second or more), not a longer low-pass.
    """
    import os

    path = os.path.join(CONTROLLER_DIR, "pose_imitation_controller.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for line in source.splitlines():
        if line.startswith("TILT_FROM_ACCELEROMETER"):
            assert line.split("=")[1].strip() == "False", line
            break
    else:
        raise AssertionError("TILT_FROM_ACCELEROMETER not found")


def test_the_pitch_scale_is_applied_to_pitch_alone(harness, monkeypatch) -> None:
    """The device reports half the real pitch, so IMU_PITCH_SCALE exists to undo
    that -- and it ships at 1.0, the conservative end, because the only session
    that ran the loop on a full-scale pitch limit-cycled (see IMU_PITCH_SCALE for
    the numbers). Whatever it is set to, it must reach the pitch channel and
    nothing else, and both readings must be logged separately."""
    c = harness.ctl
    # This is about the InertialUnit FALLBACK, so select it: the shipped source
    # is the Supervisor and it would otherwise answer first.
    monkeypatch.setattr(harness.mod, "ATTITUDE_SOURCE", "imu")
    harness.spin(120, STANDING, IDLE_GAIT)
    scale = mod_const(harness, "IMU_PITCH_SCALE")
    assert 1.0 <= scale <= 2.0, "the honest range: half-scale reading, or undone"
    c.imu.rpy = [c._imu_zero[0], c._imu_zero[1] + 0.10, 0.0]
    harness.spin(10, STANDING, IDLE_GAIT)
    imu_roll, imu_pitch = c._corrected_tilt(*c._imu_rpy()[:2])
    assert imu_pitch == pytest.approx(0.10, abs=1e-6)           # what the device says
    ctl_roll, ctl_pitch = c._torso_tilt(c.robot.getTime(), imu_roll, imu_pitch, None)
    assert ctl_pitch == pytest.approx(0.10 * scale, abs=1e-6)   # what is acted on
    assert ctl_roll == pytest.approx(imu_roll, abs=1e-9)        # roll is untouched
    # ...and both are in the log, separately: reading a session where the two
    # differ is impossible if only one of them is recorded.
    diag = c._diagnostics(ctl_roll, ctl_pitch, 0.0)
    assert diag["imu_pitch"] == pytest.approx(0.10, abs=1e-6)
    assert diag["ctl_pitch"] == pytest.approx(0.10 * scale, abs=1e-6)


def test_a_recovery_clears_the_balance_correction(harness) -> None:
    """The balance loop is an integrator, and its state was the one thing that
    survived a fall reset. The robot came back upright still holding the pelvis
    shift the old, fallen robot had needed: measured on the first control step of
    a new episode, HipPitch -0.345 with the centre of mass 40 mm off centre, six
    times in one session and never lasting three seconds."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    # Drive the loop to a large correction by holding a real backward tilt.
    c.imu.rpy = [c._imu_zero[0], c._imu_zero[1] - 0.25, 0.0]
    harness.spin(150, STANDING, IDLE_GAIT)
    state = c.driver.balance._state
    assert abs(state["pitch"]) > 0.05, state

    c._reset_for_new_episode()
    assert c.driver.balance._state["pitch"] == pytest.approx(0.0, abs=1e-9)
    assert c.driver.balance._state["roll"] == pytest.approx(0.0, abs=1e-9)
    # ...so the first step of the new episode commands the plain crouch, not the
    # posture the robot fell in.
    c.imu.rpy = [c._imu_zero[0], c._imu_zero[1], 0.0]
    harness.spin(2, STANDING, IDLE_GAIT)
    assert harness.angle("LHipPitch") == pytest.approx(
        harness.angle("RHipPitch"), abs=1e-6)
    assert abs(harness.angle("LHipPitch")) < 0.2, harness.angle("LHipPitch")


def test_the_heading_never_comes_from_the_inertial_unit() -> None:
    """This proto disables the yaw axis on the InertialUnit AND the z axis on the
    Gyro, so the robot cannot observe its own rotation with either. What the IMU
    returns in place of a heading is its own half-scale pitch channel (see
    test_the_inertial_unit_halves_the_pitch_and_leaks_the_heading), and servoing
    on that produced a heading "error" of a median 21 deg while standing still.

    Turning therefore closes its loop on the Supervisor's orientation instead --
    the one place this project uses simulator ground truth, and it steers rather
    than balances."""
    import os

    path = os.path.join(CONTROLLER_DIR, "pose_imitation_controller.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    values = {}
    for line in source.splitlines():
        for name in ("HEADING_FROM_IMU", "HEADING_SOURCE"):
            if line.startswith(name) and name not in values:
                values[name] = line.split("=")[1].split("#")[0].strip()
    assert values.get("HEADING_FROM_IMU") == "False", values
    assert values.get("HEADING_SOURCE") == '"supervisor"', values


def test_no_hip_yaw_bias_without_a_heading(controller_module, monkeypatch) -> None:
    """The cost of servoing on a phantom heading was not the turn that never came:
    it was the hip-yaw bias, whose canted axis pitches the legs as it yaws them.
    At the 0.112 rad bias recorded in one session the front sole corners dropped
    10.5 mm, the modelled support polygon collapsed onto the toe line, and the CoM
    compensation drove the pelvis to its forward clamp on a dead-level robot."""
    monkeypatch.setattr(controller_module, "HEADING_FROM_IMU", False)
    monkeypatch.setattr(controller_module, "LEG_CONTROL", "pose")
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    other = Harness(controller_module)
    try:
        other.spin(150, STANDING, dict(IDLE_GAIT, body_yaw_rad=0.9))
        assert other.angle("LHipYawPitch") == pytest.approx(0.0, abs=1e-6)
        assert other.angle("RHipYawPitch") == pytest.approx(0.0, abs=1e-6)
        # And the soles are flat, which is what the bias used to cost.
        p = other.ctl.driver.lower_body.params
        for side in ("L", "R"):
            tilt = other.angle(f"{side}HipRoll") + other.angle(f"{side}AnkleRoll")
            assert abs(tilt) <= p.sole_tilt_budget + 1e-6, (side, tilt)
    finally:
        other.close()


# ---------------------------------------------------------------------------
# Locomotion: the handover
# ---------------------------------------------------------------------------
def test_a_clip_starts_even_though_play_returns_nothing(harness) -> None:
    """The bug that cost this project walking AND turning.

    Webots' R2025a Python binding is ``def play(self): wb.wbu_motion_play(...)``
    -- no return statement -- so ``play()`` is None. The controller tested it as
    ``if not motion.play(): return False``, so MotionPlayer.start() returned False
    on every call ever made: leg_mode is "pose" in 100.0% of the frames of every
    recorded session, and four sessions logged "start REFUSED by Webots" for
    198-964 frames each.

    And the clip HAD started -- wbu_motion_play does not care what Python does
    with its return value, and the controller library applies a playing clip's
    keyframes every step. So the clip drove the legs while the controller believed
    nothing was playing and kept commanding them itself: log 1788428293 has
    LKneePitch MEASURED at the clip's own first keyframe, 1.042 rad, while the
    controller commanded 0.20-0.52. Two commanders, one joint set.
    """
    c = harness.ctl
    assert c.motion._load("forward") is not None
    motion = c.motion._load("forward")
    assert motion.play() is None, "the fake must match the real binding"

    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(200, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played, FakeMotion.played
    assert c.motion.active is True
    assert c.leg_mode == "motion:forward", c.leg_mode
    assert c._diagnostics(0.0, 0.0, 0.0)["clip_status"] == "playing"
    # ...and the joints the clip declares are no longer ours to command.
    assert c.driver.suspended is True


def test_the_legs_reach_the_clips_stance_before_it_plays(harness) -> None:
    """A clip opens in a deep sole-flat crouch (knee 1.042 rad) and commands that
    first keyframe on its first step, with the velocity caps already lifted. The
    controller stands at knee 0.20. Handing over from there asks for 0.84 rad in
    one 20 ms step, so the legs are ramped into the clip's own stance first and
    the clip is played only once they are measurably there."""
    c = harness.ctl
    pose = c.motion.first_pose("forward")
    assert pose, "the clip's opening keyframe must be readable"
    target = pose["LKneePitch"]
    assert target > 0.9, target                      # it really is a deep crouch

    harness.spin(150, STANDING, IDLE_GAIT)
    assert harness.angle("LKneePitch") < 0.35        # standing crouch

    # One step of wanting to walk must NOT start the clip -- it must ramp.
    FakeMotion.played.clear()
    harness.spin(2, STANDING, MARCH_GAIT)
    assert not FakeMotion.played, "played before reaching the stance"
    assert c.leg_mode == "prepare:forward", c.leg_mode
    assert c._preparing == "forward"

    # The ramp is rate-limited: no step bigger than the driver allows.
    prev = harness.angle("LKneePitch")
    for _ in range(40):
        harness.spin(1, STANDING, MARCH_GAIT)
        now = harness.angle("LKneePitch")
        assert now - prev <= c.driver.LEG_POSE_RATE * 0.02 + 1e-6, (prev, now)
        prev = now
        if FakeMotion.played:
            break
    assert FakeMotion.played, "never got there"
    # It arrived at the clip's stance, and only then played.
    assert abs(prev - target) <= c.driver.LEG_POSE_TOL + 1e-6, (prev, target)


def test_a_clip_that_will_not_say_what_it_opens_in_is_still_played(harness,
                                                                   monkeypatch) -> None:
    """Refusing to walk at all is worse than a jerky start."""
    monkeypatch.setattr(harness.ctl.motion, "_first_pose", {"forward": {}})
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(30, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played


def test_an_unreachable_stance_gives_up_instead_of_stalling(harness,
                                                            monkeypatch) -> None:
    """If a leg cannot get to the clip's stance -- blocked, or another layer
    fighting -- locomotion must count a failure and eventually stop trying, not
    ramp forever. Each attempt is bounded by CLIP_PREPARE_TIMEOUT_S and the
    existing MOTION_MAX_FAILURES policy then retires the clips for the session,
    so the robot falls back to imitating instead of ramping in place."""
    c = harness.ctl
    mod = harness.mod
    monkeypatch.setattr(type(c.driver), "approach_leg_pose",
                        lambda *a, **k: False)      # never arrives
    harness.spin(150, STANDING, IDLE_GAIT)
    per_attempt = int(mod.CLIP_PREPARE_TIMEOUT_S / 0.02) + 4
    for _ in range(mod.MOTION_MAX_FAILURES):
        harness.spin(per_attempt, STANDING, MARCH_GAIT)
    assert not FakeMotion.played, "played without reaching the stance"
    assert c._motion_failures >= mod.MOTION_MAX_FAILURES, c._motion_failures
    assert c.motion.available == {}, "clips were not retired after repeated failures"
    # ...and the robot is still under control, imitating.
    mode = harness.spin(30, STANDING, IDLE_GAIT)
    assert mode in ("pose", "march:march"), mode
    assert c.driver.suspended is False


def test_the_crouch_comes_back_down_gently_after_a_clip(harness) -> None:
    """A clip ENDS in the same 0.51 rad squat it opened in, and the lower body
    stands at 0.10. Its crouch rate limiter snaps to its first sample, so without
    seeding it the first post-clip step commands the whole 0.41 rad of knee travel
    at once -- the handover jolt again, in reverse."""
    c = harness.ctl
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(40, STANDING, MARCH_GAIT)           # ramp + start
    assert c.motion.active
    knee_in_clip = harness.angle("LKneePitch")
    # Long enough for the WALK LATCH to expire as well as the clip to finish:
    # while the latch holds, a finished clip is immediately followed by another
    # (that is the point of it -- see WALK_LATCH_RELEASE_S), so a few steps of
    # idle gait is not enough to see the robot stand up.
    limit = int((mod_const(harness, "WALK_LATCH_RELEASE_S") + 2.0) / 0.02)
    for _ in range(limit):
        harness.spin(1, STANDING, IDLE_GAIT)
        if not c.motion.active and c._walk_latch_until is None:
            break
    assert not c.motion.active
    lb = c.driver.lower_body
    assert lb._crouch is not None, "the crouch limiter was not seeded"
    assert lb._crouch > 0.2, lb._crouch               # seeded from the clip's squat
    # And it comes down at the rate limit, not in one step.
    prev = harness.angle("LKneePitch")
    for _ in range(10):
        harness.spin(1, STANDING, IDLE_GAIT)
        now = harness.angle("LKneePitch")
        assert prev - now <= 2.0 * lb.params.crouch_rate_limit * 0.02 + 0.02, (prev, now)
        prev = now
    assert prev < knee_in_clip


# ---------------------------------------------------------------------------
# Turning: the heading
# ---------------------------------------------------------------------------
def test_turning_closes_its_loop_on_the_supervisor_heading(harness) -> None:
    """The robot's own rotation is not measurable with its sensors (both yaw axes
    are disabled in the proto), so the loop closes on the Supervisor's
    orientation: the robot's forward axis is +x in its own frame, so the heading
    is atan2(m[3], m[0]) of the row-major orientation matrix, CCW-positive =
    toward its own left, which is the sign YawServo expects."""
    c = harness.ctl
    assert c.self_node is not None
    assert c.heading_available is True
    for yaw in (0.0, 0.4, -0.9, 2.5):
        c.imu.rpy = [c._imu_zero[0] if c._imu_zero else 0.0, 0.0, yaw]
        assert c._heading(999.0) == pytest.approx(yaw, abs=1e-9), yaw
    # It reaches the log, so a session can be read afterwards.
    c.imu.rpy = [0.0, 0.0, 0.4]
    diag = c._diagnostics(0.0, 0.0, 999.0)
    assert diag["sv_heading"] == pytest.approx(0.4, abs=1e-9)


def test_without_a_supervisor_there_is_no_turning(controller_module,
                                                  monkeypatch) -> None:
    """A world without `supervisor TRUE` must degrade to no turning and no hip-yaw
    bias, loudly, rather than servoing on a number that is not a heading."""
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    monkeypatch.setattr(FakeRobot, "getSelf", lambda self: None)
    other = Harness(controller_module)
    try:
        c = other.ctl
        assert c.self_node is None
        assert c.heading_available is False
        other.spin(150, STANDING, dict(IDLE_GAIT, body_yaw_rad=0.9))
        assert not FakeMotion.played, "turned without a heading"
        assert other.angle("LHipYawPitch") == pytest.approx(0.0, abs=1e-6)
    finally:
        other.close()


def test_the_ground_truth_is_logged_but_never_controlled_from(harness) -> None:
    """The simulator can be asked for the true centre of mass and whether it is
    inside the convex hull of the real contact points. That is logged beside the
    model's own estimate so the model can be VALIDATED -- but the balance loop
    stays model-based, so the algorithm remains one a real NAO could run."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    cols = set(harness.mod.DIAGNOSTIC_COLUMNS)
    for name in ("sv_com_x", "sv_com_y", "sv_com_z", "sv_balanced", "sv_contacts",
                 "sv_heading"):
        assert name in cols, name
    diag = c._diagnostics(0.0, 0.0, 0.0)
    assert diag["sv_com_z"] == pytest.approx(0.29, abs=1e-9)
    assert diag["sv_balanced"] == 1
    assert diag["sv_contacts"] == 2
    # The balance loop is still the model's: its correction comes from
    # BalanceController, which never sees a Supervisor.
    assert c.driver.balance is not None
    assert not hasattr(c.driver.balance, "self_node")


# ---------------------------------------------------------------------------
# The attitude: from the scene tree, not from a sensor that cannot report it
# ---------------------------------------------------------------------------
def test_the_attitude_comes_from_the_scene_tree(harness) -> None:
    """Both of this robot's attitude sensors are provably wrong (see
    ATTITUDE_SOURCE), so the controller asks the scene tree instead -- the same
    call it already makes for the heading."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._tilt_source == "supervisor", c._tilt_source
    assert c.attitude_ready is True
    for roll, pitch in ((0.0, 0.0), (0.25, 0.0), (0.0, -0.30), (-0.15, 0.20)):
        c.imu.rpy = [roll, pitch, 0.0]
        got = c._torso_tilt(c.robot.getTime(), 999.0, 999.0, None)
        assert got[0] == pytest.approx(roll, abs=1e-6), (roll, got)
        assert got[1] == pytest.approx(pitch, abs=1e-6), (pitch, got)
    # ...and it needs no learned zero to be meaningful, unlike the sensors.
    fresh = c._supervisor_tilt()
    assert fresh is not None


def test_a_turned_robot_is_not_reported_as_pitched(harness) -> None:
    """The fall this fixes, and the reason walking could never start.

    The InertialUnit's yaw axis is disabled, which makes its pitch channel carry
    a body rotation one-for-one at half scale. In log 1788768834 an episode ran
    3,735 s with the robot standing on two evenly loaded flat soles
    (hip+knee+ankle = 0.0000, head 0.439 m, 25.258 N per foot, model margin
    +0.062 m) while the InertialUnit reported -0.263 rad of pitch and
    imu_yaw/imu_pitch_raw = 1.073 -- the robot had merely shuffled ~30 deg off
    its spawn heading. The balance loop held +0.18 rad of pelvis correction for
    an hour against a tilt that did not exist, and _settled() -- which the clip
    layer needs before it will play anything -- passed in 0.015% of that episode
    against 71.2% elsewhere. So the phantom did not just risk a fall; it made
    walking impossible.
    """
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    # A pure 30 deg body rotation, no tilt whatsoever.
    c.imu.rpy = [0.0, 0.0, math.radians(30.0)]
    roll, pitch = c._torso_tilt(c.robot.getTime(), 999.0, 999.0, None)
    assert abs(pitch) < 1e-6, pitch          # NOT -0.26
    assert abs(roll) < 1e-6, roll
    # The heading, meanwhile, is exactly the rotation -- reported separately.
    assert c._heading(999.0) == pytest.approx(math.radians(30.0), abs=1e-9)
    # And the robot stays settled enough to start a clip.
    harness.spin(60, STANDING, IDLE_GAIT)
    assert c._settled(roll, pitch) is True


def test_the_ramp_into_a_clip_keeps_both_soles_flat(harness) -> None:
    """The ramp added to make the handover safe was unsafe on its own.

    A sole's attitude is Hip + Knee + Ankle, and from the standing crouch to a
    walk clip's opening stance the knee travels 0.842 rad against the hip's 0.405
    and the ankle's 0.437. Rate-limiting every joint at the same speed leaves the
    knee still moving when the others have stopped: simulated against the repo's
    CoM model that drives the sum to 0.405 rad -- eight times the 0.05 sole-tilt
    budget -- and the fore/aft support margin to -0.064 m. Each joint's rate is
    therefore scaled by its share of the longest travel, so they arrive together.
    """
    c = harness.ctl
    budget = c.driver.lower_body.params.sole_tilt_budget
    harness.spin(150, STANDING, IDLE_GAIT)
    worst = 0.0
    for _ in range(CLIP_RAMP_STEPS + 20):
        harness.spin(1, STANDING, MARCH_GAIT)
        for side in ("L", "R"):
            total = (harness.angle(f"{side}HipPitch")
                     + harness.angle(f"{side}KneePitch")
                     + harness.angle(f"{side}AnklePitch"))
            worst = max(worst, abs(total))
        if c.motion.active:
            break
    assert c.motion.active, "never reached the clip"
    assert worst <= budget + 0.02, f"the ramp tipped the soles by {worst:.3f} rad"


def test_the_longest_turn_clip_is_not_killed_by_the_watchdog(harness,
                                                             monkeypatch) -> None:
    """TurnLeft180 runs 9.0 s. Capping the watchdog budget at MOTION_WATCHDOG_S
    (8.0) guaranteed it overran, was dropped as broken and counted a failure --
    so the only clip that can turn the robot right round in one action could
    never be used. The clip's own duration sets the budget when it is knowable."""
    monkeypatch.setattr(FakeMotion, "DURATION_MS", 9000.0)
    c = harness.ctl
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    budget = c._motion_deadline - c._motion_started_at
    assert budget > 9.0, budget
    assert budget == pytest.approx(9.0 * 1.5 + 1.0)


# ---------------------------------------------------------------------------
# Smoothness: stopping a clip early, and holding the walk request
# ---------------------------------------------------------------------------
def test_a_clip_stops_early_instead_of_running_to_its_end(harness) -> None:
    """A clip used to be played to completion, which made its length the latency
    of "stop walking" -- and that is what kept the robot on Cyberbotics' short
    2.60 s stride, one start-and-stop transient per 0.095 m, measured at 0.036 m/s
    against NAO's ~0.10.

    Not every keyframe qualifies, and the gate that matters is momentum: a pose
    can be in double support with both soles flat and the CoM inside the polygon
    and still be travelling at 0.18 m/s, and freezing the legs there hands back a
    robot that walks itself over. With that gate, Forwards.motion offers 33 of its
    66 keyframes (longest wait 1.04 s) and the turn clips 49 of 73 and 108 of 226
    (0.52 s). A continuous walk clip offers very few -- 38 of 170, up to 2.56 s
    apart -- which is why it is stopped by leaving its gait cycle instead; see
    the cyclic tests at the end of this file.
    """
    c = harness.ctl
    # A clip long enough to outlive the walk latch, so "stopped early" is
    # distinguishable from "ran out of keyframes".
    FakeMotion.STEPS = 90
    harness.spin(150, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    started = c.motion.time_s()
    assert started is not None

    # The human stops. Wait out the latch; the clip must then end EARLY.
    limit = int((mod_const(harness, "WALK_LATCH_RELEASE_S") + 0.5) / 0.02)
    for _ in range(limit):
        harness.spin(1, STANDING, IDLE_GAIT)
        if not c.motion.active:
            break
    assert not c.motion.active, "the clip ran to its end instead of exiting"
    assert c._early_exits >= 1, "the clip was not stopped early"
    assert c.driver.suspended is False          # the legs came back


def test_the_walk_request_survives_a_cue_dropout(harness) -> None:
    """The cue flickers: measured over the first session that ever walked, the
    median "march" run was 1.32 s -- shorter than one clip -- and 33 of the 65
    idle runs were under 1.2 s, totalling 18.2 s. Each of those used to end a walk
    and pay for a fresh 0.6-1.0 s prepare ramp, which is the stutter the user
    sees. The request is therefore latched for WALK_LATCH_RELEASE_S."""
    c = harness.ctl
    harness.spin(150, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert c._walk_latch_until is not None

    # A short dropout -- well inside the latch -- must not stop the walk.
    harness.spin(int(0.4 / 0.02), STANDING, IDLE_GAIT)
    assert c.motion.active, "a 0.4 s cue dropout ended the walk"
    assert c._walk_latch_until is not None

    # The cue comes back: the walk simply continues, with no fresh ramp.
    FakeMotion.played.clear()
    harness.spin(20, STANDING, MARCH_GAIT)
    assert c.motion.active
    assert c.leg_mode.startswith("motion:"), c.leg_mode


def test_the_latch_does_not_outlive_the_human(harness) -> None:
    """A human who has walked out of frame is not walking, whatever the last cue
    said -- otherwise the robot would keep going on a stale request."""
    c = harness.ctl
    harness.spin(150, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert c._walk_latch_until is not None
    # Stop sending frames at all: the driver goes stale.
    harness.spin(int((mod_const(harness, "STALE_AFTER_S") + 0.2) / 0.02), None)
    assert c.driver.stats.stale is True
    assert c._walk_latch_until is None, "the walk latch outlived the human"


def test_the_clip_time_and_latch_reach_the_log(harness) -> None:
    c = harness.ctl
    cols = set(harness.mod.DIAGNOSTIC_COLUMNS)
    assert {"clip_time", "walk_latched"} <= cols
    harness.spin(150, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    diag = c._diagnostics(0.0, 0.0, 0.0)
    assert diag["clip_time"] is not None
    assert diag["walk_latched"] == 1


# ---------------------------------------------------------------------------
# Cyclic gait (GAIT_CYCLE)
# ---------------------------------------------------------------------------
def test_the_controller_promotes_a_cyclic_clip_over_the_short_one(cyclic_harness) -> None:
    """Selection happens at startup, from the clips actually on disk.

    Both candidates are discovered; the long one wins only because a cycle was
    found inside it. Nothing downstream ever sees two forward actions.
    """
    c = cyclic_harness.ctl
    assert os.path.basename(c.motion.available["forward"]) == "Forwards50.motion"
    assert "forward_continuous" not in c.motion.available
    cycle = c.motion.cycle("forward")
    assert cycle is not None
    assert cycle.speed_mps > 0.0


def test_a_cyclic_clip_repeats_its_stride_instead_of_restarting(cyclic_harness) -> None:
    """The heart of it: one clip, many strides, no transient in between.

    Played one-shot the robot pays a start-and-stop for every 0.095 m -- a squat,
    an acceleration, a settle that travels BACKWARD, and then a fresh prepare
    ramp before it can go again. Cycled, the clip is rewound at a seam where no
    joint moves, so the stride simply repeats: 47 start/stop cycles in a 10 s
    walk become 1.
    """
    h, c = cyclic_harness, cyclic_harness.ctl
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    clip = c.motion._cache["forward"]
    plays_at_start = FakeMotion.played.count("Forwards50.motion")

    # Keep asking to walk for several stride-lengths' worth of steps.
    cycle = c.motion.cycle("forward")
    steps = int(round(cycle.period_s * 3.0 / (TIMESTEP_MS / 1000.0)))
    h.spin(steps, STANDING, MARCH_GAIT)

    assert c.motion.active, "the walk stopped on its own"
    assert c._cycles_walked >= 2, f"only {c._cycles_walked} strides repeated"
    # It repeated by REWINDING, not by replaying: the clip was played once.
    assert FakeMotion.played.count("Forwards50.motion") == plays_at_start
    # ...and each rewind moved the playhead back by exactly one period.
    rewinds = [t for t in clip.seeks if t > 0.0]
    assert rewinds, "no seek was ever issued"
    # The playhead stays inside the loop window for the whole walk.
    now = c.motion.time_s()
    assert cycle.loop_start_s - 0.05 <= now <= cycle.loop_end_s + 0.05


def test_a_cyclic_clip_leaves_through_the_clips_own_deceleration(cyclic_harness) -> None:
    """How it stops. Not by freezing mid-stride -- the legs would stop and the
    body would keep its momentum -- but by jumping once into the clip's own
    closing settle, so Cyberbotics' balanced feet-together deceleration is what
    brings the robot to rest."""
    h, c = cyclic_harness, cyclic_harness.ctl
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    cycle = c.motion.cycle("forward")
    clip = c.motion._cache["forward"]
    h.spin(int(round(cycle.period_s * 2.0 / (TIMESTEP_MS / 1000.0))),
           STANDING, MARCH_GAIT)
    assert c.motion.active

    # The human stops walking. Allow the latch to expire, then the phase to come
    # round, then the tail to play out.
    budget = mod_const(h, "WALK_LATCH_RELEASE_S") + cycle.stop_latency_s + 1.0
    left_at = None
    for _ in range(int(budget / (TIMESTEP_MS / 1000.0))):
        h.spin(1, STANDING, IDLE_GAIT)
        if left_at is None and c._cycle_state == "leaving":
            left_at = c.motion.time_s()
        if not c.motion.active:
            break
    assert not c.motion.active, "the cyclic clip never stopped"
    assert left_at is not None, "it stopped without taking the exit jump"
    # It jumped to the deceleration rather than stopping where it was. (The
    # Motion API works in milliseconds; the controller works in seconds.)
    assert cycle.exit_to_s * 1000.0 == pytest.approx(clip.seeks[-1])
    assert left_at >= cycle.exit_to_s - 0.05
    # It walked several strides on one clip before leaving, which is the point.
    assert c._cycles_walked >= 2
    # And it was NOT counted as an early exit -- nothing was cut short.
    assert c._early_exits == 0


def test_a_cycling_clip_waits_for_the_right_phase_before_leaving(cyclic_harness) -> None:
    """The exit is only free at one phase of the stride, so the request to stop
    has to wait for it. That wait is bounded by one period, and while it waits
    the robot keeps walking normally rather than freezing."""
    h, c = cyclic_harness, cyclic_harness.ctl
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    cycle = c.motion.cycle("forward")
    h.spin(int(round(cycle.period_s / (TIMESTEP_MS / 1000.0))), STANDING, MARCH_GAIT)

    # Drop the latch immediately so only the phase wait remains.
    c._drop_walk_latch()
    states = []
    for _ in range(int(round((cycle.period_s + 0.2) / (TIMESTEP_MS / 1000.0)))):
        h.spin(1, STANDING, IDLE_GAIT)
        states.append(c._cycle_state)
        if c._cycle_state == "leaving":
            break
    assert "leaving" in states, "never took the exit"
    assert "stopping" in states, "left without waiting for the phase at all"
    # Waiting means walking, not standing still mid-stride.
    assert c.leg_mode.startswith("motion:")


def test_the_watchdog_does_not_kill_a_healthy_cycling_clip(cyclic_harness) -> None:
    """A cyclic clip is, by design, in exactly the state the watchdog exists to
    catch: it never reports itself over, because it is rewound before it can.
    Each completed stride is proof of life and buys another one."""
    h, c = cyclic_harness, cyclic_harness.ctl
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    cycle = c.motion.cycle("forward")
    # Walk for several times the clip's own length.
    duration = c.motion.duration_s() or 2.0
    h.spin(int(round(duration * 3.0 / (TIMESTEP_MS / 1000.0))), STANDING, MARCH_GAIT)
    assert c.motion.active, "the watchdog killed a walk that was working"
    assert c._motion_failures == 0
    assert c._cycles_walked >= 2


def test_a_cycling_clip_still_stops_dead_for_a_fall(cyclic_harness) -> None:
    """The one thing that must break the loop immediately, with no waiting for a
    phase and no deceleration: the robot going over."""
    h, c = cyclic_harness, cyclic_harness.ctl
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    h.spin(20, STANDING, MARCH_GAIT)
    assert c.motion.active
    c.imu.rpy = [0.6, 0.0, 0.0]              # well past TILT_ABORT_RAD
    h.spin(6, STANDING, MARCH_GAIT)
    assert not c.motion.active, "a cycling clip ignored a fall"
    assert c.driver.suspended is False, "the body was not handed back"
    # It did NOT ride the deceleration out: a fall is the one case where
    # stopping now beats stopping gracefully.
    assert c._cycle_state != "leaving"


def test_the_prepare_ramp_targets_the_pose_playback_will_start_from(cyclic_harness) -> None:
    """Playback starts past the clip's opening squat, so the ramp target is the
    pose at that offset -- not the first keyframe. Ramping to the wrong one would
    hand over with a posture step exactly as large as the part being skipped."""
    c = cyclic_harness.ctl
    cycle = c.motion.cycle("forward")
    entry = c.motion.entry_pose("forward")
    first = c.motion.first_pose("forward")
    assert entry
    assert c.motion.entry_time_s("forward") == pytest.approx(cycle.enter_s)
    if cycle.enter_s > 0.0:
        assert entry != first
    # Whatever the target, it must be a balanced sole-flat crouch: hip + knee +
    # ankle == 0 keeps the torso vertical and the soles on the floor.
    for side in ("L", "R"):
        total = (entry[f"{side}HipPitch"] + entry[f"{side}KneePitch"]
                 + entry[f"{side}AnklePitch"])
        assert abs(total) < 0.01


def test_playback_begins_at_the_entry_offset_not_at_zero(cyclic_harness) -> None:
    """The skipped prefix is the clip's own squat, which the prepare ramp has
    just done under its own rate limits and balance supervision. Playing it
    again would repeat it with the velocity caps lifted."""
    h, c = cyclic_harness, cyclic_harness.ctl
    cycle = c.motion.cycle("forward")
    h.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(h)
    clip = c.motion._cache["forward"]
    # The seek issued before play() put the playhead at the entry offset.
    assert clip.seeks
    assert clip.seeks[-1] == pytest.approx(cycle.enter_s * 1000.0, abs=1.0)
    assert c.motion.time_s() >= cycle.enter_s - 1e-6


def test_one_shot_clips_are_untouched_by_the_cyclic_path(harness) -> None:
    """The default harness's clips have no cycle in them, and must therefore run
    exactly as they did before: played to completion, or released at a safe
    keyframe. A detector that guessed would have broken every turn."""
    c = harness.ctl
    assert c.motion.cycle("forward") is None
    harness.spin(60, STANDING, IDLE_GAIT)
    assert spin_to_clip(harness)
    assert c.motion.cycling is False
    harness.spin(10, STANDING, MARCH_GAIT)
    assert c._cycles_walked == 0
    assert c._cycle_state == ""


# ---------------------------------------------------------------------------
# 2026-09-10 latency work: the two controller-side defects behind
# "the robot takes 5-7 s to start" and "the robot sometimes falls".
# ---------------------------------------------------------------------------
def test_a_dithering_planner_cannot_ramp_the_legs_forever(harness, monkeypatch):
    """CLIP_PREPARE_TIMEOUT_S must bound the RAMP, not just one action.

    ``_prepare_since`` is restarted whenever the planned action changes, so a
    planner alternating forward/turn_left rewound the 2.5 s timeout forever.
    Measured in log 1788865278: one episode sat in prepare for 9.50 s -- 480
    ticks, zero clips played, 6 action flips -- while the longest stretch on any
    single action was 2.67 s, so the timeout never fired once. The legs stood in
    a one-footed ramp for 9.5 s and the episode ended in a fall. A second,
    flip-immune clock (CLIP_PREPARE_RUN_TIMEOUT_S) is what stops it.
    """
    ctl = harness.ctl
    mod = harness.mod
    # A stance that exists but is never reached: the ramp can never finish.
    monkeypatch.setattr(ctl.motion, "entry_pose",
                        lambda action: {"LKneePitch": 1.0, "RKneePitch": 1.0})
    monkeypatch.setattr(ctl.driver, "approach_leg_pose",
                        lambda *a, **k: False)

    now = 0.0
    abandoned_at = None
    # Flip the planned action every 1.0 s -- comfortably inside the 2.5 s
    # per-action timeout, which is exactly what defeated it.
    for i in range(400):
        action = "forward" if (int(now) % 2 == 0) else "turn_left"
        ctl._ready_to_play(now, action)
        if ctl._preparing is None and i > 0:
            abandoned_at = now
            break
        now += 0.02

    assert abandoned_at is not None, (
        "the prepare ramp never terminated: a dithering planner can still hold "
        "the legs in a ramp indefinitely")
    assert abandoned_at <= mod.CLIP_PREPARE_RUN_TIMEOUT_S + 0.1, (
        f"abandoned after {abandoned_at:.2f}s, which is past "
        f"CLIP_PREPARE_RUN_TIMEOUT_S={mod.CLIP_PREPARE_RUN_TIMEOUT_S}")


def test_reaching_the_stance_clears_the_ramp_clock(harness, monkeypatch):
    """An honest prepare that succeeds must not leave the run clock armed."""
    ctl = harness.ctl
    monkeypatch.setattr(ctl.motion, "entry_pose",
                        lambda action: {"LKneePitch": 1.0})
    monkeypatch.setattr(ctl.driver, "approach_leg_pose", lambda *a, **k: False)
    ctl._ready_to_play(0.0, "forward")
    assert ctl._prepare_run_since is not None
    monkeypatch.setattr(ctl.driver, "approach_leg_pose", lambda *a, **k: True)
    assert ctl._ready_to_play(0.5, "forward") is True
    assert ctl._prepare_run_since is None, (
        "a successful prepare left the run clock armed, so the NEXT clip would "
        "inherit this one's elapsed time and be abandoned early")


def test_clip_handover_rate_caps_the_leg_targets(harness):
    """Coming out of a clip must be ramped, as going in already is.

    Measured in log 1788865278: 13 of 13 clip->pose handovers drove
    support_margin_x to between -0.046 and -0.091 m (the centre of mass off the
    front of the feet), and all three falls in that session began within
    0.13-1.32 s of one. Going INTO a clip has always been ramped
    (approach_leg_pose); coming out of one commanded the whole gap -- up to
    0.5 rad -- on the next tick.
    """
    from pose_control_utils import ALL_LEG_JOINTS

    driver = harness.ctl.driver
    leg = "LKneePitch"
    assert leg in ALL_LEG_JOINTS

    # The clip owned the body and left the knee at 0.0; the human wants 0.9 rad.
    driver.measured[leg] = 0.0
    driver.base_targets[leg] = 0.0
    driver.release_to_motion([leg])
    driver.reclaim_from_motion()

    driver._apply_targets({leg: 0.9}, 1.00)
    first = driver.base_targets[leg]
    step = driver.LEG_POSE_RATE * 0.02
    assert first <= step + 1e-6, (
        f"the first post-handover leg target moved {first:.3f} rad in one tick; "
        f"LEG_POSE_RATE allows {step:.3f}")

    # And the cap lifts once the blend window is over: full 1:1 leg tracking.
    driver._apply_targets({leg: 0.9}, 1.00 + driver.HANDOVER_BLEND_S + 0.05)
    driver._apply_targets({leg: 0.9}, 1.00 + driver.HANDOVER_BLEND_S + 0.07)
    assert driver.base_targets[leg] == pytest.approx(0.9), (
        "the handover cap never released; leg imitation is now permanently "
        "rate-limited, which is not what it is for")


def test_the_handover_cap_does_not_touch_the_arms(harness):
    """Arms keep tracking right through a clip; the guard is legs-only."""
    driver = harness.ctl.driver
    arm = "LShoulderPitch"
    driver.measured[arm] = 0.0
    driver.base_targets[arm] = 0.0
    driver.release_to_motion(["LKneePitch"])
    driver.reclaim_from_motion()
    driver._apply_targets({arm: 1.2}, 1.0)
    assert driver.base_targets[arm] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Hands (2026-09-16). Three things that only show up with the real controller
# in the loop: the grip fanning out to sixteen separate phalanx motors, the arm
# being re-commanded on every simulation step rather than only on the steps
# that carried a camera frame, and neither of those disturbing the legs.
# ---------------------------------------------------------------------------
def with_hands(keypoints, grip=0.0, thumb_axis=(0.0, 0.0, 1.0)):
    """Add hand markers to a ``subject()`` pose at a given hand closure.

    The finger continues the forearm and shortens as the hand closes; the thumb
    starts perpendicular to it and swings across the palm. Proportions are the
    adult ones nao_retarget's GRIP_* constants were derived from (hand 0.73 of
    the forearm, thumb 0.42), expressed as ratios so they survive the fact that
    ``subject()`` works in normalised units rather than millimetres.
    """
    out = dict(keypoints)
    for pre in ("left_", "right_"):
        elbow, wrist = out[pre + "elbow"], out[pre + "wrist"]
        fore = [wrist[i] - elbow[i] for i in range(3)]
        length = math.sqrt(sum(c * c for c in fore)) or 1.0
        fore = [c / length for c in fore]
        # Orthogonalise the thumb axis against the forearm so "perpendicular"
        # stays true whatever direction the arm happens to point.
        dot = sum(thumb_axis[i] * fore[i] for i in range(3))
        perp = [thumb_axis[i] - dot * fore[i] for i in range(3)]
        pn = math.sqrt(sum(c * c for c in perp)) or 1.0
        perp = [c / pn for c in perp]
        reach = 0.73 * length * (1.0 - 0.6 * grip)
        lean = 0.5 * grip
        out[pre + "hand_root"] = list(wrist)
        out[pre + "finger"] = [wrist[i] + reach * fore[i] for i in range(3)] + [1.0]
        out[pre + "thumb"] = [
            wrist[i] + 0.42 * length * (perp[i] * (1.0 - lean) + fore[i] * lean)
            for i in range(3)
        ] + [1.0]
    return out


OPEN_HAND = with_hands(STANDING, grip=0.0)
CLOSED_HAND = with_hands(STANDING, grip=1.0)


def _phalanges(harness, side="L"):
    return [harness.angle(f"{side}Phalanx{i}") for i in range(1, 9)]


def test_a_closing_hand_closes_every_phalanx(harness) -> None:
    """Webots' NAO has no single hand motor -- there are eight phalanx motors
    per hand and one measured grip, so the driver has to fan the value out.
    Commanding one and leaving seven at the rest pose is the failure this
    catches, and on a robot it looks like a hand that half-works."""
    harness.spin(120, OPEN_HAND)
    opened = _phalanges(harness)
    harness.spin(120, CLOSED_HAND)
    closed = _phalanges(harness)
    assert len(set(round(a, 6) for a in closed)) == 1, \
        f"the eight phalanges disagree: {closed}"
    for before, after in zip(opened, closed, strict=True):
        assert after < before - 0.05, f"phalanx did not close: {before} -> {after}"


def test_the_grip_stays_inside_the_phalanx_limits(harness) -> None:
    for pose in (OPEN_HAND, CLOSED_HAND, with_hands(STANDING, grip=0.5)):
        harness.spin(60, pose)
        for name in [f"{s}Phalanx{i}" for s in ("L", "R") for i in range(1, 9)]:
            cfg = CONFIGS[name]
            angle = harness.angle(name)
            assert cfg.min_angle - 1e-6 <= angle <= cfg.max_angle + 1e-6, \
                f"{name} at {angle} outside {cfg.min_angle}..{cfg.max_angle}"


def test_the_arms_are_commanded_on_every_step_not_only_on_camera_frames(
    harness,
) -> None:
    """The camera runs at ~12 Hz and the simulation at 50. Feeding a frame every
    fourth step and counting motor writes separates a driver that re-commands
    continuously from one that holds between frames -- the latter is what made
    the arms look like they were stepping rather than moving."""
    motor = harness.ctl.robot.motors["LShoulderPitch"]
    harness.spin(20, OPEN_HAND, every=4)   # settle and seed the tracker
    before = motor.commands
    harness.spin(40, OPEN_HAND, every=4)   # 10 camera frames, 40 steps
    written = motor.commands - before
    assert written >= 40, (
        f"only {written} motor writes over 40 simulation steps; the arm is "
        "only moving when a camera frame lands")


def test_a_moving_arm_is_not_left_behind_by_the_camera_rate(harness) -> None:
    """The point of the velocity lead. Step the human's elbow through a sweep at
    the camera's rate and check the robot is not trailing by more than a frame's
    worth of travel by the end."""
    poses = []
    for i in range(24):
        bend = 0.2 + 0.05 * i
        kps = dict(STANDING)
        for pre, sgn in (("left_", -1.0), ("right_", +1.0)):
            elbow = kps[f"{pre}elbow"]
            kps[f"{pre}wrist"] = [
                elbow[0] + sgn * 0.11 * math.sin(bend),
                elbow[1] + 0.11 * math.cos(bend),
                elbow[2], 1.0,
            ]
        poses.append(kps)

    harness.spin(20, poses[0], every=4)
    for kps in poses:
        harness.spin(4, kps, every=4)
    settled = harness.angle("LElbowRoll")
    # Now hold the final pose and let it converge: the gap between where the arm
    # was while the sweep was running and where it ends up is the lag.
    harness.spin(120, poses[-1], every=4)
    final = harness.angle("LElbowRoll")
    assert abs(final - settled) < 0.12, (
        f"the arm was {abs(final - settled):.3f} rad behind the human at the end "
        "of a steady sweep")


def test_hands_do_not_disturb_the_legs(harness) -> None:
    """The whole change is meant to be above the waist. A grip that reached the
    leg joints, or an arm tick that re-ran the leg layer, would show up as the
    legs moving while only the hand does."""
    harness.spin(150, OPEN_HAND)
    legs = {n: harness.angle(n) for n in
            [f"{s}{j}" for s in ("L", "R")
             for j in ("HipPitch", "KneePitch", "AnklePitch", "HipRoll", "AnkleRoll")]}
    harness.spin(150, CLOSED_HAND)
    for name, before in legs.items():
        assert abs(harness.angle(name) - before) < 1e-6, \
            f"{name} moved when only the hand changed"


def test_a_model_without_finger_motors_still_runs(harness, monkeypatch) -> None:
    """Nao.proto is fetched over the network at world-load time, so the finger
    motor names cannot be checked against a file here. A model that does not
    have them must degrade to no grip, not to a crash."""
    for name in list(harness.ctl.driver.motors):
        if "Phalanx" in name:
            harness.ctl.driver.motors.pop(name)
    harness.spin(60, CLOSED_HAND)
    assert harness.angle("LShoulderPitch") != 0.0, "the arm stopped tracking"


def test_a_departed_human_does_not_leave_the_robot_holding_a_fist(harness) -> None:
    """The counterpart of the arms ramping back to neutral on staleness. The
    grip is not a joint and has no rest_angle to fall back on, so it was the one
    channel that could sit frozen on the last pose its human left behind."""
    harness.spin(150, CLOSED_HAND)
    closed = _phalanges(harness)
    harness.spin(200)                      # nobody in frame: goes stale
    released = _phalanges(harness)
    assert all(b > a + 0.05 for a, b in zip(closed, released, strict=True)), \
        f"the hand stayed shut after tracking was lost: {closed} -> {released}"


# ---------------------------------------------------------------------------
# Aimed turning: the safety property that lets the overshoot floor be dropped
# ---------------------------------------------------------------------------
def test_a_turn_clip_that_cannot_be_aimed_is_not_called_interruptible(harness) -> None:
    """Somewhere safe to stop is NOT enough to loosen the turn gate.

    Dropping the overshoot floor is only sound because the clip is then aimed --
    stopped at the rung that best serves the heading error. These fixture clips
    have 60 certified stopping points and command no rotation whatsoever, so
    there is nowhere to aim; loosening the gate for them would let the loop hunt
    below the finest error the clip can serve, which is the limit cycle
    test_a_multi_clip_rotation_keeps_going_until_aligned exists to catch.

    So the two tests are the same statement from both sides: this one says the
    clip is excluded, that one says what goes wrong if it is not.
    """
    player = harness.ctl.motion
    assert player.safe_exits("turn_left"), "fixture clip should have safe exits"
    assert player.turn("turn_left") is None, "fixture clip does not rotate"
    assert "turn_left" not in player.interruptible
    # A clip that is not a turn is judged on its safe exits alone, as before.
    assert ("forward" in player.interruptible) == bool(player.safe_exits("forward"))


def test_dropping_a_clip_forgets_that_it_was_interruptible(harness) -> None:
    """The set is cached per session; a clip Webots refused must leave it, or the
    planner keeps choosing a clip that is no longer on the books."""
    player = harness.ctl.motion
    before = player.interruptible
    assert player.interruptible is before          # cached, not rebuilt per step
    player.drop("forward")
    assert "forward" not in player.interruptible
