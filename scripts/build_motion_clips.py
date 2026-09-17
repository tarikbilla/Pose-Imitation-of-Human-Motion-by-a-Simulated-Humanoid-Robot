#!/usr/bin/env python3
"""Build the robot's locomotion clip library, and refuse to ship a bad clip.

Run this once to populate ``main/controllers/pose_imitation_controller/motions/``,
which ``walk_motion.default_motion_search_dirs`` searches BEFORE the Webots
install -- so every clip written here shadows the Cyberbotics original of the
same name, and the controller picks it up with no code change.

The library has two halves, held to two different standards, because they are two
different kinds of thing.

**Generated** (``squat``, ``leg raises``) are synthesised from NAO's own link
lengths and are quasi-static: the centre of mass never leaves the support
polygon, so they are stable stopped at any keyframe. These must pass every check
in :mod:`clip_safety`, with no exceptions. If one fails, it is a bug in the
generator and the build stops.

**Refined** (walking, turning, side steps) start from the Cyberbotics clips.
Those genuinely walk the robot, and re-deriving a dynamic gait from scratch is a
research project, not a build step -- so their gait keyframes are kept exactly as
authored and only the defects that are properties of the FILE are fixed:

  * the arm and head joints are stripped, so the upper-body imitation keeps
    running through a clip instead of being suspended by it (4 of the 10 shipped
    clips drive the arms; ``Backwards`` drives the head too);
  * the clip is bookended with eased ramps to and from the controller's standing
    pose, so the 1.05 rad posture gap -- the gap the ``prepare:`` ramp and the
    clip-to-pose handover have to absorb on every play -- becomes zero.

What they cannot fix is the gait's own balance: a dynamic gait leaves the support
polygon on purpose, and Cyberbotics' leave it by 8-23 mm (centre of mass) and
24-55 mm (capture point). So refined clips are held to a NON-REGRESSION bar
instead: every other check must pass outright, and the balance margin must be no
worse than the original they were derived from. That is an honest standard -- it
says "we made this strictly better and broke nothing" -- and it is the strongest
one available without authoring new gait dynamics.

Usage::

    python scripts/build_motion_clips.py            # build and report
    python scripts/build_motion_clips.py --check    # verify only, write nothing
    python scripts/build_motion_clips.py --list     # show the planned library
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main" / "libraries"))

from clip_forge import (  # noqa: E402
    make_leg_raise,
    make_squat,
    mirror_clip,
    refine_clip,
    retime_for_velocity,
    write_motion,
)
from clip_safety import certify  # noqa: E402
from walk_motion import default_motion_search_dirs, motion_poses  # noqa: E402

OUT_DIR = ROOT / "main" / "controllers" / "pose_imitation_controller" / "motions"


def find_source(name: str) -> str | None:
    """Locate a Cyberbotics clip by filename, skipping our own output directory."""
    for directory in default_motion_search_dirs():
        if Path(directory).resolve() == OUT_DIR.resolve():
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


# (output name, kind, how to build it)
#
# Turn angles: Webots ships 40/60/180 left and 40/60 right. The missing coarse
# right turn is not cosmetic -- plan_action picks the LARGEST turn clip that will
# not overshoot, so without it an about-face to the right had to be served by
# repeated 60 deg clips, each paying its own ramp, settle and handover.
GENERATED = [
    ("Squat", "static", lambda: make_squat()),
    ("RaiseLegLeft", "static", lambda: make_leg_raise("L")),
    ("RaiseLegRight", "static", lambda: make_leg_raise("R")),
]

REFINED = [
    ("Forwards", "Forwards.motion", False),
    ("Forwards50", "Forwards50.motion", False),
    ("Backwards", "Backwards.motion", False),
    ("TurnLeft40", "TurnLeft40.motion", False),
    ("TurnLeft60", "TurnLeft60.motion", False),
    ("TurnLeft180", "TurnLeft180.motion", False),
    ("TurnRight40", "TurnRight40.motion", False),
    ("TurnRight60", "TurnRight60.motion", False),
    # Mirrored from the left clip: NAO is symmetric about the same plane, so the
    # mirror is balanced exactly as well as its original (verified to 0.0000 mm
    # of margin difference across every keyframe).
    ("TurnRight180", "TurnLeft180.motion", True),
    ("SideStepLeft", "SideStepLeft.motion", False),
    ("SideStepRight", "SideStepRight.motion", False),
]

# Checks a refined clip must pass outright. "support" is absent on purpose: see
# the non-regression rule in the module docstring.
REFINED_MUST_PASS = ("limits", "velocity", "legs_only", "posture",
                     "momentum", "seam", "format", "model")


def build(check_only: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    written = 0

    print(f"{'clip':<16s} {'kind':<8s} {'verdict':<8s} "
          f"{'CoM':>9s} {'capture':>9s} {'vel':>6s} {'gap':>6s}  note")
    print("-" * 104)

    for name, kind, make in GENERATED:
        clip = make()
        cert = certify(name, clip, kind=kind)
        note = "generated; quasi-static"
        if not cert.passed:
            failures += 1
            note = "FAILED: " + str(cert.violations[0])
        elif not check_only:
            write_motion(str(OUT_DIR / f"{name}.motion"), clip)
            written += 1
        _report(name, kind, cert, note)

    for name, source_file, do_mirror in REFINED:
        source = find_source(source_file)
        if source is None:
            print(f"{name:<16s} {'refined':<8s} {'SKIP':<8s} "
                  f"{'':>9s} {'':>9s} {'':>6s} {'':>6s}  no source clip on this install")
            continue
        raw = motion_poses(source)
        before = certify(name, raw, kind="dynamic", require_legs_only=False)
        clip = refine_clip(mirror_clip(raw) if do_mirror else raw)
        # Slow the clip if it outruns the motors. Almost always a no-op; it is
        # Forwards50 -- the one clip that can walk continuously -- that needs it.
        stretched = retime_for_velocity(clip)
        slowed = stretched[-1][0] > clip[-1][0] + 1e-6
        clip = stretched
        cert = certify(name, clip, kind="dynamic")

        blocking = [v for v in cert.violations if v.check in REFINED_MUST_PASS]
        # Non-regression: the gait's own balance may be no worse than the clip we
        # inherited it from. Equal is fine -- we did not touch those keyframes.
        regressed = cert.min_capture_margin_m < before.min_capture_margin_m - 1e-6
        if blocking:
            failures += 1
            note = "FAILED: " + str(blocking[0])
        elif regressed:
            failures += 1
            note = (f"FAILED: balance regressed "
                    f"{before.min_capture_margin_m*1000:+.1f} -> "
                    f"{cert.min_capture_margin_m*1000:+.1f} mm")
        else:
            note = (f"refined from {os.path.basename(source)}"
                    + (" (mirrored)" if do_mirror else ""))
            if before.extra_joints:
                note += f"; freed {len(before.extra_joints)} upper-body joints"
            if slowed:
                note += (f"; slowed {before.duration_s:.2f}->{cert.duration_s:.2f}s "
                         "to stay inside the motors")
            if not check_only:
                write_motion(str(OUT_DIR / f"{name}.motion"), clip)
                written += 1
        _report(name, "refined", cert, note, ok=not blocking and not regressed)

    print("-" * 104)
    if check_only:
        print(f"{failures} failing clip(s); nothing written (--check).")
    else:
        print(f"{written} clip(s) written to {OUT_DIR}")
        if failures:
            print(f"{failures} clip(s) FAILED certification and were not written.")
    return 1 if failures else 0


def _report(name, kind, cert, note, ok=None):
    verdict = ("PASS" if (cert.passed if ok is None else ok) else "FAIL")
    print(f"{name:<16s} {kind:<8s} {verdict:<8s} "
          f"{cert.min_support_margin_m*1000:+8.1f}mm "
          f"{cert.min_capture_margin_m*1000:+8.1f}mm "
          f"{cert.peak_velocity_fraction*100:5.1f}% "
          f"{cert.posture_gap_rad:5.2f}  {note}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="certify everything but write nothing")
    parser.add_argument("--list", action="store_true",
                        help="list the planned library and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("generated (quasi-static, must pass every check):")
        for name, kind, _ in GENERATED:
            print(f"   {name:<16s} {kind}")
        print("\nrefined from Cyberbotics (legs-only + bookended, non-regression):")
        for name, source, mirrored in REFINED:
            print(f"   {name:<16s} <- {source}" + ("  (mirrored)" if mirrored else ""))
        print(f"\noutput: {OUT_DIR}")
        return 0

    return build(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
