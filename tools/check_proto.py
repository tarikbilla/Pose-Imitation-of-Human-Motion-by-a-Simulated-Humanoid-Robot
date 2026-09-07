import argparse
import os
import re
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import mass_tables as mt

PROTO_DIR = os.path.join(TOOLS, "..", "webots", "protos")
PROTO = os.path.join(PROTO_DIR, "AtlasA3.proto")
VENDOR = os.path.join(PROTO_DIR, "vendor")
MASS = re.compile(r"^\s*mass\s+([\d.eE+-]+)\s*$", re.M)
TOLERANCE = 1e-4


def scan_main(text):
    masses = []
    physics = 0
    sensors = 0
    stale = 0
    local = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("mass "):
            masses.append(float(stripped.split()[1]))
        elif stripped == "physics Physics {":
            physics += 1
        elif stripped == "PositionSensor {":
            sensors += 1
        elif "USE DEFAULT_PHYSICS" in stripped:
            stale += 1
        elif stripped.startswith('EXTERNPROTO "vendor/'):
            local += 1
    return masses, physics, sensors, stale, local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default=PROTO)
    parser.add_argument("--vendor", default=VENDOR)
    parser.add_argument("--masses", default=None, choices=mt.VARIANTS)
    args = parser.parse_args()

    with open(args.proto, "r", encoding="utf-8") as handle:
        text = handle.read()

    masses, physics, sensors, stale, local = scan_main(text)
    problems = []

    variant = args.masses
    if variant is None:
        for name in mt.VARIANTS:
            if 'Mass variant "' + name + '"' in text:
                variant = name
                break
    if variant is None:
        print("FEHLER: kein Profil-Marker im PROTO-Kopf", file=sys.stderr)
        return 1

    table = mt.build(variant)
    summary = mt.summarise(table)

    vendored = {}
    for name in sorted(os.listdir(args.vendor)):
        if not name.endswith(".proto"):
            continue
        with open(os.path.join(args.vendor, name), "r", encoding="utf-8") as handle:
            found = MASS.search(handle.read())
        if found:
            vendored[name[:-6]] = float(found.group(1))

    if physics != 29:
        problems.append(f"29 Physics-Nodes erwartet, {physics} gefunden")
    if sensors != 28:
        problems.append(f"28 PositionSensors erwartet, {sensors} gefunden")
    if stale:
        problems.append(f"{stale} verbliebene USE DEFAULT_PHYSICS")
    if local != 28:
        problems.append(f"28 lokale EXTERNPROTO erwartet, {local} gefunden")
    if len(vendored) != 28:
        problems.append(f"28 vendored Sub-PROTOs erwartet, {len(vendored)} gefunden")
    if text.count("{") != text.count("}"):
        problems.append(f"geschweifte Klammern {text.count('{')}/{text.count('}')}")
    if text.count("[") != text.count("]"):
        problems.append(f"eckige Klammern {text.count('[')}/{text.count(']')}")

    worst = 0.0
    for segment, value in vendored.items():
        expected = table.get(segment)
        if expected is None:
            problems.append(f"unbekanntes Segment {segment}")
            continue
        worst = max(worst, abs(value - expected["mass"]))
    if worst > TOLERANCE:
        problems.append(f"Segmentmasse weicht um {worst:.6f} kg ab")

    head = table[mt.HEAD_SEGMENT]["mass"]
    total = sum(vendored.values()) + head
    if abs(total - summary["total"]) > 1e-3:
        problems.append(f"Gesamtmasse {total:.3f} statt {summary['total']:.3f} kg")

    residue = [value for value in masses
               if abs(value - mt.build(variant)[mt.HEAD_SEGMENT]["mass"]) > TOLERANCE]
    if any(value > 0.01 for value in residue):
        problems.append("Platzhalter-Physics traegt mehr als 0.01 kg")

    print(f"Profil            {variant}")
    print(f"Sub-PROTOs        {len(vendored)}")
    print(f"Physics-Nodes     {physics}")
    print(f"PositionSensors   {sensors}")
    print(f"Kopfmasse         {head:.3f} kg")
    print(f"Gesamtmasse       {total:.3f} kg")
    print(f"Armanteil         {summary['arm_percent']:.1f} %")
    print(f"Beinanteil        {summary['leg_percent']:.1f} %")
    print(f"largest deviation {worst:.6f} kg")

    if problems:
        print()
        for problem in problems:
            print(f"FEHLER: {problem}", file=sys.stderr)
        return 1
    print("check             passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
