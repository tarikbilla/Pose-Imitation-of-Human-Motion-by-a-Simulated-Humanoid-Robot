import argparse
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mass_tables

WEBOTS_VERSION = "R2025a"
BASE_URL = (
    f"https://raw.githubusercontent.com/cyberbotics/webots/{WEBOTS_VERSION}"
    "/projects/robots/boston_dynamics/atlas/protos"
)
SOURCE_URL = f"{BASE_URL}/Atlas.proto"

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTO_DIR = os.path.join(A3_ROOT, "webots", "protos")
OUTPUT_PATH = os.path.join(PROTO_DIR, "AtlasA3.proto")
VENDOR_DIRNAME = "vendor"
KIN_VENDOR_DIRNAME = "vendor_kin"
VENDOR_DIR = os.path.join(PROTO_DIR, VENDOR_DIRNAME)
KIN_OUTPUT_PATH = os.path.join(PROTO_DIR, "AtlasA3Kin.proto")
CACHE_PATH = os.path.join(A3_ROOT, "results", "_Atlas_upstream.proto")
SUBPROTO_CACHE = os.path.join(A3_ROOT, "results", "_subprotos")

PLACEHOLDER_MASS = 0.001
PLACEHOLDER_INERTIA = 1e-6

CONTROL_GAIN = float(os.environ.get("A3_CONTROL_GAIN", "1000.0"))
DEVICE_OPEN = re.compile(r"^(\s*)device RotationalMotor \{\s*$")
NAME_FIELD = re.compile(r'^\s*name\s+"([^"]+)"\s*$')
EXTERNPROTO = re.compile(r'^EXTERNPROTO\s+"([^"]+)"\s*$')
MESH_NODE = re.compile(r"^\s*(\w+(?:Solid|Mesh))\s*\{\s*$")
PHYSICS_USE = re.compile(r"^(\s*)physics USE DEFAULT_PHYSICS\s*$")
PHYSICS_DEF = re.compile(r"^(\s*)physics DEF DEFAULT_PHYSICS Physics \{\s*$")
SUB_MASS = re.compile(r"^(\s*)mass\s+([\d.eE+-]+)\s*$")
SUB_INERTIA = re.compile(r"^(\s*)inertiaMatrix\s+\[(.*)\]\s*$")


def fetch_source(use_cache):
    if use_cache and os.path.isfile(CACHE_PATH):
        print(f"using cached upstream   {CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            return handle.read()
    print(f"downloading             {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        text = response.read().decode("utf-8")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def fetch_subproto(name, use_cache):
    path = os.path.join(SUBPROTO_CACHE, name)
    if use_cache and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
        text = response.read().decode("utf-8")
    os.makedirs(SUBPROTO_CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def externproto_names(lines):
    names = []
    for line in lines:
        match = EXTERNPROTO.match(line)
        if match:
            names.append(os.path.basename(match.group(1)))
    return names


def localise_externprotos(lines, vendored, folder=VENDOR_DIRNAME):
    result = []
    count = 0
    for line in lines:
        match = EXTERNPROTO.match(line)
        if not match:
            result.append(line)
            continue
        name = os.path.basename(match.group(1))
        if name in vendored:
            result.append(f'EXTERNPROTO "{folder}/{name}"')
        else:
            result.append(f'EXTERNPROTO "{BASE_URL}/{name}"')
        count += 1
    return result, count


def strip_subproto_physics(text):
    lines = text.splitlines()
    result = []
    skip = 0
    for index, line in enumerate(lines):
        if skip > 0:
            skip -= 1
            continue
        external = EXTERNPROTO.match(line)
        if external and not external.group(1).startswith('http'):
            result.append('EXTERNPROTO "' + BASE_URL + '/' + external.group(1) + '"')
            continue
        if line.strip().startswith('physics Physics {'):
            depth = 1
            offset = index + 1
            while depth > 0 and offset < len(lines):
                depth += lines[offset].count('{') - lines[offset].count('}')
                offset += 1
            skip = offset - index - 1
            continue
        result.append(line)
    return chr(10).join(result) + chr(10)


def raise_max_velocity(lines, value, gain=None):
    result = []
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("maxVelocity "):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(indent + "maxVelocity " + str(value))
            if gain is not None:
                result.append(indent + "controlPID " + str(gain) + " 0 0")
            count += 1
            continue
        result.append(line)
    return result, count


def remove_link_physics(lines):
    result = []
    removed = 0
    skip = 0
    for index, line in enumerate(lines):
        if skip > 0:
            skip -= 1
            continue
        if PHYSICS_USE.match(line):
            removed += 1
            continue
        if PHYSICS_DEF.match(line):
            depth = 1
            offset = index + 1
            while depth > 0 and offset < len(lines):
                depth += lines[offset].count('{') - lines[offset].count('}')
                offset += 1
            skip = offset - index - 1
            removed += 1
            continue
        result.append(line)
    return result, removed


def rewrite_subproto(text, entry):
    lines = text.splitlines()
    original = None
    for line in lines:
        match = SUB_MASS.match(line)
        if match:
            original = float(match.group(2))
            break
    if original is None or original <= 0.0:
        return None, None

    factor = entry["mass"] / original
    result = []
    seen_mass = False
    for line in lines:
        external = EXTERNPROTO.match(line)
        if external and not external.group(1).startswith("http"):
            result.append(f'EXTERNPROTO "{BASE_URL}/{external.group(1)}"')
            continue
        mass_match = SUB_MASS.match(line)
        if mass_match and not seen_mass:
            seen_mass = True
            result.append(f"{mass_match.group(1)}mass {entry['mass']:.6g}")
            continue
        inertia_match = SUB_INERTIA.match(line)
        if inertia_match:
            values = [float(v) for v in
                      inertia_match.group(2).replace(",", " ").split()]
            scaled = " ".join(f"{v * factor:.6g}" for v in values)
            result.append(f"{inertia_match.group(1)}inertiaMatrix [ {scaled} ]")
            continue
        result.append(line)
    return chr(10).join(result) + chr(10), original


def vendor_subprotos(names, table, use_cache, kinematic=False):
    directory = (os.path.join(PROTO_DIR, KIN_VENDOR_DIRNAME) if kinematic
                 else VENDOR_DIR)
    os.makedirs(directory, exist_ok=True)
    vendored = {}
    skipped = []
    for name in names:
        segment = name[:-6] if name.endswith(".proto") else name
        entry = table.get(segment)
        text = fetch_subproto(name, use_cache)
        if kinematic:
            target = os.path.join(directory, name)
            with open(target, "w", encoding="utf-8", newline=chr(10)) as handle:
                handle.write(strip_subproto_physics(text))
            vendored[name] = (segment, 0.0, 0.0)
            continue
        if entry is None:
            skipped.append(name)
            continue
        rewritten, original = rewrite_subproto(text, entry)
        if rewritten is None:
            skipped.append(name)
            continue
        target = os.path.join(directory, name)
        with open(target, "w", encoding="utf-8", newline=chr(10)) as handle:
            handle.write(rewritten)
        vendored[name] = (segment, original, entry["mass"])
    return vendored, skipped


def header_for(proto_name, mass_variant, table):
    if mass_variant == "kinematic":
        note = [
            "#   2. ALL Physics nodes are removed, in the main PROTO and in every",
            "#      vendored sub-PROTO. The robot is not simulated by ODE: no",
            "#      gravity, no contact forces, no reaction from the floor. Joint",
            "#      angles are set directly and the base pose is placed by the",
            "#      Supervisor so the support sole rests on the ground.",
            "#      Locomotion comes from root motion. See a3/docs/M8_RESULTS.md.",
        ]
    elif mass_variant is None:
        note = [
            "#   2. Physics is LEFT AT THE STOCK VALUE on purpose: the shared",
            "#      DEFAULT_PHYSICS node (mass 0.001, inertiaMatrix [ 1 1 1 0 0 0 ])",
            "#      stays merged into every link. Control variant for the A/B test.",
        ]
    else:
        summary = mass_tables.summarise(table)
        note = [
            "#   2. Segment masses are REDISTRIBUTED. The stock robot already carries",
            "#      its real Boston Dynamics distribution (89.000 kg, summed from the",
            "#      28 sub-PROTOs), which puts 27.3% of its mass in the arms against",
            "#      10% for a human. Each sub-PROTO is vendored into protos/vendor/",
            "#      with mass and inertia rescaled; centerOfMass is kept unchanged, so",
            "#      segment geometry stays correct.",
            f'#      Mass variant "{mass_variant}": {summary["total"]:.2f} kg total,',
            f"#      {summary['arm_percent']:.1f}% in the arms, "
            f"{summary['leg_percent']:.1f}% in the legs.",
            "#   2b. The shared DEFAULT_PHYSICS placeholder that Webots merges into",
            f"#      every link is reduced to a residue (mass {PLACEHOLDER_MASS},",
            f"#      inertia {PLACEHOLDER_INERTIA}). Its stock inertiaMatrix",
            "#      [ 1 1 1 0 0 0 ] made the ankles 83.7x too inertial.",
            "#      See a3/docs/M6_RESULTS.md.",
        ]
    lines = [
        f"#VRML_SIM {WEBOTS_VERSION} utf8",
        "# Generated by a3/tools/build_atlas_proto.py -- do not edit by hand.",
        f"# Upstream: {SOURCE_URL}",
        "#",
        "# Differences from the stock Atlas PROTO:",
        "#   1. A PositionSensor is added to every one of the 28 hinge joints.",
        '#      Named "<MotorName>S", following the Webots NAO convention.',
        *note,
        "#   3. supervisor defaults to TRUE.",
        "",
    ]
    return chr(10).join(lines)


def add_position_sensors(lines):
    result = []
    index = 0
    added = []
    while index < len(lines):
        match = DEVICE_OPEN.match(lines[index])
        if not match:
            result.append(lines[index])
            index += 1
            continue

        indent = match.group(1)
        body = []
        depth = 1
        cursor = index + 1
        while cursor < len(lines) and depth > 0:
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            if depth > 0:
                body.append(lines[cursor])
            cursor += 1

        motor_name = None
        for entry in body:
            name_match = NAME_FIELD.match(entry)
            if name_match:
                motor_name = name_match.group(1)
                break
        if motor_name is None:
            raise ValueError(f"RotationalMotor without name near line {index + 1}")

        result.append(f"{indent}device [")
        result.append(f"{indent}  RotationalMotor {{")
        for entry in body:
            result.append("  " + entry if entry.strip() else entry)
        result.append(f"{indent}  }}")
        result.append(f"{indent}  PositionSensor {{")
        result.append(f'{indent}    name "{motor_name}S"')
        result.append(f"{indent}  }}")
        result.append(f"{indent}]")

        added.append(motor_name)
        index = cursor
    return result, added


def physics_block(indent, mass, com, inertia):
    com_text = " ".join(f"{v:.6g}" for v in com)
    inertia_text = " ".join(f"{v:.6g}" for v in inertia)
    return [
        f"{indent}physics Physics {{",
        f"{indent}  density -1",
        f"{indent}  mass {mass:.6g}",
        f"{indent}  centerOfMass [ {com_text} ]",
        f"{indent}  inertiaMatrix [ {inertia_text} ]",
        f"{indent}}}",
    ]


def rewrite_link_physics(lines, head_entry):
    result = []
    pending = {}
    depth = 0
    written = []
    skip = 0
    residue = [PLACEHOLDER_INERTIA] * 3 + [0.0] * 3
    for index, line in enumerate(lines):
        if skip > 0:
            skip -= 1
            depth += line.count("{") - line.count("}")
            continue
        match = MESH_NODE.match(line)
        if match:
            pending.setdefault(depth, []).append(match.group(1))
        use = PHYSICS_USE.match(line)
        define = PHYSICS_DEF.match(line)
        if use or define:
            indent = (use or define).group(1)
            found = pending.get(depth) or pending.get(depth + 1)
            segment = found.pop() if found else None
            if segment == mass_tables.HEAD_SEGMENT and head_entry is not None:
                result.extend(physics_block(indent, head_entry["mass"],
                                            head_entry["com"],
                                            head_entry["inertia"]))
            else:
                result.extend(physics_block(indent, PLACEHOLDER_MASS,
                                            [0.0, 0.0, 0.0], residue))
            written.append(segment or "?")
            if define:
                inner = 1
                offset = index + 1
                while inner > 0 and offset < len(lines):
                    inner += lines[offset].count("{") - lines[offset].count("}")
                    offset += 1
                skip = offset - index - 1
            depth += line.count("{") - line.count("}")
            continue
        result.append(line)
        depth += line.count("{") - line.count("}")
    return result, written


def rename_proto(lines, proto_name):
    result = []
    renamed = 0
    for line in lines:
        if line.startswith("PROTO Atlas ["):
            result.append(f"PROTO {proto_name} [")
            renamed += 1
        elif re.match(r'^\s*field\s+SFString\s+name\s+"Atlas"', line):
            result.append(line.replace('"Atlas"', f'"{proto_name}"'))
        elif re.match(r'^\s*field\s+SFBool\s+supervisor\s+FALSE', line):
            result.append(line.replace("FALSE", "TRUE "))
        else:
            result.append(line)
    return result, renamed


def strip_upstream_header(lines):
    return [line for line in lines if not line.startswith("#VRML_SIM")]


def build(source, proto_name, mass_variant, use_cache, kinematic=False):
    lines = strip_upstream_header(source.splitlines())
    names = externproto_names(lines)
    lines, renamed = rename_proto(lines, proto_name)
    lines, sensors = add_position_sensors(lines)

    table = None
    vendored = {}
    written = []
    if kinematic:
        vendored, _ = vendor_subprotos(names, {}, use_cache, kinematic=True)
        lines, _ = localise_externprotos(lines, set(vendored),
                                         KIN_VENDOR_DIRNAME)
        lines, count = remove_link_physics(lines)
        lines, velocities = raise_max_velocity(lines, 60, CONTROL_GAIN)
        print(f"maxVelocity raised     {velocities} (controlPID "
              f"{CONTROL_GAIN} 0 0)")
        written = ["kinematic"] * count
        return lines, len(names), renamed, sensors, written, vendored, None
    if mass_variant is None:
        lines, _ = localise_externprotos(lines, set())
    else:
        table = mass_tables.build(mass_variant)
        vendored, _ = vendor_subprotos(names, table, use_cache)
        lines, _ = localise_externprotos(lines, set(vendored))
        lines, written = rewrite_link_physics(
            lines, table.get(mass_tables.HEAD_SEGMENT))
    return lines, len(names), renamed, sensors, written, vendored, table


def main():
    parser = argparse.ArgumentParser(description="Generate AtlasA3.proto.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--masses", choices=mass_tables.VARIANTS, default="core",
                        help="per-segment mass distribution written into the PROTO")
    parser.add_argument("--kinematic", action="store_true",
                        help="also emit AtlasA3Kin.proto without any Physics")
    parser.add_argument("--with-unfixed", action="store_true",
                        help="also emit AtlasA3Unfixed.proto as the A/B control")
    args = parser.parse_args()

    source = fetch_source(not args.no_cache)
    print(f"upstream lines          {len(source.splitlines())}")

    variants = [("AtlasA3", OUTPUT_PATH, args.masses)]
    if args.kinematic:
        variants.append(("AtlasA3Kin", KIN_OUTPUT_PATH, "kinematic"))
    if args.with_unfixed:
        variants.append(("AtlasA3Unfixed",
                         os.path.join(PROTO_DIR, "AtlasA3Unfixed.proto"), None))

    for proto_name, path, mass_variant in variants:
        kinematic = mass_variant == "kinematic"
        lines, externs, renamed, sensors, written, vendored, table = build(
            source, proto_name, None if kinematic else mass_variant,
            not args.no_cache, kinematic=kinematic)
        print()
        print(f"--- {proto_name} ---")
        print(f"EXTERNPROTO rewritten   {externs}")
        print(f"PROTO renamed           {renamed}")
        print(f"PositionSensors added   {len(sensors)}")

        if len(sensors) != 28:
            print(f"FATAL: expected 28 motors, found {len(sensors)}", file=sys.stderr)
            return 1
        if renamed != 1:
            print("FATAL: PROTO header not renamed", file=sys.stderr)
            return 1

        if kinematic:
            print(f"sub-PROTOs vendored     {len(vendored)}")
            print(f"Physics nodes removed   {len(written)}")
            if len(written) != 29:
                print(f"FATAL: expected 29 physics removals, got {len(written)}",
                      file=sys.stderr)
                return 1
        elif mass_variant is not None:
            print(f"sub-PROTOs vendored     {len(vendored)}")
            print(f"link physics rewritten  {len(written)}")
            if len(vendored) != 28:
                print(f"FATAL: expected 28 vendored sub-PROTOs, got {len(vendored)}",
                      file=sys.stderr)
                return 1
            if len(written) != 29 or len(set(written)) != 29:
                print(f"FATAL: link physics {len(written)} written, "
                      f"{len(set(written))} distinct", file=sys.stderr)
                return 1
            total = (sum(new for _, _, new in vendored.values())
                     + table[mass_tables.HEAD_SEGMENT]["mass"])
            summary = mass_tables.summarise(table)
            print(f"total mass              {total:.3f} kg")
            print(f"arm share               {summary['arm_percent']:.1f} %")
            if abs(total - summary["total"]) > 1e-3:
                print(f"FATAL: mass mismatch {total:.3f} vs {summary['total']:.3f}",
                      file=sys.stderr)
                return 1

        header = header_for(proto_name, mass_variant, table)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline=chr(10)) as handle:
            handle.write(header)
            handle.write(chr(10).join(lines))
            handle.write(chr(10))
        print(f"written                 {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
