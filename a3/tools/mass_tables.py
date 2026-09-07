import os

import yaml

MODEL_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "configs", "atlas_model.yaml")

HEAD_SEGMENT = "HeadMesh"
HEAD_COM = [0.0, 0.0, 0.18]
HEAD_RADIUS = 0.10

ARM_L = ("LClav", "LScap", "LUarm", "LLarm", "LFarm", "LHand")
ARM_R = ("RClav", "RScap", "RUarm", "RLarm", "RFarm", "RHand")
LEG_L = ("LUleg", "LLleg", "LFoot", "LTalus")
LEG_R = ("RUleg", "RLleg", "RFoot", "RTalus")
TORSO_LOW = ("Pelvis", "LUglut", "LLglut", "RUglut", "RLglut")
TORSO_MID = ("Ltorso", "Mtorso")
TORSO_HIGH = ("Utorso",)

GROUPS = [("arm_l", ARM_L), ("arm_r", ARM_R), ("leg_l", LEG_L), ("leg_r", LEG_R),
          ("torso", TORSO_LOW + TORSO_MID + TORSO_HIGH)]

PROFILES = {
    "human": {"head": 0.081, "torso": 0.497, "arm": 0.050, "leg": 0.161,
              "low_bias": None},
    "legs": {"head": 0.050, "torso": 0.310, "arm": 0.040, "leg": 0.280,
             "low_bias": 0.55},
    "core": {"head": 0.030, "torso": 0.750, "arm": 0.020, "leg": 0.090,
             "low_bias": 0.80},
    "combo": {"head": 0.040, "torso": 0.460, "arm": 0.030, "leg": 0.220,
              "low_bias": 0.65},
}

VARIANTS = ("webots", "spec") + tuple(PROFILES)
SPEC_HEAD_MASS = 0.001


def group_of(segment):
    for name, prefixes in GROUPS:
        if segment.startswith(prefixes):
            return name
    return None


def load_spec(path=MODEL_YAML):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    table = {}
    for name, entry in data["segments"].items():
        inertia = list(entry["inertia"])
        if len(inertia) == 3:
            inertia = inertia + [0.0, 0.0, 0.0]
        table[name] = {"mass": float(entry["mass"]),
                       "com": list(entry.get("com", [0.0, 0.0, 0.0])),
                       "inertia": [float(v) for v in inertia]}
    return table


def head_entry(mass):
    value = 0.4 * mass * HEAD_RADIUS ** 2
    return {"mass": mass, "com": list(HEAD_COM),
            "inertia": [value, value, value, 0.0, 0.0, 0.0]}


def scale(entry, factor):
    return {"mass": entry["mass"] * factor, "com": list(entry["com"]),
            "inertia": [v * factor for v in entry["inertia"]]}


def _rescale(table, keys, target):
    current = sum(table[k]["mass"] for k in keys)
    for key in keys:
        table[key] = scale(table[key], target / current)


def build(variant, spec=None):
    spec = spec or load_spec()
    total = sum(e["mass"] for e in spec.values())

    if variant == "webots":
        flat = {"mass": 0.001, "com": [0.0, 0.0, 0.0],
                "inertia": [1e-6] * 3 + [0.0] * 3}
        table = {name: dict(flat) for name in spec}
        table[HEAD_SEGMENT] = dict(flat)
        return table

    if variant == "spec":
        table = {name: scale(entry, 1.0) for name, entry in spec.items()}
        table[HEAD_SEGMENT] = head_entry(SPEC_HEAD_MASS)
        return table

    profile = PROFILES[variant]
    table = {name: scale(entry, 1.0) for name, entry in spec.items()}
    _rescale(table, [k for k in table if group_of(k) == "arm_l"],
             profile["arm"] * total)
    _rescale(table, [k for k in table if group_of(k) == "arm_r"],
             profile["arm"] * total)
    _rescale(table, [k for k in table if group_of(k) == "leg_l"],
             profile["leg"] * total)
    _rescale(table, [k for k in table if group_of(k) == "leg_r"],
             profile["leg"] * total)

    torso_mass = profile["torso"] * total
    low = [k for k in table if k.startswith(TORSO_LOW)]
    mid = [k for k in table if k.startswith(TORSO_MID)]
    high = [k for k in table if k.startswith(TORSO_HIGH)]
    if profile["low_bias"] is None:
        _rescale(table, low + mid + high, torso_mass)
    else:
        share = profile["low_bias"]
        _rescale(table, low, share * torso_mass)
        _rescale(table, mid, 0.15 * torso_mass)
        _rescale(table, high, (1.0 - share - 0.15) * torso_mass)

    table[HEAD_SEGMENT] = head_entry(profile["head"] * total)
    return table


def summarise(table):
    total = sum(e["mass"] for e in table.values())
    def share(prefixes):
        return 100.0 * sum(e["mass"] for k, e in table.items()
                           if k.startswith(prefixes)) / total
    return {"total": total,
            "arm_percent": share(ARM_L + ARM_R),
            "leg_percent": share(LEG_L + LEG_R),
            "torso_percent": share(TORSO_LOW + TORSO_MID + TORSO_HIGH),
            "head_percent": share((HEAD_SEGMENT,))}
