import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "..", "webots", "controllers", "a3_wholebody"))

import mass_tables as mt
from atlaskin import SEGMENT_OF, AtlasModel

GRAVITY = 9.81
NEEDED_SHIFT = 0.089
ARM_JOINTS = ("LArmUsy", "LArmShx", "LArmEly", "LArmElx", "LArmUwy", "LArmMwx",
              "RArmUsy", "RArmShx", "RArmEly", "RArmElx", "RArmUwy", "RArmMwx")
UPPER = ARM_JOINTS + ("BackLbz", "BackMby", "BackUbx", "NeckAy")


def apply(model, table):
    for joint in model.names:
        segment = SEGMENT_OF.get(joint)
        entry = table.get(segment) if segment else None
        model.mass[joint] = entry["mass"] if entry else 0.0
        model.com_local[joint] = (np.asarray(entry["com"], dtype=np.float64)
                                  if entry else np.zeros(3))
    head = table[mt.HEAD_SEGMENT]
    model.mass["NeckAy"] = head["mass"]
    model.com_local["NeckAy"] = np.asarray(head["com"], dtype=np.float64)
    pelvis = table["PelvisSolid"]
    model.root_mass = pelvis["mass"]
    model.root_com = np.asarray(pelvis["com"], dtype=np.float64)
    model.total_mass = model.root_mass + sum(model.mass.values())


def polygon(model, neutral):
    left = model.sole(neutral, "L")
    right = model.sole(neutral, "R")
    ankle_x = 0.5 * (left[0] + right[0]) - 0.048
    return (ankle_x - 0.082, ankle_x + 0.178,
            right[1] - 0.0624, left[1] + 0.0624)


def margin(com, poly):
    return min(com[0] - poly[0], poly[1] - com[0],
               com[1] - poly[2], poly[3] - com[1])


def shift_effort(model, neutral):
    limit = 0.436
    for theta in np.linspace(0.0, limit, 400):
        angles = dict(neutral)
        angles["LLegMhx"] = theta
        angles["RLegMhx"] = theta
        angles["LLegLax"] = -theta
        angles["RLegLax"] = -theta
        left = model.sole(angles, "L")
        right = model.sole(angles, "R")
        mid_y = 0.5 * float(left[1] + right[1])
        if abs(float(model.com(angles)[1]) - mid_y) >= NEEDED_SHIFT:
            return theta, 100.0 * theta / limit
    return float("nan"), float("nan")


def main():
    model = AtlasModel()
    spec = mt.load_spec()
    neutral = {n: 0.0 for n in model.names}
    poly = polygon(model, neutral)
    sole_z = float(model.sole(neutral, "L")[2])

    rng = np.random.default_rng(20260905)
    full = [{**neutral, **{j: float(rng.uniform(*model.limits[j])) for j in UPPER}}
            for _ in range(4000)]
    arms = [{**neutral,
             **{j: float(rng.uniform(*model.limits[j])) for j in ARM_JOINTS}}
            for _ in range(2000)]

    print(f"{'Profil':10s}{'CoM h':>8s}{'omega':>7s}{'stabil':>8s}"
          f"{'Marge p05':>10s}{'Arm-Hub':>9s}{'theta89':>9s}{'Limit':>8s}")
    report = {}
    for variant in mt.VARIANTS:
        apply(model, mt.build(variant, spec))
        base = model.com(neutral)
        height = float(base[2]) - sole_z
        omega = float(np.sqrt(GRAVITY / height))
        margins = np.array([margin(model.com(p), poly) for p in full])
        swing = max(float(np.hypot(*(model.com(p)[:2] - base[:2]))) for p in arms)
        theta, effort = shift_effort(model, neutral)
        report[variant] = {
            "com_height_m": round(height, 4),
            "omega": round(omega, 3),
            "stable_fraction": round(float((margins > 0).mean()), 4),
            "margin_p05_m": round(float(np.percentile(margins, 5)), 4),
            "arm_excursion_m": round(swing, 4),
            "shift_theta_rad": round(theta, 4),
            "shift_effort_percent": round(effort, 1),
            "shift_needed_m": NEEDED_SHIFT,
        }
        print(f"{variant:10s}{height:8.3f}{omega:7.2f}"
              f"{100 * (margins > 0).mean():7.1f}%{np.percentile(margins, 5):10.4f}"
              f"{swing:9.4f}{theta:9.3f}{effort:8.0f}%")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "mass_compare.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"polygon": [round(v, 4) for v in poly],
                   "variants": report}, handle, indent=2)


if __name__ == "__main__":
    main()
