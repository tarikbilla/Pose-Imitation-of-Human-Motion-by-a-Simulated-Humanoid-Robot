import glob
import json
import os
import sys

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def load():
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "m1_stance_*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            runs.append(json.load(handle))
    return runs


def main():
    runs = load()
    if not runs:
        print("no stance results found")
        return 1

    runs.sort(key=lambda r: (not r["admittance"], r["impulse_ns"]))

    print("=" * 78)
    print("M1 stance / impulse rejection")
    print("=" * 78)
    header = f"{'tag':<14}{'impulse':>9}{'admit':>7}{'peak lat':>10}{'residual':>10}{'contacts':>10}{'fallen':>8}"
    print(header)
    print("-" * 78)
    for run in runs:
        print(
            f"{run['tag']:<14}"
            f"{run['impulse_ns']:>7.0f} Ns"
            f"{'on' if run['admittance'] else 'off':>7}"
            f"{run['peak_lateral_error_m'] * 1000:>8.1f} mm"
            f"{run['residual_lateral_error_m'] * 1000:>8.1f} mm"
            f"{run['min_contacts']:>10}"
            f"{'YES' if run['fallen'] else 'no':>8}"
        )
    print("-" * 78)

    for mode in (True, False):
        subset = [r for r in runs if r["admittance"] == mode and r["impulse_ns"] > 0]
        survived = [r["impulse_ns"] for r in subset if not r["fallen"]]
        failed = [r["impulse_ns"] for r in subset if r["fallen"]]
        label = "with admittance" if mode else "without admittance"
        if survived or failed:
            best = max(survived) if survived else 0
            worst = min(failed) if failed else None
            print(
                f"{label:<20} max survived {best:>5.0f} Ns"
                + (f"   first failure {worst:.0f} Ns" if worst is not None else "   no failure in range")
            )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
