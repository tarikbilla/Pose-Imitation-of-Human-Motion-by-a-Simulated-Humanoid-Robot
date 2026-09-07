import argparse
import json
import os
import statistics


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    report = {}
    for variant, entries in groups.items():
        good = [e for e in entries if e.get("ok")]
        if not good:
            report[variant] = {"runs": len(entries), "usable": 0}
            continue
        falls = sum(1 for e in good if e["fallen"])
        report[variant] = {
            "runs": len(good),
            "falls": falls,
            "imit_median_deg": round(statistics.median(
                e["imit_median_deg"] for e in good), 2),
            "imit_spread_deg": round(
                max(e["imit_median_deg"] for e in good)
                - min(e["imit_median_deg"] for e in good), 2),
            "imit_p95_deg": round(statistics.median(
                e["imit_p95_deg"] for e in good), 2),
            "packets_median": int(statistics.median(e["packets"] for e in good)),
            "duration_median_s": round(statistics.median(
                e["duration_s"] for e in good), 1),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for path in args.files:
        data = load(path)
        rows = data["rows"]
        amp = data.get("amplify", 1.0)
        print(f"\n=== {os.path.basename(path)}   amplify={amp}  "
              f"arm_scale={data.get('arm_scale')}  "
              f"safety={data.get('safety')} ===")
        report = aggregate(rows)
        print(f"{'Profil':9s}{'Laeufe':>7s}{'Stuerze':>9s}{'imit_med':>10s}"
              f"{'spread':>10s}{'p95':>8s}{'packets':>8s}{'duration':>8s}")
        for variant, entry in report.items():
            if not entry.get("runs"):
                print(f"{variant:9s}  keine verwertbaren Laeufe")
                continue
            flag = "" if entry["falls"] == 0 else "  <-- error measured only up to the fall"
            print(f"{variant:9s}{entry['runs']:7d}{entry['falls']:9d}"
                  f"{entry['imit_median_deg']:10.2f}{entry['imit_spread_deg']:10.2f}"
                  f"{entry['imit_p95_deg']:8.2f}{entry['packets_median']:8d}"
                  f"{entry['duration_median_s']:8.1f}{flag}")


if __name__ == "__main__":
    main()
