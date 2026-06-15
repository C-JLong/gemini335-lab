from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp"


def load_stream_timestamps(csv_path: Path, stream_name: str) -> tuple[list[int], list[int]]:
    device_ts: list[int] = []
    system_ts: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stream"] != stream_name:
                continue
            try:
                device_ts.append(int(row["timestamp_us"]))
                system_ts.append(int(row["system_timestamp_us"]))
            except Exception:
                continue
    return device_ts, system_ts


def nearest_abs_deltas(source: list[int], target: list[int]) -> list[int]:
    if not source or not target:
        return []
    result: list[int] = []
    for value in source:
        idx = bisect_left(target, value)
        candidates = []
        if idx < len(target):
            candidates.append(abs(target[idx] - value))
        if idx > 0:
            candidates.append(abs(target[idx - 1] - value))
        if candidates:
            result.append(min(candidates))
    return result


def summarize(values: list[int]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return float(ordered[idx])
    return {
        "count": n,
        "mean_us": round(sum(ordered) / n, 3),
        "median_us": pct(0.5),
        "p90_us": pct(0.9),
        "p95_us": pct(0.95),
        "max_us": float(ordered[-1]),
        "min_us": float(ordered[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare per-stream timestamps across two exported bag frame indexes.")
    parser.add_argument("--csv-a", required=True, help="First frame_index.csv path")
    parser.add_argument("--csv-b", required=True, help="Second frame_index.csv path")
    parser.add_argument("--label-a", default="device_a", help="Label for first device")
    parser.add_argument("--label-b", default="device_b", help="Label for second device")
    args = parser.parse_args()

    csv_a = Path(args.csv_a).resolve()
    csv_b = Path(args.csv_b).resolve()
    if not csv_a.exists():
        raise FileNotFoundError(csv_a)
    if not csv_b.exists():
        raise FileNotFoundError(csv_b)

    report = {
        "csv_a": str(csv_a),
        "csv_b": str(csv_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "streams": {},
    }

    for stream_name in ["COLOR", "DEPTH", "LEFT_IR", "RIGHT_IR"]:
        a_dev, a_sys = load_stream_timestamps(csv_a, stream_name)
        b_dev, b_sys = load_stream_timestamps(csv_b, stream_name)
        stream_report = {
            "count_a": len(a_dev),
            "count_b": len(b_dev),
            "device_timestamp_delta_us": summarize(nearest_abs_deltas(a_dev, b_dev)),
            "system_timestamp_delta_us": summarize(nearest_abs_deltas(a_sys, b_sys)),
            "first_device_timestamp_delta_us": abs(b_dev[0] - a_dev[0]) if a_dev and b_dev else None,
            "first_system_timestamp_delta_us": abs(b_sys[0] - a_sys[0]) if a_sys and b_sys else None,
        }
        report["streams"][stream_name] = stream_report

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = TMP_ROOT / "dual_bag_timestamp_compare.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved comparison report: {out_path}")
    for stream_name, stream_report in report["streams"].items():
        dev = stream_report["device_timestamp_delta_us"]
        sys = stream_report["system_timestamp_delta_us"]
        if dev is None or sys is None:
            print(f"[RESULT] {stream_name}: insufficient data")
            continue
        print(
            f"[RESULT] {stream_name}: "
            f"device_ts median={dev['median_us']} us, p95={dev['p95_us']} us, max={dev['max_us']} us | "
            f"system_ts median={sys['median_us']} us, p95={sys['p95_us']} us, max={sys['max_us']} us"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
