from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    OBFrameAggregateOutputMode,
    OBPlaybackStatus,
    OBSensorType,
    Pipeline,
    PlaybackDevice,
)


FRAME_INDEX_FIELDS = ["stream", "frame_index", "timestamp_us", "system_timestamp_us", "width", "height"]
DEPTH_RAW_INDEX_FIELDS = [
    "filename",
    "depth_frame_index",
    "timestamp_us",
    "system_timestamp_us",
    "width",
    "height",
]
RGBD_MATCHED_FIELDS = [
    "color_frame_index",
    "color_timestamp_us",
    "color_system_timestamp_us",
    "color_width",
    "color_height",
    "color_path",
    "nearest_depth_frame_index",
    "nearest_depth_timestamp_us",
    "nearest_time_diff_us",
    "nearest_time_diff_ms",
    "depth_frame_index",
    "depth_timestamp_us",
    "depth_system_timestamp_us",
    "depth_width",
    "depth_height",
    "depth_path",
    "time_diff_us",
    "time_diff_ms",
    "match_status",
]


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)()
    except Exception:
        return default


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_frame_index(frame_index_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not frame_index_csv.exists():
        raise FileNotFoundError(f"frame_index.csv not found: {frame_index_csv}")

    with frame_index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"frame_index.csv is empty: {frame_index_csv}")
        missing = [field for field in FRAME_INDEX_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"frame_index.csv missing required fields: {missing}")

        color_rows: list[dict[str, Any]] = []
        depth_rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = {
                "stream": row["stream"],
                "frame_index": int_or_none(row["frame_index"]),
                "timestamp_us": int_or_none(row["timestamp_us"]),
                "system_timestamp_us": int_or_none(row["system_timestamp_us"]),
                "width": int_or_none(row["width"]),
                "height": int_or_none(row["height"]),
            }
            if parsed["stream"] == "COLOR":
                color_rows.append(parsed)
            elif parsed["stream"] == "DEPTH":
                depth_rows.append(parsed)

    if not color_rows:
        raise ValueError(f"No COLOR rows found in {frame_index_csv}")
    if not depth_rows:
        raise ValueError(f"No DEPTH rows found in {frame_index_csv}")

    return color_rows, depth_rows


def reset_output_dir(depth_dir: Path, overwrite: bool) -> None:
    if depth_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Depth output already exists: {depth_dir}. Re-run with --overwrite to replace it."
            )
        resolved = depth_dir.resolve()
        if resolved.name != "depth_raw":
            raise ValueError(f"Refusing to remove unexpected directory: {resolved}")
        shutil.rmtree(resolved)
    depth_dir.mkdir(parents=True, exist_ok=True)


def enable_depth_stream(playback: PlaybackDevice, config: Config) -> list[str]:
    enabled_sensors: list[str] = []
    try:
        config.enable_stream(OBSensorType.DEPTH_SENSOR)
        return ["DEPTH_SENSOR"]
    except Exception:
        pass

    sensor_list = playback.get_sensor_list()
    for i in range(len(sensor_list)):
        sensor_type = sensor_list[i].get_type()
        if sensor_type != OBSensorType.DEPTH_SENSOR:
            continue
        config.enable_stream(sensor_type)
        enabled_sensors.append(getattr(sensor_type, "name", str(sensor_type)))
    if not enabled_sensors:
        raise RuntimeError("Could not enable DEPTH stream from the bag playback.")
    return enabled_sensors


def depth_frame_to_uint16(frame: Any) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    expected = width * height
    if data.size < expected:
        raise ValueError(f"Depth frame data too small: got {data.size}, expected {expected}")
    if data.size > expected:
        data = data[:expected]
    return data.reshape(height, width)


def get_depth_scale_info(frame: Any, existing_scale: float | None) -> tuple[float | None, str]:
    if existing_scale is not None:
        return existing_scale, "sdk_depth_scale_to_mm"
    scale = float_or_none(safe_attr(frame, "get_depth_scale"))
    if scale is None:
        return None, "unknown"
    return scale, "sdk_depth_scale_to_mm"


def export_depth_frames_from_bag(
    bag_path: Path,
    export_dir: Path,
    depth_rows: list[dict[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    depth_dir = export_dir / "depth_raw"
    reset_output_dir(depth_dir, overwrite)

    raw_index_path = export_dir / "depth_raw_index.csv"
    summary_path = export_dir / "depth_export_summary.json"

    playback = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback)
    config = Config()
    try:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION)
    except Exception:
        pass
    enabled_sensors = enable_depth_stream(playback, config)

    exported_rows: list[dict[str, Any]] = []
    exported_timestamps: list[int] = []
    exported_widths: set[int] = set()
    exported_heights: set[int] = set()
    sdk_index_available = False
    sdk_index_used = False
    depth_scale: float | None = None
    depth_unit = "unknown"
    final_status = playback.get_playback_status()

    print(f"[INFO] Exporting raw depth from: {bag_path}")
    print(f"[INFO] Depth output dir        : {depth_dir}")
    print("[INFO] Raw depth PNG format   : uint16 single-channel, no normalization, no colormap")

    with raw_index_path.open("w", newline="", encoding="utf-8") as index_f:
        writer = csv.DictWriter(index_f, fieldnames=DEPTH_RAW_INDEX_FIELDS)
        writer.writeheader()

        pipeline.start(config)
        try:
            empty_loops = 0
            read_depth_count = 0
            while True:
                frames = pipeline.wait_for_frames(100)
                status = playback.get_playback_status()
                final_status = status

                if frames is None:
                    empty_loops += 1
                    if status == OBPlaybackStatus.STOPPED and empty_loops > 10:
                        break
                    continue

                empty_loops = 0
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue

                read_depth_count += 1
                expected_row = depth_rows[read_depth_count - 1] if read_depth_count <= len(depth_rows) else None

                sdk_index = int_or_none(safe_attr(depth_frame, "get_index"))
                sdk_index_available = sdk_index_available or sdk_index is not None
                expected_index = expected_row["frame_index"] if expected_row is not None else read_depth_count
                if sdk_index is not None and expected_row is not None and sdk_index == expected_index:
                    depth_frame_index = sdk_index
                    sdk_index_used = True
                else:
                    depth_frame_index = expected_index

                depth_scale, depth_unit = get_depth_scale_info(depth_frame, depth_scale)
                depth_image = depth_frame_to_uint16(depth_frame)
                width = int(depth_frame.get_width())
                height = int(depth_frame.get_height())
                timestamp_us = int_or_none(safe_attr(depth_frame, "get_timestamp_us"))
                system_timestamp_us = int_or_none(safe_attr(depth_frame, "get_system_timestamp_us"))

                filename = f"depth_{depth_frame_index:06d}.png"
                output_path = depth_dir / filename
                if not cv2.imwrite(str(output_path), depth_image):
                    raise RuntimeError(f"Failed to write depth PNG: {output_path}")

                row = {
                    "filename": filename,
                    "depth_frame_index": depth_frame_index,
                    "timestamp_us": timestamp_us,
                    "system_timestamp_us": system_timestamp_us,
                    "width": width,
                    "height": height,
                }
                writer.writerow(row)
                exported_rows.append(row)
                if timestamp_us is not None:
                    exported_timestamps.append(timestamp_us)
                exported_widths.add(width)
                exported_heights.add(height)

                if read_depth_count % 1000 == 0:
                    print(f"[INFO] Exported depth frames: {read_depth_count}")

        finally:
            pipeline.stop()

    expected_depth_frames = len(depth_rows)
    exported_depth_frames = len(exported_rows)
    missing_count = max(0, expected_depth_frames - exported_depth_frames)
    extra_count = max(0, exported_depth_frames - expected_depth_frames)
    if expected_depth_frames != exported_depth_frames:
        print(
            "[WARNING] Depth PNG count mismatch: "
            f"expected {expected_depth_frames}, exported {exported_depth_frames}, "
            f"missing_count {missing_count}, extra_count {extra_count}"
        )

    frame_index_first_ts = depth_rows[0]["timestamp_us"]
    frame_index_last_ts = depth_rows[-1]["timestamp_us"]
    exported_first_ts = exported_rows[0]["timestamp_us"] if exported_rows else None
    exported_last_ts = exported_rows[-1]["timestamp_us"] if exported_rows else None
    first_diff = (
        abs(exported_first_ts - frame_index_first_ts)
        if exported_first_ts is not None and frame_index_first_ts is not None
        else None
    )
    last_diff = (
        abs(exported_last_ts - frame_index_last_ts)
        if exported_last_ts is not None and frame_index_last_ts is not None
        else None
    )

    expected_widths = sorted({row["width"] for row in depth_rows if row["width"] is not None})
    expected_heights = sorted({row["height"] for row in depth_rows if row["height"] is not None})
    width_match = sorted(exported_widths) == expected_widths
    height_match = sorted(exported_heights) == expected_heights

    png_validation = validate_depth_pngs(depth_dir, exported_rows)

    summary = {
        "source_bag": str(bag_path),
        "depth_output_dir": str(depth_dir),
        "depth_raw_index_csv": str(raw_index_path),
        "enabled_sensors": enabled_sensors,
        "final_playback_status": playback_status_name(final_status),
        "depth_format": "uint16_single_channel_png",
        "depth_width": sorted(exported_widths),
        "depth_height": sorted(exported_heights),
        "expected_depth_width": expected_widths,
        "expected_depth_height": expected_heights,
        "depth_resolution_matches_frame_index": bool(width_match and height_match),
        "num_depth_frames": exported_depth_frames,
        "expected_depth_frames": expected_depth_frames,
        "exported_depth_frames": exported_depth_frames,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "first_depth_timestamp_us": exported_first_ts,
        "last_depth_timestamp_us": exported_last_ts,
        "frame_index_first_depth_timestamp_us": frame_index_first_ts,
        "frame_index_last_depth_timestamp_us": frame_index_last_ts,
        "exported_first_depth_timestamp_us": exported_first_ts,
        "exported_last_depth_timestamp_us": exported_last_ts,
        "first_timestamp_diff_us": first_diff,
        "last_timestamp_diff_us": last_diff,
        "depth_unit": depth_unit,
        "depth_scale": depth_scale if depth_scale is not None else "unknown",
        "sdk_depth_frame_index_available": sdk_index_available,
        "sdk_depth_frame_index_used": sdk_index_used,
        "whether_matched_frame_index_csv": exported_depth_frames == expected_depth_frames
        and first_diff == 0
        and last_diff == 0
        and width_match
        and height_match,
        "png_validation": png_validation,
    }
    write_json(summary, summary_path)
    print(f"[INFO] depth_raw_index.csv      : {raw_index_path}")
    print(f"[INFO] depth_export_summary   : {summary_path}")
    return summary


def validate_depth_pngs(depth_dir: Path, exported_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not exported_rows:
        return {"checked_files": 0, "all_checked_uint16_single_channel": False, "samples": []}

    sample_indices = sorted({0, len(exported_rows) // 2, len(exported_rows) - 1})
    samples: list[dict[str, Any]] = []
    all_ok = True
    for idx in sample_indices:
        row = exported_rows[idx]
        path = depth_dir / str(row["filename"])
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        ok = image is not None and image.dtype == np.uint16 and len(image.shape) == 2
        all_ok = all_ok and ok
        samples.append(
            {
                "filename": row["filename"],
                "readable": image is not None,
                "dtype": str(image.dtype) if image is not None else None,
                "shape": list(image.shape) if image is not None else None,
                "uint16_single_channel": bool(ok),
            }
        )
    return {
        "checked_files": len(samples),
        "all_checked_uint16_single_channel": bool(all_ok),
        "samples": samples,
    }


def playback_status_name(status: Any) -> str:
    try:
        return status.name
    except Exception:
        return str(status)


def logical_color_path(frame_index: int | None) -> str:
    if frame_index is None:
        return ""
    return f"color/color_{frame_index:06d}.png"


def logical_depth_path(frame_index: int | None) -> str:
    if frame_index is None:
        return ""
    return f"depth_raw/depth_{frame_index:06d}.png"


def nearest_depth_row(
    color_timestamp_us: int | None,
    depth_rows: list[dict[str, Any]],
    depth_timestamps: list[int],
) -> tuple[dict[str, Any] | None, int | None]:
    if color_timestamp_us is None or not depth_timestamps:
        return None, None
    pos = bisect.bisect_left(depth_timestamps, color_timestamp_us)
    candidates: list[int] = []
    if pos < len(depth_timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None, None
    best_idx = min(candidates, key=lambda i: abs(depth_timestamps[i] - color_timestamp_us))
    diff_us = abs(depth_timestamps[best_idx] - color_timestamp_us)
    return depth_rows[best_idx], diff_us


def time_diff_statistics(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "num_over_5ms": 0,
            "num_over_10ms": 0,
            "num_over_16_7ms": 0,
            "num_over_33ms": 0,
        }
    return {
        "count": len(values_ms),
        "min": min(values_ms),
        "median": median(values_ms),
        "mean": mean(values_ms),
        "max": max(values_ms),
        "num_over_5ms": sum(1 for value in values_ms if value > 5.0),
        "num_over_10ms": sum(1 for value in values_ms if value > 10.0),
        "num_over_16_7ms": sum(1 for value in values_ms if value > 16.7),
        "num_over_33ms": sum(1 for value in values_ms if value > 33.0),
    }


def build_rgbd_matched_index(
    color_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    export_dir: Path,
    max_time_diff_ms: float,
) -> dict[str, Any]:
    matched_csv_path = export_dir / "rgbd_matched_index.csv"
    summary_path = export_dir / "rgbd_sync_summary.json"

    sorted_depth_rows = sorted(
        [row for row in depth_rows if row["timestamp_us"] is not None],
        key=lambda row: row["timestamp_us"],
    )
    depth_timestamps = [row["timestamp_us"] for row in sorted_depth_rows]
    max_time_diff_us = int(round(max_time_diff_ms * 1000.0))

    matched_rows: list[dict[str, Any]] = []
    nearest_diffs_ms: list[float] = []
    matched_diffs_ms: list[float] = []
    matched_depth_ids: list[int] = []

    for color_row in color_rows:
        nearest_row, nearest_diff_us = nearest_depth_row(
            color_row["timestamp_us"], sorted_depth_rows, depth_timestamps
        )
        nearest_diff_ms = nearest_diff_us / 1000.0 if nearest_diff_us is not None else None
        if nearest_diff_ms is not None:
            nearest_diffs_ms.append(nearest_diff_ms)

        row = {
            "color_frame_index": color_row["frame_index"],
            "color_timestamp_us": color_row["timestamp_us"],
            "color_system_timestamp_us": color_row["system_timestamp_us"],
            "color_width": color_row["width"],
            "color_height": color_row["height"],
            "color_path": logical_color_path(color_row["frame_index"]),
            "nearest_depth_frame_index": nearest_row["frame_index"] if nearest_row else "",
            "nearest_depth_timestamp_us": nearest_row["timestamp_us"] if nearest_row else "",
            "nearest_time_diff_us": nearest_diff_us if nearest_diff_us is not None else "",
            "nearest_time_diff_ms": nearest_diff_ms if nearest_diff_ms is not None else "",
            "depth_frame_index": "",
            "depth_timestamp_us": "",
            "depth_system_timestamp_us": "",
            "depth_width": "",
            "depth_height": "",
            "depth_path": "",
            "time_diff_us": "",
            "time_diff_ms": "",
            "match_status": "missing_depth",
        }

        if nearest_row is not None and nearest_diff_us is not None and nearest_diff_us <= max_time_diff_us:
            row.update(
                {
                    "depth_frame_index": nearest_row["frame_index"],
                    "depth_timestamp_us": nearest_row["timestamp_us"],
                    "depth_system_timestamp_us": nearest_row["system_timestamp_us"],
                    "depth_width": nearest_row["width"],
                    "depth_height": nearest_row["height"],
                    "depth_path": logical_depth_path(nearest_row["frame_index"]),
                    "time_diff_us": nearest_diff_us,
                    "time_diff_ms": nearest_diff_ms,
                    "match_status": "matched",
                }
            )
            matched_diffs_ms.append(nearest_diff_ms)
            if nearest_row["frame_index"] is not None:
                matched_depth_ids.append(nearest_row["frame_index"])

        matched_rows.append(row)

    with matched_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RGBD_MATCHED_FIELDS)
        writer.writeheader()
        writer.writerows(matched_rows)

    depth_id_counts = Counter(matched_depth_ids)
    reused_counts = [count for count in depth_id_counts.values() if count >= 2]
    num_matched = len(matched_diffs_ms)
    num_missing = len(color_rows) - num_matched
    summary = {
        "rgbd_matched_index_csv": str(matched_csv_path),
        "num_color_frames": len(color_rows),
        "num_depth_frames": len(depth_rows),
        "num_matched": num_matched,
        "num_missing_depth": num_missing,
        "match_rate": num_matched / len(color_rows) if color_rows else 0.0,
        "max_time_diff_ms": max_time_diff_ms,
        "num_unique_depth_matched": len(depth_id_counts),
        "num_reused_depth_frame_ids": len(reused_counts),
        "num_extra_depth_reuse_matches": sum(count - 1 for count in reused_counts),
        "nearest_time_diff_ms_statistics": time_diff_statistics(nearest_diffs_ms),
        "matched_time_diff_ms_statistics": time_diff_statistics(matched_diffs_ms),
        "note": (
            "This index performs timestamp synchronization only. It does not spatially align "
            "DEPTH pixels to COLOR pixels."
        ),
    }
    write_json(summary, summary_path)

    if num_missing > 0:
        print(
            "[WARNING] Some COLOR frames have no DEPTH frame within threshold: "
            f"{num_missing}/{len(color_rows)} missing_depth"
        )
    if reused_counts:
        print(
            "[WARNING] Some DEPTH frames were matched by multiple COLOR frames: "
            f"{len(reused_counts)} reused depth frame ids"
        )
    print(f"[INFO] rgbd_matched_index.csv : {matched_csv_path}")
    print(f"[INFO] rgbd_sync_summary     : {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export true raw DEPTH PNGs from an Orbbec .bag and build a COLOR-DEPTH "
            "timestamp synchronization index. This does not perform depth-to-color spatial alignment."
        )
    )
    parser.add_argument("--bag", required=True, help="Path to the original Orbbec .bag recording.")
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Existing bag_export directory containing frame_index.csv.",
    )
    parser.add_argument(
        "--max-time-diff-ms",
        type=float,
        default=10.0,
        help="Maximum nearest COLOR-DEPTH timestamp difference for a matched row. Default: 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing depth_raw directory before exporting.",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag).resolve()
    export_dir = Path(args.export_dir).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    frame_index_csv = export_dir / "frame_index.csv"
    color_rows, depth_rows = read_frame_index(frame_index_csv)

    export_depth_frames_from_bag(bag_path, export_dir, depth_rows, args.overwrite)
    build_rgbd_matched_index(color_rows, depth_rows, export_dir, args.max_time_diff_ms)

    print("[INFO] Finished true depth export and COLOR-DEPTH timestamp sync.")
    print("[INFO] Reminder: timestamp sync is not depth-to-color spatial alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
