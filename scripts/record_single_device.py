from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pyorbbecsdk import Context, OBError, Pipeline, RecordDevice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

STOP_REQUESTED = False


def on_signal(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def choose_device(serial: str | None):
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count == 0:
        raise RuntimeError("No Orbbec device found. Please connect the camera and try again.")

    if serial:
        for index in range(count):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            if info.get_serial_number() == serial:
                return device
        raise RuntimeError(f"Device with serial '{serial}' was not found.")

    if count > 1:
        raise RuntimeError(
            f"Detected {count} devices. Please pass --serial to choose one device explicitly."
        )

    return device_list.get_device_by_index(0)


def sensor_name(sensor: Any) -> str:
    try:
        return sensor.name
    except Exception:
        return str(sensor)


def frame_type_name(frame_type: Any) -> str:
    try:
        return frame_type.name
    except Exception:
        return str(frame_type)


def build_session_paths(name: str, output_root: Path) -> tuple[Path, Path, Path]:
    session_dir = output_root / f"{timestamp_name()}_{name}"
    bag_path = session_dir / "recording.bag"
    meta_path = session_dir / "metadata.json"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir, bag_path, meta_path


def save_metadata(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record one Orbbec device to a .bag file for a fixed duration."
    )
    parser.add_argument("--name", required=True, help="Experiment name, for example desk_scan_01")
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Recording duration in seconds, for example 10",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="Optional device serial number. Required later when multiple devices are connected.",
    )
    parser.add_argument(
        "--output-root",
        default=str(RAW_ROOT),
        help="Root directory for the recording session. Defaults to project data/raw.",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0 seconds.")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir, bag_path, meta_path = build_session_paths(args.name, output_root)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    device = choose_device(args.serial)
    info = device.get_device_info()
    pipeline = Pipeline(device)
    config = pipeline.get_config()
    recorder = None
    frame_counts: Counter[str] = Counter()
    enabled_sensors: list[str] = []
    skipped_sensors: list[str] = []

    metadata: dict[str, Any] = {
        "session_name": session_dir.name,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds_requested": args.duration,
        "output_bag": str(bag_path),
        "device": {
            "name": info.get_name(),
            "serial": info.get_serial_number(),
            "connection_type": info.get_connection_type(),
            "vid": info.get_vid(),
            "pid": info.get_pid(),
            "firmware_version": info.get_firmware_version(),
        },
        "enabled_sensors": enabled_sensors,
        "skipped_sensors": skipped_sensors,
        "frame_counts": {},
    }

    save_metadata(meta_path, metadata)

    print(f"[INFO] Session directory : {session_dir}")
    print(f"[INFO] Output bag       : {bag_path}")
    print(f"[INFO] Device           : {info.get_name()} ({info.get_serial_number()})")
    print(f"[INFO] Connection       : {info.get_connection_type()}")

    try:
        try:
            device.timer_sync_with_host()
            print("[INFO] Host/device time sync requested.")
        except OBError as exc:
            print(f"[WARN] timer_sync_with_host failed: {exc}")

        sensor_list = device.get_sensor_list()
        for index in range(len(sensor_list)):
            sensor = sensor_list[index]
            sensor_type = sensor.get_type()
            name = sensor_name(sensor_type)
            try:
                config.enable_stream(sensor_type)
                enabled_sensors.append(name)
            except Exception as exc:  # SDK raises mixed exception types here
                skipped_sensors.append(f"{name}: {exc}")

        recorder = RecordDevice(device, str(bag_path))

        def on_frameset(frameset: Any) -> None:
            for i in range(frameset.get_count()):
                frame = frameset.get_frame_by_index(i)
                frame_counts[frame_type_name(frame.get_type())] += 1

        pipeline.start(config, on_frameset)
        print("[INFO] Recording started. Press Ctrl+C to stop early.")

        start = time.time()
        next_report = start + 1.0
        while not STOP_REQUESTED:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break
            if time.time() >= next_report:
                print(f"[INFO] Recording... {elapsed:.1f}/{args.duration:.1f} s")
                next_report += 1.0
            time.sleep(0.05)

        actual_duration = time.time() - start
        print(f"[INFO] Stopping recording after {actual_duration:.2f} s")

        metadata["duration_seconds_actual"] = round(actual_duration, 3)
        metadata["finished_at_local"] = datetime.now().isoformat(timespec="seconds")
        metadata["frame_counts"] = dict(frame_counts)
        metadata["enabled_sensors"] = enabled_sensors
        metadata["skipped_sensors"] = skipped_sensors
        save_metadata(meta_path, metadata)
        return 0

    finally:
        recorder = None
        try:
            pipeline.stop()
        except Exception:
            pass
        if bag_path.exists():
            print(f"[INFO] Saved bag file  : {bag_path}")
        if meta_path.exists():
            print(f"[INFO] Saved metadata  : {meta_path}")


if __name__ == "__main__":
    raise SystemExit(main())
