from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pyorbbecsdk import Pipeline, RecordDevice, Context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
STOP_REQUESTED = False


def on_signal(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def frame_type_name(frame_type: Any) -> str:
    try:
        return frame_type.name
    except Exception:
        return str(frame_type)


def safe_call(method, default=None):
    try:
        return method()
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record two Orbbec devices simultaneously into separate .bag files."
    )
    parser.add_argument("--name", required=True, help="Experiment name, for example dual_sit_01")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds")
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0.")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() < 2:
        raise RuntimeError(f"Need 2 devices, but only detected {device_list.get_count()}.")

    session_root = RAW_ROOT / f"{timestamp_name()}_{args.name}"
    session_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Session root : {session_root}")

    device_infos: list[dict[str, Any]] = []
    pipelines: list[Pipeline] = []
    recorders: list[RecordDevice | None] = []
    frame_counts: list[Counter[str]] = [Counter(), Counter()]
    timestamp_samples: list[dict[str, list[dict[str, Any]]]] = [
        {
            "COLOR_FRAME": [],
            "DEPTH_FRAME": [],
            "LEFT_IR_FRAME": [],
            "RIGHT_IR_FRAME": [],
            "ACCEL_FRAME": [],
            "GYRO_FRAME": [],
        },
        {
            "COLOR_FRAME": [],
            "DEPTH_FRAME": [],
            "LEFT_IR_FRAME": [],
            "RIGHT_IR_FRAME": [],
            "ACCEL_FRAME": [],
            "GYRO_FRAME": [],
        },
    ]

    try:
        pending_starts: list[tuple[Pipeline, Any, Any]] = []

        for i in range(2):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            serial = info.get_serial_number()
            device_dir = session_root / serial
            device_dir.mkdir(parents=True, exist_ok=False)

            try:
                device.timer_sync_with_host()
                print(f"[INFO] Host/device time sync requested for {serial}.")
            except Exception as exc:
                print(f"[WARN] timer_sync_with_host failed for {serial}: {exc}")

            pipeline = Pipeline(device)
            config = pipeline.get_config()
            enabled_sensors: list[str] = []
            skipped_sensors: list[str] = []
            sensors = device.get_sensor_list()
            for j in range(len(sensors)):
                sensor_type = sensors[j].get_type()
                try:
                    config.enable_stream(sensor_type)
                    enabled_sensors.append(getattr(sensor_type, "name", str(sensor_type)))
                except Exception as exc:
                    skipped_sensors.append(f"{sensor_type}: {exc}")

            bag_path = device_dir / "recording.bag"
            recorder = RecordDevice(device, str(bag_path))

            def make_callback(device_index: int):
                def _callback(frameset: Any) -> None:
                    for idx in range(frameset.get_count()):
                        frame = frameset.get_frame_by_index(idx)
                        name = frame_type_name(frame.get_type())
                        frame_counts[device_index][name] += 1
                        if name in timestamp_samples[device_index] and len(timestamp_samples[device_index][name]) < 40:
                            timestamp_samples[device_index][name].append(
                                {
                                    "frame_index": frame_counts[device_index][name],
                                    "timestamp_us": safe_call(frame.get_timestamp_us),
                                    "system_timestamp_us": safe_call(frame.get_system_timestamp_us),
                                    "width": safe_call(frame.get_width),
                                    "height": safe_call(frame.get_height),
                                }
                            )

                return _callback

            pipelines.append(pipeline)
            recorders.append(recorder)
            pending_starts.append((pipeline, config, make_callback(i)))
            device_infos.append(
                {
                    "serial": serial,
                    "device_dir": str(device_dir),
                    "bag_path": str(bag_path),
                    "device": {
                        "name": info.get_name(),
                        "connection_type": info.get_connection_type(),
                        "firmware_version": info.get_firmware_version(),
                        "vid": info.get_vid(),
                        "pid": info.get_pid(),
                    },
                    "enabled_sensors": enabled_sensors,
                    "skipped_sensors": skipped_sensors,
                }
            )

        for pipeline, config, callback in pending_starts:
            pipeline.start(config, callback)

        print("[INFO] Both devices are recording. Press Ctrl+C to stop early.")
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
        print(f"[INFO] Stopping dual recording after {actual_duration:.2f} s")

    finally:
        recorders = [None for _ in recorders]
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        print("[INFO] Pipelines stopped.")

    timestamp_probe: dict[str, Any] = {}
    if len(device_infos) == 2:
        for stream_name in ["COLOR_FRAME", "DEPTH_FRAME", "LEFT_IR_FRAME", "RIGHT_IR_FRAME"]:
            samples_a = timestamp_samples[0].get(stream_name, [])
            samples_b = timestamp_samples[1].get(stream_name, [])
            if samples_a and samples_b:
                first_a = samples_a[0]
                first_b = samples_b[0]
                timestamp_probe[stream_name] = {
                    "device_a_serial": device_infos[0]["serial"],
                    "device_b_serial": device_infos[1]["serial"],
                    "first_timestamp_delta_us": first_b["timestamp_us"] - first_a["timestamp_us"],
                    "first_system_timestamp_delta_us": first_b["system_timestamp_us"] - first_a["system_timestamp_us"],
                }

    summary = {
        "session_root": str(session_root),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds_requested": args.duration,
        "duration_seconds_actual": round(actual_duration, 3),
        "timestamp_probe": timestamp_probe,
        "devices": [],
    }

    for i, info in enumerate(device_infos):
        payload = {
            "session_root": str(session_root),
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds_requested": args.duration,
            "duration_seconds_actual": round(actual_duration, 3),
            "serial": info["serial"],
            "bag_path": info["bag_path"],
            "device": info["device"],
            "enabled_sensors": info["enabled_sensors"],
            "skipped_sensors": info["skipped_sensors"],
            "frame_counts": dict(frame_counts[i]),
            "timestamp_samples": timestamp_samples[i],
        }
        summary["devices"].append(payload)
        meta_path = Path(info["device_dir"]) / "metadata.json"
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    (session_root / "session_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[INFO] Session summary saved.")
    for i, info in enumerate(device_infos):
        print(f"[INFO] {info['serial']} -> {info['bag_path']}")
        print(f"       frame_counts = {dict(frame_counts[i])}")
    if timestamp_probe:
        print("[INFO] Timestamp probe:")
        for stream_name, probe in timestamp_probe.items():
            print(
                f"  {stream_name}: device_ts_delta_us={probe['first_timestamp_delta_us']} "
                f"system_ts_delta_us={probe['first_system_timestamp_delta_us']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
