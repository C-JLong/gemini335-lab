from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pyorbbecsdk import Context, Pipeline, RecordDevice


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple simultaneous recording for two Gemini 335 devices.")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ctx = Context()
    devs = ctx.query_devices()
    if devs.get_count() < 2:
        raise RuntimeError(f"Need 2 devices, found {devs.get_count()}.")

    session_root = RAW_ROOT / f"{timestamp_name()}_{args.name}"
    session_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Session root : {session_root}")

    devices = []
    pipelines = []
    configs = []
    recorders = []
    frame_counts = []

    try:
        for i in range(2):
            dev = devs.get_device_by_index(i)
            info = dev.get_device_info()
            serial = info.get_serial_number()
            device_dir = session_root / serial
            device_dir.mkdir(parents=True, exist_ok=False)

            try:
                dev.timer_sync_with_host()
            except Exception as exc:
                print(f"[WARN] timer_sync_with_host failed for {serial}: {exc}")

            pipe = Pipeline(dev)
            cfg = pipe.get_config()
            sensors = dev.get_sensor_list()
            enabled_sensors = []
            skipped_sensors = []
            for j in range(len(sensors)):
                st = sensors[j].get_type()
                try:
                    cfg.enable_stream(st)
                    enabled_sensors.append(getattr(st, "name", str(st)))
                except Exception as exc:
                    skipped_sensors.append(f"{st}: {exc}")

            bag_path = device_dir / "recording.bag"
            rec = RecordDevice(dev, str(bag_path))
            counts: Counter[str] = Counter()

            def make_cb(local_counts: Counter[str]):
                def _cb(frames):
                    for k in range(frames.get_count()):
                        frame = frames.get_frame_by_index(k)
                        local_counts[frame_type_name(frame.get_type())] += 1

                return _cb

            devices.append(
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
                    "callback": make_cb(counts),
                }
            )
            pipelines.append(pipe)
            configs.append(cfg)
            recorders.append(rec)
            frame_counts.append(counts)

        for i in range(2):
            pipelines[i].start(configs[i], devices[i]["callback"])
            print(f"[INFO] Started device {i}: {devices[i]['serial']}")

        start = time.time()
        print("[INFO] Both devices are recording.")
        while not STOP_REQUESTED:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break
            time.sleep(0.05)

        actual_duration = time.time() - start
        print(f"[INFO] Stopping after {actual_duration:.2f} s")

    finally:
        recorders = [None for _ in recorders]
        for pipe in pipelines:
            try:
                pipe.stop()
            except Exception:
                pass
        print("[INFO] Pipelines stopped.")

    summary = {
        "session_root": str(session_root),
        "duration_seconds_requested": args.duration,
        "duration_seconds_actual": round(actual_duration, 3),
        "devices": [],
    }

    for i in range(2):
        device_info = devices[i].copy()
        device_info.pop("callback", None)
        device_info["frame_counts"] = dict(frame_counts[i])
        meta_path = Path(device_info["device_dir"]) / "metadata.json"
        meta_path.write_text(json.dumps(device_info, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["devices"].append(device_info)
        print(f"[INFO] {device_info['serial']} frame_counts = {device_info['frame_counts']}")

    (session_root / "session_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] Saved summary: {session_root / 'session_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
