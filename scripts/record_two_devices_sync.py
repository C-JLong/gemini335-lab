from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pyorbbecsdk import Context, OBMultiDeviceSyncMode, Pipeline, RecordDevice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
CONFIG_ROOT = PROJECT_ROOT / "config"
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


def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    name = sync_mode_str.upper()
    if name == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    if name == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    if name == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    if name == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    if name == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    if name == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    if name == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    raise ValueError(f"Unsupported sync mode: {sync_mode_str}")


def load_sync_config(config_path: Path) -> dict[str, dict]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {item["serial_number"]: item["config"] for item in payload["devices"]}


def apply_sync_config(device, desired: dict) -> dict[str, Any]:
    sync_cfg = device.get_multi_device_sync_config()
    sync_cfg.mode = sync_mode_from_str(desired["mode"])
    sync_cfg.color_delay_us = int(desired.get("color_delay_us", 0))
    sync_cfg.depth_delay_us = int(desired.get("depth_delay_us", 0))
    sync_cfg.trigger_out_enable = bool(desired.get("trigger_out_enable", False))
    sync_cfg.trigger_out_delay_us = int(desired.get("trigger_out_delay_us", 0))
    sync_cfg.frames_per_trigger = int(desired.get("frames_per_trigger", 1))
    device.set_multi_device_sync_config(sync_cfg)
    return {
        "mode": str(sync_cfg.mode),
        "color_delay_us": sync_cfg.color_delay_us,
        "depth_delay_us": sync_cfg.depth_delay_us,
        "trigger_out_enable": sync_cfg.trigger_out_enable,
        "trigger_out_delay_us": sync_cfg.trigger_out_delay_us,
        "frames_per_trigger": sync_cfg.frames_per_trigger,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record two Gemini 335 devices with project sync config applied.")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    parser.add_argument(
        "--config",
        default=str(CONFIG_ROOT / "multi_device_sync_config.json"),
        help="Path to project sync config JSON.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Sync config not found: {config_path}")
    desired_map = load_sync_config(config_path)

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
    actual_duration = 0.0

    try:
        for i in range(2):
            dev = devs.get_device_by_index(i)
            info = dev.get_device_info()
            serial = info.get_serial_number()
            if serial not in desired_map:
                raise RuntimeError(f"Serial {serial} not found in sync config {config_path}.")

            device_dir = session_root / serial
            device_dir.mkdir(parents=True, exist_ok=False)
            applied_sync = apply_sync_config(dev, desired_map[serial])

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
                    "sync_config_path": str(config_path),
                    "applied_sync_config": applied_sync,
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

        ctx.enable_multi_device_sync(60000)
        start = time.time()
        print("[INFO] Both devices are recording with sync enabled.")
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
        "sync_config_path": str(config_path),
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
