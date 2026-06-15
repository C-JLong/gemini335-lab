from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pyorbbecsdk import Context, OBMultiDeviceSyncMode, OBSensorType, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
TMP_ROOT = PROJECT_ROOT / "tmp"


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


def load_sync_config(config_path: Path) -> dict[str, dict]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {item["serial_number"]: item["config"] for item in payload["devices"]}


def apply_sync_config(device, serial: str, desired: dict) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Apply hardware sync config and probe timestamp alignment.")
    parser.add_argument(
        "--config",
        default=str(CONFIG_ROOT / "multi_device_sync_config.json"),
        help="Path to project sync config JSON.",
    )
    parser.add_argument("--seconds", type=float, default=3.0, help="Probe duration in seconds.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Sync config not found: {config_path}")

    desired_map = load_sync_config(config_path)
    ctx = Context()
    devs = ctx.query_devices()
    if devs.get_count() < 2:
        raise RuntimeError(f"Need 2 devices, found {devs.get_count()}.")

    devices = []
    pipelines = []
    callbacks = []
    samples = []

    try:
        for i in range(devs.get_count()):
            dev = devs.get_device_by_index(i)
            info = dev.get_device_info()
            serial = info.get_serial_number()
            if serial not in desired_map:
                raise RuntimeError(f"Serial {serial} not found in sync config {config_path}.")

            applied = apply_sync_config(dev, serial, desired_map[serial])

            pipe = Pipeline(dev)
            cfg = pipe.get_config()
            try:
                cfg.enable_stream(OBSensorType.COLOR_SENSOR)
            except Exception:
                pass
            try:
                cfg.enable_stream(OBSensorType.DEPTH_SENSOR)
            except Exception:
                pass

            sample_bucket = defaultdict(list)

            def make_cb(bucket, serial_number: str):
                def _cb(frames):
                    for k in range(frames.get_count()):
                        frame = frames.get_frame_by_index(k)
                        name = frame_type_name(frame.get_type())
                        if name in {"COLOR_FRAME", "DEPTH_FRAME"} and len(bucket[name]) < 120:
                            bucket[name].append(
                                {
                                    "serial": serial_number,
                                    "timestamp_us": safe_call(frame.get_timestamp_us),
                                    "system_timestamp_us": safe_call(frame.get_system_timestamp_us),
                                }
                            )

                return _cb

            devices.append(
                {
                    "index": i,
                    "serial": serial,
                    "name": info.get_name(),
                    "connection_type": info.get_connection_type(),
                    "firmware_version": info.get_firmware_version(),
                    "applied_sync_config": applied,
                }
            )
            pipelines.append((pipe, cfg))
            callbacks.append(make_cb(sample_bucket, serial))
            samples.append(sample_bucket)

        for i, (pipe, cfg) in enumerate(pipelines):
            pipe.start(cfg, callbacks[i])

        ctx.enable_multi_device_sync(60000)
        time.sleep(args.seconds)

    finally:
        for pipe, _cfg in pipelines:
            try:
                pipe.stop()
            except Exception:
                pass

    report = {
        "config_path": str(config_path),
        "duration_seconds": args.seconds,
        "devices": devices,
        "probes": {},
    }

    if len(samples) >= 2:
        for stream_name in ["COLOR_FRAME", "DEPTH_FRAME"]:
            a = samples[0].get(stream_name, [])
            b = samples[1].get(stream_name, [])
            if a and b:
                count = min(len(a), len(b))
                device_deltas = [
                    abs((b[idx]["timestamp_us"] or 0) - (a[idx]["timestamp_us"] or 0))
                    for idx in range(count)
                ]
                system_deltas = [
                    abs((b[idx]["system_timestamp_us"] or 0) - (a[idx]["system_timestamp_us"] or 0))
                    for idx in range(count)
                ]
                report["probes"][stream_name] = {
                    "paired_count": count,
                    "first_device_timestamp_delta_us": device_deltas[0],
                    "first_system_timestamp_delta_us": system_deltas[0],
                    "mean_device_timestamp_delta_us": round(sum(device_deltas) / count, 3),
                    "mean_system_timestamp_delta_us": round(sum(system_deltas) / count, 3),
                    "max_device_timestamp_delta_us": max(device_deltas),
                    "max_system_timestamp_delta_us": max(system_deltas),
                }

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = TMP_ROOT / "two_device_hardware_sync_probe.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved sync probe report: {out_path}")
    for stream_name, probe in report["probes"].items():
        print(
            f"[RESULT] {stream_name}: first_device_delta_us={probe['first_device_timestamp_delta_us']} "
            f"mean_device_delta_us={probe['mean_device_timestamp_delta_us']} "
            f"first_system_delta_us={probe['first_system_timestamp_delta_us']} "
            f"mean_system_delta_us={probe['mean_system_timestamp_delta_us']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
