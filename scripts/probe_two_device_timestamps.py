from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pyorbbecsdk import Context, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp"


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
    ctx = Context()
    devs = ctx.query_devices()
    if devs.get_count() < 2:
        raise RuntimeError(f"Need 2 devices, found {devs.get_count()}.")

    devices = []
    pipelines = []
    configs = []
    samples = [defaultdict(list), defaultdict(list)]

    try:
        for i in range(2):
            dev = devs.get_device_by_index(i)
            info = dev.get_device_info()
            serial = info.get_serial_number()

            try:
                dev.timer_sync_with_host()
            except Exception:
                pass

            pipe = Pipeline(dev)
            cfg = pipe.get_config()
            sensor_list = dev.get_sensor_list()
            for j in range(len(sensor_list)):
                st = sensor_list[j].get_type()
                try:
                    cfg.enable_stream(st)
                except Exception:
                    pass

            def make_cb(device_index: int, serial_number: str):
                def _cb(frames):
                    for k in range(frames.get_count()):
                        frame = frames.get_frame_by_index(k)
                        name = frame_type_name(frame.get_type())
                        if name in {"COLOR_FRAME", "DEPTH_FRAME"} and len(samples[device_index][name]) < 80:
                            samples[device_index][name].append(
                                {
                                    "serial": serial_number,
                                    "timestamp_us": safe_call(frame.get_timestamp_us),
                                    "system_timestamp_us": safe_call(frame.get_system_timestamp_us),
                                }
                            )

                return _cb

            devices.append({"serial": serial})
            pipelines.append(pipe)
            configs.append((cfg, make_cb(i, serial)))

        for i in range(2):
            pipelines[i].start(configs[i][0], configs[i][1])

        start = time.time()
        while time.time() - start < 3.0:
            time.sleep(0.05)

    finally:
        for pipe in pipelines:
            try:
                pipe.stop()
            except Exception:
                pass

    report = {
        "devices": devices,
        "samples": samples,
        "probes": {},
    }

    for stream_name in ["COLOR_FRAME", "DEPTH_FRAME"]:
        a = samples[0].get(stream_name, [])
        b = samples[1].get(stream_name, [])
        if a and b:
            report["probes"][stream_name] = {
                "first_device_timestamp_delta_us": b[0]["timestamp_us"] - a[0]["timestamp_us"],
                "first_system_timestamp_delta_us": b[0]["system_timestamp_us"] - a[0]["system_timestamp_us"],
            }

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out = TMP_ROOT / "two_device_timestamp_probe.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved timestamp probe: {out}")
    for stream_name, probe in report["probes"].items():
        print(
            f"[RESULT] {stream_name}: device_ts_delta_us={probe['first_device_timestamp_delta_us']} "
            f"system_ts_delta_us={probe['first_system_timestamp_delta_us']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
