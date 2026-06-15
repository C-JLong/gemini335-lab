from __future__ import annotations

import time
from collections import defaultdict

from pyorbbecsdk import Config, Context, OBError, OBSensorType, Pipeline


def main() -> int:
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    print(f"[INFO] Detected devices: {count}")
    if count < 2:
        print("[ERROR] Need at least 2 devices for this probe.")
        return 1

    pipelines: list[Pipeline] = []
    frame_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"color": 0, "depth": 0})

    try:
        for i in range(2):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            serial = info.get_serial_number()
            print(f"[INFO] Preparing device {i}: {serial} ({info.get_connection_type()})")

            pipeline = Pipeline(device)
            config = Config()

            try:
                color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = color_profiles.get_default_video_stream_profile()
                config.enable_stream(color_profile)
            except OBError as exc:
                print(f"[WARN] Failed to enable color on {serial}: {exc}")

            try:
                depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                config.enable_stream(depth_profile)
            except OBError as exc:
                print(f"[WARN] Failed to enable depth on {serial}: {exc}")

            def make_callback(device_serial: str):
                def _callback(frames):
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if color_frame is not None:
                        frame_counts[device_serial]["color"] += 1
                    if depth_frame is not None:
                        frame_counts[device_serial]["depth"] += 1

                return _callback

            pipeline.start(config, make_callback(serial))
            pipelines.append(pipeline)

        print("[INFO] Both pipelines started. Probing for 5 seconds...")
        start = time.time()
        while time.time() - start < 5.0:
            time.sleep(0.25)

        elapsed = time.time() - start
        print(f"[INFO] Probe duration: {elapsed:.2f} s")
        for serial, counts in frame_counts.items():
            color_fps = counts["color"] / elapsed
            depth_fps = counts["depth"] / elapsed
            print(
                f"[RESULT] {serial} | color_frames={counts['color']} ({color_fps:.2f} fps) | "
                f"depth_frames={counts['depth']} ({depth_fps:.2f} fps)"
            )

        if len(frame_counts) < 2:
            print("[WARN] Less than 2 devices produced frames.")
            return 2

        return 0

    finally:
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        print("[INFO] Pipelines stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
