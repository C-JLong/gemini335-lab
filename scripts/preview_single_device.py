from __future__ import annotations

import argparse

import cv2
from pyorbbecsdk import Config, Context, OBError, OBSensorType, Pipeline, VideoStreamProfile
from pyorbbecsdk.examples.utils import frame_to_bgr_image


def choose_device(serial: str | None):
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count == 0:
        raise RuntimeError("No Orbbec device found.")

    if serial:
        for index in range(count):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            if info.get_serial_number() == serial:
                return device
        raise RuntimeError(f"Device with serial '{serial}' was not found.")

    if count > 1:
        print("[WARN] Multiple devices detected. Using device index 0.")

    return device_list.get_device_by_index(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview one connected Orbbec device.")
    parser.add_argument("--serial", default=None, help="Optional device serial number.")
    args = parser.parse_args()

    device = choose_device(args.serial)
    info = device.get_device_info()
    serial = info.get_serial_number()
    pipeline = Pipeline(device)
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except OBError as exc:
        raise RuntimeError(f"Failed to enable color stream: {exc}") from exc

    window_name = f"Orbbec Preview - {serial}"
    pipeline.start(config)
    print(f"[INFO] Previewing {info.get_name()} | serial={serial}")
    print("[INFO] Press Q in the preview window to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            image = frame_to_bgr_image(color_frame)
            if image is None:
                continue

            cv2.putText(
                image,
                f"{info.get_name()}  serial={serial}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Press Q to quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
