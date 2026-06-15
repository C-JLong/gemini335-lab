from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from pyorbbecsdk import Config, Context, OBError, OBSensorType, Pipeline, VideoStreamProfile
from pyorbbecsdk.examples.utils import frame_to_bgr_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"


def timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_output_dir(name: str) -> Path:
    output_dir = RAW_ROOT / "stereo_calibration" / f"{timestamp_name()}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_metadata(
    output_dir: Path,
    session_name: str,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    devices: list[dict],
) -> None:
    payload = {
        "session_name": session_name,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "pattern_type": "chessboard",
        "inner_corners": {"cols": pattern_cols, "rows": pattern_rows},
        "square_size_mm": square_size_mm,
        "devices": devices,
        "notes": [
            "This capture set is intended for stereo extrinsic calibration.",
            "Each saved index should contain the same chessboard pose from both cameras.",
        ],
    }
    (output_dir / "capture_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def open_color_pipeline(device) -> tuple[Pipeline, Config, VideoStreamProfile]:
    pipeline = Pipeline(device)
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except OBError as exc:
        raise RuntimeError(f"Failed to enable color stream: {exc}") from exc
    return pipeline, config, color_profile


def annotate_display(image, pattern_size, found, refined_corners, capture_index, title):
    display = image.copy()
    if found and refined_corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, refined_corners, found)
        status = "DETECTED"
        status_color = (0, 220, 0)
    else:
        status = "NOT DETECTED"
        status_color = (0, 0, 255)

    cv2.putText(display, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        display,
        status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Saved pairs: {capture_index}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "SPACE=save pair  Q=quit",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def detect_pattern(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    refined_corners = None
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, refined_corners


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture synchronized chessboard image pairs for stereo calibration.")
    parser.add_argument("--name", required=True, help="Session name, for example stereo_board_01")
    parser.add_argument("--inner-cols", type=int, default=14)
    parser.add_argument("--inner-rows", type=int, default=14)
    parser.add_argument("--square-size-mm", type=float, default=50.0)
    args = parser.parse_args()

    pattern_size = (args.inner_cols, args.inner_rows)
    session_dir = build_output_dir(args.name)
    left_dir = session_dir / "cam0"
    right_dir = session_dir / "cam1"
    left_dir.mkdir(parents=True, exist_ok=False)
    right_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Output directory: {session_dir}")
    print(f"[INFO] Pattern: {args.inner_cols} x {args.inner_rows}")
    print(f"[INFO] Square size: {args.square_size_mm} mm")

    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() < 2:
        raise RuntimeError(f"Need 2 devices, found {device_list.get_count()}.")

    devices_meta = []
    pipelines = []

    try:
        for i in range(2):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            pipeline, config, color_profile = open_color_pipeline(device)
            pipelines.append((pipeline, config))
            devices_meta.append(
                {
                    "index": i,
                    "serial": info.get_serial_number(),
                    "name": info.get_name(),
                    "connection_type": info.get_connection_type(),
                    "firmware_version": info.get_firmware_version(),
                    "color_profile": {
                        "width": color_profile.get_width(),
                        "height": color_profile.get_height(),
                        "fps": color_profile.get_fps(),
                    },
                }
            )

        save_metadata(
            session_dir,
            session_dir.name,
            args.inner_cols,
            args.inner_rows,
            args.square_size_mm,
            devices_meta,
        )

        for pipeline, config in pipelines:
            pipeline.start(config)

        print("[INFO] Both cameras started.")
        print("[INFO] Controls:")
        print("       SPACE = save current pair only if chessboard is detected in both views")
        print("       Q     = quit")

        capture_index = 0
        while True:
            images = []
            detections = []
            for pipeline, _config in pipelines:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    images = []
                    break
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    images = []
                    break
                image = frame_to_bgr_image(color_frame)
                if image is None:
                    images = []
                    break
                found, refined_corners = detect_pattern(image, pattern_size)
                images.append(image)
                detections.append((found, refined_corners))

            if len(images) != 2:
                continue

            display0 = annotate_display(
                images[0], pattern_size, detections[0][0], detections[0][1], capture_index, "Camera 0"
            )
            display1 = annotate_display(
                images[1], pattern_size, detections[1][0], detections[1][1], capture_index, "Camera 1"
            )

            combined = cv2.hconcat([display0, display1])
            cv2.imshow("Stereo Calibration Capture", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not all(found for found, _corners in detections):
                    print("[WARN] Chessboard must be detected in both cameras. Pair not saved.")
                    continue

                capture_index += 1
                left_path = left_dir / f"color_{capture_index:03d}.png"
                right_path = right_dir / f"color_{capture_index:03d}.png"
                left_overlay = left_dir / f"overlay_{capture_index:03d}.png"
                right_overlay = right_dir / f"overlay_{capture_index:03d}.png"
                cv2.imwrite(str(left_path), images[0])
                cv2.imwrite(str(right_path), images[1])
                cv2.imwrite(str(left_overlay), display0)
                cv2.imwrite(str(right_overlay), display1)
                print(f"[INFO] Saved pair #{capture_index:03d}")

    finally:
        for pipeline, _config in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print(f"[INFO] Capture finished. Total saved pairs: {locals().get('capture_index', 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
