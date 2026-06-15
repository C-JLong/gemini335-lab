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
    output_dir = RAW_ROOT / "calibration" / f"{timestamp_name()}_{name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_metadata(
    output_dir: Path,
    session_name: str,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    serial: str,
    color_profile: VideoStreamProfile,
) -> None:
    payload = {
        "session_name": session_name,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "pattern_type": "chessboard",
        "inner_corners": {
            "cols": pattern_cols,
            "rows": pattern_rows,
        },
        "square_size_mm": square_size_mm,
        "device_serial": serial,
        "color_profile": {
            "width": color_profile.get_width(),
            "height": color_profile.get_height(),
            "fps": color_profile.get_fps(),
        },
        "notes": [
            "This capture set is intended for single-camera intrinsic calibration.",
            "Pattern size uses inner corners, not square count.",
        ],
    }
    (output_dir / "capture_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture chessboard images for single-camera intrinsic calibration."
    )
    parser.add_argument("--name", required=True, help="Session name, for example chessboard_intrinsic_01")
    parser.add_argument(
        "--inner-cols",
        type=int,
        default=14,
        help="Number of inner corners horizontally. If the board has 15 squares, this is usually 14.",
    )
    parser.add_argument(
        "--inner-rows",
        type=int,
        default=14,
        help="Number of inner corners vertically. If the board has 15 squares, this is usually 14.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=50.0,
        help="Square edge length in millimeters.",
    )
    args = parser.parse_args()

    pattern_size = (args.inner_cols, args.inner_rows)
    session_dir = build_output_dir(args.name)
    print(f"[INFO] Output directory: {session_dir}")
    print(f"[INFO] Using chessboard inner corners: {args.inner_cols} x {args.inner_rows}")
    print(f"[INFO] Square size: {args.square_size_mm} mm")

    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        raise RuntimeError("No Orbbec device found.")
    if device_list.get_count() > 1:
        print("[WARN] Multiple devices detected. This script will use device index 0.")

    device = device_list.get_device_by_index(0)
    info = device.get_device_info()
    pipeline = Pipeline(device)
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except OBError as exc:
        raise RuntimeError(f"Failed to enable color stream: {exc}") from exc

    save_metadata(
        session_dir,
        session_dir.name,
        args.inner_cols,
        args.inner_rows,
        args.square_size_mm,
        info.get_serial_number(),
        color_profile,
    )

    pipeline.start(config)
    print("[INFO] Camera started.")
    print("[INFO] Controls:")
    print("       SPACE = save current frame only if chessboard is detected")
    print("       Q     = quit")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    capture_index = 0
    window_name = "Calibration Capture"

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

            display = image.copy()
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
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, pattern_size, refined_corners, found)
                status = "DETECTED"
                status_color = (0, 220, 0)
            else:
                status = "NOT DETECTED"
                status_color = (0, 0, 255)

            cv2.putText(
                display,
                f"Pattern {args.inner_cols}x{args.inner_rows} | {status}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Saved images: {capture_index}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "SPACE=save  Q=quit",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not found or refined_corners is None:
                    print("[WARN] Chessboard not detected. Frame not saved.")
                    continue

                capture_index += 1
                image_path = session_dir / f"color_{capture_index:03d}.png"
                overlay_path = session_dir / f"overlay_{capture_index:03d}.png"
                cv2.imwrite(str(image_path), image)
                cv2.imwrite(str(overlay_path), display)
                print(f"[INFO] Saved {image_path.name}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Capture finished. Total saved images: {capture_index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
