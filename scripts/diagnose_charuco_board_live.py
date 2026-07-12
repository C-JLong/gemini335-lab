from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from tkinter import Tk

import cv2
from pyorbbecsdk import Config, Context, OBError, OBSensorType, Pipeline, VideoStreamProfile
from pyorbbecsdk.examples.utils import frame_to_bgr_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DICT_NAME_TO_ID = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


def timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_screen_size() -> tuple[int, int]:
    try:
        root = Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1366, 768


def fit_to_screen(image, screen_width: int, screen_height: int):
    max_width = int(screen_width * 0.96)
    max_height = int(screen_height * 0.86)
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


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


def create_detector(dict_name: str, detect_inverted: bool):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAME_TO_ID[dict_name])
    params = cv2.aruco.DetectorParameters()
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = detect_inverted
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_all_markers(image, detectors):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []
    for dict_name, detector in detectors:
        corners, ids, _rejected = detector.detectMarkers(gray)
        count = 0 if ids is None else len(ids)
        results.append((count, dict_name, corners, ids))
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def annotate(image, title: str, results, detect_inverted: bool):
    display = image.copy()
    best_count, best_dict, best_corners, best_ids = results[0]
    if best_count > 0:
        cv2.aruco.drawDetectedMarkers(display, best_corners, best_ids)
    id_preview = "-"
    if best_ids is not None:
        ids = sorted(int(item) for item in best_ids.flatten().tolist())
        id_preview = ",".join(str(item) for item in ids[:20])

    color = (0, 220, 0) if best_count > 0 else (0, 0, 255)
    cv2.putText(display, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        display,
        f"Best: {best_dict} markers={best_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    y = 100
    for count, dict_name, _corners, _ids in results:
        cv2.putText(
            display,
            f"{dict_name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    cv2.putText(
        display,
        f"IDs: {id_preview}",
        (20, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"inverted={detect_inverted}  S=save  Q=quit",
        (20, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def main() -> int:
    parser = argparse.ArgumentParser(description="Live diagnose whether a ChArUco board's ArUco markers are detectable.")
    parser.add_argument("--name", default="charuco_diagnose")
    parser.add_argument("--detect-inverted", action="store_true", help="Also allow inverted black/white markers.")
    parser.add_argument(
        "--layout",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Preview layout. vertical is better for a laptop screen.",
    )
    args = parser.parse_args()

    screen_width, screen_height = get_screen_size()
    output_dir = RAW_ROOT / "diagnostics" / f"{timestamp_name()}_{args.name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    detectors = [(name, create_detector(name, args.detect_inverted)) for name in DICT_NAME_TO_ID]

    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() < 1:
        raise RuntimeError("No Orbbec devices found.")

    pipelines = []
    serials = []
    try:
        for i in range(min(2, device_list.get_count())):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            pipeline, config, _profile = open_color_pipeline(device)
            pipelines.append((pipeline, config))
            serials.append(info.get_serial_number())

        for pipeline, config in pipelines:
            pipeline.start(config)

        print(f"[INFO] Output directory: {output_dir}")
        print("[INFO] Point the new board at the cameras.")
        print("[INFO] If all marker counts stay 0, the board is not being recognized as an OpenCV 5x5 ArUco board.")
        print("[INFO] Controls: S = save current diagnostic frames, Q = quit")

        save_index = 0
        while True:
            displays = []
            raw_images = []
            for i, (pipeline, _config) in enumerate(pipelines):
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    displays = []
                    break
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    displays = []
                    break
                image = frame_to_bgr_image(color_frame)
                if image is None:
                    displays = []
                    break
                raw_images.append((serials[i], image))
                results = detect_all_markers(image, detectors)
                displays.append(annotate(image, f"Camera {i} {serials[i]}", results, args.detect_inverted))

            if not displays:
                continue

            combined = cv2.vconcat(displays) if args.layout == "vertical" else cv2.hconcat(displays)
            combined = fit_to_screen(combined, screen_width, screen_height)
            cv2.namedWindow("ChArUco Board Diagnose", cv2.WINDOW_NORMAL)
            cv2.imshow("ChArUco Board Diagnose", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                save_index += 1
                cv2.imwrite(str(output_dir / f"diagnostic_overlay_{save_index:03d}.png"), combined)
                for serial, image in raw_images:
                    cv2.imwrite(str(output_dir / f"{serial}_raw_{save_index:03d}.png"), image)
                print(f"[INFO] Saved diagnostic frame #{save_index:03d}")

    finally:
        for pipeline, _config in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
