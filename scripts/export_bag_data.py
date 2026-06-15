from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBPlaybackStatus,
    OBSensorType,
    Pipeline,
    PlaybackDevice,
)
from pyorbbecsdk.examples.utils import frame_to_bgr_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)()
    except Exception:
        return default


def process_depth(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(frame.get_height(), frame.get_width())
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    except Exception:
        return None


def process_ir(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    ir_data = np.asanyarray(frame.get_data())
    width = frame.get_width()
    height = frame.get_height()
    ir_format = frame.get_format()

    if ir_format == OBFormat.Y8:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    elif ir_format == OBFormat.MJPG:
        ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
        if ir_data is None:
            return None
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    else:
        ir_data = np.frombuffer(ir_data, dtype=np.uint16)
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint16
        image_dtype = cv2.CV_16UC1
        max_data = 255

    cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
    return cv2.cvtColor(ir_data.astype(data_type), cv2.COLOR_GRAY2BGR)


@dataclass
class StreamExport:
    name: str
    filename: str
    writer: cv2.VideoWriter | None = None
    frame_count: int = 0
    fps: float = 30.0
    width: int = 0
    height: int = 0

    def ensure_writer(self, output_dir: Path, frame: Any, image: np.ndarray) -> None:
        if self.writer is not None:
            return
        try:
            self.fps = float(frame.get_stream_profile().as_video_stream_profile().get_fps())
        except Exception:
            self.fps = 30.0
        self.height, self.width = image.shape[:2]
        output_path = output_dir / self.filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    def write(self, output_dir: Path, frame: Any, image: np.ndarray) -> None:
        self.ensure_writer(output_dir, frame, image)
        self.writer.write(image)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def default_output_dir(bag_path: Path) -> Path:
    if bag_path.parent.parent == RAW_ROOT:
        session_name = bag_path.parent.name
        return PROCESSED_ROOT / session_name / "bag_export"
    return bag_path.parent / "bag_export"


def playback_status_name(status: Any) -> str:
    try:
        return status.name
    except Exception:
        return str(status)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RGB/depth/IR/IMU data from an Orbbec .bag recording.")
    parser.add_argument("--bag", required=True, help="Path to a .bag file.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    bag_path = Path(args.bag).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(bag_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    playback = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback)
    config = Config()

    try:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION)
    except Exception:
        pass

    enabled_sensors: list[str] = []
    sensor_list = playback.get_sensor_list()
    for i in range(len(sensor_list)):
        sensor_type = sensor_list[i].get_type()
        try:
            config.enable_stream(sensor_type)
            enabled_sensors.append(getattr(sensor_type, "name", str(sensor_type)))
        except Exception:
            continue

    exports = {
        "color": StreamExport("color", "color_preview.mp4"),
        "depth": StreamExport("depth", "depth_preview.mp4"),
        "left_ir": StreamExport("left_ir", "left_ir_preview.mp4"),
        "right_ir": StreamExport("right_ir", "right_ir_preview.mp4"),
    }

    imu_csv_path = output_dir / "imu.csv"
    frame_csv_path = output_dir / "frame_index.csv"
    summary_path = output_dir / "export_summary.json"

    imu_row_count = 0
    frame_rows = 0
    final_status = playback.get_playback_status()

    print(f"[INFO] Bag file     : {bag_path}")
    print(f"[INFO] Output dir   : {output_dir}")
    print(f"[INFO] Duration(ms) : {playback.get_duration()}")

    with imu_csv_path.open("w", newline="", encoding="utf-8") as imu_f, frame_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as frame_f:
        imu_writer = csv.writer(imu_f)
        frame_writer = csv.writer(frame_f)
        imu_writer.writerow(
            ["stream", "timestamp_us", "system_timestamp_us", "x", "y", "z", "temperature"]
        )
        frame_writer.writerow(
            ["stream", "frame_index", "timestamp_us", "system_timestamp_us", "width", "height"]
        )

        pipeline.start(config)
        try:
            empty_loops = 0
            while True:
                frames = pipeline.wait_for_frames(100)
                status = playback.get_playback_status()
                final_status = status

                if frames is None:
                    empty_loops += 1
                    if status == OBPlaybackStatus.STOPPED and empty_loops > 10:
                        break
                    continue

                empty_loops = 0

                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    color_img = frame_to_bgr_image(color_frame)
                    if color_img is not None:
                        exports["color"].write(output_dir, color_frame, color_img)
                        frame_writer.writerow(
                            [
                                "COLOR",
                                exports["color"].frame_count,
                                safe_attr(color_frame, "get_timestamp_us"),
                                safe_attr(color_frame, "get_system_timestamp_us"),
                                safe_attr(color_frame, "get_width"),
                                safe_attr(color_frame, "get_height"),
                            ]
                        )
                        frame_rows += 1

                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_img = process_depth(depth_frame)
                    if depth_img is not None:
                        exports["depth"].write(output_dir, depth_frame, depth_img)
                        frame_writer.writerow(
                            [
                                "DEPTH",
                                exports["depth"].frame_count,
                                safe_attr(depth_frame, "get_timestamp_us"),
                                safe_attr(depth_frame, "get_system_timestamp_us"),
                                safe_attr(depth_frame, "get_width"),
                                safe_attr(depth_frame, "get_height"),
                            ]
                        )
                        frame_rows += 1

                left_ir_frame = frames.get_left_ir_frame()
                if left_ir_frame is not None:
                    left_ir_img = process_ir(left_ir_frame)
                    if left_ir_img is not None:
                        exports["left_ir"].write(output_dir, left_ir_frame, left_ir_img)
                        frame_writer.writerow(
                            [
                                "LEFT_IR",
                                exports["left_ir"].frame_count,
                                safe_attr(left_ir_frame, "get_timestamp_us"),
                                safe_attr(left_ir_frame, "get_system_timestamp_us"),
                                safe_attr(left_ir_frame, "get_width"),
                                safe_attr(left_ir_frame, "get_height"),
                            ]
                        )
                        frame_rows += 1

                right_ir_frame = frames.get_right_ir_frame()
                if right_ir_frame is not None:
                    right_ir_img = process_ir(right_ir_frame)
                    if right_ir_img is not None:
                        exports["right_ir"].write(output_dir, right_ir_frame, right_ir_img)
                        frame_writer.writerow(
                            [
                                "RIGHT_IR",
                                exports["right_ir"].frame_count,
                                safe_attr(right_ir_frame, "get_timestamp_us"),
                                safe_attr(right_ir_frame, "get_system_timestamp_us"),
                                safe_attr(right_ir_frame, "get_width"),
                                safe_attr(right_ir_frame, "get_height"),
                            ]
                        )
                        frame_rows += 1

                accel_frame = frames.get_accel_frame()
                if accel_frame is not None:
                    imu_writer.writerow(
                        [
                            "ACCEL",
                            safe_attr(accel_frame, "get_timestamp_us"),
                            safe_attr(accel_frame, "get_system_timestamp_us"),
                            safe_attr(accel_frame, "get_x"),
                            safe_attr(accel_frame, "get_y"),
                            safe_attr(accel_frame, "get_z"),
                            safe_attr(accel_frame, "get_temperature"),
                        ]
                    )
                    imu_row_count += 1

                gyro_frame = frames.get_gyro_frame()
                if gyro_frame is not None:
                    imu_writer.writerow(
                        [
                            "GYRO",
                            safe_attr(gyro_frame, "get_timestamp_us"),
                            safe_attr(gyro_frame, "get_system_timestamp_us"),
                            safe_attr(gyro_frame, "get_x"),
                            safe_attr(gyro_frame, "get_y"),
                            safe_attr(gyro_frame, "get_z"),
                            safe_attr(gyro_frame, "get_temperature"),
                        ]
                    )
                    imu_row_count += 1

        finally:
            pipeline.stop()
            for export in exports.values():
                export.close()

    summary = {
        "bag_file": str(bag_path),
        "output_dir": str(output_dir),
        "playback_duration_ms": playback.get_duration(),
        "final_playback_status": playback_status_name(final_status),
        "enabled_sensors": enabled_sensors,
        "video_exports": {
            name: {
                "filename": export.filename,
                "frames": export.frame_count,
                "fps": export.fps,
                "width": export.width,
                "height": export.height,
            }
            for name, export in exports.items()
        },
        "imu_rows": imu_row_count,
        "frame_index_rows": frame_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Export finished.")
    for name, export in exports.items():
        print(f"[INFO] {name:<8} -> {export.filename} ({export.frame_count} frames)")
    print(f"[INFO] imu.csv rows     : {imu_row_count}")
    print(f"[INFO] frame_index.csv : {frame_rows}")
    print(f"[INFO] summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
