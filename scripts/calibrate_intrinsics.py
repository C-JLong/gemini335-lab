from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def build_object_points(inner_cols: int, inner_rows: int, square_size_mm: float) -> np.ndarray:
    objp = np.zeros((inner_rows * inner_cols, 3), np.float32)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_mm
    return objp


def build_output_dir(session_name: str) -> Path:
    output_dir = PROCESSED_ROOT / "calibration" / session_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_capture_info(capture_dir: Path) -> dict:
    info_path = capture_dir / "capture_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing capture metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def calibrate_from_images(
    capture_dir: Path,
    inner_cols: int,
    inner_rows: int,
    square_size_mm: float,
) -> tuple[dict, list[dict], tuple[int, int]]:
    image_paths = sorted(capture_dir.glob("color_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No color_*.png images found in {capture_dir}")

    pattern_size = (inner_cols, inner_rows)
    objp = build_object_points(inner_cols, inner_rows, square_size_mm)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    per_image_results: list[dict] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            per_image_results.append(
                {
                    "file": image_path.name,
                    "used": False,
                    "reason": "failed_to_read",
                }
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

        if not found or corners is None:
            per_image_results.append(
                {
                    "file": image_path.name,
                    "used": False,
                    "reason": "pattern_not_detected",
                }
            )
            continue

        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp.copy())
        image_points.append(refined_corners)
        per_image_results.append(
            {
                "file": image_path.name,
                "used": True,
                "corner_count": int(len(refined_corners)),
            }
        )

    if image_size is None:
        raise RuntimeError("Failed to infer image size from calibration images.")

    if len(object_points) < 8:
        raise RuntimeError(
            f"Only {len(object_points)} valid images were detected. At least 8 are recommended."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_view_errors = []
    total_error = 0.0
    total_points = 0
    for idx, obj_points in enumerate(object_points):
        projected, _ = cv2.projectPoints(
            obj_points,
            rvecs[idx],
            tvecs[idx],
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(image_points[idx], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))
        total_error += float(error) * len(projected)
        total_points += len(projected)

    mean_error = total_error / total_points if total_points else None

    calibration = {
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error_pixels": float(mean_error) if mean_error is not None else None,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "used_image_count": len(object_points),
        "total_image_count": len(image_paths),
        "per_view_reprojection_error_pixels": per_view_errors,
    }
    return calibration, per_image_results, image_size


def write_opencv_yaml(output_path: Path, calibration: dict) -> None:
    fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", calibration["image_width"])
    fs.write("image_height", calibration["image_height"])
    fs.write("camera_matrix", np.array(calibration["camera_matrix"], dtype=np.float64))
    fs.write(
        "distortion_coefficients",
        np.array(calibration["distortion_coefficients"], dtype=np.float64),
    )
    fs.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute single-camera intrinsic calibration from saved chessboard images."
    )
    parser.add_argument(
        "--capture-dir",
        required=True,
        help="Calibration image directory under data/raw/calibration.",
    )
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.exists():
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    capture_info = load_capture_info(capture_dir)
    inner_cols = int(capture_info["inner_corners"]["cols"])
    inner_rows = int(capture_info["inner_corners"]["rows"])
    square_size_mm = float(capture_info["square_size_mm"])

    output_dir = build_output_dir(capture_dir.name)
    print(f"[INFO] Capture directory: {capture_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Pattern: {inner_cols} x {inner_rows} inner corners")
    print(f"[INFO] Square size: {square_size_mm} mm")
    print("[INFO] This is an offline calculation only. It does not modify the camera.")

    calibration, per_image_results, _ = calibrate_from_images(
        capture_dir,
        inner_cols,
        inner_rows,
        square_size_mm,
    )

    payload = {
        "source_capture_dir": str(capture_dir),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "device_serial": capture_info.get("device_serial"),
        "pattern_type": capture_info.get("pattern_type"),
        "inner_corners": capture_info.get("inner_corners"),
        "square_size_mm": square_size_mm,
        "camera_factory_settings_modified": False,
        "notes": [
            "This file contains user-computed intrinsic calibration results.",
            "It does not overwrite or change the camera factory parameters.",
        ],
        "calibration": calibration,
        "image_usage": per_image_results,
    }

    json_path = output_dir / "intrinsics_result.json"
    yaml_path = output_dir / "intrinsics_opencv.yaml"
    payload_path = output_dir / "summary.txt"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_opencv_yaml(yaml_path, calibration)
    payload_path.write_text(
        "\n".join(
            [
                f"Capture dir: {capture_dir}",
                f"Used images: {calibration['used_image_count']} / {calibration['total_image_count']}",
                f"RMS reprojection error: {calibration['rms_reprojection_error']:.6f}",
                "Factory settings modified: No",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "[INFO] Calibration complete. "
        f"Used {calibration['used_image_count']} / {calibration['total_image_count']} images."
    )
    print(f"[INFO] RMS reprojection error: {calibration['rms_reprojection_error']:.6f}")
    print(f"[INFO] Wrote {json_path.name} and {yaml_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
