from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


DICT_NAME_TO_ID = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
}


def create_board(cols: int, rows: int, square_mm: float, marker_mm: float, dict_name: str):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAME_TO_ID[dict_name])
    board = cv2.aruco.CharucoBoard((cols, rows), square_mm, marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def build_output_dir(session_name: str) -> Path:
    output_dir = PROCESSED_ROOT / "calibration" / session_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_charuco(image_path: Path, detector):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None, "failed_to_read"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 6:
        return image.shape[1::-1], None, None, "insufficient_charuco_corners"
    return image.shape[1::-1], charuco_corners, charuco_ids, None


def build_charuco_object_points(board, charuco_ids):
    all_points = board.getChessboardCorners()
    ids = charuco_ids.flatten().astype(int)
    return np.array([all_points[idx] for idx in ids], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute single-camera intrinsics from ChArUco images.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--cols", type=int, required=True, help="Board square count horizontally")
    parser.add_argument("--rows", type=int, required=True, help="Board square count vertically")
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--dictionary", required=True, choices=sorted(DICT_NAME_TO_ID))
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.exists():
        raise FileNotFoundError(capture_dir)

    board, detector = create_board(
        args.cols,
        args.rows,
        args.square_size_mm,
        args.marker_size_mm,
        args.dictionary,
    )

    image_paths = sorted(capture_dir.glob("color_*.png"))
    if not image_paths:
        raise RuntimeError(f"No color_*.png in {capture_dir}")

    all_image_points = []
    all_object_points = []
    per_image_results = []
    image_size = None

    for image_path in image_paths:
        detected_size, corners, ids, error = detect_charuco(image_path, detector)
        if image_size is None and detected_size is not None:
            image_size = tuple(detected_size)
        used = corners is not None and ids is not None
        per_image_results.append(
            {
                "file": image_path.name,
                "used": used,
                "charuco_corner_count": 0 if ids is None else int(len(ids)),
                "reason": error,
            }
        )
        if used:
            all_image_points.append(corners.astype(np.float32))
            all_object_points.append(build_charuco_object_points(board, ids))

    if image_size is None:
        raise RuntimeError("Failed to infer image size.")
    if len(all_image_points) < 8:
        raise RuntimeError(f"Only {len(all_image_points)} usable images found.")

    camera_matrix_init = np.array(
        [[1000.0, 0.0, image_size[0] / 2.0], [0.0, 1000.0, image_size[1] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_init = np.zeros((5, 1), dtype=np.float64)
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_object_points,
        imagePoints=all_image_points,
        imageSize=image_size,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_init,
        flags=flags,
        criteria=criteria,
    )

    calibration = {
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "used_image_count": len(all_image_points),
        "total_image_count": len(image_paths),
    }

    output_dir = build_output_dir(capture_dir.name + "_charuco")
    payload = {
        "source_capture_dir": str(capture_dir),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "pattern_type": "charuco",
        "board": {
            "cols": args.cols,
            "rows": args.rows,
            "square_size_mm": args.square_size_mm,
            "marker_size_mm": args.marker_size_mm,
            "dictionary": args.dictionary,
        },
        "camera_factory_settings_modified": False,
        "calibration": calibration,
        "image_usage": per_image_results,
    }

    json_path = output_dir / "intrinsics_result.json"
    summary_path = output_dir / "summary.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                f"Capture dir: {capture_dir}",
                f"Board: {args.cols}x{args.rows}, square={args.square_size_mm}mm, marker={args.marker_size_mm}mm",
                f"Dictionary: {args.dictionary}",
                f"Used images: {len(all_image_points)} / {len(image_paths)}",
                f"RMS reprojection error: {rms:.6f}",
                "Factory settings modified: No",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Used images: {len(all_image_points)} / {len(image_paths)}")
    print(f"[INFO] RMS reprojection error: {rms:.6f}")
    print(f"[INFO] Wrote {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
