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


def load_intrinsics(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    calib = payload["calibration"]
    k = np.array(calib["camera_matrix"], dtype=np.float64)
    d = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    image_size = (int(calib["image_width"]), int(calib["image_height"]))
    return k, d, image_size


def detect_charuco_points(image_path: Path, detector, board):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None, "failed_to_read"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 6:
        return image.shape[1::-1], None, None, "insufficient_charuco_corners"
    return image.shape[1::-1], charuco_corners, charuco_ids, None


def build_correspondence(corners_a, ids_a, corners_b, ids_b, board):
    ids_a = ids_a.flatten()
    ids_b = ids_b.flatten()
    common_ids = sorted(set(ids_a.tolist()) & set(ids_b.tolist()))
    if len(common_ids) < 6:
        return None, None, None

    obj_points = []
    img_a = []
    img_b = []
    for cid in common_ids:
        idx_a = int(np.where(ids_a == cid)[0][0])
        idx_b = int(np.where(ids_b == cid)[0][0])
        obj_points.append(board.getChessboardCorners()[cid])
        img_a.append(corners_a[idx_a][0])
        img_b.append(corners_b[idx_b][0])
    return (
        np.array(obj_points, dtype=np.float32),
        np.array(img_a, dtype=np.float32).reshape(-1, 1, 2),
        np.array(img_b, dtype=np.float32).reshape(-1, 1, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute stereo extrinsics from ChArUco image pairs.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--left-intrinsics", required=True)
    parser.add_argument("--right-intrinsics", required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--dictionary", required=True, choices=sorted(DICT_NAME_TO_ID))
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    left_dir = capture_dir / "cam0"
    right_dir = capture_dir / "cam1"
    if not left_dir.exists() or not right_dir.exists():
        raise RuntimeError("Missing cam0/cam1 directories.")

    board, detector = create_board(
        args.cols, args.rows, args.square_size_mm, args.marker_size_mm, args.dictionary
    )
    left_k, left_d, left_size = load_intrinsics(Path(args.left_intrinsics).resolve())
    right_k, right_d, right_size = load_intrinsics(Path(args.right_intrinsics).resolve())
    if left_size != right_size:
        raise RuntimeError(f"Image size mismatch: {left_size} vs {right_size}")

    left_by_name = {p.name: p for p in sorted(left_dir.glob("color_*.png"))}
    right_by_name = {p.name: p for p in sorted(right_dir.glob("color_*.png"))}
    shared_names = sorted(set(left_by_name) & set(right_by_name))
    if len(shared_names) < 8:
        raise RuntimeError(f"Only {len(shared_names)} shared image pairs.")

    object_points = []
    left_points = []
    right_points = []
    pair_usage = []

    for name in shared_names:
        size_a, corners_a, ids_a, err_a = detect_charuco_points(left_by_name[name], detector, board)
        size_b, corners_b, ids_b, err_b = detect_charuco_points(right_by_name[name], detector, board)
        used = False
        common_count = 0
        if corners_a is not None and corners_b is not None and tuple(size_a) == left_size and tuple(size_b) == right_size:
            objp, img_a, img_b = build_correspondence(corners_a, ids_a, corners_b, ids_b, board)
            if objp is not None and len(objp) >= 6:
                object_points.append(objp)
                left_points.append(img_a)
                right_points.append(img_b)
                used = True
                common_count = len(objp)
        pair_usage.append(
            {
                "file": name,
                "used": used,
                "common_charuco_corner_count": common_count,
                "left_error": err_a,
                "right_error": err_b,
            }
        )

    if len(object_points) < 8:
        raise RuntimeError(f"Only {len(object_points)} usable stereo pairs.")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, camera_matrix1, dist1, camera_matrix2, dist2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_k,
        left_d,
        right_k,
        right_d,
        left_size,
        criteria=criteria,
        flags=flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix1, dist1, camera_matrix2, dist2, left_size, R, T
    )

    baseline_mm = float(np.linalg.norm(T))
    output_dir = PROCESSED_ROOT / "stereo_calibration" / (capture_dir.name + "_charuco")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "left_intrinsics_path": str(Path(args.left_intrinsics).resolve()),
        "right_intrinsics_path": str(Path(args.right_intrinsics).resolve()),
        "used_pair_count": len(object_points),
        "total_pair_count": len(shared_names),
        "stereo_rms_reprojection_error": float(rms),
        "baseline_mm": baseline_mm,
        "rotation_matrix": R.tolist(),
        "translation_vector_mm": T.reshape(-1).tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "rectification": {
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi1": list(roi1),
            "roi2": list(roi2),
        },
        "pair_usage": pair_usage,
    }

    json_path = output_dir / "stereo_extrinsics_result.json"
    summary_path = output_dir / "summary.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                f"Capture dir: {capture_dir}",
                f"Board: {args.cols}x{args.rows}, square={args.square_size_mm}mm, marker={args.marker_size_mm}mm",
                f"Dictionary: {args.dictionary}",
                f"Used stereo pairs: {len(object_points)} / {len(shared_names)}",
                f"Stereo RMS reprojection error: {rms:.6f}",
                f"Baseline (mm): {baseline_mm:.3f}",
                "Factory settings modified: No",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Used stereo pairs: {len(object_points)} / {len(shared_names)}")
    print(f"[INFO] Stereo RMS reprojection error: {rms:.6f}")
    print(f"[INFO] Baseline (mm): {baseline_mm:.3f}")
    print(f"[INFO] Wrote {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
