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


def detect_corners(image_path: Path, pattern_size: tuple[int, int]):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, "failed_to_read"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found or corners is None:
        return image.shape[1::-1], None, "pattern_not_detected"
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return image.shape[1::-1], refined, None


def load_intrinsics(intrinsics_path: Path):
    payload = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    calib = payload["calibration"]
    k = np.array(calib["camera_matrix"], dtype=np.float64)
    d = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    image_size = (int(calib["image_width"]), int(calib["image_height"]))
    return k, d, image_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute stereo extrinsics from paired chessboard images.")
    parser.add_argument("--capture-dir", required=True, help="Stereo calibration capture directory.")
    parser.add_argument("--left-intrinsics", required=True, help="Left camera intrinsics_result.json")
    parser.add_argument("--right-intrinsics", required=True, help="Right camera intrinsics_result.json")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.exists():
        raise FileNotFoundError(capture_dir)

    capture_info_path = capture_dir / "capture_info.json"
    if not capture_info_path.exists():
        raise FileNotFoundError(capture_info_path)
    capture_info = json.loads(capture_info_path.read_text(encoding="utf-8"))

    left_dir = capture_dir / "cam0"
    right_dir = capture_dir / "cam1"
    left_images = sorted(left_dir.glob("color_*.png"))
    right_images = sorted(right_dir.glob("color_*.png"))
    if not left_images or not right_images:
        raise RuntimeError("No stereo image pairs found.")

    left_by_name = {p.name: p for p in left_images}
    right_by_name = {p.name: p for p in right_images}
    shared_names = sorted(set(left_by_name) & set(right_by_name))
    if len(shared_names) < 8:
        raise RuntimeError(f"Only {len(shared_names)} shared pairs found. At least 8 are recommended.")

    inner_cols = int(capture_info["inner_corners"]["cols"])
    inner_rows = int(capture_info["inner_corners"]["rows"])
    square_size_mm = float(capture_info["square_size_mm"])
    pattern_size = (inner_cols, inner_rows)
    objp = build_object_points(inner_cols, inner_rows, square_size_mm)

    left_k, left_d, left_size = load_intrinsics(Path(args.left_intrinsics).resolve())
    right_k, right_d, right_size = load_intrinsics(Path(args.right_intrinsics).resolve())
    if left_size != right_size:
        raise RuntimeError(f"Image size mismatch: left={left_size}, right={right_size}")

    object_points = []
    left_points = []
    right_points = []
    pair_report = []

    for name in shared_names:
        left_size_found, left_corners, left_err = detect_corners(left_by_name[name], pattern_size)
        right_size_found, right_corners, right_err = detect_corners(right_by_name[name], pattern_size)

        used = left_corners is not None and right_corners is not None
        pair_report.append(
            {
                "file": name,
                "used": used,
                "left_error": left_err,
                "right_error": right_err,
            }
        )
        if not used:
            continue
        if tuple(left_size_found) != left_size or tuple(right_size_found) != right_size:
            continue

        object_points.append(objp.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)

    if len(object_points) < 8:
        raise RuntimeError(f"Only {len(object_points)} usable stereo pairs found. At least 8 are recommended.")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )

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

    output_dir = PROCESSED_ROOT / "stereo_calibration" / capture_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_capture_dir": str(capture_dir),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "camera_factory_settings_modified": False,
        "left_intrinsics_path": str(Path(args.left_intrinsics).resolve()),
        "right_intrinsics_path": str(Path(args.right_intrinsics).resolve()),
        "inner_corners": capture_info["inner_corners"],
        "square_size_mm": square_size_mm,
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
        "pair_usage": pair_report,
    }

    json_path = output_dir / "stereo_extrinsics_result.json"
    summary_path = output_dir / "summary.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                f"Capture dir: {capture_dir}",
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
