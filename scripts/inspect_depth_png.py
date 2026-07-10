from __future__ import annotations

"""
Inspect raw Gemini335 depth PNG values.

depth_raw PNG files often look almost black in normal image viewers. That is
normal because they are uint16 single-channel true depth images, not 8-bit
pseudo-color images made for human viewing. Judge whether a depth_raw image is
healthy by checking dtype, shape, nonzero pixel ratio, and valid depth
statistics, not by whether it looks bright in a normal viewer.
"""

import argparse
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np


LOW_NONZERO_RATIO_WARNING = 0.05


def format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def load_depth_png(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Depth PNG not found: {path}")
    if path.suffix.lower() != ".png":
        print(f"[WARNING] Input is not a PNG file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cv2 failed to read image: {path}")
    return image


def image_stats(image: np.ndarray) -> dict[str, Any]:
    flat = image.reshape(-1)
    nonzero = flat[flat > 0]
    nonzero_count = int(nonzero.size)
    nonzero_ratio = nonzero_count / int(flat.size) if flat.size else 0.0

    stats: dict[str, Any] = {
        "dtype": str(image.dtype),
        "shape": tuple(int(x) for x in image.shape),
        "min": int(np.min(flat)) if flat.size else None,
        "max": int(np.max(flat)) if flat.size else None,
        "mean": float(np.mean(flat)) if flat.size else None,
        "median": float(np.median(flat)) if flat.size else None,
        "nonzero_count": nonzero_count,
        "nonzero_ratio": nonzero_ratio,
        "valid_min": int(np.min(nonzero)) if nonzero.size else None,
        "valid_max": int(np.max(nonzero)) if nonzero.size else None,
        "valid_mean": float(np.mean(nonzero)) if nonzero.size else None,
        "valid_median": float(np.median(nonzero)) if nonzero.size else None,
    }
    return stats


def print_stats(path: Path, stats: dict[str, Any], image: np.ndarray) -> None:
    print("=" * 80)
    print(f"file: {path}")
    print(f"dtype: {stats['dtype']}")
    print(f"shape: {stats['shape']}")
    print(f"min: {format_number(stats['min'])}")
    print(f"max: {format_number(stats['max'])}")
    print(f"mean: {format_number(stats['mean'])}")
    print(f"median: {format_number(stats['median'])}")
    print(f"nonzero_count: {stats['nonzero_count']}")
    print(f"nonzero_ratio: {format_ratio(stats['nonzero_ratio'])}")
    print(f"valid_min: {format_number(stats['valid_min'])}")
    print(f"valid_max: {format_number(stats['valid_max'])}")
    print(f"valid_mean: {format_number(stats['valid_mean'])}")
    print(f"valid_median: {format_number(stats['valid_median'])}")

    if image.dtype != np.uint16:
        print(f"[WARNING] Expected uint16 depth PNG, got {image.dtype}.")
    if image.ndim != 2:
        print(f"[WARNING] Expected single-channel 2D image, got shape {image.shape}.")
    if stats["nonzero_ratio"] < LOW_NONZERO_RATIO_WARNING:
        print(
            "[WARNING] nonzero_ratio is very low "
            f"({format_ratio(stats['nonzero_ratio'])}); depth may be mostly invalid."
        )


def inspect_pixel(image: np.ndarray, u: int | None, v: int | None) -> None:
    if u is None and v is None:
        return
    if u is None or v is None:
        raise ValueError("Both --u and --v must be provided to inspect a pixel.")
    if image.ndim < 2:
        raise ValueError(f"Cannot inspect pixel for image with shape {image.shape}")

    height, width = image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise IndexError(f"Pixel out of bounds: u={u}, v={v}, image width={width}, height={height}")

    value = image[v, u]
    print(f"depth[v, u] at v={v}, u={u}: {value}")
    print("[INFO] Pixel indexing reminder: depth[v, u], where v is row and u is column.")


def inspect_window(image: np.ndarray, u: int | None, v: int | None, window: int | None) -> None:
    if window is None:
        return
    if window <= 0:
        raise ValueError("--window must be a positive integer.")
    if u is None or v is None:
        raise ValueError("--window requires both --u and --v.")
    if image.ndim < 2:
        raise ValueError(f"Cannot inspect window for image with shape {image.shape}")

    height, width = image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise IndexError(f"Window center out of bounds: u={u}, v={v}, image width={width}, height={height}")

    radius = window // 2
    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    patch = image[y0:y1, x0:x1]
    valid = patch[patch > 0]

    print(f"window: requested={window}x{window}, actual_rows={y0}:{y1}, actual_cols={x0}:{x1}")
    print(f"window_valid_count: {int(valid.size)}")
    if valid.size:
        print(f"window_valid_min: {int(np.min(valid))}")
        print(f"window_valid_max: {int(np.max(valid))}")
        print(f"window_valid_mean: {float(np.mean(valid)):.3f}")
        print(f"window_valid_median: {float(np.median(valid)):.3f}")
    else:
        print("window_valid_min: N/A")
        print("window_valid_max: N/A")
        print("window_valid_mean: N/A")
        print("window_valid_median: N/A")
        print("[WARNING] No valid depth value > 0 in this window.")


def save_visualization(image: np.ndarray, path: Path, min_depth: int, max_depth: int) -> Path:
    if max_depth <= min_depth:
        raise ValueError("--vis-max-depth must be greater than --vis-min-depth.")

    if image.ndim != 2:
        raise ValueError(f"Visualization expects a single-channel depth image, got shape {image.shape}")

    clipped = np.clip(image, min_depth, max_depth)
    normalized = ((clipped - min_depth) * 255.0 / (max_depth - min_depth)).astype(np.uint8)
    normalized[image == 0] = 0
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[image == 0] = (0, 0, 0)

    vis_path = path.with_name(f"{path.stem}_vis.png")
    if not cv2.imwrite(str(vis_path), color):
        raise RuntimeError(f"Failed to save visualization: {vis_path}")
    print(f"visualization: {vis_path}")
    print("[INFO] Visualization is 8-bit pseudo-color for viewing only, not for real depth calculation.")
    return vis_path


def inspect_one(
    path: Path,
    u: int | None,
    v: int | None,
    window: int | None,
    save_vis: bool,
    vis_min_depth: int,
    vis_max_depth: int,
) -> dict[str, Any]:
    image = load_depth_png(path)
    stats = image_stats(image)
    print_stats(path, stats, image)
    inspect_pixel(image, u, v)
    inspect_window(image, u, v, window)
    if save_vis:
        save_visualization(image, path, vis_min_depth, vis_max_depth)
    return stats


def collect_depth_files(depth_dir: Path, sample_count: int) -> list[Path]:
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not depth_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {depth_dir}")

    files = sorted(depth_dir.glob("depth_*.png"))
    if sample_count > 0:
        files = files[:sample_count]
    if not files:
        raise FileNotFoundError(f"No depth_*.png files found in: {depth_dir}")
    return files


def print_summary(all_stats: list[dict[str, Any]]) -> None:
    valid_values: list[float] = []
    nonzero_ratios: list[float] = []
    num_uint16 = 0
    num_single_channel = 0

    for stats in all_stats:
        if stats["dtype"] == "uint16":
            num_uint16 += 1
        if len(stats["shape"]) == 2:
            num_single_channel += 1
        nonzero_ratios.append(float(stats["nonzero_ratio"]))
        if stats["valid_min"] is not None:
            valid_values.append(float(stats["valid_min"]))
        if stats["valid_max"] is not None:
            valid_values.append(float(stats["valid_max"]))
        if stats["valid_median"] is not None:
            valid_values.append(float(stats["valid_median"]))

    total_checked = len(all_stats)
    print("=" * 80)
    print("SUMMARY")
    print(f"total_checked: {total_checked}")
    print(f"num_uint16: {num_uint16}")
    print(f"num_single_channel: {num_single_channel}")
    print(f"average_nonzero_ratio: {format_ratio(float(np.mean(nonzero_ratios)) if nonzero_ratios else None)}")
    print(f"global_valid_min: {format_number(min(valid_values) if valid_values else None)}")
    print(f"global_valid_max: {format_number(max(valid_values) if valid_values else None)}")
    print(f"sampled_valid_median: {format_number(median(valid_values) if valid_values else None)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect true uint16 single-channel depth_raw PNG values. A dark-looking "
            "depth PNG can be normal; use numeric stats instead of a normal image viewer."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--depth", help="Path to one depth PNG.")
    input_group.add_argument("--depth-dir", help="Directory containing depth_*.png files.")
    parser.add_argument("--sample-count", type=int, default=10, help="Directory mode: inspect first N files. <=0 means all.")
    parser.add_argument("--u", type=int, default=None, help="Pixel column coordinate for depth[v, u].")
    parser.add_argument("--v", type=int, default=None, help="Pixel row coordinate for depth[v, u].")
    parser.add_argument("--window", type=int, default=None, help="Optional local window size around --u/--v, e.g. 7.")
    parser.add_argument("--save-vis", action="store_true", help="Save *_vis.png pseudo-color preview for viewing only.")
    parser.add_argument("--vis-min-depth", type=int, default=300, help="Visualization clip minimum depth. Default: 300.")
    parser.add_argument("--vis-max-depth", type=int, default=4000, help="Visualization clip maximum depth. Default: 4000.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.depth:
        inspect_one(
            Path(args.depth).resolve(),
            args.u,
            args.v,
            args.window,
            args.save_vis,
            args.vis_min_depth,
            args.vis_max_depth,
        )
        return 0

    depth_dir = Path(args.depth_dir).resolve()
    files = collect_depth_files(depth_dir, args.sample_count)
    print(f"[INFO] Inspecting {len(files)} depth PNG file(s) from: {depth_dir}")
    all_stats = [
        inspect_one(path, args.u, args.v, args.window, args.save_vis, args.vis_min_depth, args.vis_max_depth)
        for path in files
    ]
    print_summary(all_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
