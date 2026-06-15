# Gemini 335 Lab Project

Python-based workspace for Orbbec Gemini 335 single-camera and dual-camera experiments.

This repository is designed for a practical lab workflow:

- verify devices with Orbbec Viewer
- record `.bag` files from one or two cameras
- export recorded data into preview videos and CSV indexes
- calibrate single-camera intrinsics
- calibrate dual-camera extrinsics
- keep raw data, processed data, scripts, and configuration separated

## Current Scope

This project currently supports:

- single-camera recording
- dual-camera synchronized recording
- ChArUco intrinsic calibration
- ChArUco stereo extrinsic calibration
- `.bag` export into RGB / depth / IR preview videos and IMU tables

The current ChArUco board parameters used in this project are:

- board squares: `15 x 15`
- `squareLength = 50 mm`
- `markerLength = 37 mm`
- dictionary: `DICT_5X5_250`

## Repository Layout

- `scripts/`
  Main acquisition, calibration, export, and validation scripts.
- `config/`
  Configuration files, especially dual-camera sync configuration.
- `docs/`
  Small supporting documentation.
- `data/`
  Project data root.
- `third_party/`
  Vendor software placeholders.
- `logs/`
  Runtime logs.
- `tmp/`
  Temporary local outputs.

Inside `data/`:

- `data/inbox/`
  External files copied into the project.
- `data/raw/`
  Raw recordings and calibration captures.
- `data/processed/`
  Calibration results and export results.
- `data/exports/`
  Final deliverables you may want to share.

Raw and processed experiment data are intentionally ignored by Git.

## Environment

Recommended environment:

- Windows 64-bit
- Python `3.13 x64`
- `pyorbbecsdk2==2.1.1`

Install Python dependency:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements-python.txt
```

## Typical Workflow

### 1. Viewer Health Check

Open Orbbec Viewer from its own extracted directory. Do not copy `OrbbecViewer.exe` out by itself.

Check:

- device is detected
- color stream works
- depth stream works
- left / right IR work
- IMU updates

### 2. Single-Camera Recording

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\record_single_device.py --name subject1_sit_test --duration 10
```

### 3. Dual-Camera Synchronized Recording

Before recording, update:

- `config/multi_device_sync_config.json`

so the serial numbers match the actual two cameras.

Then record:

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\record_two_devices_sync.py --name 001test --duration 10
```

### 4. Export a `.bag`

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\export_bag_data.py --bag .\data\raw\<session>\recording.bag
```

### 5. Single-Camera ChArUco Intrinsics

Capture images:

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\capture_calibration_images.py --name charuco_intrinsic_camA
```

Compute intrinsics:

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\calibrate_intrinsics_charuco.py --capture-dir .\data\raw\calibration\<capture_dir> --cols 15 --rows 15 --square-size-mm 50 --marker-size-mm 37 --dictionary DICT_5X5_250
```

### 6. Dual-Camera ChArUco Extrinsics

Capture paired stereo images:

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\capture_stereo_calibration_images.py --name stereo_board_01
```

Compute stereo extrinsics:

```powershell
Set-Location <project-root>
.\.venv\Scripts\python.exe .\scripts\calibrate_stereo_extrinsics_charuco.py --capture-dir .\data\raw\stereo_calibration\<capture_dir> --left-intrinsics .\data\processed\calibration\<left_intrinsics_dir>\intrinsics_result.json --right-intrinsics .\data\processed\calibration\<right_intrinsics_dir>\intrinsics_result.json --cols 15 --rows 15 --square-size-mm 50 --marker-size-mm 37 --dictionary DICT_5X5_250
```

## Important Notes

- Single-camera intrinsics do not need to be recomputed every day, but recomputing them at the formal experiment site is recommended.
- Dual-camera extrinsics must be recomputed whenever the relative camera placement changes.
- Vendor SDK archives, extracted Viewer files, local migration bundles, and experimental data are not intended to live in GitHub history.

## Upload Policy For This Repo

This repository is intended to contain:

- scripts
- configs
- lightweight docs
- dependency manifest
- empty data directory placeholders

This repository should not contain:

- large `.bag` files
- exported videos
- local calibration image archives
- extracted vendor binaries
- temporary handoff zip packages
