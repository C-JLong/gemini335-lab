# Single-Camera Recording Workflow

## Why `.bag` instead of `.mp4`

For your experiment, `.bag` is the better raw recording format because it preserves:

- multiple streams together
- timestamps
- color and depth synchronization context
- IMU and other sensor data when available

This is better than recording only a visual video file, because later you can still export frames if needed.

## Environment

This project uses a dedicated virtual environment:

- Python executable: `F:\GitHub\gemini335\.venv\Scripts\python.exe`

Dependencies are listed in:

- `requirements-python.txt`

## Record One Device

Example:

```powershell
.\.venv\Scripts\python.exe .\scripts\record_single_device.py --name sit_test_01 --duration 10
```

What it does:

1. creates a new folder under `data/raw/`
2. records one device into `recording.bag`
3. saves `metadata.json` next to the bag file
4. stops automatically after the requested number of seconds

## Output Layout

Example output:

```text
data/raw/20260612_101500_sit_test_01/
  recording.bag
  metadata.json
```

## Notes

- If multiple devices are connected later, pass `--serial`.
- This script is for stable raw acquisition first.
- Point cloud is not recorded as the primary raw format; it can be regenerated later from depth plus calibration.
