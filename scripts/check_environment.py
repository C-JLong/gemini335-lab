from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "multi_device_sync_config.json"
VIEWER_EXE = (
    PROJECT_ROOT
    / "third_party"
    / "orbbec"
    / "viewer"
    / "OrbbecViewer_v2.6.3_win_x64"
    / "OrbbecViewer.exe"
)


def print_check(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "WARN"
    print(f"[{status}] {name}: {detail}")


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    git_path = shutil.which("git")
    print_check("Git on PATH", git_path is not None, git_path or "not found in PATH")

    try:
        import cv2

        print_check("OpenCV", True, cv2.__version__)
    except Exception as exc:
        print_check("OpenCV", False, str(exc))

    try:
        import numpy as np

        print_check("NumPy", True, np.__version__)
    except Exception as exc:
        print_check("NumPy", False, str(exc))

    device_serials: list[str] = []
    try:
        import pyorbbecsdk
        from pyorbbecsdk import Context

        print_check("pyorbbecsdk", True, str(Path(pyorbbecsdk.__file__).parent))
        ctx = Context()
        devices = ctx.query_devices()
        count = devices.get_count()
        print_check("Orbbec devices", count > 0, f"{count} detected")
        for index in range(count):
            device = devices.get_device_by_index(index)
            info = device.get_device_info()
            serial = info.get_serial_number()
            device_serials.append(serial)
            print(
                "  "
                f"{index}: {info.get_name()} | serial={serial} | "
                f"connection={info.get_connection_type()} | fw={info.get_firmware_version()}"
            )
    except Exception as exc:
        print_check("pyorbbecsdk / devices", False, str(exc))

    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        configured = [item["serial_number"] for item in payload["devices"]]
        print_check("Sync config", len(configured) == 2, ", ".join(configured))
        if device_serials:
            missing = sorted(set(device_serials) - set(configured))
            extra = sorted(set(configured) - set(device_serials))
            print_check(
                "Config matches connected devices",
                not missing and not extra,
                f"connected_not_configured={missing}; configured_not_connected={extra}",
            )
    except Exception as exc:
        print_check("Sync config", False, str(exc))

    print_check("Orbbec Viewer", VIEWER_EXE.exists(), str(VIEWER_EXE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
