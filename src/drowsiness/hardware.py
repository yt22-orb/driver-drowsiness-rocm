"""Verify that PyTorch can access the expected AMD ROCm device."""

from __future__ import annotations

import argparse
import json
import platform

import torch


def hardware_report() -> dict[str, object]:
    available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "accelerator_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if available
        else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-rocm", action="store_true")
    parser.add_argument("--expected-gpu", default="7800 XT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = hardware_report()
    print(json.dumps(report, indent=2))
    if args.require_rocm:
        if not report["torch_hip"]:
            raise SystemExit("FAIL: this is not a ROCm/HIP PyTorch build")
        if not report["accelerator_available"]:
            raise SystemExit("FAIL: ROCm build is installed but no accelerator is available")
        if not any(args.expected_gpu.lower() in name.lower() for name in report["devices"]):
            message = f"FAIL: expected GPU containing {args.expected_gpu!r} was not detected"
            raise SystemExit(message)
        print("PASS: expected ROCm GPU is available")


if __name__ == "__main__":
    main()
