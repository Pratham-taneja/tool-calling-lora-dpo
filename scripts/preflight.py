from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run local environment checks before model or dataset work."""

    report = {
        "project_root": str(PROJECT_ROOT),
        "python": python_report(),
        "platform": platform_report(),
        "nvidia_smi": nvidia_smi_report(),
        "torch": torch_report(),
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["torch"]["cuda_available"]:
        print("\nFAIL: PyTorch cannot see CUDA. Stop before training stack work.")
        return 1

    print("\nPASS: CUDA preflight is ready for model smoke tests.")
    return 0


def python_report() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "project_local_venv": is_project_local_venv(Path(sys.executable)),
    }


def platform_report() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "error": "nvidia-smi was not found on PATH",
        }

    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            name, memory_total_mb, driver_version = parts
            gpus.append(
                {
                    "name": name,
                    "memory_total_mb": int(memory_total_mb),
                    "driver_version": driver_version,
                }
            )

    return {
        "available": True,
        "gpus": gpus,
    }


def torch_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {
            "installed": False,
            "cuda_available": False,
            "error": repr(error),
        }

    cuda_available = bool(torch.cuda.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def is_project_local_venv(executable: Path) -> bool:
    try:
        executable.relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
