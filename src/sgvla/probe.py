"""Observe the execution host without importing or changing its ML environment."""

from __future__ import annotations

import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

from sgvla.io import utc_now, write_json

GCS_METADATA = (
    "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?"
    "prefix=checkpoints%2Fpi0_libero%2F&maxResults=3&fields=items(name%2Csize)"
)


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def http_check(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {"ok": response.status == 200, "status": response.status}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def probe(runtime, *, network=True):
    runtime.work.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(runtime.work)
    report = {
        "observed_at_utc": utc_now(),
        "system": platform.system(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "render_libraries": {name: ctypes.util.find_library(name) for name in ("EGL", "OSMesa")},
        "disk": {
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "quota_verified": False,
            "note": "Filesystem capacity may be virtual; actual molab quota is unknown.",
        },
        "root": os.geteuid() == 0 if hasattr(os, "geteuid") else None,
        "tools": {name: shutil.which(name) for name in ("git", "uv", "apt-get", "docker")},
        "network": (
            {
                "gcs": http_check(GCS_METADATA),
                "huggingface": http_check("https://huggingface.co/api/models?limit=1"),
            }
            if network
            else {"status": "not_checked"}
        ),
        "gpu_kernels": "not_checked; run backend-check in the server environment",
        "headless_render": "not_checked; run render in the client environment",
    }
    write_json(runtime.work / "docs" / "molab_probe.json", report)
    return report
