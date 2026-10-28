"""G1 is based on artifacts from real GPU and simulator jobs, never a manual checkbox."""

from __future__ import annotations

from pathlib import Path

from sgvla.io import digest, read_json, utc_now, write_json


def check_g1(runtime):
    expected = runtime.config_hash
    checks = {}
    try:
        backend = read_json(runtime.work / "docs/backend_check.json")
        checks["gpu_kernel"] = (
            backend["config_sha256"] == expected
            and backend[runtime.config["backend"]]["ok"] is True
        )
    except (OSError, KeyError, ValueError):
        checks["gpu_kernel"] = False
    for stage in ("dummy", "smoke"):
        try:
            pointer = read_json(runtime.work / "docs" / (stage + ".json"))
            path = Path(pointer["metrics_path"])
            result = read_json(path)
            valid = (
                digest(path) == pointer["metrics_sha256"]
                and result["status"] == "completed"
                and result["config_sha256"] == expected
            )
            if stage == "dummy":
                valid = valid and result["finite"] is True and result["shape"] == [50, 7]
                valid = valid and result["server_metadata"]["sgvla"]["config_sha256"] == expected
            else:
                valid = valid and result["steps"] > 0
                valid = valid and result["decoded_video_frames"] == result["steps"]
                for name in ("video", "wrist_video"):
                    valid = valid and digest(result[name]) == result[name + "_sha256"]
            checks[stage] = bool(valid)
        except (OSError, KeyError, ValueError):
            checks[stage] = False
    report = {
        "gate": "G1",
        "checked_at_utc": utc_now(),
        "checks": checks,
        "passed": all(checks.values()),
        "config_sha256": expected,
    }
    write_json(runtime.work / "docs/g1.json", report)
    return report
