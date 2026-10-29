"""Append-only experiment index; metrics stay in per-run JSON artifacts."""

from __future__ import annotations

import csv
import platform

from sgvla.io import digest, file_lock, read_json, utc_now
from sgvla.runtime import REPO, git_commit

FIELDS = [
    "run_id",
    "git_commit",
    "config_path",
    "config_sha256",
    "data_manifest_hash",
    "seed",
    "hardware",
    "wall_clock_seconds",
    "metrics_json_path",
    "run_type",
    "recorded_at_utc",
]


def register(runtime, run_id, run_type, metrics_path, elapsed, *, manifest_path=None):
    probe_path = runtime.work / "docs/molab_probe.json"
    hardware = (
        read_json(probe_path).get("nvidia_smi", {})
        if probe_path.exists()
        else {"system": platform.system(), "gpu": "unobserved"}
    )
    config = runtime.config
    data_manifest = manifest_path or runtime.results / "runs" / run_id / "manifest.json"
    row = dict(
        zip(
            FIELDS,
            [
                run_id,
                git_commit(REPO),
                str(runtime.config_path),
                runtime.config_hash,
                digest(data_manifest),
                config["seed"],
                str(hardware),
                round(elapsed, 6) if elapsed is not None else "",
                str(metrics_path),
                run_type,
                utc_now(),
            ],
        )
    )
    registry = runtime.results / "registry.csv"
    with file_lock(runtime.results / ".registry.lock"):
        if registry.exists():
            with registry.open(newline="", encoding="utf-8") as stream:
                if any(existing["run_id"] == run_id for existing in csv.DictReader(stream)):
                    return
        fresh = not registry.exists()
        with registry.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if fresh:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
