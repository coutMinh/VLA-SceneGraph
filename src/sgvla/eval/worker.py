"""Client-runtime entry point used by CLI and detached Molab jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgvla.eval.engine import completed, evaluation_root, run_batch
from sgvla.eval.manifest import build_manifest, validate_manifest
from sgvla.eval.reporting import review_first_batch, summarize
from sgvla.io import read_json, write_json
from sgvla.runtime import Runtime


def run(runtime, mode, manifest_path=None, limit=5):
    manifest_path = manifest_path or runtime.work / "configs/e0_manifest.json"
    if mode == "plan":
        from sgvla.eval.scenes import verify_scenes, write_report

        report = verify_scenes(runtime.upstream)
        report_path = runtime.work / "docs/scene_report.json"
        write_report(report, report_path)
        manifest = build_manifest(report, runtime.config)
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise RuntimeError("Existing manifest differs; use a new --manifest path")
        write_json(manifest_path, manifest)
        return {"manifest": str(manifest_path), "episodes": len(manifest["episodes"])}
    manifest = read_json(manifest_path)
    validate_manifest(manifest, runtime.config)
    if mode == "parity":
        from sgvla.eval.parity import run_parity

        return run_parity(runtime, manifest)
    if mode == "batch":
        result = run_batch(runtime, manifest, limit)
        if any(e["status"] == "infrastructure_error" for e in result["episodes"]):
            raise RuntimeError("Batch stopped on infrastructure error; inspect last_batch.json")
        return result
    if mode == "review":
        return review_first_batch(runtime, manifest)
    if mode == "all":
        while True:
            done = completed(runtime, manifest, deep=True)
            if all(done.values()):
                return summarize(runtime, manifest)
            first = [e["episode_id"] for e in manifest["episodes"][:5]]
            review_path = evaluation_root(runtime, manifest) / "first_batch_review.json"
            if all(done[key] for key in first) and not review_path.exists():
                review_first_batch(runtime, manifest)
            result = run_batch(runtime, manifest, limit)
            if any(e["status"] == "infrastructure_error" for e in result["episodes"]):
                raise RuntimeError("Pilot stopped on infrastructure error; resume after fixing it")
            if not result["episodes"]:
                raise RuntimeError(
                    "No evaluation progress; inspect manifest and completion records"
                )
    raise ValueError("Unknown evaluator mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("plan", "parity", "batch", "review", "all"), required=True
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    result = run(Runtime(args.work, args.config), args.mode, args.manifest, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
