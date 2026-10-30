"""Part 3 CLI; heavy simulation runs in the existing private client environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sgvla.io import read_json
from sgvla.runtime import Runtime

TASKS = (
    "verify",
    "verify-temporal",
    "verify-landmarks",
    "audit-e0",
    "plan",
    "batch",
    "all",
    "review",
    "status",
    "report",
    "episode",
)


def manifest_path(runtime, condition):
    if condition not in ("S1", "T1", "T1-ctrl", "S2", "S2-ctrl"):
        raise ValueError("Specify a supported condition")
    return runtime.work / "configs/part3" / (condition + ".json")


def dispatch(runtime, action, condition=None, limit=5, episode_id=None):
    relations = runtime.results / "part3/verification-v2/verification.json"
    temporal = runtime.results / "part3/temporal_verification/temporal_verification.json"
    landmarks = runtime.results / "part3/landmark_verification/landmark_verification.json"
    if action == "verify":
        from sgvla.analysis.verify import verify

        return verify(runtime, relations.parent)
    if action == "verify-temporal":
        from sgvla.analysis.temporal_verify import verify_temporal

        return verify_temporal(runtime, relations, temporal.parent)
    if action == "audit-e0":
        from sgvla.analysis.outcomes import audit_e0

        return audit_e0(runtime, runtime.results / "part3/e0_review")
    if action == "verify-landmarks":
        from sgvla.analysis.landmarks import verify_landmarks

        return verify_landmarks(runtime, relations, landmarks.parent)
    if action == "plan":
        from sgvla.analysis.manifest import build, freeze

        manifest = build(
            runtime,
            condition,
            relations,
            temporal if condition.startswith("T1") else None,
            landmarks if condition.startswith("S2") else None,
        )
        path = manifest_path(runtime, condition)
        freeze(path, manifest)
        return {
            "manifest": str(path),
            "condition": condition,
            "episodes": len(manifest["episodes"]),
            "excluded": manifest["excluded"],
        }
    if action == "status":
        from sgvla.analysis.engine import summarize

        return {
            c: summarize(runtime, read_json(manifest_path(runtime, c)))
            for c in ("S1", "T1", "T1-ctrl", "S2", "S2-ctrl")
            if manifest_path(runtime, c).exists()
        }
    if action == "report":
        from sgvla.analysis.reporting import aggregate

        return aggregate(runtime)
    manifest = read_json(manifest_path(runtime, condition))
    if action == "episode":
        from sgvla.analysis.engine import completed

        item = completed(runtime, manifest).get(episode_id)
        if not item:
            raise ValueError("Episode is incomplete or incompatible")
        return {
            "episode_id": episode_id,
            "metrics": item[1],
            "agentview": str(item[2] / "agentview.mp4"),
            "wrist": str(item[2] / "wrist.mp4"),
        }
    if action == "review":
        from sgvla.analysis.engine import review

        return review(runtime, manifest)
    if action in ("batch", "all"):
        from sgvla.analysis.engine import completed, root, run_batch, summarize

        while True:
            done = completed(runtime, manifest)
            if all(done.values()):
                return {"status": "completed", "summary": summarize(runtime, manifest)}
            first = [e["episode_id"] for e in manifest["episodes"][:5]]
            if (
                all(done[eid] for eid in first)
                and not (root(runtime, manifest) / "review.json").exists()
            ):
                return {
                    "status": "awaiting_first_batch_review",
                    "summary": summarize(runtime, manifest),
                }
            result = run_batch(runtime, manifest, limit)
            if any(e["status"] == "infrastructure_error" for e in result["episodes"]):
                raise RuntimeError("Infrastructure error; inspect last_batch.json before resume")
            if action == "batch":
                return result
            if not result["episodes"]:
                raise RuntimeError("No progress; inspect condition artifacts")
    raise ValueError("Unknown action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", required=True, choices=(*TASKS, "start", "jobs", "stop"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--condition", choices=("S1", "T1", "T1-ctrl", "S2", "S2-ctrl"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--episode-id")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    runtime = Runtime(args.work.resolve(), args.config.resolve())
    os.environ.update(runtime.env())
    if args.action == "start":
        from sgvla.analysis.jobs import start

        result = start(runtime, args.task, args.condition, args.limit)
    elif args.action == "jobs":
        from sgvla.analysis.jobs import list_jobs

        result = list_jobs(runtime)
    elif args.action == "stop":
        from sgvla.analysis.jobs import stop

        result = stop(runtime, args.job_id)
    else:
        result = dispatch(runtime, args.action, args.condition, args.limit, args.episode_id)
    if result is not None:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
