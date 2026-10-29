"""Atomic per-episode completion, validation/resume and infrastructure retry records."""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from sgvla.eval.manifest import fingerprint, validate_manifest
from sgvla.io import digest, file_lock, read_json, utc_now, write_json
from sgvla.registry import register
from sgvla.runtime import REPO, git_commit

ARTIFACTS = (
    "metrics.json",
    "trajectory.npz",
    "agentview.mp4",
    "wrist.mp4",
    "manifest.json",
    "config.json",
)


def implementation_hash():
    return fingerprint(
        {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / "src").rglob("*.py"))}
    )


def evaluation_root(runtime, manifest):
    return runtime.results / "part2/e0" / fingerprint(manifest)


def context(runtime, manifest):
    return {
        "manifest_sha256": validate_manifest(manifest, runtime.config),
        "policy_sha256": runtime.policy_hash,
        "implementation_sha256": implementation_hash(),
    }


def ensure_sources(runtime, manifest):
    if git_commit(runtime.upstream) != runtime.config["openpi_commit"]:
        raise ValueError("Runtime openpi commit differs")
    if git_commit(runtime.upstream / "third_party/libero") != runtime.config["libero_commit"]:
        raise ValueError("Runtime LIBERO commit differs")
    if digest(runtime.upstream / "examples/libero/main.py") != manifest["official_client_sha256"]:
        raise ValueError("Official client source changed")
    for repo, paths in (
        (runtime.upstream, ["src/openpi", "packages/openpi-client", "examples/libero"]),
        (runtime.upstream / "third_party/libero", ["libero"]),
    ):
        result = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *paths])
        if result.returncode:
            raise ValueError("Tracked upstream sources have local modifications: " + str(repo))


def artifact_hashes(folder):
    return {name: digest(folder / name) for name in ARTIFACTS}


def validate_completion(folder, entry, expected, *, deep=True):
    """Missing, interrupted, stale, changed or corrupt output is never skipped."""
    try:
        record = read_json(folder / "complete.json")
        if record["context"] != expected or record["entry_sha256"] != fingerprint(entry):
            return None
        attempt = record["attempt"]
        if not isinstance(attempt, str) or not attempt.startswith("attempt-"):
            return None
        if Path(attempt).name != attempt:
            return None
        output = folder / attempt
        if record["artifacts"] != artifact_hashes(output):
            return None
        metrics = read_json(output / "metrics.json")
        if (
            metrics["status"] != "completed"
            or type(metrics["success"]) is not bool
            or type(metrics["steps"]) is not int
            or not 1 <= metrics["steps"] <= 220
            or metrics["initial_state_sha256"] != entry["initial_state_sha256"]
            or metrics["policy_seed"] != entry["policy_seed"]
            or metrics["target_object"] != entry["target_object"]
        ):
            return None
        provenance = read_json(output / "manifest.json")
        if provenance["entry"] != entry or provenance["context"] != expected:
            return None
        if deep:
            from sgvla.eval.trace import validate_trace

            validate_trace(output, metrics)
        return record, metrics, output
    except (OSError, ValueError, KeyError, TypeError, EOFError, RuntimeError, IndexError):
        return None


def completed(runtime, manifest, *, deep=False):
    expected, root = context(runtime, manifest), evaluation_root(runtime, manifest)
    return {
        entry["episode_id"]: validate_completion(
            root / "episodes" / entry["episode_id"], entry, expected, deep=deep
        )
        for entry in manifest["episodes"]
    }


def reconcile_registry(runtime, item):
    record, metrics, output = item
    from sgvla.runtime import Runtime

    snapshot = Runtime(runtime.work, output / "config.json")
    register(
        snapshot,
        record["run_id"],
        "E0",
        output / "metrics.json",
        metrics["wall_clock_seconds"],
        manifest_path=output / "manifest.json",
    )


def recover_interrupted(runtime, episode, entry, expected, current_attempt):
    """Index abandoned attempts on resume; a killed process cannot write its final row."""
    from sgvla.runtime import Runtime

    for output in sorted(episode.glob("attempt-*")):
        if output.name == current_attempt or not (output / "manifest.json").exists():
            continue
        try:
            provenance = read_json(output / "manifest.json")
        except (OSError, ValueError):
            continue
        if not isinstance(provenance, dict) or not (output / "config.json").exists():
            continue
        if provenance.get("entry") != entry or provenance.get("context") != expected:
            continue
        # Completed/error attempts already have a measured row or reconcile separately.
        if (output / "metrics.json").exists() or (output / "error.json").exists():
            continue
        recovery = {
            "status": "interrupted",
            "episode_id": entry["episode_id"],
            "wall_clock_seconds": None,
            "recovered_at_utc": utc_now(),
            "note": "No final metrics; elapsed time is unknown and excluded from rates",
        }
        if not (output / "interrupted.json").exists():
            write_json(output / "interrupted.json", recovery)
        register(
            Runtime(runtime.work, output / "config.json"),
            entry["episode_id"] + "-" + output.name,
            "E0_interrupted",
            output / "interrupted.json",
            None,
            manifest_path=output / "manifest.json",
        )


def execute_one(runtime, manifest, entry, execute_episode):
    expected = context(runtime, manifest)
    episode = evaluation_root(runtime, manifest) / "episodes" / entry["episode_id"]
    with file_lock(episode / ".lock"):
        existing = validate_completion(episode, entry, expected)
        recover_interrupted(
            runtime, episode, entry, expected, existing[0]["attempt"] if existing else None
        )
        if existing:
            reconcile_registry(runtime, existing)
            return {"episode_id": entry["episode_id"], "status": "skipped"}
        attempt = "attempt-" + uuid.uuid4().hex[:12]
        output = episode / attempt
        output.mkdir(parents=True, exist_ok=False)
        run_id = entry["episode_id"] + "-" + attempt
        started = time.monotonic()
        write_json(output / "config.json", runtime.config)
        write_json(
            output / "manifest.json",
            {
                "entry": entry,
                "context": expected,
                "evaluation_manifest": manifest,
                "git_commit": git_commit(REPO),
                "created_at_utc": utc_now(),
            },
        )
        try:
            metrics = execute_episode(runtime, entry, output)
            if (
                metrics.get("status") != "completed"
                or type(metrics.get("success")) is not bool
                or type(metrics.get("steps")) is not int
                or not 1 <= metrics["steps"] <= runtime.config["max_steps"]
                or metrics.get("initial_state_sha256") != entry["initial_state_sha256"]
                or metrics.get("policy_seed") != entry["policy_seed"]
                or metrics.get("target_object") != entry["target_object"]
            ):
                raise ValueError("Episode returned incompatible metrics")
            metrics["wall_clock_seconds"] = time.monotonic() - started
            write_json(output / "metrics.json", metrics)
            from sgvla.eval.trace import validate_trace

            validate_trace(output, metrics)
            record = {
                "context": expected,
                "entry_sha256": fingerprint(entry),
                "attempt": attempt,
                "run_id": run_id,
                "artifacts": artifact_hashes(output),
                "completed_at_utc": utc_now(),
            }
            # The pointer is published only after all artifacts have been decoded/validated.
            write_json(episode / "complete.json", record)
            reconcile_registry(runtime, (record, metrics, output))
            return {"episode_id": entry["episode_id"], "status": "completed", **metrics}
        except Exception as exc:
            metrics = {
                "status": "infrastructure_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_clock_seconds": time.monotonic() - started,
                "episode_id": entry["episode_id"],
            }
            write_json(output / "error.json", metrics)
            # Error records have their own registry rows, never completed task-failure labels.
            register(
                runtime,
                run_id,
                "E0_infrastructure_error",
                output / "error.json",
                metrics["wall_clock_seconds"],
                manifest_path=output / "manifest.json",
            )
            return metrics


def run_batch(runtime, manifest, limit=5):
    from sgvla.eval.parity import require_parity
    from sgvla.eval.reporting import require_review, summarize
    from sgvla.eval.run_libero import run_episode

    if type(limit) is not int or not 5 <= limit <= 10:
        raise ValueError("Batch size must be 5-10 episodes (last batch may be shorter)")
    ensure_sources(runtime, manifest)
    require_parity(runtime, manifest)
    root = evaluation_root(runtime, manifest)
    with file_lock(root / ".batch.lock"):
        done = completed(runtime, manifest, deep=True)
        for item in done.values():
            if item:
                reconcile_registry(runtime, item)
        first_ids = [e["episode_id"] for e in manifest["episodes"][:5]]
        first_complete = all(done[key] for key in first_ids)
        if first_complete:
            require_review(runtime, manifest)
        pending = [e for e in manifest["episodes"] if not done[e["episode_id"]]]
        # First five must be finished and reviewed before crossing the inspection boundary.
        if not first_complete:
            pending = [e for e in pending if e["episode_id"] in first_ids]
        results = []
        for entry in pending[:limit]:
            item = execute_one(runtime, manifest, entry, run_episode)
            results.append(item)
            print(f"{entry['episode_id']}: {item['status']}", flush=True)
            if item["status"] == "infrastructure_error":
                break  # stop on infrastructure errors; resume retries the same episode.
        report = summarize(runtime, manifest)
        write_json(root / "summary.json", report)
        write_json(root / "last_batch.json", {"episodes": results, "summary": report})
        return {"episodes": results, "summary": report}
