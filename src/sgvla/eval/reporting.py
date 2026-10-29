"""Reduced-pilot rates, uncertainty and a recorded first-batch storage inspection."""

from __future__ import annotations

import math
import shutil

from sgvla.eval.engine import completed, context, evaluation_root
from sgvla.eval.manifest import fingerprint
from sgvla.io import digest, read_json, utc_now, write_json


def wilson(successes, total):
    if not total:
        return None
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(runtime, manifest):
    done = completed(runtime, manifest)
    rows = []
    for task_id in sorted({e["task_id"] for e in manifest["episodes"]}):
        entries = [e for e in manifest["episodes"] if e["task_id"] == task_id]
        metrics = [done[e["episode_id"]][1] for e in entries if done[e["episode_id"]]]
        count, successes = len(metrics), sum(m["success"] for m in metrics)
        rows.append(
            {
                "task_id": task_id,
                "completed": count,
                "planned": len(entries),
                "successes": successes,
                "success_rate": successes / count if count else None,
                "wilson_95": wilson(successes, count),
            }
        )
    n, wins = sum(row["completed"] for row in rows), sum(row["successes"] for row in rows)
    return {
        "context": context(runtime, manifest),
        "output_dir": str(evaluation_root(runtime, manifest)),
        "scope": "reduced Spatial pilot; not an official full-suite benchmark",
        "completed": n,
        "planned": len(manifest["episodes"]),
        "pending_or_invalid": len(manifest["episodes"]) - n,
        "successes": wins,
        "success_rate": wins / n if n else None,
        "wilson_95": wilson(wins, n),
        "per_task": rows,
        "uncertainty_note": (
            "Episode-level Wilson interval; not adjusted for within-task dependence"
        ),
    }


def review_first_batch(runtime, manifest):
    done = completed(runtime, manifest, deep=True)
    first = [e["episode_id"] for e in manifest["episodes"][:5]]
    if len(first) != 5 or not all(done[key] for key in first):
        raise RuntimeError("Complete five validated episodes before first-batch inspection")
    root = evaluation_root(runtime, manifest)
    selected = [done[key] for key in first]
    used = sum(sum(p.stat().st_size for p in item[2].iterdir() if p.is_file()) for item in selected)
    remaining = sum(item is None for item in done.values())
    estimate = math.ceil(used / 5 * remaining * 1.5)
    free = shutil.disk_usage(runtime.work).free
    latencies = sorted(v for _, m, _ in selected for v in m["policy_call_seconds"])
    report = {
        "context": context(runtime, manifest),
        "completion_sha256": {
            key: digest(root / "episodes" / key / "complete.json") for key in first
        },
        "reviewed_at_utc": utc_now(),
        "validation": "All arrays checked and both videos decoded for every first-batch episode",
        "failures": [key for key in first if not done[key][1]["success"]],
        "elapsed_seconds": sum(item[1]["wall_clock_seconds"] for item in selected),
        "artifact_bytes": used,
        "free_bytes": free,
        "remaining_storage_estimate_bytes_with_50pct_margin": estimate,
        "policy_call_seconds_median": latencies[len(latencies) // 2],
        "policy_call_seconds_max": max(latencies),
        "passed": free > estimate,
        "manual_trajectory_review": "pending; use saved videos for Part 3 failure taxonomy",
    }
    write_json(root / "first_batch_review.json", report)
    if not report["passed"]:
        raise RuntimeError("Insufficient estimated disk space to finish the pilot")
    return report


def require_review(runtime, manifest):
    root = evaluation_root(runtime, manifest)
    path = root / "first_batch_review.json"
    if not path.exists():
        raise RuntimeError("Run eval-review to inspect the first batch before continuing")
    review = read_json(path)
    if review.get("context") != context(runtime, manifest) or review.get("passed") is not True:
        raise RuntimeError("First-batch review is incompatible or failed")
    expected = {
        e["episode_id"]: digest(root / "episodes" / e["episode_id"] / "complete.json")
        for e in manifest["episodes"][:5]
    }
    if fingerprint(expected) != fingerprint(review["completion_sha256"]):
        raise RuntimeError("First-batch artifacts changed; rerun eval-review")
    remaining = sum(v is None for v in completed(runtime, manifest).values())
    required = math.ceil(review["artifact_bytes"] / 5 * remaining * 1.5)
    if shutil.disk_usage(runtime.work).free <= required:
        raise RuntimeError("Free disk space fell below the remaining pilot estimate")
