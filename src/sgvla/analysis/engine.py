"""Part 3 atomic episodes, resume, explicit batch review and condition summaries."""

from __future__ import annotations

import math
import shutil
import time
import uuid

from sgvla.analysis.manifest import validate
from sgvla.eval.engine import implementation_hash
from sgvla.eval.manifest import fingerprint
from sgvla.eval.reporting import wilson
from sgvla.eval.scenes import BOWL1, BOWL2
from sgvla.io import digest, file_lock, read_json, utc_now, write_json
from sgvla.registry import register
from sgvla.runtime import REPO, Runtime, git_commit

ARTIFACTS = (
    "metrics.json",
    "trajectory.npz",
    "agentview.mp4",
    "wrist.mp4",
    "manifest.json",
    "config.json",
    "intervention.json",
)


def context(runtime, manifest):
    return {
        "manifest_sha256": validate(manifest, runtime.config),
        "policy_sha256": runtime.policy_hash,
        "implementation_sha256": implementation_hash(),
    }


def root(runtime, manifest):
    return runtime.results / "part3" / manifest["condition"] / fingerprint(manifest)


def validate_trace(folder, metrics):
    import imageio.v2 as imageio
    import numpy as np

    prefix = (
        metrics["settling_steps"]
        + metrics["history_steps"]
        + metrics["post_history_steps"]
        + metrics.get("landmark_settling_steps", 0)
    )
    steps, n = metrics["steps"], metrics["frames"]
    condition = metrics.get("condition", "S1")
    expected_history = 40 if condition.startswith("T1") else 0
    expected_post = 10 if condition.startswith("T1") else 0
    expected_landmark = 20 if condition.startswith("S2") else 0
    if (
        metrics["settling_steps"] != 10
        or metrics["history_steps"] != expected_history
        or metrics["post_history_steps"] != expected_post
        or metrics.get("landmark_settling_steps", 0) != expected_landmark
    ):
        raise ValueError("Prelude does not match condition protocol")
    if n != prefix + steps + 1:
        raise ValueError("Misaligned Part 3 trace length")
    with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
        names = trace["object_names"].tolist()
        if names != metrics["object_names"] or not {BOWL1, BOWL2} <= set(names):
            raise ValueError("Missing identity trace")
        for key, shape in {
            "state": (n, 8),
            "object_poses": (n, len(names), 7),
            "actions": (n - 1, 7),
            "bowl_on_plate": (n, 2),
        }.items():
            if trace[key].shape != shape or not np.isfinite(trace[key]).all():
                raise ValueError("Invalid trace " + key)
        if trace["sim_state"].shape[0] != n or not np.isfinite(trace["sim_state"]).all():
            raise ValueError("Missing complete simulator state")
        phase = (
            [0] * metrics["settling_steps"]
            + [3] * metrics["history_steps"]
            + [4] * metrics["post_history_steps"]
            + [5] * metrics.get("landmark_settling_steps", 0)
            + [1] * steps
            + [2]
        )
        if trace["phase"].tolist() != phase:
            raise ValueError("Invalid prelude/policy phase alignment")
        calls = list(range(prefix, prefix + steps, metrics["replan_steps"]))
        if (
            trace["policy_call_frames"].tolist() != calls
            or trace["policy_chunks"].shape != (len(calls), 50, 7)
            or len(metrics["policy_observation_hashes"]) != len(calls)
            or trace["policy_call_seconds"].shape != (len(calls),)
            or not np.isfinite(trace["policy_chunks"]).all()
            or not np.isfinite(trace["policy_call_seconds"]).all()
            or (trace["policy_call_seconds"] < 0).any()
            or not np.array_equal(trace["policy_call_seconds"], metrics["policy_call_seconds"])
        ):
            raise ValueError("Invalid policy-call evidence")
        actions = np.concatenate([c[: metrics["replan_steps"]] for c in trace["policy_chunks"]])[
            :steps
        ]
        if not np.array_equal(actions, trace["actions"][prefix:]):
            raise ValueError("Executed actions differ from returned chunks")
        selected = 0 if metrics["target_object"] == BOWL1 else 1
        if bool(trace["bowl_on_plate"][-1, selected]) != metrics["success"]:
            raise ValueError("Success differs from selected target predicate")
    for camera in ("agentview", "wrist"):
        count = 0
        with imageio.get_reader(str(folder / (camera + ".mp4"))) as reader:
            for frame in reader:
                if frame.shape != (224, 224, 3):
                    raise ValueError("Wrong camera dimensions")
                count += 1
        if count != n:
            raise ValueError("Truncated camera video")


def hashes(folder):
    return {name: digest(folder / name) for name in ARTIFACTS}


def completion(folder, entry, expected, deep=False):
    try:
        pointer = read_json(folder / "complete.json")
        if pointer["context"] != expected or pointer["entry_sha256"] != fingerprint(entry):
            return None
        attempt = pointer["attempt"]
        if (
            not isinstance(attempt, str)
            or not attempt.startswith("attempt-")
            or "/" in attempt
            or "\\" in attempt
        ):
            return None
        output = folder / attempt
        if pointer["artifacts"] != hashes(output):
            return None
        provenance = read_json(output / "manifest.json")
        if provenance["entry"] != entry or provenance["context"] != expected:
            return None
        metrics = read_json(output / "metrics.json")
        if (
            metrics["status"] != "completed"
            or metrics.get("condition") != entry["condition"]
            or type(metrics["success"]) is not bool
            or type(metrics["steps"]) is not int
            or not 1 <= metrics["steps"] <= 220
            or metrics["target_object"] != entry["target_object"]
            or metrics["policy_seed"] != entry["policy_seed"]
            or metrics["initial_state_sha256"] != entry["initial_state_sha256"]
            or metrics["prompt"] != entry["prompt"]
        ):
            return None
        if deep:
            validate_trace(output, metrics)
        return pointer, metrics, output
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, IndexError):
        return None


def completed(runtime, manifest, deep=False):
    expected, output = context(runtime, manifest), root(runtime, manifest)
    return {
        e["episode_id"]: completion(output / "episodes" / e["episode_id"], e, expected, deep)
        for e in manifest["episodes"]
    }


def registry_row(runtime, record, metrics, folder, run_type):
    register(
        Runtime(runtime.work, folder / "config.json"),
        record["run_id"],
        run_type,
        folder / "metrics.json",
        metrics["wall_clock_seconds"],
        manifest_path=folder / "manifest.json",
    )


def execute_one(runtime, manifest, entry, execute=None):
    from sgvla.analysis.rollout import run_episode

    execute = execute or run_episode
    expected = context(runtime, manifest)
    episode = root(runtime, manifest) / "episodes" / entry["episode_id"]
    with file_lock(episode / ".lock"):
        previous = completion(episode, entry, expected, deep=True)
        if previous:
            registry_row(runtime, *previous, "P3_" + entry["condition"])
            return {"episode_id": entry["episode_id"], "status": "skipped"}
        for abandoned in sorted(episode.glob("attempt-*")):
            if (
                (abandoned / "manifest.json").exists()
                and not (abandoned / "metrics.json").exists()
                and not (abandoned / "error.json").exists()
            ):
                recovery = {
                    "status": "interrupted",
                    "wall_clock_seconds": None,
                    "recovered_at_utc": utc_now(),
                    "episode_id": entry["episode_id"],
                }
                write_json(abandoned / "interrupted.json", recovery)
                register(
                    Runtime(runtime.work, abandoned / "config.json"),
                    entry["episode_id"] + "-" + abandoned.name,
                    "P3_interrupted",
                    abandoned / "interrupted.json",
                    None,
                    manifest_path=abandoned / "manifest.json",
                )
        output = episode / ("attempt-" + uuid.uuid4().hex[:12])
        output.mkdir(parents=True)
        run_id = entry["episode_id"] + "-" + output.name
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
        started = time.monotonic()
        try:
            metrics = execute(runtime, entry, output)
            metrics["wall_clock_seconds"] = time.monotonic() - started
            if (
                metrics.get("status") != "completed"
                or metrics.get("condition") != entry["condition"]
                or type(metrics.get("success")) is not bool
                or type(metrics.get("steps")) is not int
                or not 1 <= metrics["steps"] <= 220
                or metrics.get("target_object") != entry["target_object"]
                or metrics.get("policy_seed") != entry["policy_seed"]
                or metrics.get("initial_state_sha256") != entry["initial_state_sha256"]
                or metrics.get("prompt") != entry["prompt"]
            ):
                raise ValueError("Incompatible episode metrics")
            validate_trace(output, metrics)
            write_json(output / "metrics.json", metrics)
            record = {
                "context": expected,
                "entry_sha256": fingerprint(entry),
                "attempt": output.name,
                "run_id": run_id,
                "artifacts": hashes(output),
                "completed_at_utc": utc_now(),
            }
            write_json(episode / "complete.json", record)
            registry_row(runtime, record, metrics, output, "P3_" + entry["condition"])
            return {"episode_id": entry["episode_id"], "status": "completed", **metrics}
        except Exception as exc:
            error = {
                "status": "infrastructure_error",
                "episode_id": entry["episode_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_clock_seconds": time.monotonic() - started,
            }
            write_json(output / "error.json", error)
            register(
                runtime,
                run_id,
                "P3_infrastructure_error",
                output / "error.json",
                error["wall_clock_seconds"],
                manifest_path=output / "manifest.json",
            )
            return error


def summarize(runtime, manifest):
    done = completed(runtime, manifest)
    metrics = [item[1] for item in done.values() if item]
    n, success = len(metrics), sum(m["success"] for m in metrics)
    return {
        "context": context(runtime, manifest),
        "condition": manifest["condition"],
        "planned": len(done),
        "completed": n,
        "successes": success,
        "success_rate": success / n if n else None,
        "wilson_95": wilson(success, n),
        "pending_or_invalid": len(done) - n,
        "task_failures": n - success,
        "infrastructure_error_attempts": len(
            list(root(runtime, manifest).glob("episodes/*/attempt-*/error.json"))
        ),
        "exclusive_first_target_successes": sum(
            m["exclusive_first_target_success"] for m in metrics
        ),
        "wrong_first_placements": sum(m["wrong_first_placement"] for m in metrics),
        "ties": sum(m["first_placed"] == "tie" for m in metrics),
        "no_placement": sum(m["first_placed"] is None for m in metrics),
        "output_dir": str(root(runtime, manifest)),
        "excluded": manifest["excluded"],
        "uncertainty_note": "Episode-level interval; task dependence not adjusted",
        "manual_label_validation": "pending; pose failure labels are provisional",
    }


def review(runtime, manifest):
    done = completed(runtime, manifest, deep=True)
    ids = [e["episode_id"] for e in manifest["episodes"][:5]]
    if not all(done[eid] for eid in ids):
        raise ValueError("Finish the first batch before review")
    used = sum(sum(p.stat().st_size for p in done[eid][2].iterdir() if p.is_file()) for eid in ids)
    remaining = sum(item is None for item in done.values())
    required = math.ceil(used / len(ids) * remaining * 1.5)
    report = {
        "context": context(runtime, manifest),
        "episodes": ids,
        "completion_sha256": {
            eid: digest(root(runtime, manifest) / "episodes" / eid / "complete.json") for eid in ids
        },
        "passed": shutil.disk_usage(runtime.work).free > required,
        "remaining_storage_estimate_bytes": required,
        "task_failures": [eid for eid in ids if not done[eid][1]["success"]],
        "manual_trajectory_review": "separate visual review required",
        "created_at_utc": utc_now(),
    }
    write_json(root(runtime, manifest) / "review.json", report)
    if not report["passed"]:
        raise RuntimeError("Insufficient estimated storage")
    return report


def run_batch(runtime, manifest, limit=5):
    from sgvla.analysis.baseline import load_baseline
    from sgvla.analysis.manifest import build
    from sgvla.eval.engine import ensure_sources

    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Batch size must be 1-10")
    validate(manifest, runtime.config)
    verification = runtime.work / manifest["verification_path"]
    temporal = (
        runtime.work / manifest["temporal_verification_path"]
        if manifest["temporal_verification_path"]
        else None
    )
    landmarks = (
        runtime.work / manifest["landmark_verification_path"]
        if manifest["landmark_verification_path"]
        else None
    )
    if build(runtime, manifest["condition"], verification, temporal, landmarks) != manifest:
        raise ValueError("Frozen condition differs from verified sources")
    ensure_sources(runtime, manifest["e0_manifest"])
    _, baseline_root, _, _ = load_baseline(runtime)
    if digest(baseline_root / "summary.json") != manifest["baseline_summary_sha256"]:
        raise ValueError("Frozen baseline summary changed")
    output = root(runtime, manifest)
    with file_lock(runtime.results / "part3/.evaluation.lock"):
        done = completed(runtime, manifest, deep=True)
        first = [e["episode_id"] for e in manifest["episodes"][:5]]
        pending = [e for e in manifest["episodes"] if not done[e["episode_id"]]]
        if pending and all(done[eid] for eid in first):
            report = read_json(output / "review.json") if (output / "review.json").exists() else {}
            checksums = {eid: digest(output / "episodes" / eid / "complete.json") for eid in first}
            if (
                report.get("context") != context(runtime, manifest)
                or not report.get("passed")
                or report.get("completion_sha256") != checksums
            ):
                raise RuntimeError("Review the first batch before continuing this condition")
        else:
            pending = [entry for entry in pending if entry["episode_id"] in first]
        results = []
        for entry in pending[:limit]:
            result = execute_one(runtime, manifest, entry)
            results.append(result)
            print(entry["episode_id"], result["status"], result.get("success"), flush=True)
            if result["status"] == "infrastructure_error":
                break
        summary = summarize(runtime, manifest)
        write_json(output / "summary.json", summary)
        write_json(output / "last_batch.json", {"episodes": results, "summary": summary})
        return {"episodes": results, "summary": summary}
