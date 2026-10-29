"""Instrument the unmodified official eval_libero, not a second copy of our loop."""

from __future__ import annotations

import time
import uuid
from contextlib import ExitStack
from unittest.mock import patch

from sgvla.eval.engine import context, ensure_sources, evaluation_root
from sgvla.io import digest, file_lock, read_json, utc_now, write_json
from sgvla.registry import register

TOLERANCES = {"action_atol": 1e-5, "action_rtol": 0, "policy_inputs": "byte_identical"}


def compare(wrapper, reference):
    import numpy as np

    checks = {
        "observations": wrapper["hashes"] == reference["hashes"],
        "success": wrapper["success"] == reference["success"],
        "steps": wrapper["steps"] == reference["steps"],
    }
    for name in ("chunks", "actions"):
        a, b = np.asarray(wrapper[name]), np.asarray(reference[name])
        checks[name] = bool(
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, rtol=0, atol=TOLERANCES["action_atol"])
        )
    return {"passed": all(checks.values()), "checks": checks}


def official_episode(runtime, entry, folder):
    import numpy as np
    from libero.libero import benchmark

    from sgvla.baselines.openpi.rollout import official_module
    from sgvla.baselines.openpi.runner import connect
    from sgvla.baselines.openpi.seeded import SeededClient
    from sgvla.eval.run_libero import load_task
    from sgvla.eval.trace import observation_hash

    official = official_module(runtime)
    real_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task, state = load_task(runtime, entry, real_suite)
    base_client = SeededClient(connect(runtime), entry["policy_seed"])
    captured = {"hashes": [], "chunks": [], "actions": [], "success": False, "error": None}
    environments = []

    class SelectedSuite:
        n_tasks = 1

        def get_task(self, task_id):
            if task_id != 0:
                raise ValueError("Parity only exposes one selected task")
            return task

        def get_task_init_states(self, task_id):
            return [state]

    class RecordingClient:
        def infer(self, obs):
            captured["hashes"].append(observation_hash(obs))
            try:
                result = base_client.infer(obs)
                captured["chunks"].append(np.asarray(result["actions"]).copy())
                return result
            except Exception as exc:
                captured["error"] = str(exc)
                raise

    original_factory = official._get_libero_env

    def factory(*args, **kwargs):
        env, prompt = original_factory(*args, **kwargs)
        environments.append(env)

        class RecordingEnv:
            def reset(self):
                return env.reset()

            def set_init_state(self, initial):
                return env.set_init_state(initial)

            def step(self, action):
                try:
                    result = env.step(action)
                except Exception as exc:
                    captured["error"] = str(exc)
                    raise
                captured["actions"].append(np.asarray(action).copy())
                captured["success"] = bool(result[2])
                return result

        return RecordingEnv(), prompt

    args = official.Args(
        host=runtime.config["host"],
        port=runtime.config["port"],
        resize_size=runtime.config["resize_size"],
        replan_steps=runtime.config["replan_steps"],
        task_suite_name="libero_spatial",
        num_steps_wait=runtime.config["settling_steps"],
        num_trials_per_task=1,
        video_out_path=str(folder),
        seed=runtime.config["seed"],
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    official.benchmark,
                    "get_benchmark_dict",
                    return_value={"libero_spatial": SelectedSuite},
                )
            )
            stack.enter_context(
                patch.object(
                    official._websocket_client_policy,
                    "WebsocketClientPolicy",
                    return_value=RecordingClient(),
                )
            )
            stack.enter_context(patch.object(official, "_get_libero_env", side_effect=factory))
            official.eval_libero(args)
    finally:
        for env in environments:
            env.close()
    if captured["error"] is not None:
        raise RuntimeError("Official loop swallowed an infrastructure error: " + captured["error"])
    captured["actions"] = np.asarray(captured["actions"])[runtime.config["settling_steps"] :]
    captured["steps"] = len(captured["actions"])
    if not captured["hashes"] or captured["steps"] < 1:
        raise RuntimeError("Official client produced no policy execution evidence")
    np.savez_compressed(
        folder / "official.npz", actions=captured["actions"], chunks=captured["chunks"]
    )
    write_json(folder / "observations.json", {"hashes": captured["hashes"]})
    return captured


def run_parity(runtime, manifest):
    import numpy as np

    from sgvla.eval.run_libero import run_episode
    from sgvla.eval.trace import validate_trace

    ensure_sources(runtime, manifest)
    root = evaluation_root(runtime, manifest)
    with file_lock(root / ".parity.lock"):
        folder = root / "parity" / ("attempt-" + uuid.uuid4().hex[:12])
        folder.mkdir(parents=True)
        started = time.monotonic()
        write_json(
            folder / "manifest.json", {"context": context(runtime, manifest), "workload": manifest}
        )
        comparisons = []
        report = {
            "context": context(runtime, manifest),
            "task_id": 0,
            "initial_states": [0, 1, 2],
            "tolerances": TOLERANCES,
            "passed": False,
            "comparisons": comparisons,
            "protocol": "unmodified official eval_libero; one selected state per fresh invocation",
        }
        try:
            for state_id in (0, 1, 2):
                entry = next(
                    e
                    for e in manifest["episodes"]
                    if e["task_id"] == 0 and e["init_state_id"] == state_id
                )
                wrapper_dir, reference_dir = (
                    folder / f"state{state_id}/wrapper",
                    folder / f"state{state_id}/official",
                )
                wrapper_dir.mkdir(parents=True)
                reference_dir.mkdir(parents=True)
                metrics = run_episode(runtime, entry, wrapper_dir)
                write_json(wrapper_dir / "metrics.json", metrics)
                validate_trace(wrapper_dir, metrics)
                with np.load(wrapper_dir / "trajectory.npz", allow_pickle=False) as trace:
                    wrapper = {
                        "hashes": metrics["policy_observation_hashes"],
                        "success": metrics["success"],
                        "steps": metrics["steps"],
                        "chunks": trace["policy_chunks"].copy(),
                        "actions": trace["actions"][metrics["settling_steps"] :].copy(),
                    }
                reference = official_episode(runtime, entry, reference_dir)
                comparison = {"init_state_id": state_id, **compare(wrapper, reference)}
                write_json(
                    reference_dir / "outcome.json",
                    {"success": reference["success"], "steps": reference["steps"]},
                )
                comparisons.append(comparison)
            report["passed"] = all(item["passed"] for item in comparisons)
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["wall_clock_seconds"] = time.monotonic() - started
        report["created_at_utc"] = utc_now()
        report["artifact_sha256"] = {
            str(p.relative_to(root)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()
        }
        metrics_path = folder / "report.json"
        write_json(metrics_path, report)
        register(
            runtime,
            "parity-" + folder.name,
            "official_client_parity",
            metrics_path,
            report["wall_clock_seconds"],
            manifest_path=folder / "manifest.json",
        )
        write_json(
            root / "parity.json",
            {"report": str(metrics_path.relative_to(root)), "sha256": digest(metrics_path)},
        )
        if not report["passed"]:
            raise RuntimeError("Official-client parity failed; inspect " + str(metrics_path))
        return report


def require_parity(runtime, manifest):
    root = evaluation_root(runtime, manifest)
    try:
        pointer = read_json(root / "parity.json")
        report_path = root / pointer["report"]
        report_path.resolve().relative_to(root.resolve())
        report = read_json(report_path)
        if (
            digest(report_path) != pointer["sha256"]
            or report["context"] != context(runtime, manifest)
            or report["passed"] is not True
            or report["task_id"] != 0
            or report["initial_states"] != [0, 1, 2]
            or report["tolerances"] != TOLERANCES
            or len(report["comparisons"]) != 3
            or not all(item["passed"] for item in report["comparisons"])
            or sorted(item["init_state_id"] for item in report["comparisons"]) != [0, 1, 2]
            or not all(
                item["checks"]
                == dict.fromkeys(("observations", "success", "steps", "chunks", "actions"), True)
                for item in report["comparisons"]
            )
            or not report["artifact_sha256"]
        ):
            raise ValueError("Incompatible parity evidence")
        for relative, checksum in report["artifact_sha256"].items():
            path = root / relative
            path.resolve().relative_to(root.resolve())
            if digest(path) != checksum:
                raise ValueError("Parity evidence changed")
        return report
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Run a compatible passing eval-parity before E0") from exc
