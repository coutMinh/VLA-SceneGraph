"""Verify stability and byte-identical policy inputs before permitting T1 pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sgvla.analysis.interventions import temporal_prelude
from sgvla.analysis.scene import RULES, TABLE_TASKS, settle
from sgvla.eval.trace import Trace, observation_hash
from sgvla.io import digest, read_json, write_json
from sgvla.runtime import Runtime


def history_prompt(original):
    return (
        original.replace("from table center", "at table center")
        .replace("pick up the black bowl ", "pick up the black bowl that was ", 1)
        .replace(" and place it on the plate", " at the start and place it on the plate")
    )


def verify_temporal(runtime, verification, output):
    import numpy as np

    from sgvla.baselines.openpi.rollout import observation_element, official_module
    from sgvla.eval.run_libero import load_task

    source = read_json(verification)
    original = read_json(runtime.work / "configs/e0_manifest.json")
    entries = {e["episode_id"]: e for e in original["episodes"]}
    official = official_module(runtime)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in source["rows"]:
        record = {
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "init_state_id": row["init_state_id"],
            "eligible": False,
        }
        if row["task_id"] not in TABLE_TASKS:
            record["exclusion_reason"] = "different_supports_or_drawer_not_verified"
        elif not row["original_relation"]["eligible"]:
            record["exclusion_reason"] = "initial_target_relation_not_unique"
        else:
            entry = entries[row["episode_id"]]
            prompt = history_prompt(entry["prompt"])
            observations, evidence = [], []
            for swap in (False, True):
                np.random.seed(runtime.config["seed"])
                task, state = load_task(runtime, entry)
                env, _ = official._get_libero_env(
                    task, official.LIBERO_ENV_RESOLUTION, runtime.config["seed"]
                )
                folder = output / row["episode_id"] / ("swap" if swap else "control")
                folder.mkdir(parents=True, exist_ok=True)
                trace = None
                try:
                    env.reset()
                    obs = env.set_init_state(state)
                    trace = Trace(folder, env, official, runtime.config["resize_size"])
                    obs = settle(env, obs, official.LIBERO_DUMMY_ACTION, 10, trace, prompt)
                    obs, check = temporal_prelude(env, obs, official, prompt, swap, trace)
                    trace.capture(obs, prompt, 2)
                    trace.save()
                    element = observation_element(
                        obs, prompt, official, runtime.config["resize_size"]
                    )
                    np.savez_compressed(folder / "endpoint.npz", **element)
                    observations.append(element)
                    evidence.append({k: v for k, v in check.items() if k != "endpoint_sim_state"})
                finally:
                    if trace is not None:
                        trace.close()
                    env.close()
            equal = {
                k: bool(np.array_equal(observations[0][k], observations[1][k]))
                for k in observations[0]
            }
            record.update(
                eligible=all(equal.values()) and all(e["stable"] for e in evidence),
                endpoint_equal=equal,
                endpoint_hashes=[observation_hash(x) for x in observations],
                prompt=prompt,
                stability=evidence,
                exclusion_reason=None,
            )
            if not record["eligible"]:
                record["exclusion_reason"] = "endpoint_input_mismatch_or_unstable_intervention"
            print(row["episode_id"], "T1", record["eligible"], equal, flush=True)
        rows.append(record)
        write_json(
            output / "temporal_verification.json",
            {
                "schema": 1,
                "rules": RULES,
                "relation_verification_sha256": digest(verification),
                "examined": len(rows),
                "complete": len(rows) == len(source["rows"]),
                "eligible": sum(r["eligible"] for r in rows),
                "rows": rows,
            },
        )
    return read_json(output / "temporal_verification.json")


def main():
    parser = argparse.ArgumentParser()
    for name in ("work", "config", "verification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    runtime = Runtime(args.work, args.config)
    os.environ.update(runtime.env())
    verify_temporal(runtime, args.verification, args.output)


if __name__ == "__main__":
    main()
