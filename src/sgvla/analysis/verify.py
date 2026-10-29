"""Read-only simulator relation verification before condition manifests are enabled."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from sgvla.analysis.scene import RULES, TABLE_TASKS, poses, relation, settle
from sgvla.eval.manifest import fingerprint
from sgvla.eval.scenes import BOWL1, BOWL2
from sgvla.io import digest, read_json, utc_now, write_json
from sgvla.runtime import Runtime


def verify(runtime, output):
    import numpy as np

    from sgvla.baselines.openpi.rollout import observation_element, official_module
    from sgvla.eval.engine import ensure_sources
    from sgvla.eval.run_libero import load_task

    started = time.monotonic()
    manifest = read_json(runtime.work / "configs/e0_manifest.json")
    ensure_sources(runtime, manifest)
    source = read_json(runtime.work / "docs/scene_report.json")
    official = official_module(runtime)
    rows = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["episodes"]:
        if entry["init_state_id"] > 2:
            continue
        np.random.seed(runtime.config["seed"])
        task, state = load_task(runtime, entry)
        env, prompt = official._get_libero_env(
            task, official.LIBERO_ENV_RESOLUTION, runtime.config["seed"]
        )
        try:
            env.reset()
            obs = env.set_init_state(state)
            obs = settle(env, obs, official.LIBERO_DUMMY_ACTION)
            row = source["tasks"][entry["task_id"]]
            s1 = relation(env, row, BOWL2)
            original = relation(env, row, BOWL1, original=True)
            position = poses(env)
            element = observation_element(obs, prompt, official, runtime.config["resize_size"])
            snapshot = output / (entry["episode_id"] + ".npz")
            np.savez_compressed(
                snapshot,
                sim_state=env.get_sim_state(),
                object_names=np.asarray(list(position)),
                object_poses=list(position.values()),
                agentview=element["observation/image"],
                wrist=element["observation/wrist_image"],
                proprio=element["observation/state"],
            )
            record = {
                "episode_id": entry["episode_id"],
                "task_id": entry["task_id"],
                "init_state_id": entry["init_state_id"],
                "initial_state_sha256": entry["initial_state_sha256"],
                "poses_world_xyz_wxyz": {name: p.tolist() for name, p in position.items()},
                "s1": s1,
                "original_relation": original,
                "temporal_source_candidate": entry["task_id"] in TABLE_TASKS,
                "temporal_status": "pending_swap_and_endpoint_verification",
                "snapshot": str(snapshot),
                "snapshot_sha256": digest(snapshot),
                "s1_exclusion_reason": None
                if s1["eligible"]
                else "relation_not_unique_under_declared_rules",
            }
            rows.append(record)
            write_json(output / (entry["episode_id"] + ".json"), record)
            print(entry["episode_id"], "S1", s1["eligible"], flush=True)
        finally:
            env.close()
    report = {
        "schema": 1,
        "phase": "relations_only",
        "rules": RULES,
        "manifest_sha256": fingerprint(manifest),
        "policy_sha256": runtime.policy_hash,
        "source_report_sha256": digest(runtime.work / "docs/scene_report.json"),
        "created_at_utc": utc_now(),
        "wall_clock_seconds": time.monotonic() - started,
        "s1_eligible": sum(row["s1"]["eligible"] for row in rows),
        "examined": len(rows),
        "rows": rows,
    }
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = Runtime(args.work, args.config)
    os.environ.update(runtime.env())
    report = verify(runtime, args.output)
    print("verification", report["examined"], report["s1_eligible"], flush=True)


if __name__ == "__main__":
    main()
