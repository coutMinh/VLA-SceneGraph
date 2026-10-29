"""Immutable workload definitions, independent of policy/server compatibility."""

from __future__ import annotations

import hashlib
import json

POLICY_KEYS = ("openpi_commit", "model_config", "checkpoint", "backend", "num_steps")
LOOP_KEYS = ("task_suite", "seed", "resize_size", "replan_steps", "settling_steps", "max_steps")
PROTOCOL = "sgvla-seeded-noise-v1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def policy_fingerprint(config):
    return fingerprint({"protocol": PROTOCOL, **{key: config[key] for key in POLICY_KEYS}})


def loop_settings(config):
    return {key: config[key] for key in LOOP_KEYS}


def episode_seed(seed, task_id, state_id):
    # Conditions share the same noise sequence for a paired task/state.
    return int(fingerprint(["policy", seed, task_id, state_id])[:8], 16)


def build_manifest(report, config, n_per_task=5):
    if type(n_per_task) is not int or not 1 <= n_per_task <= 5:
        raise ValueError("This reduced pilot supports the predeclared initial states 0-4 only")
    episodes = []
    for task in report["tasks"]:
        for state_id in range(n_per_task):
            episodes.append(
                {
                    "episode_id": f"E0-t{task['task_id']:02d}-s{state_id:02d}",
                    "condition": "E0",
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "init_state_id": state_id,
                    "initial_state_sha256": task["state_sha256"][state_id],
                    "init_file_sha256": task["init_file_sha256"],
                    "bddl_sha256": task["bddl_sha256"],
                    "prompt": task["official_instruction"],
                    "target_object": task["e0_target"],
                    "policy_seed": episode_seed(config["seed"], task["task_id"], state_id),
                }
            )
    manifest = {
        "schema": 1,
        "kind": "reduced_spatial_pilot",
        "policy_sha256": policy_fingerprint(config),
        "evaluation": loop_settings(config),
        "libero_commit": config["libero_commit"],
        "scene_report_sha256": fingerprint(report),
        "official_client_sha256": report["official_client_sha256"],
        "rng_contract": PROTOCOL,
        "episodes": episodes,
    }
    validate_manifest(manifest, config)
    return manifest


def validate_manifest(manifest, config):
    official = {"resize_size": 224, "replan_steps": 5, "settling_steps": 10, "max_steps": 220}
    if any(config[key] != value for key, value in official.items()):
        raise ValueError("E0 requires the declared official Spatial loop settings")
    if config["task_suite"] != "libero_spatial" or config["backend"] != "jax":
        raise ValueError("The reduced pilot is verified only for Spatial on JAX")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Invalid simulator seed")
    if manifest.get("schema") != 1 or manifest.get("rng_contract") != PROTOCOL:
        raise ValueError("Unsupported evaluation manifest")
    if manifest["policy_sha256"] != policy_fingerprint(config):
        raise ValueError("Manifest policy differs from runtime")
    if manifest["evaluation"] != loop_settings(config):
        raise ValueError("Manifest evaluation settings differ from runtime")
    if manifest["libero_commit"] != config["libero_commit"]:
        raise ValueError("Manifest LIBERO commit differs")
    entries = manifest["episodes"]
    ids = set()
    for entry in entries:
        task, state = entry["task_id"], entry["init_state_id"]
        expected_id = f"E0-t{task:02d}-s{state:02d}"
        if (
            type(task) is not int
            or type(state) is not int
            or not 0 <= task < 10
            or not 0 <= state < 5
            or entry["episode_id"] != expected_id
            or entry["episode_id"] in ids
            or entry["condition"] != "E0"
            or entry["target_object"] != "akita_black_bowl_1"
            or entry["prompt"] != entry["task_name"].replace("_", " ")
            or entry["policy_seed"] != episode_seed(config["seed"], task, state)
        ):
            raise ValueError("Invalid, duplicate or non-E0 episode")
        for key in ("initial_state_sha256", "init_file_sha256", "bddl_sha256"):
            if len(entry[key]) != 64 or any(c not in "0123456789abcdef" for c in entry[key]):
                raise ValueError("Invalid source checksum")
        ids.add(entry["episode_id"])
    if not entries:
        raise ValueError("Empty evaluation manifest")
    return fingerprint(manifest)
