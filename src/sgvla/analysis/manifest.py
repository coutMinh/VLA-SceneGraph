"""Freeze Part 3 conditions only after independent scene verification."""

from __future__ import annotations

import copy
from pathlib import Path

from sgvla.analysis.scene import RULES, TABLE_TASKS
from sgvla.eval.manifest import fingerprint, policy_fingerprint
from sgvla.eval.manifest import validate_manifest as validate_e0
from sgvla.eval.scenes import BOWL1, BOWL2
from sgvla.io import digest, read_json, write_json

CONDITIONS = ("S1", "T1", "T1-ctrl", "S2", "S2-ctrl")
S1_FRAGMENTS = (
    "next to the ramekin",
    "next to the cookie box",
    "next to the plate",
    "on the wooden cabinet",
    "on the wooden cabinet",
    "on the cookie box",
    "on the stove",
    "on the wooden cabinet",
    "next to the ramekin",
    "on the stove",
)


def build(runtime, condition, verification, temporal=None, landmarks=None):
    from sgvla.analysis.baseline import load_baseline
    from sgvla.analysis.temporal_verify import history_prompt

    if condition not in CONDITIONS:
        raise ValueError("Condition has no verified implementation")
    e0, root, summary, _ = load_baseline(runtime)
    verify = read_json(verification)
    if verify["rules"] != RULES or verify["examined"] != 30:
        raise ValueError("Incomplete/incompatible relation verification")
    if verify["manifest_sha256"] != fingerprint(e0):
        raise ValueError("Verification uses another E0 manifest")
    rows = {row["episode_id"]: row for row in verify["rows"]}
    temporal_report = read_json(temporal) if temporal else None
    if condition.startswith("T1"):
        if (
            not temporal_report
            or not temporal_report["complete"]
            or temporal_report["examined"] != 30
            or temporal_report["rules"] != RULES
            or temporal_report["relation_verification_sha256"] != digest(verification)
        ):
            raise ValueError("T1 requires complete compatible endpoint/stability verification")
        temporal_rows = {row["episode_id"]: row for row in temporal_report["rows"]}
    if condition.startswith("S2"):
        landmark_report = read_json(landmarks) if landmarks else None
        if (
            not landmark_report
            or not landmark_report["complete"]
            or landmark_report["examined"] != 30
            or landmark_report["rules"] != RULES
            or landmark_report["relation_verification_sha256"] != digest(verification)
        ):
            raise ValueError("S2 requires compatible paired landmark verification")
        landmark_rows = {r["episode_id"]: r for r in landmark_report["rows"]}
    source = read_json(runtime.work / "docs/scene_report.json")
    episodes, excluded = [], []
    for original in e0["episodes"]:
        if original["init_state_id"] > 2:
            continue
        row = rows[original["episode_id"]]
        check_row = (
            row
            if condition == "S1"
            else (
                temporal_rows[original["episode_id"]]
                if condition.startswith("T1")
                else landmark_rows[original["episode_id"]]
            )
        )
        eligible = row["s1"]["eligible"] if condition == "S1" else check_row["eligible"]
        if not eligible:
            reason = (
                row["s1_exclusion_reason"] if condition == "S1" else check_row["exclusion_reason"]
            )
            excluded.append({"paired_e0_id": original["episode_id"], "reason": reason})
            continue
        entry = copy.deepcopy(original)
        entry.update(
            episode_id=condition + original["episode_id"][2:],
            condition=condition,
            paired_e0_id=original["episode_id"],
            original_prompt=original["prompt"],
            prompt=source["tasks"][entry["task_id"]]["s1_instruction"]
            if condition == "S1"
            else history_prompt(original["prompt"])
            if condition.startswith("T1")
            else original["prompt"],
            target_object=BOWL2 if condition in ("S1", "S2") else BOWL1,
        )
        if condition.startswith("T1"):
            check = temporal_rows[original["episode_id"]]
            if not all(check["endpoint_equal"].values()) or len(set(check["endpoint_hashes"])) != 1:
                raise ValueError("Temporal endpoint inputs differ")
            entry["endpoint_sha256"] = check["endpoint_hashes"][0]
        if condition.startswith("S2"):
            selection = check_row["selections"][condition]
            entry["landmark_intervention"] = selection["recipe"]
            entry["endpoint_sha256"] = selection["endpoint_sha256"]
            endpoint_file = Path(landmarks).parent / (
                original["episode_id"] + "-" + condition + ".npz"
            )
            entry["endpoint_reference_path"] = str(
                endpoint_file.resolve().relative_to(runtime.work.resolve())
            )
        episodes.append(entry)
    if not episodes:
        raise ValueError("No verified episodes for " + condition)
    result = {
        "schema": 1,
        "kind": "part3_verified_conditions",
        "condition": condition,
        "rules": RULES,
        "policy_sha256": runtime.policy_hash,
        "e0_manifest": e0,
        "baseline_context": summary["context"],
        "baseline_summary_sha256": digest(root / "summary.json"),
        "verification_sha256": digest(verification),
        "verification_path": str(Path(verification).resolve().relative_to(runtime.work.resolve())),
        "temporal_verification_path": str(
            Path(temporal).resolve().relative_to(runtime.work.resolve())
        )
        if temporal
        else None,
        "temporal_verification_sha256": digest(temporal) if temporal else None,
        "landmark_verification_path": str(
            Path(landmarks).resolve().relative_to(runtime.work.resolve())
        )
        if landmarks
        else None,
        "landmark_verification_sha256": digest(landmarks) if landmarks else None,
        "episodes": episodes,
        "excluded": excluded,
        "scope": "predeclared states 0-2, with pre-policy verification exclusions",
    }
    validate(result, runtime.config)
    return result


def validate(manifest, config):
    from sgvla.analysis.temporal_verify import history_prompt

    if manifest.get("schema") != 1 or manifest.get("kind") != "part3_verified_conditions":
        raise ValueError("Unsupported Part 3 manifest")
    condition = manifest["condition"]
    if condition not in CONDITIONS or manifest["rules"] != RULES:
        raise ValueError("Unsupported condition or changed protocol")
    e0 = manifest["e0_manifest"]
    validate_e0(e0, config)
    if manifest["policy_sha256"] != policy_fingerprint(config):
        raise ValueError("Policy differs from frozen workload")
    originals = {e["episode_id"]: e for e in e0["episodes"] if e["init_state_id"] <= 2}
    ids, pairs = set(), set()
    for entry in manifest["episodes"]:
        original = originals[entry["paired_e0_id"]]
        if (
            entry["condition"] != condition
            or entry["episode_id"] != condition + original["episode_id"][2:]
            or entry["episode_id"] in ids
            or entry["paired_e0_id"] in pairs
            or entry["original_prompt"] != original["prompt"]
            or entry["target_object"] != (BOWL2 if condition in ("S1", "S2") else BOWL1)
        ):
            raise ValueError("Invalid Part 3 episode identity")
        for key in (
            "task_id",
            "task_name",
            "init_state_id",
            "initial_state_sha256",
            "init_file_sha256",
            "bddl_sha256",
            "policy_seed",
        ):
            if entry[key] != original[key]:
                raise ValueError("Paired source/seed differs: " + key)
        expected_prompt = (
            "pick up the black bowl "
            + S1_FRAGMENTS[entry["task_id"]]
            + " and place it on the plate"
            if condition == "S1"
            else history_prompt(original["prompt"])
            if condition.startswith("T1")
            else original["prompt"]
        )
        if entry["prompt"] != expected_prompt:
            raise ValueError("Instruction differs from predeclared wording")
        if condition.startswith("T1") and len(entry.get("endpoint_sha256", "")) != 64:
            raise ValueError("Missing verified temporal input hash")
        if condition.startswith("T1") and entry["task_id"] not in TABLE_TASKS:
            raise ValueError("Temporal support configuration not verified")
        if condition.startswith("S2"):
            from sgvla.analysis.landmarks import LANDMARK_TASKS

            reference_path = Path(entry.get("endpoint_reference_path", ""))
            if (
                entry["task_id"] not in LANDMARK_TASKS
                or entry["landmark_intervention"]["landmark"] != LANDMARK_TASKS[entry["task_id"]]
                or len(entry.get("endpoint_sha256", "")) != 64
                or not entry.get("endpoint_reference_path")
                or reference_path.is_absolute()
                or ".." in reference_path.parts
            ):
                raise ValueError("Unsupported landmark intervention")
        ids.add(entry["episode_id"])
        pairs.add(entry["paired_e0_id"])
    excluded = {row["paired_e0_id"] for row in manifest["excluded"]}
    if not ids or pairs & excluded or pairs | excluded != set(originals):
        raise ValueError("Incomplete declared subset or overlapping exclusions")
    return fingerprint(manifest)


def freeze(path, manifest):
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Manifest already exists and differs; choose another protocol directory")
    write_json(path, manifest)
