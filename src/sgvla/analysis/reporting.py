"""Generate paired comparisons and a traceable human-review queue from artifacts."""

from __future__ import annotations

import math

from sgvla.analysis.engine import completed, root, summarize
from sgvla.io import digest, read_json, write_json


def paired_statistics(pairs):
    """Exact two-sided binomial test on discordant matched binary outcomes."""
    n = len(pairs)
    lost = sum(bool(a) and not bool(b) for a, b in pairs)
    gained = sum(not bool(a) and bool(b) for a, b in pairs)
    discordant = lost + gained
    p = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(lost, gained) + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "matched_n": n,
        "baseline_only_success": lost,
        "condition_only_success": gained,
        "discordant_pairs": discordant,
        "success_difference": (gained - lost) / n if n else None,
        "exact_discordant_two_sided_p": p,
        "note": "Exploratory episode-paired test; task clustering and multiplicity not adjusted",
    }


def temporal_prefix(left, right):
    import numpy as np

    a, b = left[1], right[1]
    lengths = []
    for metrics in (a, b):
        placed = [
            v["first_on_plate_frame"]
            for v in metrics["pose_evidence"].values()
            if v["first_on_plate_frame"] is not None
        ]
        prefix = metrics["policy_start_frame"]
        lengths.append(min(placed) - prefix if placed else metrics["steps"])
    length = min(lengths)
    calls = math.ceil(length / a["replan_steps"])
    input_equal = a["policy_observation_hashes"][:calls] == b["policy_observation_hashes"][:calls]
    with np.load(left[2] / "trajectory.npz", allow_pickle=False) as ta:
        with np.load(right[2] / "trajectory.npz", allow_pickle=False) as tb:
            pa, pb = a["policy_start_frame"], b["policy_start_frame"]
            actions_equal = bool(
                np.allclose(
                    ta["actions"][pa : pa + length],
                    tb["actions"][pb : pb + length],
                    atol=1e-5,
                    rtol=0,
                )
            )
    total = int(a["exclusive_first_target_success"]) + int(b["exclusive_first_target_success"])
    return {
        "compared_policy_steps": length,
        "inputs_equal_until_first_placement": input_equal,
        "actions_equal_until_first_placement": actions_equal,
        "exclusive_first_target_success_sum": total,
        "bound_applicable_to_recorded_pair": input_equal and actions_equal,
        "bound_satisfied": total <= 1 if input_equal and actions_equal else None,
        "unrestricted_target_success_sum": int(a["success"]) + int(b["success"]),
    }


def aggregate(runtime):
    from sgvla.analysis.baseline import load_baseline

    _, baseline_root, baseline_summary, baseline = load_baseline(runtime)
    output = {
        "baseline_context": baseline_summary["context"],
        "conditions": {},
        "manual_review_queue": [],
        "temporal_pairs": [],
        "limitations": [
            "Verification exclusions restrict task/state coverage",
            "Failure-cause labels remain provisional until visually validated",
        ],
    }
    done_conditions = {}
    for condition in ("S1", "T1", "T1-ctrl", "S2", "S2-ctrl"):
        path = runtime.work / "configs/part3" / (condition + ".json")
        if not path.exists():
            continue
        manifest = read_json(path)
        done = completed(runtime, manifest)
        done_conditions[condition] = (manifest, done)
        pairs, rows = [], []
        for entry in manifest["episodes"]:
            item = done[entry["episode_id"]]
            if item:
                bm = baseline[entry["paired_e0_id"]][1]
                pairs.append((bm["success"], item[1]["success"]))
                rows.append(
                    {
                        "episode_id": entry["episode_id"],
                        "paired_e0_id": entry["paired_e0_id"],
                        "baseline_success": bm["success"],
                        "success": item[1]["success"],
                        "first_placed": item[1]["first_placed"],
                        "exclusive_first_target_success": item[1]["exclusive_first_target_success"],
                        "completion_sha256": digest(
                            root(runtime, manifest)
                            / "episodes"
                            / entry["episode_id"]
                            / "complete.json"
                        ),
                        "baseline_completion_sha256": digest(
                            baseline_root / "episodes" / entry["paired_e0_id"] / "complete.json"
                        ),
                    }
                )
        output["conditions"][condition] = {
            "summary": summarize(runtime, manifest),
            "paired_comparison": paired_statistics(pairs),
            "episodes": rows,
        }
        failures = [r for r in rows if not r["success"]][:3]
        successes = [r for r in rows if r["success"]][:2]
        for row in failures + successes:
            folder = done[row["episode_id"]][2]
            output["manual_review_queue"].append(
                {
                    "condition": condition,
                    **row,
                    "agentview": str(folder / "agentview.mp4"),
                    "wrist": str(folder / "wrist.mp4"),
                    "review_status": "pending",
                }
            )
    if {"T1", "T1-ctrl"} <= set(done_conditions):
        left_manifest, left = done_conditions["T1"]
        _, right = done_conditions["T1-ctrl"]
        for entry in left_manifest["episodes"]:
            other_id = "T1-ctrl" + entry["paired_e0_id"][2:]
            if left[entry["episode_id"]] and right.get(other_id):
                output["temporal_pairs"].append(
                    {
                        "paired_e0_id": entry["paired_e0_id"],
                        **temporal_prefix(left[entry["episode_id"]], right[other_id]),
                    }
                )
    write_json(runtime.results / "part3/report.json", output)
    return output
