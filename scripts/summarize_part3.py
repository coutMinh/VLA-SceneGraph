"""Build a concise exploratory report from the machine-generated Part 3 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgvla.analysis.reporting import paired_statistics


def summarize(report: dict) -> dict:
    conditions = report["conditions"]
    direct_spatial = None
    if {"S2", "S2-ctrl"} <= set(conditions):
        treated = {row["paired_e0_id"]: row for row in conditions["S2"]["episodes"]}
        control = {row["paired_e0_id"]: row for row in conditions["S2-ctrl"]["episodes"]}
        if treated.keys() != control.keys():
            raise ValueError("S2 and S2-ctrl must contain the same paired E0 states")
        keys = sorted(treated)
        direct_spatial = {
            "pairing": keys,
            "comparison": "S2 success versus S2-ctrl success",
            "paired_statistics": paired_statistics(
                [(control[key]["success"], treated[key]["success"]) for key in keys]
            ),
        }

    temporal = report.get("temporal_pairs", [])
    applicable = [row for row in temporal if row["bound_applicable_to_recorded_pair"]]
    temporal_summary = {
        "matched_pairs": len(temporal),
        "identical_input_and_action_prefix_pairs": len(applicable),
        "exclusive_first_target_bound_satisfied": sum(
            row["bound_satisfied"] is True for row in applicable
        ),
        "exclusive_first_target_bound_violations": sum(
            row["bound_satisfied"] is False for row in applicable
        ),
        "pairs_with_divergent_prefix": len(temporal) - len(applicable),
        "unrestricted_both_target_success_pairs": sum(
            row["unrestricted_target_success_sum"] == 2 for row in temporal
        ),
        "interpretation": (
            "The exclusive first-target bound applies only when observations and actions match "
            "through the first placement. Unrestricted rollouts may place both objects."
        ),
    }
    rows = {}
    implementation_hashes = {}
    for name, item in conditions.items():
        summary = item["summary"]
        paired = item["paired_comparison"]
        implementation_hashes[name] = summary["context"]["implementation_sha256"]
        rows[name] = {
            "planned": summary["planned"],
            "completed": summary["completed"],
            "successes": summary["successes"],
            "task_failures": summary["task_failures"],
            "infrastructure_error_attempts": summary["infrastructure_error_attempts"],
            "wilson_95": summary["wilson_95"],
            "exclusive_first_target_successes": summary["exclusive_first_target_successes"],
            "wrong_first_placements": summary["wrong_first_placements"],
            "no_placement": summary["no_placement"],
            "matched_e0": paired["matched_n"],
            "e0_only_success": paired["baseline_only_success"],
            "condition_only_success": paired["condition_only_success"],
            "paired_success_difference": paired["success_difference"],
            "paired_exact_two_sided_p": paired["exact_discordant_two_sided_p"],
            "excluded_states": len(summary["excluded"]),
        }
    return {
        "baseline_context": report["baseline_context"],
        "condition_implementation_hashes": implementation_hashes,
        "conditions": rows,
        "direct_S2_vs_S2_ctrl": direct_spatial,
        "temporal_prefix": temporal_summary,
        "limitations": report["limitations"]
        + [
            "Small, selected task/state sets; episode intervals do not adjust for task clustering.",
            "S2 landmark displacement magnitudes were not matched between treatment and control.",
            "Failure-cause descriptions remain coarse where sampled video cannot isolate "
            "contact or release.",
        ],
    }


def markdown(result: dict) -> str:
    lines = [
        "# Part 3 exploratory results",
        "",
        "These results describe the verified subset of LIBERO Spatial episodes. They are "
        "exploratory; they do not establish that a scene graph improves the policy.",
        "",
        f"Policy fingerprint: `{result['baseline_context']['policy_sha256']}`  ",
        f"Baseline source commit: "
        f"`{result['baseline_context'].get('source_commit', 'not recorded')}`  ",
        f"Baseline implementation SHA-256: "
        f"`{result['baseline_context']['implementation_sha256']}`",
        "",
        "| Condition | Completed / planned | Success | First-target success | Wrong first object | "
        "No placement | Matched E0: E0-only / condition-only | Paired p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["conditions"].items():
        p = row["paired_exact_two_sided_p"]
        p_text = "—" if p is None else f"{p:.4g}"
        lines.append(
            f"| {name} | {row['completed']} / {row['planned']} | {row['successes']} | "
            f"{row['exclusive_first_target_successes']} | {row['wrong_first_placements']} | "
            f"{row['no_placement']} | {row['e0_only_success']} / "
            f"{row['condition_only_success']} | {p_text} |"
        )
    lines.extend(["", "Implementation hash by condition:", ""])
    lines.extend(
        f"- {name}: `{implementation_hash}`"
        for name, implementation_hash in result["condition_implementation_hashes"].items()
    )
    spatial = result["direct_S2_vs_S2_ctrl"]
    lines.extend(["", "## Direct landmark comparison", ""])
    if spatial is None:
        lines.append("Pending: both S2 and S2-ctrl must finish.")
    else:
        stats = spatial["paired_statistics"]
        lines.append(
            f"S2 versus S2-ctrl, matched on {stats['matched_n']} initial states: "
            f"S2-only successes {stats['condition_only_success']}, "
            f"control-only successes {stats['baseline_only_success']}, "
            f"difference {stats['success_difference']:.3f}, "
            f"exact two-sided p={stats['exact_discordant_two_sided_p']:.4g}."
        )
    temporal = result["temporal_prefix"]
    lines.extend(
        [
            "",
            "## Temporal prefix check",
            "",
            f"{temporal['identical_input_and_action_prefix_pairs']} of "
            f"{temporal['matched_pairs']} pairs matched observations and actions through first "
            f"placement; the exclusive first-target bound held in "
            f"{temporal['exclusive_first_target_bound_satisfied']} and was violated in "
            f"{temporal['exclusive_first_target_bound_violations']}. "
            f"{temporal['pairs_with_divergent_prefix']} pairs diverged before that point.",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("results/part3/report.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = summarize(report)
    output_dir = args.output_dir or args.report.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
