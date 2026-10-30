"""Aggregate cross-suite metrics into summary.json / summary.md (standard library only).

    python -m sgvla.xsuite.report RESULTS_DIR [OUT_DIR]

RESULTS_DIR holds one folder per episode with metrics.json (+ intervention.json).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from sgvla.xsuite.spec import QUESTIONS, SUITES


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, round(centre - half, 4)), min(1.0, round(centre + half, 4))]


def exact_mcnemar(b, c):
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def goal_objects(m):
    """Objects named by the success predicate (both sub-goals for Long)."""
    if m["suite"] == "libero_10":
        return [w.split()[1] for w in m["watch"]]
    return [m["target_object"]]


def outcome(m):
    """Coarse label from pose-derived signals; videos remain the reference."""
    if m["success"]:
        return "success"
    if m["suite"] == "libero_10":
        final = {w.split()[1]: v for w, v in zip(m["watch"], m["watch_final"])}
        undo = m.get("undo") or {}
        if undo.get("applied"):
            return (
                "returned_subgoal_not_redone"
                if not final[undo["object"]]
                else "other_subgoal_missing"
            )
        return "remaining_subgoal_not_done" if any(final.values()) else "no_subgoal_done"
    lifted = m.get("first_lifted")
    if lifted is None:
        return "no_lift"
    if lifted != m["target_object"]:
        return "wrong_object_first"
    return "target_lifted_not_placed"


def load(results):
    rows = []
    for path in sorted(Path(results).glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m.get("status") != "completed":
            rows.append(
                {
                    "episode_id": m.get("episode_id", path.parent.name),
                    "status": m.get("status"),
                    "reason": m.get("reason"),
                }
            )
            continue
        watch = dict(zip(m["watch"], m["watch_final"]))
        rows.append(
            {
                "episode_id": m["episode_id"],
                "suite": m["suite"],
                "condition": m["condition"],
                "task_id": int(m["episode_id"].split("-t")[1][:2]),
                "init_state_id": int(m["episode_id"].split("-s")[-1]),
                "status": "completed",
                "success": m["success"],
                "steps": m["steps"],
                "prompt": m["prompt"],
                "target_object": m["target_object"],
                "first_lifted": m.get("first_lifted"),
                "outcome": outcome(m),
                "wrong_first_lift": m.get("first_lifted") not in (None, *goal_objects(m)),
                "watch_final": watch,
                "undo": m.get("undo", {}),
                "wall_clock_seconds": m.get("wall_clock_seconds"),
            }
        )
    return rows


def summarize(rows):
    done = [r for r in rows if r["status"] == "completed"]
    groups = defaultdict(list)
    for r in done:
        groups[(r["suite"], r["condition"])].append(r)
    table = []
    for suite, spec in SUITES.items():
        for condition in spec["conditions"]:
            g = groups.get((suite, condition), [])
            k = sum(r["success"] for r in g)
            labels = defaultdict(int)
            for r in g:
                labels[r["outcome"]] += 1
            row = {
                "suite": suite,
                "condition": spec["short"] + "-" + condition,
                "n": len(g),
                "successes": k,
                "rate": round(k / len(g), 4) if g else None,
                "wilson95": wilson(k, len(g)),
                "outcomes": dict(labels),
                "wrong_object_lifted_first": sum(r["wrong_first_lift"] for r in g),
                "question": QUESTIONS.get(spec["short"] + "-" + condition, "official baseline"),
            }
            if condition != "E0" and g:
                base = {
                    (r["task_id"], r["init_state_id"]): r["success"]
                    for r in groups.get((suite, "E0"), [])
                }
                pairs = [
                    (base[(r["task_id"], r["init_state_id"])], r["success"])
                    for r in g
                    if (r["task_id"], r["init_state_id"]) in base
                ]
                b = sum(1 for e, c in pairs if e and not c)
                c = sum(1 for e, c in pairs if c and not e)
                row["paired_vs_E0"] = {
                    "pairs": len(pairs),
                    "e0_only": b,
                    "condition_only": c,
                    "exact_mcnemar_p": round(exact_mcnemar(b, c), 5),
                }
            table.append(row)
    excluded = [r for r in rows if r["status"] != "completed"]
    return {"conditions": table, "excluded": excluded, "episodes": done}


def markdown(summary):
    lines = [
        "| Condition | n | Success | Wilson 95% | Paired vs E0 (E0-only / cond-only, p) "
        "| Wrong object lifted first | Outcomes |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in summary["conditions"]:
        if not row["n"]:
            continue
        lo, hi = row["wilson95"]
        paired = row.get("paired_vs_E0")
        paired_text = (
            f"{paired['e0_only']} / {paired['condition_only']}, p={paired['exact_mcnemar_p']}"
            if paired
            else "—"
        )
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(row["outcomes"].items()))
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['successes']}/{row['n']} | "
            f"[{lo:.2f}, {hi:.2f}] | {paired_text} | {row['wrong_object_lifted_first']} "
            f"| {outcomes} |"
        )
    lines += [
        "",
        "| Episode | Prompt | Target | Success | Steps | First lifted | Outcome |",
        "|---|---|---|---|---:|---|---|",
    ]
    for r in summary["episodes"]:
        lines.append(
            f"| {r['episode_id']} | {r['prompt']} | {r['target_object']} | {r['success']} | "
            f"{r['steps']} | {r['first_lifted']} | {r['outcome']} |"
        )
    if summary["excluded"]:
        lines += ["", "Excluded before policy start:"]
        lines += [f"- {r['episode_id']}: {r.get('reason')}" for r in summary["excluded"]]
    return "\n".join(lines) + "\n"


def main():
    results = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else results
    summary = summarize(load(results))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
