"""Combine immutable Part 3 report snapshots without conflating source contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

LEGACY_CONDITIONS = ("S1", "T1", "T1-ctrl")
CURRENT_CONDITIONS = ("S2", "S2-ctrl")


def merge_reports(legacy: dict, current: dict) -> dict:
    if legacy["baseline_context"] != current["baseline_context"]:
        raise ValueError("Reports do not share the frozen E0 baseline context")
    result = deepcopy(current)
    for condition in LEGACY_CONDITIONS:
        item = legacy.get("conditions", {}).get(condition)
        if item is None or item["summary"]["completed"] != item["summary"]["planned"]:
            raise ValueError("Legacy condition is absent or incomplete: " + condition)
        result["conditions"][condition] = item
    for condition in CURRENT_CONDITIONS:
        item = current.get("conditions", {}).get(condition)
        if item is None:
            raise ValueError("Current report is missing condition: " + condition)
        if item["summary"]["completed"] != item["summary"]["planned"]:
            raise ValueError("Current condition is absent or incomplete: " + condition)
    if not legacy.get("temporal_pairs"):
        raise ValueError("Legacy report has no temporal pair diagnostics")
    result["temporal_pairs"] = legacy["temporal_pairs"]
    result["manual_review_queue"] = legacy.get("manual_review_queue", []) + current.get(
        "manual_review_queue", []
    )
    result["limitations"] = list(
        dict.fromkeys(legacy.get("limitations", []) + current.get("limitations", []))
    )
    result["combined_source_contexts"] = {
        condition: item["summary"]["context"] for condition, item in result["conditions"].items()
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path, help="Frozen report from the original Part 3 source")
    parser.add_argument("current", type=Path, help="Report from the tolerant S2 implementation")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    legacy_bytes = args.legacy.read_bytes()
    current_bytes = args.current.read_bytes()
    result = merge_reports(json.loads(legacy_bytes), json.loads(current_bytes))
    result["source_report_sha256"] = {
        "legacy": hashlib.sha256(legacy_bytes).hexdigest(),
        "current": hashlib.sha256(current_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
