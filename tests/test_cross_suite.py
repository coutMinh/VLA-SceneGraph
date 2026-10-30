"""CPU checks for the cross-suite manifest and report; synthetic metrics are not results."""

import json

from sgvla.cross_suite.report import exact_mcnemar, load, summarize, wilson
from sgvla.cross_suite.spec import STATES, SUITES, build


def test_manifest_is_complete_unique_and_paired():
    manifest = build(7)
    episodes = manifest["episodes"]
    expected = sum(len(s["tasks"]) * len(s["conditions"]) * len(STATES) for s in SUITES.values())
    assert len(episodes) == expected == 90
    assert len({e["episode_id"] for e in episodes}) == len(episodes)
    seeds = {}
    for e in episodes:
        key = (e["suite"], e["task_id"], e["init_state_id"])
        # Every condition of a suite reuses the same policy-noise seed for a task/state pair.
        assert seeds.setdefault(key, e["policy_seed"]) == e["policy_seed"]
        assert e["max_steps"] == SUITES[e["suite"]]["max_steps"]
    assert build(7)["sha256"] == manifest["sha256"] != build(8)["sha256"]


def test_move_conditions_name_declared_objects():
    for e in build(7)["episodes"]:
        if e["condition"] in ("move", "move-ctrl"):
            key = "move" if e["condition"] == "move" else "move_ctrl"
            assert e["moved_object"] == SUITES["libero_goal"][key][e["task_id"]]


def test_statistics():
    assert exact_mcnemar(0, 0) == 1.0
    assert abs(exact_mcnemar(0, 6) - 0.03125) < 1e-12
    low, high = wilson(9, 9)
    assert 0.65 < low < 0.71 and high == 1.0


def _metrics(eid, condition, success, lifted, target="a_1"):
    return {
        "status": "completed",
        "episode_id": eid,
        "suite": "libero_object",
        "condition": condition,
        "success": success,
        "steps": 10,
        "prompt": "p",
        "target_object": target,
        "first_lifted": lifted,
        "watch": ["in a_1 basket"],
        "watch_final": [success],
    }


def test_report_pairs_and_labels(tmp_path):
    rows = [
        _metrics("OBJ-E0-t00-s00", "E0", True, "a_1"),
        _metrics("OBJ-swap-t00-s00", "swap", False, "b_1"),
        _metrics("OBJ-swap-t00-s01", "swap", False, None),
    ]
    for m in rows:
        folder = tmp_path / m["episode_id"]
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(m))
    summary = summarize(load(tmp_path))
    swap = next(r for r in summary["conditions"] if r["condition"] == "OBJ-swap")
    assert swap["outcomes"] == {"wrong_object_first": 1, "no_lift": 1}
    assert swap["paired_vs_E0"] == {
        "pairs": 1,
        "e0_only": 1,
        "condition_only": 0,
        "exact_mcnemar_p": 1.0,
    }
