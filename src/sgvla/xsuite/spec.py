"""Predeclared cross-suite conditions. Pure Python: no simulator imports.

Every suite keeps the official instruction, step limit and paired policy noise;
conditions differ only in the declared pre-policy (or mid-episode) intervention.
"""

from __future__ import annotations

from sgvla.eval.manifest import fingerprint

STATES = (0, 1, 2)
IDLE_PRELUDE = {"settle_steps": 10, "post_intervention_steps": 20}
RULES = {
    "lift_threshold_m": 0.05,
    "stable_drift_max_m": 0.01,
    "stable_tilt_max_rad": 0.25,
    "penetration_max_m": 0.004,
    "move_radii_m": (0.12, 0.15, 0.10),
    "move_angles": 12,
    "move_clearance_m": 0.07,
    "workspace_margin_m": 0.03,
    "basket_drop_height_m": 0.12,
    "basket_settle_steps": 40,
    "undo_hold_steps": 20,
}

SUITES = {
    "libero_object": {
        "short": "OBJ",
        "max_steps": 280,
        "tasks": (0, 1, 4),
        "conditions": ("E0", "swap", "swap-ctrl", "lang"),
    },
    "libero_goal": {
        "short": "GOAL",
        "max_steps": 300,
        "tasks": (8, 6, 1),
        "conditions": ("E0", "move", "move-ctrl"),
        # Task-relevant object moved by GOAL-move; an irrelevant one by GOAL-move-ctrl.
        "move": {8: "plate_1", 6: "akita_black_bowl_1", 1: "akita_black_bowl_1"},
        "move_ctrl": {8: "cream_cheese_1", 6: "plate_1", 1: "cream_cheese_1"},
    },
    "libero_10": {
        "short": "LONG",
        "max_steps": 520,
        "tasks": (0, 1, 7),
        "conditions": ("E0", "pre", "undo"),
    },
}

QUESTIONS = {
    "OBJ-swap": "Target exchanges place with the object in the other canonical target slot."
    " Does pi0 fetch the named object, or whatever sits at the trained target position?",
    "OBJ-swap-ctrl": "Two distractors exchange places; target untouched (layout-novelty control).",
    "OBJ-lang": "Same scene; instruction names the object in the other canonical target slot."
    " Does language select the object (S1 analogue)?",
    "GOAL-move": "The task-relevant object is relocated before the policy starts."
    " Does pi0 act on the current position or the trained one (S2 analogue)?",
    "GOAL-move-ctrl": "An irrelevant object is relocated (novelty control).",
    "LONG-pre": "The first named sub-goal is already satisfied at policy start."
    " Is progress read from the scene?",
    "LONG-undo": "The first object placed in the basket is returned to its start pose after"
    " it has rested there; a policy without history sees an unfinished scene again.",
}


def episode_seed(seed, suite, task, state):
    return int(fingerprint(["xsuite-policy", seed, suite, task, state])[:8], 16)


def episode_id(suite, condition, task, state):
    return f"{SUITES[suite]['short']}-{condition}-t{task:02d}-s{state:02d}"


def build(seed, suites=None):
    episodes = []
    for suite in suites or SUITES:
        spec = SUITES[suite]
        for condition in spec["conditions"]:
            for task in spec["tasks"]:
                for state in STATES:
                    entry = {
                        "episode_id": episode_id(suite, condition, task, state),
                        "suite": suite,
                        "condition": condition,
                        "task_id": task,
                        "init_state_id": state,
                        "max_steps": spec["max_steps"],
                        "policy_seed": episode_seed(seed, suite, task, state),
                    }
                    if condition == "move":
                        entry["moved_object"] = spec["move"][task]
                    if condition == "move-ctrl":
                        entry["moved_object"] = spec["move_ctrl"][task]
                    episodes.append(entry)
    manifest = {
        "schema": 1,
        "kind": "xsuite_followup",
        "rules": RULES,
        "idle_prelude": IDLE_PRELUDE,
        "episodes": episodes,
    }
    manifest["sha256"] = fingerprint({k: v for k, v in manifest.items()})
    return manifest
