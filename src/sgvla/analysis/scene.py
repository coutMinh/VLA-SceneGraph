"""Source-verified LIBERO API boundary and declared geometric relation checks."""

from __future__ import annotations

from sgvla.eval.scenes import BOWL1, BOWL2, PLATE

RULES = {
    "near_target_max_m": 0.16,
    "near_other_min_m": 0.18,
    "near_height_max_m": 0.12,
    "between_projection_min": 0.15,
    "between_projection_max": 0.85,
    "between_perpendicular_max_m": 0.05,
    "history_steps": 40,
    "history_lift_m": 0.20,
    "post_history_steps": 10,
    "stability_drift_max_m": 0.005,
    "penetration_max_m": 0.003,
}
TABLE_TASKS = (0, 1, 2, 8)


def poses(env):
    import numpy as np

    core = env.env
    entities = {**core.objects_dict, **core.fixtures_dict}
    return {
        name: np.concatenate(
            (
                env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)],
                env.sim.data.body_xquat[env.sim.model.body_name2id(obj.root_body)],
            )
        ).copy()
        for name, obj in sorted(entities.items())
    }


def predicate(env, name, object_name, support):
    from libero.libero.envs.predicates import eval_predicate_fn

    states = env.env.object_states_dict
    return bool(eval_predicate_fn(name.lower(), states[object_name], states[support]))


def relation(env, row, target, *, original=False):
    """Return explicit measurements and uniqueness; never infer semantic certainty."""
    import numpy as np

    other = BOWL2 if target == BOWL1 else BOWL1
    geometry = poses(env)
    source = row["initial_relations"][target]
    fragment = row["official_instruction"] if original else row["s1_instruction"]
    measured = {}
    if "next to" in fragment:
        landmark = (
            "glazed_rim_porcelain_ramekin_1"
            if "next to the ramekin" in fragment
            else PLATE
            if "next to the plate" in fragment
            else "cookies_1"
        )
        distances = {
            name: float(np.linalg.norm(geometry[name][:2] - geometry[landmark][:2]))
            for name in (target, other)
        }
        dz = float(abs(geometry[target][2] - geometry[landmark][2]))
        selected = distances[target] <= RULES["near_target_max_m"]
        unique = selected and distances[other] >= RULES["near_other_min_m"]
        unique = unique and dz <= RULES["near_height_max_m"]
        measured.update(landmark=landmark, distances_xy_m=distances, target_height_delta_m=dz)
    elif "between" in fragment:
        a, b = geometry[PLATE][:2], geometry["glazed_rim_porcelain_ramekin_1"][:2]
        segment = b - a
        checks = {}
        for name in (target, other):
            t = float(np.dot(geometry[name][:2] - a, segment) / np.dot(segment, segment))
            d = float(np.linalg.norm(geometry[name][:2] - (a + t * segment)))
            checks[name] = {
                "projection": t,
                "perpendicular_m": d,
                "holds": RULES["between_projection_min"] <= t <= RULES["between_projection_max"]
                and d <= RULES["between_perpendicular_max_m"],
            }
        selected = checks[target]["holds"]
        unique = selected and not checks[other]["holds"]
        measured["between"] = checks
    else:
        selected = predicate(env, source[0], target, source[2])
        competing = predicate(env, source[0], other, source[2])
        unique = selected and not competing
        measured.update(source_predicate=source, target_holds=selected, other_holds=competing)
    return {"eligible": bool(unique), "target_relation_holds": bool(selected), **measured}


def settle(env, obs, dummy, steps=10, trace=None, prompt="", phase=0):
    for _ in range(steps):
        if trace is not None:
            trace.capture(obs, prompt, phase, dummy)
        obs, _, _, _ = env.step(dummy)
    return obs
