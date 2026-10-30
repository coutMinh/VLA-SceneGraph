"""Verify optional S2 landmark-only moves while preserving bowls and destination."""

from __future__ import annotations

import math

from sgvla.analysis.interventions import reset_controller
from sgvla.analysis.scene import RULES, poses, relation, settle
from sgvla.eval.scenes import BOWL1, BOWL2, PLATE
from sgvla.eval.trace import observation_hash
from sgvla.io import digest, read_json, write_json

LANDMARK_TASKS = {1: "glazed_rim_porcelain_ramekin_1", 6: "cookies_1"}
LANDMARK_SETTLING_STEPS = 20


def penetration(env, name):
    ids = {env.sim.model.geom_name2id(g) for g in env.env.objects_dict[name].contact_geoms}
    return max(
        [0.0]
        + [
            max(0.0, -float(env.sim.data.contact[i].dist))
            for i in range(env.sim.data.ncon)
            if env.sim.data.contact[i].geom1 in ids or env.sim.data.contact[i].geom2 in ids
        ]
    )


def apply_landmark(env, obs, official, prompt, recipe, trace=None):
    import numpy as np

    name = recipe["landmark"]
    names = env.env.objects_dict[name].joints
    if len(names) != 1 or env.sim.data.get_joint_qpos(names[0]).shape != (7,):
        raise ValueError("Landmark is not a free-joint movable object")
    before = poses(env)
    env.sim.data.set_joint_qpos(names[0], np.asarray(recipe["qpos"]))
    env.sim.data.set_joint_qvel(names[0], np.zeros(6))
    obs = env.regenerate_obs_from_state(env.get_sim_state())
    reset_controller(env)
    wanted = poses(env)
    overlap = penetration(env, name)
    obs = settle(env, obs, official.LIBERO_DUMMY_ACTION, LANDMARK_SETTLING_STEPS, trace, prompt, 5)
    after = poses(env)
    landmark_drift = float(np.linalg.norm(after[name][:3] - wanted[name][:3]))
    other_drift = max(
        float(np.linalg.norm(after[n][:3] - before[n][:3])) for n in before if n != name
    )
    plate_drift = float(np.linalg.norm(after[PLATE][:3] - before[PLATE][:3]))
    overlap = max(overlap, penetration(env, name))
    return obs, {
        "kind": "landmark_pose_assignment",
        "recipe": recipe,
        "landmark_settling_steps": LANDMARK_SETTLING_STEPS,
        "landmark_drift_m": landmark_drift,
        "other_object_drift_m": other_drift,
        "plate_drift_m": plate_drift,
        "maximum_landmark_penetration_m": overlap,
        "stable": landmark_drift <= RULES["stability_drift_max_m"]
        and other_drift <= RULES["stability_drift_max_m"]
        and plate_drift <= 0.002
        and overlap <= RULES["penetration_max_m"],
    }


def verify_landmarks(runtime, verification, output):
    import numpy as np

    from sgvla.baselines.openpi.rollout import observation_element, official_module
    from sgvla.eval.run_libero import load_task

    original = read_json(runtime.work / "configs/e0_manifest.json")
    entries = {e["episode_id"]: e for e in original["episodes"]}
    source = read_json(runtime.work / "docs/scene_report.json")
    checks = read_json(verification)
    official = official_module(runtime)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in checks["rows"]:
        record = {
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "init_state_id": row["init_state_id"],
            "eligible": False,
        }
        if row["task_id"] not in LANDMARK_TASKS:
            record["exclusion_reason"] = "destination_or_support_fixture_confound_not_redesigned"
        elif not row["original_relation"]["eligible"]:
            record["exclusion_reason"] = "original_relation_not_unique"
        else:
            entry = entries[row["episode_id"]]
            np.random.seed(runtime.config["seed"])
            task, state = load_task(runtime, entry)
            env, prompt = official._get_libero_env(
                task, official.LIBERO_ENV_RESOLUTION, runtime.config["seed"]
            )
            try:
                env.reset()
                obs = env.set_init_state(state)
                obs = settle(env, obs, official.LIBERO_DUMMY_ACTION)
                base, geometry = env.get_sim_state().copy(), poses(env)
                name = LANDMARK_TASKS[row["task_id"]]
                joint = env.env.objects_dict[name].joints[0]
                old = env.sim.data.get_joint_qpos(joint).copy()
                selections, attempts = {}, {}
                for condition, target in (("S2", BOWL2), ("S2-ctrl", BOWL1)):
                    failures = []
                    for radius in (0.13, 0.15, 0.11):
                        if condition in selections:
                            break
                        for k in range(12):
                            q = old.copy()
                            q[:2] = geometry[target][:2] + radius * np.array(
                                [math.cos(2 * math.pi * k / 12), math.sin(2 * math.pi * k / 12)]
                            )
                            displacement = float(np.linalg.norm(q[:2] - old[:2]))
                            if displacement < 0.04:
                                continue
                            recipe = {
                                "landmark": name,
                                "qpos": q.tolist(),
                                "displacement_m": displacement,
                                "radius_m": radius,
                                "search_index": k,
                            }
                            obs = env.regenerate_obs_from_state(base)
                            obs, stability = apply_landmark(env, obs, official, prompt, recipe)
                            check = relation(
                                env, source["tasks"][row["task_id"]], target, original=True
                            )
                            if stability["stable"] and check["eligible"]:
                                element = observation_element(
                                    obs, prompt, official, runtime.config["resize_size"]
                                )
                                selections[condition] = {
                                    "recipe": recipe,
                                    "stability": stability,
                                    "relation": check,
                                    "endpoint_sha256": observation_hash(element),
                                }
                                np.savez_compressed(
                                    output / (row["episode_id"] + "-" + condition + ".npz"),
                                    **element,
                                )
                                break
                            failures.append(
                                {"recipe": recipe, "stability": stability, "relation": check}
                            )
                    attempts[condition] = failures
                record.update(
                    eligible=len(selections) == 2,
                    selections=selections,
                    rejected_candidates=attempts,
                    exclusion_reason=None
                    if len(selections) == 2
                    else "no_stable_unique_paired_landmark_moves",
                )
                print(row["episode_id"], "S2", record["eligible"], flush=True)
            finally:
                env.close()
        results.append(record)
        write_json(
            output / "landmark_verification.json",
            {
                "schema": 1,
                "rules": RULES,
                "relation_verification_sha256": digest(verification),
                "complete": len(results) == 30,
                "examined": len(results),
                "eligible": sum(r["eligible"] for r in results),
                "rows": results,
                "control_note": (
                    "Same movable landmark and fixed destination; "
                    "displacement magnitudes recorded, not assumed equal"
                ),
            },
        )
    return read_json(output / "landmark_verification.json")
