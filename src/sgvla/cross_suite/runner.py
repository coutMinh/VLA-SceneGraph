"""Resumable cross-suite rollouts: official action loop, declared interventions, full traces."""

from __future__ import annotations

import collections
import os
import time

from sgvla.cross_suite.spec import IDLE_PRELUDE, RULES
from sgvla.io import digest, utc_now, write_json


def load(official, suite_name, entry, seed):
    from libero.libero import benchmark, get_libero_path

    from sgvla.eval.scenes import array_hash

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(entry["task_id"])
    state = suite.get_task_init_states(entry["task_id"])[entry["init_state_id"]]
    bddl = get_libero_path("bddl_files") + "/" + task.problem_folder + "/" + task.bddl_file
    env, prompt = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, seed)
    source = {
        "task_name": task.name,
        "official_prompt": prompt,
        "bddl_sha256": digest(bddl),
        "initial_state_sha256": array_hash(state),
    }
    return env, state, source


def watched(env, entry, extra):
    goal = [list(s) for s in env.env.parsed_problem["goal_state"]]
    if entry["suite"] == "libero_object":
        basket = goal[0][2]
        for name in extra.get("watch_objects", []):
            goal.append(["in", name, basket])
    return goal


def lift_summary(trace, start_frame, names):
    import numpy as np

    poses = np.asarray(trace.poses)
    index = {n: i for i, n in enumerate(trace.names)}
    first, rise = None, {}
    for name in names:
        z = poses[start_frame:, index[name], 2] - poses[start_frame, index[name], 2]
        rise[name] = float(z.max()) if len(z) else 0.0
        above = np.nonzero(z > RULES["lift_threshold_m"])[0]
        if len(above) and (first is None or above[0] < first[1]):
            first = (name, int(above[0]))
    return {
        "max_rise_m": rise,
        "first_lifted": first[0] if first else None,
        "first_lifted_policy_step": first[1] if first else None,
    }


def run_episode(runtime, entry, folder, dry=False):
    import numpy as np

    from sgvla.analysis.interventions import reset_controller
    from sgvla.baselines.openpi.rollout import (
        observation_element,
        official_module,
        validate_actions,
    )
    from sgvla.baselines.openpi.runner import connect
    from sgvla.baselines.openpi.seeded import SeededClient
    from sgvla.cross_suite import sim

    config = runtime.config
    np.random.seed(config["seed"])
    official = official_module(runtime)
    env, state, source = load(official, entry["suite"], entry, config["seed"])
    dummy = official.LIBERO_DUMMY_ACTION
    trace = None
    try:
        env.reset()
        obs = env.set_init_state(state)
        condition, prompt = entry["condition"], source["official_prompt"]
        intervention = {"kind": "none"}
        extra = {}
        target = sim.goal_objects(env)[0]
        if entry["suite"] == "libero_object":
            decoy = sim.initial_region_object(env, "other_object_region_0")
            extra["watch_objects"] = [decoy]
            if condition == "lang":
                prompt = "pick up the " + sim.words(decoy) + " and place it in the basket"
                target = decoy
        trace = sim.GoalTrace(
            folder, env, official, config["resize_size"], watched(env, entry, extra)
        )
        obs = sim.idle(env, obs, dummy, IDLE_PRELUDE["settle_steps"], trace, prompt, 0)
        before = sim.poses(env)
        changed = []
        try:
            if condition == "swap":
                intervention = sim.swap_xy(env, target, decoy)
                changed = [target, decoy]
            elif condition == "swap-ctrl":
                a = sim.initial_region_object(env, "other_object_region_1")
                b = sim.initial_region_object(env, "other_object_region_2")
                intervention = sim.swap_xy(env, a, b)
                changed = [a, b]
            elif condition in ("move", "move-ctrl"):
                intervention = sim.relocate(env, entry["moved_object"])
                changed = [entry["moved_object"]]
            elif condition == "pre":
                intervention = sim.place_in_basket(env, sim.goal_objects(env)[0], dummy)
                changed = [intervention["object"]]
        except ValueError as error:  # declared pre-policy exclusion, e.g. no free placement
            return {"status": "excluded", "reason": str(error), "condition": condition}
        wanted = sim.poses(env)
        if changed:
            obs = env.regenerate_obs_from_state(env.get_sim_state())
            reset_controller(env)
        obs = sim.idle(env, obs, dummy, IDLE_PRELUDE["post_intervention_steps"], trace, prompt, 5)
        check = sim.stability(env, before, wanted, changed)
        intervention["stability"] = check
        if changed and condition != "pre" and not check["stable"]:
            write_json(folder / "intervention.json", intervention)
            return {
                "status": "excluded",
                "reason": "intervention not stable before policy",
                "intervention": intervention,
            }
        goal = [list(s) for s in env.env.parsed_problem["goal_state"]]
        if condition == "pre" and not sim.eval_state(env, goal[0]):
            return {"status": "excluded", "reason": "pre-placed object left basket"}
        if dry:
            return {
                "status": "dry",
                "episode_id": entry["episode_id"],
                "prompt": prompt,
                "target_object": target,
                "intervention": intervention,
                "source": source,
            }
        start_frame = len(trace.state)
        start_qpos = {n: sim.qpos(env, n) for n in sim.movable(env)}
        # S1-style scoring: the instructed object, not the official goal, defines success.
        scored = ["in", target, goal[0][2]] if condition == "lang" else None
        undo = {"applied": False}
        hold = collections.Counter()
        client = SeededClient(connect(runtime), entry["policy_seed"])
        plan, success, steps = collections.deque(), False, 0
        for _ in range(entry["max_steps"]):
            element = observation_element(obs, prompt, official, config["resize_size"])
            if not plan:
                started = time.monotonic()
                chunk = validate_actions(client.infer(element)["actions"], config["replan_steps"])
                trace.inference(element, chunk, time.monotonic() - started)
                plan.extend(chunk[: config["replan_steps"]])
            action = plan.popleft()
            trace.capture(obs, prompt, 1, action)
            obs, _, success, _ = env.step(action.tolist())
            if scored:
                success = sim.eval_state(env, scored)
            steps += 1
            if condition == "undo" and not undo["applied"]:
                for g in goal:
                    hold[g[1]] = hold[g[1]] + 1 if sim.eval_state(env, g) else 0
                ready = [g[1] for g in goal if hold[g[1]] >= RULES["undo_hold_steps"]]
                if ready and not success:
                    name = ready[0]
                    sim.set_pose(env, name, start_qpos[name])
                    env.sim.forward()
                    obs = env.regenerate_obs_from_state(env.get_sim_state())
                    undo = {
                        "applied": True,
                        "object": name,
                        "policy_step": steps,
                        "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                        "restored_qpos": start_qpos[name].tolist(),
                    }
            if success:
                break
        trace.capture(obs, prompt, 2)
        intervention["undo"] = undo
        write_json(folder / "intervention.json", intervention)
        result = {"success": bool(success), "steps": steps}
        result.update(trace.save())
        values = np.asarray(trace.placed, dtype=bool)[start_frame:]
        result["watch"] = [" ".join(s) for s in trace.watch]
        result["watch_final"] = values[-1].tolist()
        result["watch_first_true_step"] = [
            int(np.nonzero(values[:, i])[0][0]) if values[:, i].any() else None
            for i in range(values.shape[1])
        ]
        result.update(lift_summary(trace, start_frame, sim.movable(env)))
        result.update(
            status="completed",
            episode_id=entry["episode_id"],
            suite=entry["suite"],
            condition=condition,
            prompt=prompt,
            target_object=target,
            policy_seed=entry["policy_seed"],
            policy_start_frame=start_frame,
            max_steps=entry["max_steps"],
            intervention_kind=intervention["kind"],
            undo=undo,
            source=source,
            server_metadata=client.get_server_metadata(),
        )
        return result
    finally:
        if trace is not None:
            trace.close()
        env.close()


def run_manifest(runtime, manifest, suite=None, only=None, dry=False):
    # Runtime folder keeps the original name so existing molab runs stay resumable.
    root = runtime.results / ("xsuite-dry" if dry else "xsuite") / manifest["sha256"][:16]
    write_json(root / "manifest.json", manifest)
    for entry in manifest["episodes"]:
        if (suite and entry["suite"] != suite) or (only and entry["episode_id"] not in only):
            continue
        folder = root / entry["episode_id"]
        if (folder / "complete.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        try:  # atomic claim, so several workers can share one suite
            os.close(os.open(str(folder / "claim"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        started = time.monotonic()
        try:
            result = run_episode(runtime, entry, folder, dry)
        except Exception as error:  # infrastructure errors never count as task failures
            import traceback

            write_json(
                folder / ("error-" + str(int(time.time())) + ".json"),
                {"error": repr(error), "traceback": traceback.format_exc(), "at": utc_now()},
            )
            print(entry["episode_id"], "ERROR", repr(error), flush=True)
            (folder / "claim").unlink()  # retried by the next pass
            continue
        result["wall_clock_seconds"] = time.monotonic() - started
        write_json(folder / "metrics.json", result)
        write_json(folder / "complete.json", {"at": utc_now(), "status": result["status"]})
        print(
            entry["episode_id"],
            result["status"],
            result.get("success"),
            result.get("steps"),
            result.get("first_lifted"),
            flush=True,
        )
    return root


def main():
    import argparse
    from pathlib import Path

    from sgvla.cross_suite.spec import build
    from sgvla.runtime import Runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--dry", action="store_true", help="interventions only, no policy")
    args = parser.parse_args()
    runtime = Runtime(args.work.resolve(), args.config.resolve())
    os.environ.update(runtime.env())
    manifest = build(runtime.config["seed"])
    root = run_manifest(runtime, manifest, args.suite, args.only, args.dry)
    print("done", root, flush=True)


if __name__ == "__main__":
    main()
