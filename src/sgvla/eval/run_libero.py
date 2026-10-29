"""Simulator boundary; imports run exclusively in the private Python 3.8 client env."""

from __future__ import annotations

import collections
import time
from pathlib import Path

from sgvla.eval.scenes import array_hash
from sgvla.io import digest


def run_control_loop(env, obs, prompt, client, official, config, trace):
    from sgvla.baselines.openpi.rollout import observation_element, validate_actions

    for _ in range(config["settling_steps"]):
        trace.capture(obs, prompt, 0, official.LIBERO_DUMMY_ACTION)
        obs, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
    plan, success, steps = collections.deque(), False, 0
    for _ in range(config["max_steps"]):
        element = observation_element(obs, prompt, official, config["resize_size"])
        if not plan:
            started = time.monotonic()
            chunk = validate_actions(client.infer(element)["actions"], config["replan_steps"])
            if chunk.shape != (50, 7):
                raise ValueError("Expected pi0_libero's 50x7 action chunk")
            trace.inference(element, chunk, time.monotonic() - started)
            plan.extend(chunk[: config["replan_steps"]])
        action = plan.popleft()
        trace.capture(obs, prompt, 1, action)
        obs, _, success, _ = env.step(action.tolist())
        steps += 1
        if success:
            break
    trace.capture(obs, prompt, 2)
    return {"success": bool(success), "steps": steps}


def load_task(runtime, entry, suite=None):
    from libero.libero import benchmark, get_libero_path

    suite = suite or benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(entry["task_id"])
    if task.name != entry["task_name"] or task.language != entry["prompt"]:
        raise ValueError("Runtime task order/language differs from verified manifest")
    initial_file = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if digest(initial_file) != entry["init_file_sha256"] or digest(bddl) != entry["bddl_sha256"]:
        raise ValueError("Runtime task sources differ from verified manifest")
    state = suite.get_task_init_states(entry["task_id"])[entry["init_state_id"]]
    if array_hash(state) != entry["initial_state_sha256"]:
        raise ValueError("Initial state differs from verified manifest")
    return task, state


def run_episode(runtime, entry, folder):
    import numpy as np

    from sgvla.baselines.openpi.rollout import official_module
    from sgvla.baselines.openpi.runner import connect
    from sgvla.baselines.openpi.seeded import SeededClient
    from sgvla.eval.trace import Trace

    config = runtime.config
    np.random.seed(config["seed"])
    official = official_module(runtime)
    task, state = load_task(runtime, entry)
    client = SeededClient(connect(runtime), entry["policy_seed"])
    env, prompt = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, config["seed"])
    trace = None
    try:
        env.reset()
        obs = env.set_init_state(state)
        trace = Trace(folder, env, official, config["resize_size"])
        result = run_control_loop(env, obs, prompt, client, official, config, trace)
        result.update(trace.save())
    finally:
        try:
            if trace is not None:
                trace.close()
        finally:
            env.close()
    result.update(
        status="completed",
        settling_steps=config["settling_steps"],
        replan_steps=config["replan_steps"],
        initial_state_sha256=array_hash(state),
        policy_seed=entry["policy_seed"],
        target_object=entry["target_object"],
        server_metadata=client.get_server_metadata(),
    )
    return result
