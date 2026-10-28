"""D1 single-episode loop. D2's multi-episode evaluator is deliberately not started yet."""

from __future__ import annotations

import collections
import importlib.util
import time

from sgvla.io import digest, write_json


def official_module(runtime):
    path = runtime.upstream / "examples/libero/main.py"
    spec = importlib.util.spec_from_file_location("sgvla_official_libero", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspects sys.modules when annotations are postponed upstream.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observation_element(obs, prompt, official, resize_size):
    import numpy as np
    from openpi_client import image_tools

    images = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs[key][::-1, ::-1]), resize_size, resize_size
            )
        )
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]
    return {
        "observation/image": images[0],
        "observation/wrist_image": images[1],
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                official._quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(prompt),
    }


def validate_actions(actions, replan_steps):
    import numpy as np

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan_steps:
        raise ValueError("Expected an (H, 7) action chunk with H >= replan_steps")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite actions")
    return actions


def run_steps(env, obs, prompt, client, official, config, *, render_only=False):
    """Preserve official settling, preprocessing, replanning and step-limit ordering."""
    import numpy as np

    frames, wrist_frames, states, actions, latencies = [], [], [], [], []
    plan = collections.deque()
    done = False
    for _ in range(config["settling_steps"]):
        obs, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
    limit = 5 if render_only else config["max_steps"]
    for _ in range(limit):
        element = observation_element(obs, prompt, official, config["resize_size"])
        frames.append(element["observation/image"])
        wrist_frames.append(element["observation/wrist_image"])
        states.append(element["observation/state"])
        if render_only:
            action = np.asarray(official.LIBERO_DUMMY_ACTION)
        else:
            if not plan:
                started = time.monotonic()
                chunk = validate_actions(client.infer(element)["actions"], config["replan_steps"])
                latencies.append(time.monotonic() - started)
                plan.extend(chunk[: config["replan_steps"]])
            action = plan.popleft()
        actions.append(action)
        obs, _, done, _ = env.step(action.tolist())
        if done and not render_only:
            break
    return {
        "success": bool(done),
        "steps": len(actions),
        "frames": frames,
        "wrist_frames": wrist_frames,
        "states": states,
        "actions": actions,
        "policy_call_seconds": latencies,
    }


def rollout(runtime, folder, client=None):
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark, get_libero_path

    config = runtime.config
    official = official_module(runtime)
    np.random.seed(config["seed"])
    suite = benchmark.get_benchmark_dict()[config["task_suite"]]()
    task = suite.get_task(config["task_id"])
    initial_states = suite.get_task_init_states(config["task_id"])
    init_file = (get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    from pathlib import Path

    write_json(
        folder / "initial_state_manifest.json",
        {
            "suite": config["task_suite"],
            "task_id": config["task_id"],
            "init_state_id": config["init_state_id"],
            "init_file_sha256": digest(Path(*init_file)),
        },
    )
    env, prompt = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, config["seed"])
    try:
        env.reset()
        obs = env.set_init_state(initial_states[config["init_state_id"]])
        trace = run_steps(env, obs, prompt, client, official, config, render_only=client is None)
    finally:
        env.close()
    for key, name in (("frames", "agentview.mp4"), ("wrist_frames", "wrist.mp4")):
        path = folder / name
        imageio.mimwrite(path, trace[key], fps=10)
        # Saving a file is insufficient; decode every frame to reject truncated videos.
        with imageio.get_reader(path) as reader:
            count = sum(1 for _ in reader)
        if count != trace["steps"]:
            raise RuntimeError("Video frame count differs from completed control steps")
    np.savez_compressed(folder / "trajectory.npz", actions=trace["actions"], state=trace["states"])
    return {
        "status": "completed",
        "kind": "render_probe" if client is None else "smoke",
        "suite": config["task_suite"],
        "task_id": config["task_id"],
        "init_state_id": config["init_state_id"],
        "prompt": prompt,
        "success": trace["success"],
        "steps": trace["steps"],
        "settling_steps": config["settling_steps"],
        "policy_call_seconds": trace["policy_call_seconds"],
        "video": str(folder / "agentview.mp4"),
        "wrist_video": str(folder / "wrist.mp4"),
        "video_sha256": digest(folder / "agentview.mp4"),
        "wrist_video_sha256": digest(folder / "wrist.mp4"),
        "decoded_video_frames": count,
        "benchmark_estimate": False,
    }
