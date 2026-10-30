"""Reuse the official action loop with identity-specific stopping and logged preludes."""

from __future__ import annotations

from sgvla.analysis.interventions import temporal_prelude
from sgvla.analysis.outcomes import measure_trace
from sgvla.analysis.scene import predicate, settle
from sgvla.eval.scenes import PLATE, array_hash
from sgvla.eval.trace import Trace, compare_observations, observation_hash
from sgvla.io import write_json


class TargetEnv:
    def __init__(self, env, target):
        self.raw, self.target = env, target

    def step(self, action):
        obs, reward, _original_done, info = self.raw.step(action)
        return obs, reward, predicate(self.raw, "on", self.target, PLATE), info


def run_episode(runtime, entry, folder):
    import numpy as np

    from sgvla.baselines.openpi.rollout import observation_element, official_module
    from sgvla.baselines.openpi.runner import connect
    from sgvla.baselines.openpi.seeded import SeededClient
    from sgvla.eval.run_libero import load_task, run_control_loop

    np.random.seed(runtime.config["seed"])
    official = official_module(runtime)
    task, state = load_task(runtime, {**entry, "prompt": entry["original_prompt"]})
    env, _ = official._get_libero_env(task, official.LIBERO_ENV_RESOLUTION, runtime.config["seed"])
    trace = None
    try:
        env.reset()
        obs = env.set_init_state(state)
        prompt = entry["prompt"]
        trace = Trace(folder, env, official, runtime.config["resize_size"])
        config = runtime.config.copy()
        intervention = {"kind": "none", "condition": entry["condition"]}
        endpoint_check = {"applicable": False}
        if entry["condition"].startswith("T1"):
            obs = settle(env, obs, official.LIBERO_DUMMY_ACTION, 10, trace, prompt)
            obs, intervention = temporal_prelude(
                env, obs, official, prompt, entry["condition"] == "T1", trace
            )
            if not intervention["stable"]:
                raise ValueError("Temporal intervention no longer stable")
            element = observation_element(obs, prompt, official, config["resize_size"])
            actual_hash = observation_hash(element)
            endpoint_check = {
                "applicable": True,
                "mode": "exact_sha256",
                "expected_sha256": entry["endpoint_sha256"],
                "actual_sha256": actual_hash,
                "matches": actual_hash == entry["endpoint_sha256"],
            }
            if not endpoint_check["matches"]:
                raise ValueError("Policy-start input differs from verified endpoint")
            config["settling_steps"] = 0
        elif entry["condition"].startswith("S2"):
            from sgvla.analysis.landmarks import apply_landmark

            obs = settle(env, obs, official.LIBERO_DUMMY_ACTION, 10, trace, prompt)
            obs, intervention = apply_landmark(
                env, obs, official, prompt, entry["landmark_intervention"], trace
            )
            if not intervention["stable"]:
                raise ValueError("Landmark intervention no longer stable")
            element = observation_element(obs, prompt, official, config["resize_size"])
            import numpy as np

            reference_path = runtime.work / entry["endpoint_reference_path"]
            with np.load(reference_path, allow_pickle=False) as saved:
                reference = {key: saved[key].copy() for key in saved.files}
            endpoint_check = compare_observations(reference, element)
            endpoint_check.update(
                applicable=True,
                mode="tolerant_observation",
                expected_sha256=entry["endpoint_sha256"],
                reference_path=entry["endpoint_reference_path"],
            )
            if not endpoint_check["matches"]:
                write_json(folder / "endpoint_check.json", endpoint_check)
                raise ValueError("Landmark policy input exceeds verified endpoint tolerances")
            config["settling_steps"] = 0
        write_json(folder / "endpoint_check.json", endpoint_check)
        intervention["endpoint_check"] = endpoint_check
        write_json(folder / "intervention.json", intervention)
        client = SeededClient(connect(runtime), entry["policy_seed"])
        result = run_control_loop(
            TargetEnv(env, entry["target_object"]), obs, prompt, client, official, config, trace
        )
        result.update(trace.save())
        result.update(
            status="completed",
            condition=entry["condition"],
            prompt=prompt,
            settling_steps=10,
            history_steps=intervention.get("history_steps", 0),
            post_history_steps=intervention.get("post_history_steps", 0),
            landmark_settling_steps=intervention.get("landmark_settling_steps", 0),
            replan_steps=config["replan_steps"],
            initial_state_sha256=array_hash(state),
            policy_seed=entry["policy_seed"],
            target_object=entry["target_object"],
            endpoint_check=endpoint_check,
            server_metadata=client.get_server_metadata(),
        )
        result.update(measure_trace(folder / "trajectory.npz", entry["target_object"]))
        result["phase_labels"].update(
            {"3": "scripted_history", "4": "history_stability_check", "5": "landmark_settling"}
        )
        return result
    finally:
        if trace is not None:
            trace.close()
        env.close()
