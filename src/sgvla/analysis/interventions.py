"""Explicit scripted state interventions with recorded, reproducible endpoints."""

from __future__ import annotations

import math

from sgvla.analysis.scene import RULES, poses, settle
from sgvla.eval.scenes import BOWL1, BOWL2


def reset_controller(env):
    controller = env.env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()


def joints(env):
    result = []
    for name in (BOWL1, BOWL2):
        names = env.env.objects_dict[name].joints
        if len(names) != 1:
            raise ValueError("Expected a single free joint per bowl")
        joint = names[0]
        q = env.sim.data.get_joint_qpos(joint).copy()
        v = env.sim.data.get_joint_qvel(joint).copy()
        if q.shape != (7,) or v.shape != (6,):
            raise ValueError("Expected a 7-qpos / 6-qvel free joint")
        result.append(joint)
    return result


def permuted_state(env, state, swap):
    """Preserve all non-bowl state and exchange complete bowl qpos/qvel only."""
    env.set_state(state)
    pair = joints(env)
    q = [env.sim.data.get_joint_qpos(j).copy() for j in pair]
    v = [env.sim.data.get_joint_qvel(j).copy() for j in pair]
    if swap:
        for i, joint in enumerate(pair):
            env.sim.data.set_joint_qpos(joint, q[1 - i])
            env.sim.data.set_joint_qvel(joint, v[1 - i])
    env.sim.forward()
    return env.get_sim_state().copy()


def bowl_penetration(env):
    ids = set()
    for name in (BOWL1, BOWL2):
        ids.update(env.sim.model.geom_name2id(g) for g in env.env.objects_dict[name].contact_geoms)
    return max(
        [0.0]
        + [
            max(0.0, -float(env.sim.data.contact[i].dist))
            for i in range(env.sim.data.ncon)
            if env.sim.data.contact[i].geom1 in ids or env.sim.data.contact[i].geom2 in ids
        ]
    )


def temporal_prelude(env, obs, official, prompt, swap, trace=None):
    import numpy as np

    base = env.get_sim_state().copy()
    pair = joints(env)
    original = [env.sim.data.get_joint_qpos(j).copy() for j in pair]
    maximum_penetration = 0.0
    overrides = []
    count = RULES["history_steps"]
    for k in range(count):
        if swap:
            u = (k + 1) / count
            lift = min(u / 0.25, (1 - u) / 0.25, 1.0)
            angle = math.pi * min(1.0, max(0.0, (u - 0.25) / 0.5))
            center = (original[0][:2] + original[1][:2]) / 2
            rotation = np.asarray(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            )
            commanded = []
            for i, joint in enumerate(pair):
                q = original[i].copy()
                q[:2] = center + rotation.dot(original[i][:2] - center)
                q[2] += RULES["history_lift_m"] * max(0.0, lift)
                if u == 1.0:
                    q = original[1 - i].copy()
                env.sim.data.set_joint_qpos(joint, q)
                env.sim.data.set_joint_qvel(joint, np.zeros(6))
                commanded.append(q.tolist())
            obs = env.regenerate_obs_from_state(env.get_sim_state())
            maximum_penetration = max(maximum_penetration, bowl_penetration(env))
            overrides.append({"history_step": k, "qpos": commanded})
        if trace is not None:
            trace.capture(obs, prompt, 3, official.LIBERO_DUMMY_ACTION)
        obs, _, _, _ = env.step(official.LIBERO_DUMMY_ACTION)
    # Use common non-bowl state, not any robot disturbance from scripted motion.
    # The time is advanced equally; the complete endpoint assignment is recorded.
    base[0] = env.sim.data.time
    endpoint = permuted_state(env, base, swap)
    obs = env.regenerate_obs_from_state(endpoint)
    reset_controller(env)
    wanted = poses(env)
    obs = settle(
        env, obs, official.LIBERO_DUMMY_ACTION, RULES["post_history_steps"], trace, prompt, 4
    )
    actual = poses(env)
    drift = max(float(np.linalg.norm(actual[n][:3] - wanted[n][:3])) for n in (BOWL1, BOWL2))
    maximum_penetration = max(maximum_penetration, bowl_penetration(env))
    endpoint[0] = env.sim.data.time
    obs = env.regenerate_obs_from_state(endpoint)
    reset_controller(env)
    return obs, {
        "swap": swap,
        "history_steps": count,
        "post_history_steps": RULES["post_history_steps"],
        "bowl_drift_m": drift,
        "maximum_bowl_penetration_m": maximum_penetration,
        "stable": drift <= RULES["stability_drift_max_m"]
        and maximum_penetration <= RULES["penetration_max_m"],
        "joint_overrides": overrides,
        "endpoint_assignment": "common settled full state; exchange bowl qpos/qvel only",
        "endpoint_reset_after_stability_check": True,
        "controller_goal_reset_to_endpoint": True,
        "endpoint_sim_state": endpoint.tolist(),
    }
