"""Simulator-side interventions and logging; runs only in the Python 3.8 client env."""

from __future__ import annotations

import math

from sgvla.analysis.landmarks import penetration
from sgvla.analysis.scene import poses
from sgvla.cross_suite.spec import RULES
from sgvla.eval.trace import Trace


def free_joint(env, name):
    joints = env.env.objects_dict[name].joints
    if len(joints) != 1 or env.sim.data.get_joint_qpos(joints[0]).shape != (7,):
        raise ValueError(name + " is not a free-joint object")
    return joints[0]


def movable(env):
    names = []
    for name, obj in sorted(env.env.objects_dict.items()):
        joints = obj.joints
        if len(joints) == 1 and env.sim.data.get_joint_qpos(joints[0]).shape == (7,):
            names.append(name)
    return names


def set_pose(env, name, qpos):
    import numpy as np

    joint = free_joint(env, name)
    env.sim.data.set_joint_qpos(joint, np.asarray(qpos, dtype=float))
    env.sim.data.set_joint_qvel(joint, np.zeros(6))


def qpos(env, name):
    return env.sim.data.get_joint_qpos(free_joint(env, name)).copy()


def eval_state(env, state):
    return bool(env.env._eval_predicate(state))


def tilt(q0, q1):
    import numpy as np

    dot = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return 2 * math.acos(min(1.0, dot))


def idle(env, obs, dummy, steps, trace, prompt, phase):
    for _ in range(steps):
        trace.capture(obs, prompt, phase, dummy)
        obs, _, _, _ = env.step(dummy)
    return obs


def stability(env, before, wanted, changed):
    """Drift after settling: moved objects vs. their assigned pose, others vs. before."""
    import numpy as np

    after = poses(env)
    report = {}
    for name in movable(env):
        reference = wanted[name] if name in changed else before[name]
        report[name] = {
            "drift_m": float(np.linalg.norm(after[name][:3] - reference[:3])),
            "tilt_rad": tilt(reference[3:], after[name][3:]),
        }
    worst = max(r["drift_m"] for r in report.values())
    worst_tilt = max(r["tilt_rad"] for r in report.values())
    return {
        "per_object": report,
        "max_drift_m": worst,
        "max_tilt_rad": worst_tilt,
        "stable": worst <= RULES["stable_drift_max_m"]
        and worst_tilt <= RULES["stable_tilt_max_rad"],
    }


def initial_region_object(env, region_suffix):
    for state in env.env.parsed_problem["initial_state"]:
        if len(state) == 3 and state[2].endswith(region_suffix):
            return state[1]
    raise ValueError("No object initialised in " + region_suffix)


def goal_objects(env):
    return [state[1] for state in env.env.parsed_problem["goal_state"] if len(state) == 3]


def words(name):
    return name.rsplit("_", 1)[0].replace("_", " ")


def swap_xy(env, a, b):
    qa, qb = qpos(env, a), qpos(env, b)
    na, nb = qa.copy(), qb.copy()
    na[:2], nb[:2] = qb[:2], qa[:2]
    set_pose(env, a, na)
    set_pose(env, b, nb)
    env.sim.forward()
    return {"kind": "swap_xy", "objects": [a, b], "qpos": {a: na.tolist(), b: nb.tolist()}}


def relocate(env, name):
    """Deterministic search for a free, in-workspace, non-penetrating position."""
    import numpy as np

    current = poses(env)
    others = [n for n in movable(env) if n != name]
    xy = np.asarray([current[n][:2] for n in movable(env)])
    low = xy.min(0) - RULES["workspace_margin_m"]
    high = xy.max(0) + RULES["workspace_margin_m"]
    origin = qpos(env, name)
    state = env.get_sim_state().copy()
    tried = []
    for radius in RULES["move_radii_m"]:
        for k in range(RULES["move_angles"]):
            angle = 2 * math.pi * k / RULES["move_angles"]
            q = origin.copy()
            q[0] += radius * math.cos(angle)
            q[1] += radius * math.sin(angle)
            clearance = min(float(np.linalg.norm(q[:2] - current[n][:2])) for n in others)
            inside = bool((q[:2] >= low).all() and (q[:2] <= high).all())
            record = {
                "radius_m": radius,
                "angle_rad": angle,
                "clearance_m": clearance,
                "inside": inside,
            }
            if inside and clearance >= RULES["move_clearance_m"]:
                set_pose(env, name, q)
                env.sim.forward()
                record["penetration_m"] = penetration(env, name)
                if record["penetration_m"] <= RULES["penetration_max_m"]:
                    tried.append(record)
                    return {
                        "kind": "relocate",
                        "object": name,
                        "qpos": q.tolist(),
                        "displacement_m": radius,
                        "angle_rad": angle,
                        "candidates": tried,
                    }
                env.set_state(state)
                env.sim.forward()
            tried.append(record)
    raise ValueError("No valid relocation for " + name)


def place_in_basket(env, name, dummy):
    """Drop `name` into the basket; accept the first offset where In() holds and rests."""
    import numpy as np

    basket = [n for n in env.env.objects_dict if n.startswith("basket")][0]
    region = basket + "_contain_region"
    state = env.get_sim_state().copy()
    base = poses(env)[basket]
    attempts = []
    for dx, dy in ((0, 0), (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)):
        q = qpos(env, name)
        q[0], q[1] = base[0] + dx, base[1] + dy
        q[2] = base[2] + RULES["basket_drop_height_m"]
        set_pose(env, name, q)
        env.sim.forward()
        for _ in range(RULES["basket_settle_steps"]):
            env.step(dummy)
        inside = eval_state(env, ["in", name, region])
        speed = float(np.linalg.norm(env.sim.data.get_joint_qvel(free_joint(env, name))[:3]))
        attempts.append({"offset_m": [dx, dy], "inside": inside, "speed_mps": speed})
        if inside and speed < 0.02:
            return {
                "kind": "place_in_basket",
                "object": name,
                "region": region,
                "qpos": qpos(env, name).tolist(),
                "attempts": attempts,
            }
        env.set_state(state)
        env.sim.forward()
    raise ValueError("Could not place " + name + " in " + region)


class GoalTrace(Trace):
    """Base Trace with per-frame values of the watched (goal) predicates."""

    def __init__(self, folder, env, official, resize_size, watch):
        super().__init__(folder, env, official, resize_size)
        self.watch = watch

    def capture(self, obs, prompt, phase, action=None):
        import numpy as np

        from sgvla.baselines.openpi.rollout import observation_element

        element = observation_element(obs, prompt, self.official, self.resize)
        for writer, key in zip(self.writers, ("observation/image", "observation/wrist_image")):
            writer.append_data(element[key])
        self.state.append(element["observation/state"].copy())
        data = self.env.sim.data
        self.poses.append(
            np.concatenate((data.body_xpos[self.bodies], data.body_xquat[self.bodies]), 1)
        )
        self.sim.append(self.env.get_sim_state().copy())
        self.placed.append([eval_state(self.env, s) for s in self.watch])
        self.phase.append(phase)
        if action is not None:
            self.actions.append(np.asarray(action).copy())
        return element

    def save(self):
        import numpy as np

        np.savez_compressed(
            self.folder / "trajectory.npz",
            actions=self.actions,
            state=self.state,
            object_names=np.asarray(self.names),
            object_poses=self.poses,
            sim_state=self.sim,
            phase=np.asarray(self.phase),
            watch=np.asarray([" ".join(s) for s in self.watch]),
            watch_values=np.asarray(self.placed, dtype=bool),
            policy_chunks=self.chunks,
            policy_call_seconds=self.latency,
            policy_call_frames=np.asarray(self.call_frames, dtype=np.int64),
        )
        return {
            "frames": len(self.state),
            "object_names": self.names,
            "policy_observation_hashes": self.hashes,
            "policy_call_seconds": self.latency,
            "phase_labels": {
                "0": "settling",
                "1": "policy",
                "2": "terminal",
                "5": "intervention_settling",
            },
        }
