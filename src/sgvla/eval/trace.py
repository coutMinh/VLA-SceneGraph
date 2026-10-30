"""Synchronized before-action frames, terminal state, poses and inference evidence."""

from __future__ import annotations

from sgvla.eval.manifest import fingerprint
from sgvla.eval.scenes import BOWL1, BOWL2, PLATE, array_hash


def observation_hash(element):
    return fingerprint(
        {key: array_hash(value) if key != "prompt" else value for key, value in element.items()}
    )


ENDPOINT_TOLERANCES = {
    "state_max_abs": 1e-5,
    "image_mean_abs": 0.02,
    "image_fraction_changed": 0.01,
    "image_max_abs": 64.0,
}


def compare_observations(reference, actual):
    """Compare policy inputs across MuJoCo/renderer processes with explicit tolerances."""
    import numpy as np

    if set(reference) != set(actual):
        return {
            "matches": False,
            "reason": "observation keys differ",
            "reference_keys": sorted(reference),
            "actual_keys": sorted(actual),
        }

    components = {}
    matches = True
    for key in sorted(reference):
        if key == "prompt":
            left_array = np.asarray(reference[key])
            right_array = np.asarray(actual[key])
            component = {
                "equal": left_array.item() == right_array.item(),
                "shape_equal": left_array.shape == right_array.shape,
            }
            matches = matches and component["equal"] and component["shape_equal"]
        else:
            left = np.asarray(reference[key])
            right = np.asarray(actual[key])
            if left.shape != right.shape:
                components[key] = {
                    "matches": False,
                    "reference_shape": list(left.shape),
                    "actual_shape": list(right.shape),
                }
                matches = False
                continue
            dtype_equal = left.dtype == right.dtype
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            if key == "observation/state":
                maximum = float(difference.max(initial=0.0))
                component = {
                    "dtype_equal": bool(dtype_equal),
                    "max_abs": maximum,
                    "tolerance": ENDPOINT_TOLERANCES["state_max_abs"],
                    "matches": bool(dtype_equal)
                    and bool(np.isfinite(difference).all())
                    and maximum <= ENDPOINT_TOLERANCES["state_max_abs"],
                }
            elif key in ("observation/image", "observation/wrist_image"):
                maximum = float(difference.max(initial=0.0))
                mean = float(difference.mean())
                fraction = float(np.count_nonzero(difference) / difference.size)
                component = {
                    "dtype_equal": bool(dtype_equal),
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "fraction_changed": fraction,
                    "tolerances": {
                        "max_abs": ENDPOINT_TOLERANCES["image_max_abs"],
                        "mean_abs": ENDPOINT_TOLERANCES["image_mean_abs"],
                        "fraction_changed": ENDPOINT_TOLERANCES[
                            "image_fraction_changed"
                        ],
                    },
                    "matches": bool(dtype_equal)
                    and bool(np.isfinite(difference).all())
                    and maximum <= ENDPOINT_TOLERANCES["image_max_abs"]
                    and mean <= ENDPOINT_TOLERANCES["image_mean_abs"]
                    and fraction <= ENDPOINT_TOLERANCES["image_fraction_changed"],
                }
            else:
                component = {
                    "dtype_equal": bool(dtype_equal),
                    "equal": bool(np.array_equal(left, right)),
                }
                matches = matches and component["equal"] and component["dtype_equal"]
            if "matches" in component:
                matches = matches and component["matches"]
        components[key] = component

    hash_reference = dict(reference)
    if "prompt" in hash_reference:
        hash_reference["prompt"] = np.asarray(hash_reference["prompt"]).item()
    return {
        "matches": bool(matches),
        "reference_sha256": observation_hash(hash_reference),
        "actual_sha256": observation_hash(actual),
        "tolerances": ENDPOINT_TOLERANCES,
        "components": components,
    }


class Trace:
    """Stream videos; retain compact numeric data, including settling and terminal frames."""

    def __init__(self, folder, env, official, resize_size):
        import imageio.v2 as imageio

        self.folder, self.env, self.official, self.resize = folder, env, official, resize_size
        core = env.env
        entities = {**core.objects_dict, **core.fixtures_dict}
        self.names = sorted(entities)
        self.bodies = [env.sim.model.body_name2id(entities[n].root_body) for n in self.names]
        self.writers = [
            imageio.get_writer(str(folder / name), fps=10, macro_block_size=1)
            for name in ("agentview.mp4", "wrist.mp4")
        ]
        self.state, self.poses, self.sim, self.phase, self.placed = [], [], [], [], []
        self.actions, self.chunks, self.latency, self.call_frames, self.hashes = [], [], [], [], []

    def capture(self, obs, prompt, phase, action=None):
        import numpy as np
        from libero.libero.envs.predicates import eval_predicate_fn

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
        objects = self.env.env.object_states_dict
        self.placed.append(
            [
                bool(eval_predicate_fn("on", objects[name], objects[PLATE]))
                for name in (BOWL1, BOWL2)
            ]
        )
        self.phase.append(phase)
        if action is not None:
            self.actions.append(np.asarray(action).copy())
        return element

    def inference(self, element, chunk, latency):
        self.call_frames.append(len(self.state))
        self.hashes.append(observation_hash(element))
        self.chunks.append(chunk.copy())
        self.latency.append(latency)

    def close(self):
        for writer in self.writers:
            writer.close()

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
            bowl_on_plate=self.placed,
            policy_chunks=self.chunks,
            policy_call_seconds=self.latency,
            policy_call_frames=np.asarray(self.call_frames, dtype=np.int64),
        )
        return {
            "frames": len(self.state),
            "object_names": self.names,
            "policy_observation_hashes": self.hashes,
            "policy_call_seconds": self.latency,
            "phase_labels": {"0": "settling", "1": "policy", "2": "terminal"},
            "pose_convention": "MuJoCo world xyz + quaternion wxyz; fixture root + full sim state",
        }


def validate_trace(folder, metrics):
    """Read all arrays and decode both videos before accepting or resuming an episode."""
    import imageio.v2 as imageio
    import numpy as np

    with np.load(folder / "trajectory.npz", allow_pickle=False) as trace:
        n, steps = metrics["frames"], metrics["steps"]
        count = len(metrics["object_names"])
        if n != steps + metrics["settling_steps"] + 1:
            raise ValueError("Trace frame count differs from action/settling count")
        shapes = {"state": (n, 8), "actions": (n - 1, 7), "object_poses": (n, count, 7)}
        for name, shape in shapes.items():
            if trace[name].shape != shape or not np.isfinite(trace[name]).all():
                raise ValueError("Invalid trace " + name)
        if (
            trace["object_names"].tolist() != metrics["object_names"]
            or not {BOWL1, BOWL2, PLATE} <= set(metrics["object_names"])
            or trace["sim_state"].shape[0] != n
            or not np.isfinite(trace["sim_state"]).all()
            or trace["bowl_on_plate"].shape != (n, 2)
        ):
            raise ValueError("Missing object identity or simulator state")
        expected_phase = [0] * metrics["settling_steps"] + [1] * steps + [2]
        if trace["phase"].tolist() != expected_phase:
            raise ValueError("Trace phases are not aligned")
        calls = list(range(metrics["settling_steps"], n - 1, metrics["replan_steps"]))
        if trace["policy_call_frames"].tolist() != calls:
            raise ValueError("Invalid policy-call frame indices")
        if (
            trace["policy_chunks"].shape != (len(calls), 50, 7)
            or len(metrics["policy_observation_hashes"]) != len(calls)
            or trace["policy_call_seconds"].shape != (len(calls),)
            or not np.isfinite(trace["policy_chunks"]).all()
            or not np.isfinite(trace["policy_call_seconds"]).all()
            or (trace["policy_call_seconds"] < 0).any()
        ):
            raise ValueError("Invalid policy evidence")
        if not np.array_equal(trace["policy_call_seconds"], metrics["policy_call_seconds"]):
            raise ValueError("Latency metrics differ from the trace")
        executed = np.concatenate(
            [chunk[: metrics["replan_steps"]] for chunk in trace["policy_chunks"]]
        )[:steps]
        if not np.array_equal(executed, trace["actions"][metrics["settling_steps"] :]):
            raise ValueError("Executed actions differ from the recorded policy chunks")
    for name in ("agentview.mp4", "wrist.mp4"):
        with imageio.get_reader(str(folder / name)) as reader:
            decoded = 0
            for frame in reader:
                if frame.shape != (224, 224, 3):
                    raise ValueError("Unexpected video resolution " + name)
                decoded += 1
        if decoded != n:
            raise ValueError("Invalid/truncated video " + name)
