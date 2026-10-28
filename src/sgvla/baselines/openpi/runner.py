"""Entry points run by the server (3.11) or simulator (3.8), never the molab kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import time
from pathlib import Path

from sgvla.io import digest, read_json, utc_now, write_json
from sgvla.runtime import Runtime, git_commit


def backend_check(runtime):
    report = {"observed_at_utc": utc_now(), "backend": runtime.config["backend"]}
    for name in ("jax", "torch", "numpy", "transformers"):
        report[name + "_version"] = importlib.metadata.version(name)
    try:
        import jax
        import jax.numpy as jnp

        gpu = next(device for device in jax.devices() if device.platform == "gpu")
        with jax.default_device(gpu):
            matrix = jnp.ones((128, 128))
            value = jax.jit(lambda x: x @ x)(matrix).block_until_ready()
        report["jax"] = {"ok": bool(jnp.all(value == 128)), "device": str(gpu)}
    except Exception as exc:
        report["jax"] = {"ok": False, "error": str(exc)}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        matrix = torch.ones((128, 128), device="cuda")
        value = matrix @ matrix
        torch.cuda.synchronize()
        report["pytorch"] = {
            "ok": bool((value == 128).all().item()),
            "device": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
        }
    except Exception as exc:
        report["pytorch"] = {"ok": False, "error": str(exc)}
    report["config_sha256"] = runtime.config_hash
    write_json(runtime.work / "docs/backend_check.json", report)
    if not report[runtime.config["backend"]]["ok"]:
        raise RuntimeError("Selected backend failed GPU kernels; see docs/backend_check.json")
    return report


def serve(runtime):
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    config = runtime.config
    evidence = read_json(runtime.work / "docs/backend_check.json")
    if evidence["config_sha256"] != runtime.config_hash:
        raise RuntimeError("Config changed since backend-check; rerun it")
    if not evidence[config["backend"]]["ok"]:
        raise RuntimeError("Selected backend has no passing GPU check")
    if config["backend"] == "pytorch":
        import torch

        torch.manual_seed(0)
        checkpoint = read_json(runtime.work / "docs/conversion.json")["path"]
    else:
        checkpoint = read_json(runtime.work / "docs/checkpoint.json")["path"]
    policy = policy_config.create_trained_policy(
        training_config.get_config(config["model_config"]),
        checkpoint,
        sample_kwargs={"num_steps": config["num_steps"]},
    )
    metadata = {
        **policy.metadata,
        "sgvla": {
            "config_sha256": runtime.config_hash,
            "openpi_commit": git_commit(runtime.upstream),
            "backend": config["backend"],
            "model_config": config["model_config"],
            "policy_seed": 0,
            "num_steps": config["num_steps"],
        },
    }
    websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=config["host"],
        port=config["port"],
        metadata=metadata,
    ).serve_forever()


def connect(runtime):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(runtime.config["host"], runtime.config["port"])
    metadata = client.get_server_metadata().get("sgvla", {})
    if metadata.get("config_sha256") != runtime.config_hash:
        raise RuntimeError("Server config differs from client (or this is an unmanaged server)")
    if metadata.get("openpi_commit") != runtime.config["openpi_commit"]:
        raise RuntimeError("Server upstream commit differs")
    return client


def dummy(runtime, folder):
    import numpy as np

    from sgvla.baselines.openpi.rollout import validate_actions

    client = connect(runtime)
    rng = np.random.RandomState(runtime.config["seed"])
    observation = {
        "observation/image": rng.randint(256, size=(224, 224, 3)).astype(np.uint8),
        "observation/wrist_image": rng.randint(256, size=(224, 224, 3)).astype(np.uint8),
        "observation/state": rng.rand(8),
        "prompt": "pick up the black bowl",
    }
    started = time.monotonic()
    response = client.infer(observation)
    actions = validate_actions(response["actions"], runtime.config["replan_steps"])
    if actions.shape != (50, 7):
        raise ValueError("pi0_libero config should produce a (50, 7) chunk")
    np.save(folder / "actions.npy", actions)
    return {
        "status": "completed",
        "shape": list(actions.shape),
        "finite": True,
        "wall_clock_seconds": time.monotonic() - started,
        "server_metadata": client.get_server_metadata(),
        "benchmark_estimate": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    runtime = Runtime(args.work, args.config)
    if args.stage == "download" or (
        args.stage == "server" and runtime.config["backend"] == "pytorch"
    ):
        os.environ["JAX_PLATFORMS"] = "cpu"
    if git_commit(runtime.upstream) != runtime.config["openpi_commit"]:
        raise RuntimeError("Upstream commit differs from config")
    if args.stage == "backend-check":
        backend_check(runtime)
    elif args.stage == "download":
        from sgvla.baselines.openpi.setup import download

        download(runtime)
    elif args.stage == "server":
        serve(runtime)
    else:
        from sgvla.baselines.openpi.rollout import rollout
        from sgvla.registry import register

        started = time.monotonic()
        folder = runtime.results / "runs" / args.run_id
        folder.mkdir(parents=True, exist_ok=False)
        write_json(
            folder / "manifest.json",
            {
                "openpi_commit": git_commit(runtime.upstream),
                "libero_commit": runtime.config["libero_commit"],
                "checkpoint": runtime.config["checkpoint"],
                "config_sha256": runtime.config_hash,
                "kind": args.stage,
                "created_at_utc": utc_now(),
            },
        )
        metrics_path = folder / "metrics.json"
        try:
            if args.stage == "dummy":
                result = dummy(runtime, folder)
            elif args.stage == "render":
                result = rollout(runtime, folder)
            elif args.stage == "smoke":
                result = rollout(runtime, folder, connect(runtime))
            else:
                raise ValueError("Unknown stage " + args.stage)
            result["config_sha256"] = runtime.config_hash
            write_json(metrics_path, result)
        except Exception as exc:
            write_json(metrics_path, {"status": "error", "error_type": type(exc).__name__})
            raise
        finally:
            manifest = read_json(folder / "manifest.json")
            manifest["artifacts"] = {
                path.name: digest(path)
                for path in folder.iterdir()
                if path.is_file() and path.name != "manifest.json"
            }
            write_json(folder / "manifest.json", manifest)
            register(runtime, args.run_id, args.stage, metrics_path, time.monotonic() - started)
        write_json(
            runtime.work / "docs" / (args.stage + ".json"),
            {
                "run_id": args.run_id,
                "metrics_path": str(metrics_path),
                "metrics_sha256": digest(metrics_path),
            },
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
