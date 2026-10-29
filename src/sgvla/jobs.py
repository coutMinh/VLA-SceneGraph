"""Detached jobs with unique logs, duplicate protection and restart detection."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from sgvla.io import file_lock, read_json, utc_now, write_json
from sgvla.runtime import REPO

ACTIONS = (
    "probe",
    "install-server",
    "install-client",
    "repair-client",
    "repair-rendering",
    "backend-check",
    "download",
    "convert",
    "server",
    "dummy",
    "render",
    "smoke",
    "eval-plan",
    "eval-parity",
    "eval-batch",
    "eval-review",
    "eval-all",
)


def process_identity(pid):
    """Linux start ticks distinguish a relaunched molab process from a reused PID."""
    path = Path("/proc") / str(pid) / "stat"
    try:
        # comm may contain spaces and parentheses; fields after its last ')' start at #3.
        fields = path.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return boot + ":" + fields[19]
    except (OSError, IndexError):
        return None


def status(runtime, name):
    request_path = runtime.work / "jobs" / (name + ".json")
    if not request_path.exists():
        return {"name": name, "status": "not_started"}
    request = read_json(request_path)
    result_path = runtime.work / "jobs" / (request["id"] + ".result.json")
    if result_path.exists():
        return {**request, **read_json(result_path)}
    identity = process_identity(request["pid"])
    running = identity is not None and identity == request["process_identity"]
    return {**request, "status": "running" if running else "interrupted"}


def start(runtime, name):
    if sys.platform != "linux":
        raise RuntimeError("Detached jobs target Linux molab; use 'run probe' locally")
    if name not in ACTIONS:
        raise ValueError("Unknown action: " + name)
    jobs = runtime.work / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    # All jobs share environment/cache state. Install and conversion must not race inference.
    with file_lock(jobs / ".launch.lock"):
        current = status(runtime, name)
        if current["status"] == "running":
            return current
        active = [action for action in ACTIONS if status(runtime, action)["status"] == "running"]
        mutating = {
            "install-server",
            "install-client",
            "repair-client",
            "repair-rendering",
            "convert",
        }
        if (name in mutating and active) or (set(active) & mutating):
            raise RuntimeError(
                "Finish the active job before changing its environments: " + str(active)
            )
        if (
            name in {"dummy", "smoke", "eval-parity", "eval-batch", "eval-all"}
            and "server" not in active
        ):
            raise RuntimeError("Start the policy server first and check its log")
        if name.startswith("eval-") and any(action.startswith("eval-") for action in active):
            raise RuntimeError("Finish the active evaluation job before launching another")
        run_id = name + "-" + uuid.uuid4().hex[:12]
        config_path = jobs / (run_id + ".config.json")
        write_json(config_path, runtime.config)
        log_path = runtime.results / "logs" / (run_id + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "sgvla",
            "--work",
            str(runtime.work),
            "--config",
            str(config_path),
            "_worker",
            name,
            run_id,
        ]
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=runtime.env(),
                stdout=log,
                stderr=log,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
        request = {
            "name": name,
            "id": run_id,
            "pid": process.pid,
            "process_identity": process_identity(process.pid),
            "started_at_utc": utc_now(),
            "log": str(log_path),
            "config": str(config_path),
        }
        write_json(jobs / (name + ".json"), request)
        return {**request, "status": "started"}


def worker(runtime, name, run_id):
    import traceback

    try:
        execute(runtime, name, run_id)
    except Exception as exc:
        traceback.print_exc()
        write_json(
            runtime.work / "jobs" / (run_id + ".result.json"),
            {
                "status": "failed",
                "error": type(exc).__name__,
                "finished_at_utc": utc_now(),
            },
        )
        raise
    else:
        write_json(
            runtime.work / "jobs" / (run_id + ".result.json"),
            {
                "status": "completed",
                "finished_at_utc": utc_now(),
            },
        )


def stop(runtime, name):
    """Signal only the session leader recorded by this work directory."""
    if sys.platform != "linux":
        raise RuntimeError("Managed job stop requires Linux")
    with file_lock(runtime.work / "jobs/.launch.lock"):
        current = status(runtime, name)
        if current["status"] != "running":
            return current
        pid = current["pid"]
        if process_identity(pid) != current["process_identity"] or os.getpgid(pid) != pid:
            raise RuntimeError("Process identity changed; refusing to signal it")
        os.killpg(pid, signal.SIGTERM)
        write_json(
            runtime.work / "jobs" / (current["id"] + ".result.json"),
            {
                "status": "stopped",
                "finished_at_utc": utc_now(),
            },
        )
        return {**current, "status": "stopped"}


def execute(runtime, name, run_id):
    from sgvla.baselines.openpi import setup
    from sgvla.probe import probe

    if name == "probe":
        return probe(runtime, network=runtime.config.get("probe_network", True))
    if name.startswith("install-"):
        return setup.install(runtime, name[len("install-") :])
    if name == "convert":
        return setup.convert(runtime)
    if name == "repair-client":
        from sgvla.baselines.openpi.compat import repair_client

        return repair_client(runtime)
    if name == "repair-rendering":
        from sgvla.baselines.openpi.compat import repair_rendering

        return repair_rendering(runtime)
    if name.startswith("eval-"):
        return runtime.module(
            "client",
            "sgvla.eval.worker",
            "--mode",
            name[len("eval-") :],
            "--manifest",
            runtime.config.get(
                "evaluation_manifest", str(runtime.work / "configs/e0_manifest.json")
            ),
            "--limit",
            str(runtime.config.get("eval_batch_size", 5)),
        )
    env_name = "client" if name in ("dummy", "render", "smoke") else "server"
    timeout = None if name in ("server", "download") else 1800
    return runtime.module(
        env_name,
        "sgvla.baselines.openpi.runner",
        "--stage",
        name,
        "--run-id",
        run_id,
        timeout=timeout,
    )
