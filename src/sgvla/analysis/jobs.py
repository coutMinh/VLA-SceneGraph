"""Detached Part 3 jobs with process identity checks; no new Molab session."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from sgvla.io import file_lock, read_json, utc_now, write_json
from sgvla.runtime import REPO, Runtime, git_commit


def process_identity(pid):
    try:
        fields = Path("/proc", str(int(pid)), "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None


def status(path):
    state = read_json(path / "state.json")
    live = process_identity(state["pid"])
    if live is not None and live == state["process_start_ticks"]:
        state["status"] = "running"
    elif (path / "result.json").exists():
        state["status"] = read_json(path / "result.json")["job_status"]
    else:
        state["status"] = "interrupted"
    return state


def list_jobs(runtime):
    folder = runtime.work / "jobs/part3"
    return [status(p) for p in sorted(folder.glob("job-*")) if (p / "state.json").exists()]


def start(runtime, task, condition=None, limit=5):
    from sgvla.analysis.cli import TASKS
    from sgvla.eval.engine import implementation_hash

    if task not in TASKS or task == "episode":
        raise ValueError("Unsupported managed action")
    if task in ("plan", "batch", "all", "review") and condition not in (
        "S1",
        "T1",
        "T1-ctrl",
        "S2",
        "S2-ctrl",
    ):
        raise ValueError("Condition required")
    with file_lock(runtime.work / "jobs/part3/.launch.lock"):
        if any(job["status"] == "running" for job in list_jobs(runtime)):
            raise RuntimeError("A Part 3 job is still running; inspect it before another launch")
        job_id = "job-" + uuid.uuid4().hex[:12]
        folder = runtime.work / "jobs/part3" / job_id
        folder.mkdir(parents=True)
        config = folder / "config.json"
        write_json(config, runtime.config)
        argv = [
            str(runtime.python("client")),
            "-m",
            "sgvla.analysis.jobs",
            "--work",
            str(runtime.work),
            "--config",
            str(config),
            "--job-id",
            job_id,
            "--task",
            task,
            "--limit",
            str(limit),
        ]
        if condition:
            argv += ["--condition", condition]
        log = folder / "job.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                argv,
                cwd=REPO,
                env=runtime.env(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state = {
            "job_id": job_id,
            "task": task,
            "condition": condition,
            "pid": process.pid,
            "process_start_ticks": process_identity(process.pid),
            "log": str(log),
            "argv": argv,
            "source_commit": git_commit(REPO),
            "source_sha256": implementation_hash(),
            "created_at_utc": utc_now(),
        }
        write_json(folder / "state.json", state)
        return state


def stop(runtime, job_id):
    if not job_id or not job_id.startswith("job-") or Path(job_id).name != job_id:
        raise ValueError("Invalid managed job ID")
    folder = runtime.work / "jobs/part3" / job_id
    state = status(folder)
    if state["status"] != "running":
        return state
    command = Path("/proc", str(state["pid"]), "cmdline").read_bytes()
    if (
        b"sgvla.analysis.jobs" not in command
        or job_id.encode() not in command
        or os.getpgid(state["pid"]) != state["pid"]
    ):
        raise RuntimeError("Process no longer matches this managed job")
    os.killpg(state["pid"], signal.SIGTERM)
    return {**state, "status": "stop_requested"}


def main():
    from sgvla.analysis.cli import dispatch

    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    runtime = Runtime(args.work, args.config)
    started = time.monotonic()
    folder = runtime.work / "jobs/part3" / args.job_id
    try:
        result = dispatch(runtime, args.task, args.condition, args.limit)
    except Exception as exc:
        write_json(
            folder / "result.json",
            {
                "job_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_clock_seconds": time.monotonic() - started,
            },
        )
        raise
    write_json(
        folder / "result.json",
        {
            "job_status": "completed",
            "result": result,
            "wall_clock_seconds": time.monotonic() - started,
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
