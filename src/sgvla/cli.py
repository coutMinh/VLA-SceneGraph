"""Thin orchestration CLI, usable on molab Python 3.13 without ML dependencies."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from sgvla.io import write_json
from sgvla.jobs import ACTIONS, execute, start, status, stop, worker
from sgvla.runtime import REPO, Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(os.environ.get("SGVLA_WORK", str(Path.home() / "vla_scenegraph_work"))),
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/part2_smoke.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-config")
    commands.add_parser("gate")
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("action", choices=ACTIONS)
    launch = commands.add_parser("start")
    launch.add_argument("action", choices=ACTIONS)
    halt = commands.add_parser("stop")
    halt.add_argument("action", choices=ACTIONS)
    internal = commands.add_parser("_worker")
    internal.add_argument("action", choices=ACTIONS)
    internal.add_argument("run_id")
    args = parser.parse_args()
    runtime = Runtime(args.work.expanduser().resolve(), args.config.expanduser().resolve())
    if args.command == "init-config":
        path = runtime.work / "configs/part2_smoke.json"
        if not path.exists():
            # Keep previous runtime choices and their hashes when adopting part-based names.
            legacy = runtime.work / "configs/d1.json"
            source = Runtime(runtime.work, legacy) if legacy.exists() else runtime
            write_json(path, source.config)
        result = {"config": str(path), "existing_configs_preserved": True}
    elif args.command == "status":
        result = [status(runtime, name) for name in ACTIONS]
    elif args.command == "gate":
        from sgvla.gate import check_smoke

        result = check_smoke(runtime)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    elif args.command == "start":
        result = start(runtime, args.action)
    elif args.command == "stop":
        result = stop(runtime, args.action)
    elif args.command == "_worker":
        worker(runtime, args.action, args.run_id)
        return
    else:
        run_id = args.action + "-" + uuid.uuid4().hex[:12]
        snapshot = runtime.work / "jobs" / (run_id + ".config.json")
        write_json(snapshot, runtime.config)
        result = execute(Runtime(runtime.work, snapshot), args.action, run_id)
    if result is not None:
        print(json.dumps(result, indent=2))
