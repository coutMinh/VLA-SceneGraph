"""Thin orchestration CLI, usable on molab Python 3.13 without ML dependencies."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from sgvla.io import read_json, write_json
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
    verify = commands.add_parser("eval-verify", help="CPU source/initial-state verification")
    verify.add_argument("--upstream", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--markdown", type=Path)
    manifest = commands.add_parser("eval-manifest", help="Freeze a reduced E0 workload")
    manifest.add_argument("--scene-report", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--n-per-task", type=int, default=5)
    progress = commands.add_parser("eval-status", help="Validated E0 progress; no ML imports")
    progress.add_argument("--manifest", type=Path)
    episode = commands.add_parser("eval-episode", help="Validated completed episode paths")
    episode.add_argument("--manifest", type=Path)
    episode.add_argument("--episode-id", required=True)
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
    if args.command == "eval-verify":
        from sgvla.eval.scenes import verify_scenes, write_report

        report = verify_scenes(args.upstream)
        write_report(report, args.output, args.markdown)
        result = {"report": str(args.output), "tasks": len(report["tasks"])}
    elif args.command == "eval-manifest":
        from sgvla.eval.manifest import build_manifest

        workload = build_manifest(read_json(args.scene_report), runtime.config, args.n_per_task)
        if args.output.exists() and read_json(args.output) != workload:
            raise RuntimeError("Existing manifest differs; choose a new output path")
        write_json(args.output, workload)
        result = {"manifest": str(args.output), "episodes": len(workload["episodes"])}
    elif args.command == "eval-episode":
        from sgvla.eval.engine import completed

        path = args.manifest or runtime.work / "configs/e0_manifest.json"
        item = completed(runtime, read_json(path)).get(args.episode_id)
        if not item:
            raise RuntimeError("Episode has no compatible validated completion")
        _, metrics, output = item
        result = {
            "episode_id": args.episode_id,
            "metrics": metrics,
            "agentview": str(output / "agentview.mp4"),
            "wrist": str(output / "wrist.mp4"),
        }
    elif args.command == "eval-status":
        from sgvla.eval.reporting import summarize

        path = args.manifest or runtime.work / "configs/e0_manifest.json"
        result = summarize(runtime, read_json(path))
    elif args.command == "init-config":
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
