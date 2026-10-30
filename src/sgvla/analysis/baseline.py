"""Read frozen E0 evidence using its recorded context, never the Part 3 source hash."""

from __future__ import annotations

from sgvla.eval.engine import validate_completion
from sgvla.eval.manifest import fingerprint, validate_manifest
from sgvla.io import digest, read_json


def load_baseline(runtime):
    manifest = read_json(runtime.work / "configs/e0_manifest.json")
    validate_manifest(manifest, runtime.config)
    root = runtime.results / "part2/e0" / fingerprint(manifest)
    summary = read_json(root / "summary.json")
    expected = summary["context"]
    if (
        expected["manifest_sha256"] != fingerprint(manifest)
        or expected["policy_sha256"] != runtime.policy_hash
    ):
        raise ValueError("Frozen baseline policy/manifest differs")
    snapshot_path = runtime.work / "source_snapshot.json"
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        source_files = {
            path: checksum
            for path, checksum in snapshot.get("files", {}).items()
            if path.startswith("src/") and path.endswith(".py")
        }
        if (
            not snapshot.get("local_base_commit")
            or not source_files
            or fingerprint(source_files) != expected["implementation_sha256"]
        ):
            raise ValueError("Frozen baseline implementation differs from its source snapshot")
        expected_git_commit = snapshot["local_base_commit"]
    else:
        # Compatibility with the original run, which predates source_snapshot.json.
        expected_git_commit = "1c98fd3efece9194330a038b2c62b7cb2d6a1046"
    pointer = read_json(root / "parity.json")
    report_path = root / pointer["report"]
    report_path.resolve().relative_to(root.resolve())
    parity = read_json(report_path)
    if (
        digest(report_path) != pointer["sha256"]
        or not parity["passed"]
        or parity["context"] != expected
        or parity["initial_states"] != [0, 1, 2]
        or parity["task_id"] != 0
        or len(parity["comparisons"]) != 3
    ):
        raise ValueError("Frozen E0 parity is missing or incompatible")
    for relative, checksum in parity["artifact_sha256"].items():
        path = root / relative
        path.resolve().relative_to(root.resolve())
        if digest(path) != checksum:
            raise ValueError("Frozen E0 parity artifact changed")
    done = {}
    for entry in manifest["episodes"]:
        item = validate_completion(
            root / "episodes" / entry["episode_id"], entry, expected, deep=False
        )
        if not item:
            raise ValueError("Invalid frozen E0 completion: " + entry["episode_id"])
        if (
            read_json(item[2] / "manifest.json")["git_commit"]
            != expected_git_commit
        ):
            raise ValueError("Unexpected frozen baseline source commit")
        done[entry["episode_id"]] = item
    return manifest, root, summary, done
