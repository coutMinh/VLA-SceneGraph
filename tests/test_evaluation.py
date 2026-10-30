"""CPU fixtures exercise artifact integrity and execution contracts, not model quality."""

import copy
import csv
import io
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sgvla.eval import engine
from sgvla.eval.manifest import build_manifest, fingerprint, policy_fingerprint, validate_manifest
from sgvla.eval.scenes import ArrayUnpickler, read_initial_states
from sgvla.eval.trace import compare_observations
from sgvla.io import digest, read_json, write_json
from sgvla.runtime import REPO, Runtime


def test_endpoint_comparison_allows_only_small_process_render_drift():
    reference = {
        "observation/image": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64),
        "prompt": "place the bowl on the plate",
    }
    actual = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in reference.items()}
    actual["observation/state"][0] = 1e-6
    actual["observation/image"][0, 0, 0] = 4
    result = compare_observations(reference, actual)
    assert result["matches"] is True
    assert result["reference_sha256"] != result["actual_sha256"]

    # The selected bound accepts a sparse low-intensity raster difference,
    # while a one-pixel-step increase beyond the changed-pixel limit fails.
    boundary_reference = {key: value.copy() if isinstance(value, np.ndarray) else value
                          for key, value in reference.items()}
    boundary_reference["observation/image"] = np.zeros((100, 100, 3), dtype=np.uint8)
    actual = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in boundary_reference.items()}
    actual["observation/image"].reshape(-1)[:300] = 2
    assert compare_observations(boundary_reference, actual)["matches"] is True
    actual["observation/image"].reshape(-1)[300] = 2
    assert compare_observations(boundary_reference, actual)["matches"] is False

    actual = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in reference.items()}
    actual["observation/state"][0] = 1e-3
    assert compare_observations(reference, actual)["matches"] is False
    actual["observation/state"][0] = 0
    actual["prompt"] = "place the other bowl on the plate"
    assert compare_observations(reference, actual)["matches"] is False
    actual["prompt"] = reference["prompt"]
    actual["observation/image"][:] = 255
    assert compare_observations(reference, actual)["matches"] is False


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    write_json(config, read_json(REPO / "configs/part2_smoke.json"))
    monkeypatch.setattr(engine, "git_commit", lambda _: "a" * 40)
    monkeypatch.setattr("sgvla.registry.git_commit", lambda _: "a" * 40)
    return Runtime(tmp_path, config)


@pytest.fixture
def manifest(runtime):
    report = read_json(REPO / "tests/fixtures/scene_report.json")
    return build_manifest(report, runtime.config)


def test_real_source_report_task_order_and_fixed_pilot(runtime, manifest):
    assert len(manifest["episodes"]) == 50
    entries = manifest["episodes"]
    assert entries[0]["task_name"].startswith("pick_up_the_black_bowl_between")
    assert entries[5]["task_name"].startswith("pick_up_the_black_bowl_next_to_the_ramekin")
    assert [(e["task_id"], e["init_state_id"]) for e in entries] == [
        (t, s) for t in range(10) for s in range(5)
    ]
    changed = {**runtime.config, "task_id": 9, "init_state_id": 4, "eval_batch_size": 10}
    assert policy_fingerprint(changed) == runtime.policy_hash
    assert validate_manifest(manifest, changed) == fingerprint(manifest)
    assert policy_fingerprint({**changed, "num_steps": 11}) != runtime.policy_hash


@pytest.mark.parametrize("mutation", ["duplicate", "instruction", "state", "seed", "source"])
def test_manifest_rejects_changed_workload(runtime, manifest, mutation):
    modified = copy.deepcopy(manifest)
    entry = modified["episodes"][0]
    if mutation == "duplicate":
        modified["episodes"].append(entry.copy())
    elif mutation == "instruction":
        entry["prompt"] = "pick up something else"
    elif mutation == "state":
        entry["init_state_id"] = 6
    elif mutation == "seed":
        entry["policy_seed"] += 1
    else:
        entry["bddl_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        validate_manifest(modified, runtime.config)


def test_initial_archive_validation_and_restricted_globals(tmp_path):
    path = tmp_path / "states.pruned_init"
    arrays = np.zeros((50, 92))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(arrays, protocol=2))
    np.testing.assert_equal(read_initial_states(path), arrays)
    with pytest.raises(ValueError, match="Unexpected"):
        ArrayUnpickler(io.BytesIO(pickle.dumps(Path("untrusted")))).load()
    arrays[0, 0] = np.nan
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(arrays, protocol=2))
    with pytest.raises(ValueError, match="Invalid"):
        read_initial_states(path)


def fixture_episode(runtime, entry, folder, *, success=False):
    import imageio.v2 as imageio

    steps, settling, objects = 7, 10, ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1"]
    n = steps + settling + 1
    for name in ("agentview.mp4", "wrist.mp4"):
        with imageio.get_writer(str(folder / name), fps=10) as writer:
            for _ in range(n):
                writer.append_data(np.zeros((224, 224, 3), dtype=np.uint8))
    np.savez_compressed(
        folder / "trajectory.npz",
        state=np.zeros((n, 8)),
        actions=np.zeros((n - 1, 7)),
        object_names=objects,
        object_poses=np.zeros((n, len(objects), 7)),
        sim_state=np.zeros((n, 92)),
        bowl_on_plate=np.zeros((n, 2), dtype=bool),
        phase=[0] * settling + [1] * steps + [2],
        policy_chunks=np.zeros((2, 50, 7)),
        policy_call_seconds=[0.01, 0.02],
        policy_call_frames=[10, 15],
    )
    return {
        "status": "completed",
        "success": success,
        "steps": steps,
        "frames": n,
        "settling_steps": settling,
        "replan_steps": 5,
        "object_names": objects,
        "initial_state_sha256": entry["initial_state_sha256"],
        "policy_seed": entry["policy_seed"],
        "target_object": entry["target_object"],
        "policy_observation_hashes": ["a" * 64, "b" * 64],
        "policy_call_seconds": [0.01, 0.02],
    }


def test_resume_preserves_completed_failure_and_registry(runtime, manifest):
    entry = manifest["episodes"][0]
    result = engine.execute_one(runtime, manifest, entry, fixture_episode)
    assert result["status"] == "completed" and result["success"] is False
    again = engine.execute_one(runtime, manifest, entry, lambda *_: pytest.fail("must skip"))
    assert again["status"] == "skipped"
    with (runtime.results / "registry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["run_type"] == "E0"
    assert Path(rows[0]["config_path"]).name == "config.json"
    assert engine.completed(runtime, manifest, deep=True)[entry["episode_id"]]


def test_interruption_and_infrastructure_retry_do_not_count_as_task_failure(runtime, manifest):
    entry = manifest["episodes"][0]
    episode = engine.evaluation_root(runtime, manifest) / "episodes" / entry["episode_id"]
    write_json(episode / "attempt-interrupted/manifest.json", {"status": "partial"})

    def broken(*_):
        raise ConnectionError("CPU fixture disconnect")

    assert engine.execute_one(runtime, manifest, entry, broken)["status"] == "infrastructure_error"
    assert not (episode / "complete.json").exists()
    assert engine.execute_one(runtime, manifest, entry, fixture_episode)["status"] == "completed"
    with (runtime.results / "registry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["run_type"] for r in rows] == ["E0_infrastructure_error", "E0"]
    assert (episode / "attempt-interrupted/manifest.json").exists()


@pytest.mark.parametrize(
    "change", ["truncated_video", "missing_trace", "config", "context", "pointer"]
)
def test_corrupt_or_incompatible_completion_is_retried(runtime, manifest, change):
    entry = manifest["episodes"][0]
    engine.execute_one(runtime, manifest, entry, fixture_episode)
    episode = engine.evaluation_root(runtime, manifest) / "episodes" / entry["episode_id"]
    record = read_json(episode / "complete.json")
    output = episode / record["attempt"]
    if change == "truncated_video":
        (output / "agentview.mp4").write_bytes(b"truncated")
    elif change == "missing_trace":
        (output / "trajectory.npz").unlink()
    elif change == "config":
        write_json(output / "config.json", {})
    elif change == "context":
        record["context"]["implementation_sha256"] = "old"
        write_json(episode / "complete.json", record)
    else:
        (episode / "complete.json").write_text("{interrupted", encoding="utf-8")
    assert not engine.completed(runtime, manifest)[entry["episode_id"]]
    assert engine.execute_one(runtime, manifest, entry, fixture_episode)["status"] == "completed"
    assert read_json(episode / "complete.json")["attempt"] != record["attempt"]


def test_trace_schema_checks_arrays_even_with_updated_file_hash(runtime, manifest):
    from sgvla.eval.trace import validate_trace

    folder = runtime.work / "fixture"
    folder.mkdir()
    metrics = fixture_episode(runtime, manifest["episodes"][0], folder)
    with np.load(folder / "trajectory.npz") as trace:
        arrays = dict(trace)
    arrays["object_poses"] = np.zeros((2, 3, 7))
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    with pytest.raises(ValueError, match="object_poses"):
        validate_trace(folder, metrics)


def test_control_loop_settling_replan_terminal_and_failure(monkeypatch, runtime):
    from sgvla.baselines.openpi import rollout
    from sgvla.eval.run_libero import run_control_loop

    monkeypatch.setattr(rollout, "observation_element", lambda *args: {"prompt": "task"})
    actions, captures, calls = [], [], []

    def step(action):
        actions.append(action)
        return {}, 0, len(actions) == 17, {}

    def infer(element):
        calls.append(element)
        return {"actions": np.full((50, 7), len(calls), dtype=float)}

    trace = SimpleNamespace(
        capture=lambda obs, prompt, phase, action=None: captures.append((phase, action)),
        inference=lambda *args: None,
    )
    result = run_control_loop(
        SimpleNamespace(step=step),
        {},
        "task",
        SimpleNamespace(infer=infer),
        SimpleNamespace(LIBERO_DUMMY_ACTION=[0] * 6 + [-1]),
        runtime.config,
        trace,
    )
    assert result == {"success": True, "steps": 7}
    assert [phase for phase, _ in captures] == [0] * 10 + [1] * 7 + [2]
    assert [a[0] for a in actions[10:]] == [1] * 5 + [2] * 2
    assert len(calls) == 2


def test_seeded_client_retry_and_input_integrity():
    from sgvla.baselines.openpi.seeded import RNG_KEY, SeededClient
    from sgvla.eval.manifest import PROTOCOL

    requests = []

    def infer(obs):
        requests.append(obs)
        if len(requests) == 1:
            raise ConnectionError("fixture")
        return {"actions": np.zeros((50, 7))}

    client = SimpleNamespace(
        infer=infer, get_server_metadata=lambda: {"sgvla": {"rng_protocol": PROTOCOL}}
    )
    seeded = SeededClient(client, 42)
    obs = {"prompt": "task"}
    with pytest.raises(ConnectionError):
        seeded.infer(obs)
    seeded.infer(obs)
    seeded.infer(obs)
    assert requests[0][RNG_KEY] == requests[1][RNG_KEY]
    assert requests[2][RNG_KEY]["call"] == 1
    assert RNG_KEY not in obs


def test_parity_rejects_observation_or_action_or_success_mismatch():
    from sgvla.eval.parity import compare

    reference = {
        "hashes": ["a"],
        "success": True,
        "steps": 1,
        "actions": np.zeros((1, 7)),
        "chunks": np.zeros((1, 50, 7)),
    }
    assert compare(reference, reference)["passed"]
    for key, value in (
        ("hashes", ["b"]),
        ("success", False),
        ("steps", 2),
        ("actions", np.ones((1, 7))),
        ("chunks", np.ones((1, 50, 7))),
    ):
        assert not compare(reference, {**reference, key: value})["passed"]


def test_first_batch_inspection_and_invalidation(runtime, manifest):
    from sgvla.eval.reporting import require_review, review_first_batch, summarize

    with pytest.raises(RuntimeError, match="five"):
        review_first_batch(runtime, manifest)
    for entry in manifest["episodes"][:5]:
        engine.execute_one(runtime, manifest, entry, fixture_episode)
    with pytest.raises(RuntimeError, match="eval-review"):
        require_review(runtime, manifest)
    review = review_first_batch(runtime, manifest)
    assert review["passed"] and len(review["failures"]) == 5
    require_review(runtime, manifest)
    summary = summarize(runtime, manifest)
    assert summary["completed"] == 5 and summary["successes"] == 0
    assert summary["pending_or_invalid"] == 45
    pointer = engine.evaluation_root(runtime, manifest) / "episodes/E0-t00-s00/complete.json"
    record = read_json(pointer)
    record["completed_at_utc"] = "changed"
    write_json(pointer, record)
    with pytest.raises(RuntimeError, match="changed"):
        require_review(runtime, manifest)


def test_missing_parity_blocks_e0(runtime, manifest):
    from sgvla.eval.parity import require_parity

    with pytest.raises(RuntimeError, match="eval-parity"):
        require_parity(runtime, manifest)


def test_batch_boundary_and_resume(runtime, manifest, monkeypatch):
    from sgvla.eval.reporting import review_first_batch

    monkeypatch.setattr(engine, "ensure_sources", lambda *_: None)
    monkeypatch.setattr("sgvla.eval.parity.require_parity", lambda *_: None)
    monkeypatch.setattr("sgvla.eval.run_libero.run_episode", fixture_episode)
    first = engine.run_batch(runtime, manifest, 10)
    assert len(first["episodes"]) == 5 and first["summary"]["completed"] == 5
    with pytest.raises(RuntimeError, match="eval-review"):
        engine.run_batch(runtime, manifest, 10)
    review_first_batch(runtime, manifest)
    second = engine.run_batch(runtime, manifest, 5)
    assert [e["episode_id"] for e in second["episodes"]] == [
        e["episode_id"] for e in manifest["episodes"][5:10]
    ]
    assert second["summary"]["completed"] == 10


def test_batch_stops_and_retries_an_infrastructure_error(runtime, manifest, monkeypatch):
    monkeypatch.setattr(engine, "ensure_sources", lambda *_: None)
    monkeypatch.setattr("sgvla.eval.parity.require_parity", lambda *_: None)

    def broken(*_):
        raise ConnectionError("CPU fixture")

    monkeypatch.setattr("sgvla.eval.run_libero.run_episode", broken)
    failed = engine.run_batch(runtime, manifest)
    assert len(failed["episodes"]) == 1 and failed["summary"]["completed"] == 0
    assert failed["summary"]["success_rate"] is None
    monkeypatch.setattr("sgvla.eval.run_libero.run_episode", fixture_episode)
    retried = engine.run_batch(runtime, manifest)
    assert retried["episodes"][0]["episode_id"] == failed["episodes"][0]["episode_id"]
    assert retried["summary"]["completed"] == 5


def test_killed_attempt_is_indexed_without_inventing_elapsed_time(runtime, manifest):
    entry = manifest["episodes"][0]
    episode = engine.evaluation_root(runtime, manifest) / "episodes" / entry["episode_id"]
    orphan = episode / "attempt-killed"
    write_json(
        orphan / "manifest.json", {"entry": entry, "context": engine.context(runtime, manifest)}
    )
    write_json(orphan / "config.json", runtime.config)
    engine.execute_one(runtime, manifest, entry, fixture_episode)
    with (runtime.results / "registry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["run_type"] == "E0_interrupted"
    assert rows[0]["wall_clock_seconds"] == ""
    assert read_json(orphan / "interrupted.json")["wall_clock_seconds"] is None
    assert rows[1]["run_type"] == "E0"


def test_source_checksum_detects_local_initial_file_changes(tmp_path):
    # CPU source inspection remains reproducible independently of the simulator.
    source = REPO.parent / "third_party/openpi"
    if not source.exists():
        pytest.skip("Read-only upstream checkout not present in standalone clone")
    from sgvla.eval.scenes import verify_scenes

    report = verify_scenes(source)
    assert report == read_json(REPO / "tests/fixtures/scene_report.json")
    assert all(t["initial_state_count"] == 50 for t in report["tasks"])
    assert digest(source / "examples/libero/main.py") == report["official_client_sha256"]
