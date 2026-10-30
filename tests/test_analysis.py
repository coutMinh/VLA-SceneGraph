"""CPU correctness evidence only; synthetic trajectories are not policy results."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from sgvla.analysis import engine
from sgvla.analysis.manifest import build, validate
from sgvla.analysis.outcomes import measure_trace
from sgvla.analysis.reporting import paired_statistics
from sgvla.analysis.scene import RULES
from sgvla.eval.manifest import build_manifest, fingerprint
from sgvla.eval.scenes import BOWL1, BOWL2, PLATE
from sgvla.io import read_json, write_json
from sgvla.runtime import REPO, Runtime


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = read_json(REPO / "configs/part2_smoke.json")
    write_json(tmp_path / "config.json", config)
    runtime = Runtime(tmp_path, tmp_path / "config.json")
    source = read_json(REPO / "results/part2/scene_verification/scene_report.json")
    write_json(tmp_path / "docs/scene_report.json", source)
    e0 = build_manifest(source, config)
    baseline_root = tmp_path / "baseline"
    summary = {
        "context": {"manifest_sha256": fingerprint(e0), "policy_sha256": runtime.policy_hash}
    }
    write_json(baseline_root / "summary.json", summary)
    monkeypatch.setattr(
        "sgvla.analysis.baseline.load_baseline", lambda _: (e0, baseline_root, summary, {})
    )
    report = {"rules": RULES, "examined": 30, "manifest_sha256": fingerprint(e0), "rows": []}
    for entry in e0["episodes"]:
        if entry["init_state_id"] > 2:
            continue
        report["rows"].append(
            {
                "episode_id": entry["episode_id"],
                "s1": {"eligible": entry["task_id"] != 0},
                "s1_exclusion_reason": "ambiguous" if entry["task_id"] == 0 else None,
            }
        )
    path = tmp_path / "verification.json"
    write_json(path, report)
    manifest = build(runtime, "S1", path)
    monkeypatch.setattr(engine, "implementation_hash", lambda: "c" * 64)
    monkeypatch.setattr(engine, "git_commit", lambda _: "b" * 40)
    monkeypatch.setattr("sgvla.registry.git_commit", lambda _: "b" * 40)
    return runtime, manifest


def fixture_episode(runtime, entry, folder, *, success=False):
    import imageio.v2 as imageio

    steps = 2
    history, post = (40, 10) if entry["condition"].startswith("T1") else (0, 0)
    prefix = 10 + history + post
    n = prefix + steps + 1
    for camera in ("agentview", "wrist"):
        with imageio.get_writer(str(folder / (camera + ".mp4")), fps=10) as writer:
            for _ in range(n):
                writer.append_data(np.zeros((224, 224, 3), np.uint8))
    placed = np.zeros((n, 2), bool)
    placed[-1, 0 if entry["target_object"] == BOWL1 else 1] = success
    np.savez_compressed(
        folder / "trajectory.npz",
        state=np.zeros((n, 8)),
        object_names=[BOWL1, BOWL2, PLATE],
        object_poses=np.zeros((n, 3, 7)),
        sim_state=np.zeros((n, 30)),
        actions=np.zeros((n - 1, 7)),
        phase=[0] * 10 + [3] * history + [4] * post + [1] * steps + [2],
        bowl_on_plate=placed,
        policy_chunks=np.zeros((1, 50, 7)),
        policy_call_frames=[prefix],
        policy_call_seconds=[0.1],
    )
    write_json(folder / "intervention.json", {"kind": "synthetic_test_only"})
    write_json(folder / "endpoint_check.json", {"applicable": False})
    return {
        "status": "completed",
        "success": success,
        "steps": steps,
        "frames": n,
        "object_names": [BOWL1, BOWL2, PLATE],
        "settling_steps": 10,
        "history_steps": history,
        "post_history_steps": post,
        "replan_steps": 5,
        "policy_observation_hashes": ["d" * 64],
        "policy_call_seconds": [0.1],
        "initial_state_sha256": entry["initial_state_sha256"],
        "target_object": entry["target_object"],
        "policy_seed": entry["policy_seed"],
        "prompt": entry["prompt"],
        "condition": entry["condition"],
        **measure_trace(folder / "trajectory.npz", entry["target_object"]),
    }


def test_verified_subset_retains_pair_ids_and_noise(setup):
    runtime, manifest = setup
    assert len(manifest["episodes"]) == 27
    assert len(manifest["excluded"]) == 3
    assert manifest["episodes"][0]["episode_id"] == "S1-t01-s00"
    assert all(e["target_object"] == BOWL2 for e in manifest["episodes"])
    assert validate(manifest, runtime.config) == fingerprint(manifest)


@pytest.mark.parametrize(
    "key,value",
    [("prompt", "new prompt"), ("policy_seed", 0), ("init_state_id", 4), ("target_object", BOWL1)],
)
def test_rejects_condition_mutations(setup, key, value):
    runtime, manifest = setup
    changed = copy.deepcopy(manifest)
    changed["episodes"][0][key] = value
    with pytest.raises(ValueError):
        validate(changed, runtime.config)


def test_official_bowl1_done_does_not_stop_bowl2_episode(monkeypatch):
    from sgvla.analysis.rollout import TargetEnv

    raw = SimpleNamespace(steps=0)

    def step(action):
        raw.steps += 1
        return {}, 0, True, {}

    raw.step = step
    monkeypatch.setattr("sgvla.analysis.rollout.predicate", lambda env, *args: env.steps == 2)
    target = TargetEnv(raw, BOWL2)
    assert target.step([0] * 7)[2] is False
    assert target.step([0] * 7)[2] is True


def test_completed_task_failure_resumes_and_corruption_retries(setup):
    runtime, manifest = setup
    entry = manifest["episodes"][0]
    first = engine.execute_one(runtime, manifest, entry, fixture_episode)
    assert first["status"] == "completed" and not first["success"]
    assert engine.execute_one(runtime, manifest, entry, fixture_episode)["status"] == "skipped"
    item = engine.completed(runtime, manifest)[entry["episode_id"]]
    (item[2] / "agentview.mp4").write_bytes(b"corrupt")
    assert engine.completed(runtime, manifest)[entry["episode_id"]] is None
    retried = engine.execute_one(runtime, manifest, entry, fixture_episode)
    assert retried["status"] == "completed"
    episode = engine.root(runtime, manifest) / "episodes" / entry["episode_id"]
    assert len(list(episode.glob("attempt-*"))) == 2


def test_infrastructure_error_is_pending_then_retry_succeeds(setup):
    runtime, manifest = setup
    entry = manifest["episodes"][0]

    def broken(*args):
        raise ConnectionError("synthetic policy disconnect")

    assert engine.execute_one(runtime, manifest, entry, broken)["status"] == "infrastructure_error"
    assert engine.summarize(runtime, manifest)["completed"] == 0
    engine.execute_one(runtime, manifest, entry, fixture_episode)
    assert engine.summarize(runtime, manifest)["completed"] == 1


def test_interrupted_attempt_is_retained_and_indexed(setup):
    runtime, manifest = setup
    entry = manifest["episodes"][0]
    partial = engine.root(runtime, manifest) / "episodes" / entry["episode_id"] / "attempt-killed"
    write_json(partial / "config.json", runtime.config)
    write_json(
        partial / "manifest.json", {"entry": entry, "context": engine.context(runtime, manifest)}
    )
    engine.execute_one(runtime, manifest, entry, fixture_episode)
    assert read_json(partial / "interrupted.json")["status"] == "interrupted"


def test_identity_metric_accepts_bowl2_and_rejects_wrong_terminal_goal(setup, tmp_path):
    runtime, manifest = setup
    entry = manifest["episodes"][0]
    metrics = fixture_episode(runtime, entry, tmp_path, success=True)
    engine.validate_trace(tmp_path, metrics)
    assert metrics["first_placed"] == BOWL2
    assert metrics["exclusive_first_target_success"] is True
    changed = {**metrics, "target_object": BOWL1}
    with pytest.raises(ValueError, match="selected target"):
        engine.validate_trace(tmp_path, changed)


def test_temporal_trace_counts_include_history_and_terminal(setup, tmp_path):
    runtime, manifest = setup
    entry = {**manifest["episodes"][0], "condition": "T1", "target_object": BOWL1}
    metrics = fixture_episode(runtime, entry, tmp_path)
    assert metrics["policy_start_frame"] == 60
    engine.validate_trace(tmp_path, metrics)
    with pytest.raises(ValueError, match="Prelude|trace length"):
        engine.validate_trace(tmp_path, {**metrics, "history_steps": 0})


def test_relation_gate_rejects_two_nearby_bowls(monkeypatch):
    from sgvla.analysis.scene import relation

    monkeypatch.setattr(
        "sgvla.analysis.scene.poses",
        lambda env: {
            BOWL1: np.array([0.13, 0, 0.8]),
            BOWL2: np.array([0, 0.13, 0.8]),
            "glazed_rim_porcelain_ramekin_1": np.array([0, 0, 0.8]),
        },
    )
    row = {
        "initial_relations": {BOWL2: ["On", BOWL2, "region"]},
        "s1_instruction": "pick up the black bowl next to the ramekin and place it on the plate",
    }
    assert relation(None, row, BOWL2)["eligible"] is False


def test_exact_discordant_pair_statistic():
    result = paired_statistics([(True, False)] * 4 + [(True, True)] * 3)
    assert result["exact_discordant_two_sided_p"] == 0.125
    assert result["success_difference"] == -4 / 7
    assert paired_statistics([])["success_difference"] is None


def test_stale_managed_pid_is_never_signalled(tmp_path, monkeypatch):
    from sgvla.analysis import jobs

    runtime = SimpleNamespace(work=tmp_path)
    folder = tmp_path / "jobs/part3/job-old"
    write_json(folder / "state.json", {"pid": 123, "process_start_ticks": "old"})
    monkeypatch.setattr(jobs, "process_identity", lambda _: "new")
    signalled = []
    monkeypatch.setattr(jobs.os, "killpg", lambda *args: signalled.append(args), raising=False)
    assert jobs.stop(runtime, "job-old")["status"] == "interrupted"
    assert not signalled


def test_history_prompt_is_fixed_and_grammatical():
    from sgvla.analysis.temporal_verify import history_prompt

    assert history_prompt("pick up the black bowl from table center and place it on the plate") == (
        "pick up the black bowl that was at table center at the start and place it on the plate"
    )


def test_first_batch_requires_review_before_sixth_episode(setup, monkeypatch):
    runtime, manifest = setup
    monkeypatch.setattr("sgvla.eval.engine.ensure_sources", lambda *args: None)
    monkeypatch.setattr("sgvla.analysis.rollout.run_episode", fixture_episode)
    first = engine.run_batch(runtime, manifest, 5)
    assert first["summary"]["completed"] == 5
    with pytest.raises(RuntimeError, match="Review the first batch"):
        engine.run_batch(runtime, manifest, 5)
    engine.review(runtime, manifest)
    assert engine.run_batch(runtime, manifest, 1)["summary"]["completed"] == 6


def test_temporal_plan_rejects_nonidentical_endpoint_inputs(setup):
    from sgvla.io import digest

    runtime, _ = setup
    verification = runtime.work / "verification.json"
    checks = read_json(verification)
    temporal = {
        "complete": True,
        "examined": 30,
        "rules": RULES,
        "relation_verification_sha256": digest(verification),
        "rows": [],
    }
    for row in checks["rows"]:
        task = int(row["episode_id"].split("-t")[1].split("-s")[0])
        temporal["rows"].append(
            {
                "episode_id": row["episode_id"],
                "eligible": task in (0, 1, 2, 8),
                "endpoint_equal": {"image": True, "state": True, "prompt": True},
                "endpoint_hashes": ["f" * 64, "f" * 64],
                "exclusion_reason": "unverified",
            }
        )
    path = runtime.work / "temporal.json"
    write_json(path, temporal)
    assert len(build(runtime, "T1", verification, path)["episodes"]) == 12
    temporal["rows"][0]["endpoint_equal"]["image"] = False
    write_json(path, temporal)
    with pytest.raises(ValueError, match="endpoint inputs differ"):
        build(runtime, "T1", verification, path)


def test_temporal_state_permutation_preserves_every_nonbowl_component(monkeypatch):
    from sgvla.analysis import interventions

    base = np.arange(28, dtype=float)
    env = SimpleNamespace(current=base.copy())

    def set_state(state):
        env.current = state.copy()

    env.set_state = set_state
    env.get_sim_state = lambda: env.current.copy()
    spans = {"a": (slice(1, 8), slice(15, 21)), "b": (slice(8, 15), slice(21, 27))}
    data = SimpleNamespace(
        get_joint_qpos=lambda name: env.current[spans[name][0]],
        get_joint_qvel=lambda name: env.current[spans[name][1]],
        set_joint_qpos=lambda name, value: env.current.__setitem__(spans[name][0], value),
        set_joint_qvel=lambda name, value: env.current.__setitem__(spans[name][1], value),
    )
    env.sim = SimpleNamespace(data=data, forward=lambda: None)
    monkeypatch.setattr(interventions, "joints", lambda _: ["a", "b"])
    swapped = interventions.permuted_state(env, base, True)
    np.testing.assert_array_equal(swapped[[0, 27]], base[[0, 27]])
    np.testing.assert_array_equal(swapped[1:8], base[8:15])
    np.testing.assert_array_equal(swapped[15:21], base[21:27])
    np.testing.assert_array_equal(interventions.permuted_state(env, swapped, True), base)


def test_temporal_bound_is_withheld_when_poststart_inputs_differ(setup, tmp_path):
    from sgvla.analysis.reporting import temporal_prefix

    runtime, manifest = setup
    folders = [tmp_path / "left", tmp_path / "right"]
    for folder in folders:
        folder.mkdir()
    entry = {**manifest["episodes"][0], "condition": "T1", "target_object": BOWL1}
    left = fixture_episode(runtime, entry, folders[0], success=True)
    right = fixture_episode(runtime, entry, folders[1], success=True)
    right["policy_observation_hashes"] = ["e" * 64]
    result = temporal_prefix((None, left, folders[0]), (None, right, folders[1]))
    assert result["exclusive_first_target_success_sum"] == 2
    assert result["bound_applicable_to_recorded_pair"] is False
    assert result["bound_satisfied"] is None
