"""CPU checks for evidence, process lifecycle and protocol-preserving smoke behavior."""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sgvla.gate import check_smoke
from sgvla.io import digest, read_json, write_json
from sgvla.jobs import process_identity, start, status, stop
from sgvla.runtime import REPO, Runtime


@pytest.fixture
def runtime(tmp_path):
    config = tmp_path / "configs/part2_smoke.json"
    values = read_json(REPO / "configs/part2_smoke.json")
    values["probe_network"] = False
    write_json(config, values)
    return Runtime(tmp_path, config)


def test_importing_control_plane_does_not_import_ml():
    code = "import sgvla.cli, sys; assert not {'numpy','torch','jax','libero'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_partial_gpu_evidence_does_not_pass_smoke(runtime):
    report = check_smoke(runtime)
    assert report["gate"] == "part2_smoke"
    assert read_json(runtime.work / "docs/smoke_check.json") == report
    assert report["passed"] is False
    assert not any(report["checks"].values())
    write_json(
        runtime.work / "docs/backend_check.json",
        {
            "config_sha256": runtime.config_hash,
            "jax": {"ok": True},
        },
    )
    assert check_smoke(runtime)["checks"] == {"gpu_kernel": True, "dummy": False, "smoke": False}


def make_evidence(runtime):
    fingerprint = runtime.config_hash
    write_json(
        runtime.work / "docs/backend_check.json",
        {
            "config_sha256": fingerprint,
            "jax": {"ok": True},
        },
    )
    for stage in ("dummy", "smoke"):
        folder = runtime.results / "runs" / stage
        folder.mkdir(parents=True)
        result = {"status": "completed", "config_sha256": fingerprint}
        if stage == "dummy":
            result.update(
                finite=True,
                shape=[50, 7],
                server_metadata={"sgvla": {"config_sha256": fingerprint}},
            )
        else:
            result.update(steps=7, decoded_video_frames=7, success=False)
            for name in ("video", "wrist_video"):
                video = folder / (name + ".mp4")
                video.write_bytes(b"test fixture; real decoder is checked by the runtime")
                result[name] = str(video)
                result[name + "_sha256"] = digest(video)
        metrics = folder / "metrics.json"
        write_json(metrics, result)
        write_json(
            runtime.work / "docs" / (stage + ".json"),
            {
                "metrics_path": str(metrics),
                "metrics_sha256": digest(metrics),
            },
        )


def test_smoke_gate_accepts_completed_episode_even_if_policy_failed_task(runtime):
    make_evidence(runtime)
    assert check_smoke(runtime)["passed"] is True


@pytest.mark.parametrize("change", ["config", "video", "metrics"])
def test_smoke_gate_rejects_stale_or_changed_artifacts(runtime, change):
    make_evidence(runtime)
    if change == "config":
        config = runtime.config
        config["render_backend"] = "egl" if config["render_backend"] == "osmesa" else "osmesa"
        write_json(runtime.config_path, config)
    elif change == "video":
        (runtime.results / "runs/smoke/video.mp4").write_bytes(b"truncated")
    else:
        path = runtime.results / "runs/dummy/metrics.json"
        result = read_json(path)
        result["shape"] = [1, 7]
        write_json(path, result)
    assert check_smoke(runtime)["passed"] is False


def test_registry_is_idempotent_and_preserves_provenance(runtime, monkeypatch):
    from sgvla import registry

    monkeypatch.setattr(registry, "git_commit", lambda _: "a" * 40)
    folder = runtime.results / "runs/test"
    write_json(folder / "manifest.json", {"kind": "CPU fixture"})
    write_json(folder / "metrics.json", {"status": "completed"})
    for _ in range(2):
        registry.register(runtime, "test", "fixture", folder / "metrics.json", 1.25)
    with (runtime.results / "registry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["data_manifest_hash"] == digest(folder / "manifest.json")
    assert rows[0]["config_sha256"] == runtime.config_hash
    assert rows[0]["metrics_json_path"] == str(folder / "metrics.json")


def test_smoke_matches_official_action_order_and_early_termination(runtime, monkeypatch):
    np = pytest.importorskip("numpy")
    from sgvla.baselines.openpi import rollout

    # Fake only the IO boundary; the actual loop must settle, replan at 5 and stop at done.
    def element(obs, prompt, official, resize_size):
        return {
            "observation/image": np.zeros((2, 2, 3)),
            "observation/wrist_image": np.zeros((2, 2, 3)),
            "observation/state": np.zeros(8),
            "prompt": prompt,
        }

    monkeypatch.setattr(rollout, "observation_element", element)
    calls, executed = [], []

    def step(action):
        executed.append(action)
        return {}, 0, len(executed) == 17, {}

    def infer(obs):
        calls.append(obs)
        base = (len(calls) - 1) * 100
        return {"actions": np.repeat(np.arange(base, base + 50)[:, None], 7, axis=1)}

    official = SimpleNamespace(LIBERO_DUMMY_ACTION=[0] * 6 + [-1])
    trace = rollout.run_steps(
        SimpleNamespace(step=step),
        {},
        "task",
        SimpleNamespace(infer=infer),
        official,
        runtime.config,
    )
    assert executed[:10] == [official.LIBERO_DUMMY_ACTION] * 10
    assert [a[0] for a in executed[10:]] == [0, 1, 2, 3, 4, 100, 101]
    assert len(calls) == 2 and trace["steps"] == 7 and trace["success"] is True


def test_invalid_chunks_are_rejected_before_simulator_step():
    np = pytest.importorskip("numpy")
    from sgvla.baselines.openpi.rollout import validate_actions

    for actions in (np.zeros((4, 7)), np.zeros((50, 8)), np.full((50, 7), np.nan)):
        with pytest.raises(ValueError):
            validate_actions(actions, 5)


def test_init_config_does_not_overwrite_runtime_choices(runtime):
    config = runtime.config
    config["seed"] += 5
    write_json(runtime.config_path, config)
    legacy = runtime.work / "configs/d1.json"
    write_json(legacy, {**config, "seed": config["seed"] + 1})
    subprocess.run(
        [sys.executable, "-m", "sgvla", "--work", str(runtime.work), "init-config"],
        check=True,
        capture_output=True,
    )
    assert runtime.config == config


def test_init_config_migrates_legacy_values_and_preserves_hash(tmp_path):
    legacy = tmp_path / "configs/d1.json"
    values = read_json(REPO / "configs/part2_smoke.json")
    values.update(seed=19, render_backend="egl")
    write_json(legacy, values)
    original = legacy.read_bytes()
    expected_hash = Runtime(tmp_path, legacy).config_hash
    result = subprocess.run(
        [sys.executable, "-m", "sgvla", "--work", str(tmp_path), "init-config"],
        check=True,
        text=True,
        capture_output=True,
    )
    migrated = Path(json.loads(result.stdout)["config"])
    assert migrated == tmp_path / "configs/part2_smoke.json"
    assert read_json(migrated) == values
    assert Runtime(tmp_path, migrated).config_hash == expected_hash
    assert legacy.read_bytes() == original


@pytest.mark.skipif(sys.platform != "linux", reason="molab process lifecycle uses Linux /proc")
def test_detached_probe_has_completion_duplicate_protection_and_restart_detection(runtime):
    first = start(runtime, "probe")
    again = start(runtime, "probe")
    if again["status"] == "running":
        assert first["id"] == again["id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and status(runtime, "probe")["status"] == "running":
        time.sleep(0.2)
    result = status(runtime, "probe")
    assert result["status"] == "completed"
    assert (runtime.work / "docs/molab_probe.json").exists()
    request_path = runtime.work / "jobs/probe.json"
    request = read_json(request_path)
    (runtime.work / "jobs" / (request["id"] + ".result.json")).unlink()
    request["process_identity"] = "a different boot:start identity"
    write_json(request_path, request)
    assert status(runtime, "probe")["status"] == "interrupted"
    assert process_identity(999999999) is None


def test_source_is_parseable_by_python38_grammar():
    import ast

    for path in (REPO / "src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_config_hash_ignores_formatting_but_tracks_values(runtime):
    expected = runtime.config_hash
    runtime.config_path.write_text(json.dumps(runtime.config, indent=4), encoding="utf-8")
    assert runtime.config_hash == expected
    values = runtime.config
    values["seed"] += 1
    write_json(runtime.config_path, values)
    assert runtime.config_hash != expected


@pytest.mark.skipif(sys.platform != "linux", reason="molab process groups require Linux")
def test_stop_refuses_a_reused_pid_and_can_stop_its_own_group(runtime):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        request = {
            "name": "server",
            "id": "stop-test",
            "pid": process.pid,
            "process_identity": "incorrect identity",
        }
        write_json(runtime.work / "jobs/server.json", request)
        assert stop(runtime, "server")["status"] == "interrupted"
        assert process.poll() is None
        request["process_identity"] = process_identity(process.pid)
        write_json(runtime.work / "jobs/server.json", request)
        assert stop(runtime, "server")["status"] == "stopped"
        process.wait(timeout=5)
        assert process.returncode < 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
