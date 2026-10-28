"""Install pinned upstream stacks outside the implementation repository (Linux only)."""

from __future__ import annotations

import shutil
import subprocess
import sys

from sgvla.io import read_json, utc_now, write_json
from sgvla.runtime import git_commit


def ensure_upstream(runtime):
    config = runtime.config
    upstream = runtime.upstream
    if not (upstream / ".git").exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        runtime.call(["git", "clone", "--no-checkout", config["openpi_url"], upstream])
        runtime.call(["git", "checkout", "--detach", config["openpi_commit"]], cwd=upstream)
    if git_commit(upstream) != config["openpi_commit"]:
        raise RuntimeError("Upstream commit differs from config; use a new work directory")
    changes = subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    if changes:
        raise RuntimeError("Upstream has tracked modifications; refusing to proceed: " + changes)
    runtime.call(["git", "submodule", "update", "--init", "third_party/libero"], cwd=upstream)
    if git_commit(upstream / "third_party" / "libero") != config["libero_commit"]:
        raise RuntimeError("LIBERO submodule does not match the verified pin")


def uv_command(runtime):
    binary = shutil.which("uv")
    if binary:
        return [binary]
    tools = runtime.work / "tools"
    if not (tools / "uv" / "__main__.py").exists():
        runtime.call(
            [sys.executable, "-m", "pip", "install", "--target", tools, "--no-deps", "uv==0.9.5"]
        )
    return [sys.executable, "-m", "uv"]


def install(runtime, target):
    if sys.platform != "linux":
        raise RuntimeError("Install the GPU/simulator environments on Linux molab, not Windows")
    ensure_upstream(runtime)
    uv = uv_command(runtime)
    env = runtime.env()
    env_path = runtime.work / "envs" / target
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if target == "server":
        env["UV_PROJECT_ENVIRONMENT"] = str(env_path)
        subprocess.run(
            [*uv, "sync", "--frozen", "--no-dev", "--python", "3.11"],
            cwd=runtime.upstream,
            env=env,
            check=True,
        )
    elif target == "client":
        runtime.call([*uv, "venv", "--allow-existing", "--python", "3.8", env_path])
        runtime.call(
            [
                *uv,
                "pip",
                "sync",
                "--python",
                runtime.python("client"),
                runtime.upstream / "examples/libero/requirements.txt",
                runtime.upstream / "third_party/libero/requirements.txt",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu113",
                "--index-strategy",
                "unsafe-best-match",
            ]
        )
        runtime.call(
            [
                *uv,
                "pip",
                "install",
                "--python",
                runtime.python("client"),
                "-e",
                runtime.upstream / "packages/openpi-client",
                "-e",
                runtime.upstream / "third_party/libero",
                "typing_extensions>=4.12,<5",
            ]
        )
        configure_libero(runtime)
    else:
        raise ValueError(target)
    freeze = subprocess.check_output(
        [*uv, "pip", "freeze", "--python", str(runtime.python(target))],
        env=env,
        text=True,
    )
    folder = runtime.work / "docs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (target + "_freeze.txt")).write_text(freeze, encoding="utf-8")
    write_json(
        folder / (target + "_install.json"),
        {
            "installed_at_utc": utc_now(),
            "openpi_commit": git_commit(runtime.upstream),
            "python": str(runtime.python(target)),
            "target": target,
        },
    )


def configure_libero(runtime):
    """Avoid LIBERO's interactive first-import prompt without altering its source."""
    root = runtime.upstream / "third_party/libero/libero/libero"
    folder = runtime.work / "libero_config"
    folder.mkdir(parents=True, exist_ok=True)
    # JSON string values are valid YAML scalars, including paths with spaces.
    import json

    paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "assets": root / "assets",
        "datasets": runtime.work / "data/libero",
    }
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    text = "".join(f"{key}: {json.dumps(str(value))}\n" for key, value in paths.items())
    (folder / "config.yaml").write_text(text, encoding="utf-8")


def download(runtime):
    from openpi.shared import download as upstream_download

    path = upstream_download.maybe_download(runtime.config["checkpoint"])
    if not (path / "params/_METADATA").is_file() or not (path / "assets").is_dir():
        raise RuntimeError("Downloaded checkpoint lacks params metadata or assets")
    write_json(
        runtime.work / "docs/checkpoint.json",
        {
            "downloaded_at_utc": utc_now(),
            "uri": runtime.config["checkpoint"],
            "path": str(path),
        },
    )


def convert(runtime):
    """Use the official converter; keep all patches and output outside upstream."""
    env = runtime.env()
    # Conversion restores arrays on CPU; it should not depend on working JAX CUDA kernels.
    env["JAX_PLATFORMS"] = "cpu"
    python = runtime.python("server")
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    from pathlib import Path

    site = Path(site).resolve()
    site.relative_to((runtime.work / "envs/server").resolve())
    source = runtime.upstream / "src/openpi/models_pytorch/transformers_replace"
    shutil.copytree(source, site / "transformers", dirs_exist_ok=True)
    checkpoint = Path(read_json(runtime.work / "docs/checkpoint.json")["path"])
    output = runtime.work / "checkpoints/pi0_libero_pytorch"
    subprocess.run(
        [
            str(python),
            str(runtime.upstream / "examples/convert_jax_model_to_pytorch.py"),
            "--checkpoint-dir",
            str(checkpoint),
            "--config-name",
            runtime.config["model_config"],
            "--output-path",
            str(output),
        ],
        cwd=runtime.upstream,
        env=env,
        check=True,
    )
    # At the pin the converter restores ROOT/params but copies PARENT/assets.
    # Correct only our output, never the upstream converter.
    shutil.copytree(checkpoint / "assets", output / "assets", dirs_exist_ok=True)
    write_json(
        runtime.work / "docs/conversion.json",
        {
            "converted_at_utc": utc_now(),
            "path": str(output),
            "deviation": "Copied ROOT/assets into converted output; upstream copy path differs",
            "source": str(checkpoint),
            "transformers_patched_in_isolated_env": True,
        },
    )
