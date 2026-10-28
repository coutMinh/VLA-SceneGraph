"""Layout and subprocess boundaries. The molab kernel needs only the standard library."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sgvla.io import read_json

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Runtime:
    work: Path
    config_path: Path

    @property
    def config(self):
        config = read_json(self.config_path)
        if config["backend"] not in ("jax", "pytorch"):
            raise ValueError("backend must be jax or pytorch")
        if config["render_backend"] not in ("egl", "osmesa"):
            raise ValueError("render_backend must be egl or osmesa")
        if config["model_config"] != "pi0_libero":
            raise ValueError("D1 adapter is verified only for pi0_libero")
        if config["task_suite"] != "libero_spatial" or config["max_steps"] != 220:
            raise ValueError("D1 smoke is Spatial only, with the official 220-step limit")
        for key in ("replan_steps", "num_steps", "resize_size"):
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        return config

    @property
    def config_hash(self):
        """Hash config values, so formatting changes do not invalidate a job snapshot."""
        encoded = json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def upstream(self):
        return self.work / "third_party" / "openpi"

    @property
    def results(self):
        return self.work / "results"

    def python(self, name):
        folder = self.work / "envs" / name
        return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def env(self):
        env = os.environ.copy()
        env.update(
            PYTHONUNBUFFERED="1",
            GIT_LFS_SKIP_SMUDGE="1",
            UV_LINK_MODE="copy",
            UV_CACHE_DIR=str(self.work / "cache" / "uv"),
            UV_PYTHON_INSTALL_DIR=str(self.work / "python"),
            OPENPI_DATA_HOME=str(self.work / "cache" / "openpi"),
            HF_HOME=str(self.work / "cache" / "huggingface"),
            LIBERO_CONFIG_PATH=str(self.work / "libero_config"),
            MUJOCO_GL=self.config["render_backend"],
            PYOPENGL_PLATFORM=self.config["render_backend"],
            PYTHONPATH=os.pathsep.join(
                [
                    str(REPO / "src"),
                    str(self.upstream / "third_party" / "libero"),
                    str(self.work / "tools"),
                ]
            ),
        )
        # Avoid reserving most of the GPU while another homework job is using it.
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        return env

    def call(self, command, *, cwd=None, timeout=None):
        subprocess.run(
            [str(arg) for arg in command],
            cwd=cwd or REPO,
            env=self.env(),
            check=True,
            timeout=timeout,
        )

    def module(self, env_name, module, *args, timeout=None):
        python = self.python(env_name)
        if not python.exists():
            raise RuntimeError(f"Missing {env_name} environment; run install-{env_name}")
        return self.call(
            [python, "-m", module, "--work", self.work, "--config", self.config_path, *args],
            timeout=timeout,
        )


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
