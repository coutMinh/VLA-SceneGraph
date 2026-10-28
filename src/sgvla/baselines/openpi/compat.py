"""Repair the observed legacy PyTorch ELF metadata issue in the private client env."""

from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess

from sgvla.io import digest, utc_now, write_json


def torch_import(runtime):
    return subprocess.run(
        [str(runtime.python("client")), "-c", "import torch; print(torch.__version__)"],
        env=runtime.env(),
        capture_output=True,
        text=True,
        timeout=60,
    )


def repair_client(runtime):
    check = torch_import(runtime)
    if check.returncode == 0:
        print("Client torch import already passes; no binary changes needed.")
        return
    if "cannot enable executable stack" not in check.stderr:
        raise RuntimeError("Client error is not the verified execstack issue:\n" + check.stderr)
    if shutil.which("patchelf") is None:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise RuntimeError("Install patchelf with --clear-execstack support, then retry")
        runtime.call(["apt-get", "update"], timeout=180)
        runtime.call(
            ["apt-get", "install", "-y", "--no-install-recommends", "patchelf"], timeout=180
        )
    help_result = subprocess.run(["patchelf", "--help"], capture_output=True, text=True)
    if "--clear-execstack" not in help_result.stdout + help_result.stderr:
        raise RuntimeError("Installed patchelf does not support --clear-execstack")
    root = (runtime.work / "envs/client").resolve()
    libraries = root / "lib/python3.8/site-packages/torch/lib"
    repairs = []
    for path in sorted(libraries.glob("*.so")):
        path.resolve().relative_to(root)
        flag = subprocess.check_output(["patchelf", "--print-execstack", str(path)], text=True)
        if flag.strip().split()[-1] != "X":
            continue
        before = digest(path)
        runtime.call(["patchelf", "--clear-execstack", path])
        repairs.append({"path": str(path), "before_sha256": before, "after_sha256": digest(path)})
    check = torch_import(runtime)
    write_json(
        runtime.work / "docs/client_compatibility.json",
        {
            "observed_at_utc": utc_now(),
            "issue": "legacy torch executable GNU_STACK",
            "patchelf_version": subprocess.check_output(
                ["patchelf", "--version"], text=True
            ).strip(),
            "repairs": repairs,
            "torch_import_ok": check.returncode == 0,
            "torch_version": check.stdout.strip(),
            "error": check.stderr if check.returncode else None,
        },
    )
    if check.returncode:
        raise RuntimeError("Client import still fails after ELF repair:\n" + check.stderr)
    print("Client torch import passes after clearing executable GNU_STACK metadata.")


def repair_rendering(runtime):
    """Install the system loaders observed missing from the headless molab image."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("Install libopengl0 and libosmesa6 on the host, then retry rendering")
    runtime.call(["apt-get", "update"], timeout=180)
    runtime.call(
        ["apt-get", "install", "-y", "--no-install-recommends", "libopengl0", "libosmesa6"],
        timeout=180,
    )
    versions = subprocess.check_output(
        ["dpkg-query", "-W", "libopengl0", "libosmesa6"], text=True
    ).strip()
    write_json(
        runtime.work / "docs/render_libraries.json",
        {
            "installed_at_utc": utc_now(),
            "packages": versions,
            "OpenGL": ctypes.util.find_library("OpenGL"),
            "OSMesa": ctypes.util.find_library("OSMesa"),
        },
    )
    print(versions)
