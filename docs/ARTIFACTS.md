# Artifacts and experiment provenance by Part

## What belongs in the codebase

- Part 1: `study/` contains reviewed source rows, generated matrix and reading notes.
- Part 2: [the smoke package](../results/part2/smoke_2026-10-28/README.md) retains the original config, GPU/probe/gate reports, dummy actions, smoke metrics/manifests/trajectory and two videos. `results/registry.csv` is the original collected registry, including diagnostic render attempts.
- Parts 3 and 4 have no completed experiment artifacts yet. Future reviewed result packages use `results/part3/` and `results/part4/` with an experiment ID and run ID in their reports.
- Runtime venvs, checkpoints, notebook output caches, job pointers/PID state and repeated install/debug logs stay outside the implementation repository.

## Complete raw archive and historical snapshot

The previous 98-file `docs/evidence/` tree was copied byte-for-byte and verified before removal from the current codebase. The local archive is at `Homework/NganLe/artifacts/part2/2026-10-28/evidence/` in the parent workspace. It contains the source metadata listing and full runtime `docs/`, `jobs/`, logs, registry and run artifacts. Its archival README files retain their original wording and are not current instructions.

The same complete evidence is also available in the immutable published [commit 6dd7208 snapshot](https://github.com/coutMinh/VLA-SceneGraph/tree/6dd7208/docs/evidence). Environments and checkpoint weights were never included in that archive.

## Path and label mapping

Original runtime root: `/home/marimo/vla_scenegraph_work`.

| Original location/label | Current report/storage |
|---|---|
| `$SGVLA_WORK/results/runs/dummy-d575feeb4253/` and `smoke-a081cbb0b0ab/` | `results/part2/smoke_2026-10-28/runs/<original-run-id>/` |
| `$SGVLA_WORK/results/registry.csv` | `results/registry.csv`, original rows and paths preserved |
| `$SGVLA_WORK/configs/d1.json` used in the run | Retained `config.json`; current source/runtime default is `configs/part2_smoke.json` |
| Historical `G1`, `g1.json` | Part 2 smoke check; raw report is `checks/g1.json`; new CLI output is `docs/smoke_check.json` with label `part2_smoke` |
| Other raw runtime paths | Local archive `evidence/molab/work/<runtime-relative-path>` or the immutable Git snapshot above |

Absolute paths inside original metrics/registry were not rewritten. Run IDs, raw manifest hashes, config values, metrics and video bytes remain unchanged. Git line-ending conversion is disabled under `results/` to preserve those hashes across platforms. Read the Part 2 package README for file links; do not attempt to open original molab absolute paths on Windows. A completed smoke validates execution only; no E0 success rate can be derived from it.

The notebook source moved from `d1.py` to the external `part2_reproduce.py`. Generated `__marimo__` JSON is disposable local UI state and is ignored. Existing runtime config values are migrated without overwriting choices or changing their hash. This reorganization collected no new GPU results.
