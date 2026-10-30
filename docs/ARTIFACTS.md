# Artifacts and experiment provenance by Part

## What belongs in the codebase

- Part 1: `study/` contains source rows, generated matrix, comparison synthesis, diffusion/flow-matching note, and reading notes. Prepared notes do not certify the student's personal reading.
- Part 2: [the smoke package](../results/part2/smoke_2026-10-28/README.md) retains the original config, GPU/probe/gate reports, dummy actions, smoke metrics/manifests/trajectory and two videos. `results/registry.csv` is the original collected registry, including diagnostic render attempts.
- Part 3: historical reports/reviews remain under `results/part3/`; the fresh 2026-10-30 run has a complete local archive with all 105 videos, traces and manifests at `Homework/NganLe/artifacts/part2/2026-10-30-rerun/`. See the fresh-run section below. Part 4 has no experiment artifacts yet.
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


## Fresh Spatial rerun - 2026-10-30

Local review assets and machine-readable reports are retained at
`Homework/NganLe/artifacts/part2/2026-10-30-rerun/` in the parent workspace. This
folder contains the final Part 3 report, source snapshot and deployment bundle, all
condition-level first/final visual-review ZIPs and contact sheets, five JSON review
notes, the exact 12 S2 processed policy inputs, the S2 visibility audit, and paired
S2 overview images. The report SHA-256 is
`57e662b315f4d58e481573be1fac45bce7c6f1c0ceaea7aaa5833e9c4b517bb5`.

The complete validated archive is
`Homework/NganLe/artifacts/part2/2026-10-30-rerun/spatial-rerun-2026-10-30-evidence.zip` (68,431,677 bytes; SHA-256 `0dc29398ee65d255732c5fd3aa5ca099b0ba2d4dd1129e961b37d90f72824c86`). It contains 105 indexed episodes and 1,291 files whose hashes were checked after download. The ZIP CRC passed. A copy of its manifest and the local verification record are `archive_ready.json` and `archive_validation.json`; the extracted evidence tree is `raw-evidence/`.
It contains all 105 indexed episodes with both camera videos, numerical traces, metrics, manifests, the runtime registry, source snapshot, runtime-adjustment record, protocols, logs, and reviews. The archive build deep-validated video decoding and numeric traces before packaging; local member hashes matched that validated manifest after download. The new run is separate from the historical 48/50 E0 report.

The top-level `results/registry.csv` preserves its six earlier rows and appends all 108 rows from the fresh runtime registry. The archive retains the original runtime registry file. Fresh registry paths use the original Molab absolute paths; use `exports/episode_index.csv` and the extracted archive tree for local navigation.
