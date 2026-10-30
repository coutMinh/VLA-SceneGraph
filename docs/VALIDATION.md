# Validation · Part 2 setup and smoke

## Initial E0 implementation validation — 2026-10-29

- Windows Python 3.12: **37 passed, 2 skipped**. The skips are existing Linux
  process-lifecycle checks. At this initial implementation-validation stage, GPU/model
  inference and physics were not run or mocked as research results.
- Tests include real FFmpeg MP4 encoding/decoding, partial/killed attempts, corrupt
  videos/traces/config/pointers, completed task-failure resume, infrastructure-error
  retry/denominator exclusion, duplicate registry protection and interrupted-run indexing.
- Batch tests enforce the five-episode inspection boundary and resume at the next
  manifest entry. Parity comparisons reject observation/action/outcome differences;
  missing parity prevents E0. The later live official-client/GPU comparison passed; see
  the runtime record below.
- The source verifier read all pinned Spatial BDDL/initial-state archives using NumPy
  and a restricted archive reader (no torch/MuJoCo). The generated report reproduces
  exactly; `configs/e0_manifest.json` matches the manifest builder and contains the
  declared task/state selection. Source evidence is in `results/part2/scene_verification/`.
- Ruff lint and format checks pass. Source parses under Python 3.8 grammar; importing
  CLI/evaluator control modules does not import NumPy/JAX/torch/LIBERO.
- Both external notebooks (`part2_reproduce.py`, `part2_e0.py`) pass strict marimo
  structure checks. The sandbox prints the same nonfatal inaccessible user-config
  diagnostics observed previously; the checker exits successfully.
- Original retained smoke/dummy artifact hashes still match their manifests.
  No upstream source edits, remote pushes or new GPU experiments were performed.

## Live Molab runtime validation — 2026-10-29

- Molab was fast-forwarded from the older checkout to the latest pushed implementation,
  commit `1c98fd3efece9194330a038b2c62b7cb2d6a1046`, before setup or evaluation.
- RTX PRO 6000 Blackwell Server Edition; policy server used JAX 0.5.3 on CUDA. The
  pinned pi0_libero checkpoint loaded. The official client completed rendering with OSMesa.
- Parity passed for task 0 and initial states 0, 1, 2. Policy inputs were byte-identical;
  actions, chunks, observations, step counts and success results matched (atol 1e-5,
  rtol 0), with policy RNG controlled.
- Fixed manifest: 50 Spatial episodes, tasks 0-9 × states 0-4, seed 7; SHA-256
  `fe252dd6412e9b88c1ae02e7356826a983bf6c335404013cc4b8f96e0897bd82`. The first batch
  passed artifact review before the remaining workload ran.
- Final validated count: 50/50; 48 success, 2 task failures at the 220-step cap, 0
  infrastructure errors, 0 pending/invalid. Overall success 0.96, Wilson 95% interval
  [0.865399, 0.988961]; interval does not adjust for within-task dependence. Task 1 and
  task 7 each had 4/5 successes; all other tasks had 5/5.
- The two failed episodes were `E0-t01-s01` and `E0-t07-s03`. The evaluator's saved
  videos/traces passed decoding and integrity checks. Manual visual trajectory review
  remains pending. Raw run and summary are on Molab at
  `/home/marimo/vla_scenegraph_work/results/part2/e0/fe252dd6412e9b88c1ae02e7356826a983bf6c335404013cc4b8f96e0897bd82/`.
  A concise local record is [here](../results/part2/e0/molab-2026-10-29/README.md).

The pilot is a reduced run, not a full official-suite reproduction. Its result does not
establish scene-graph improvements; S1/S2/T1 comparison conditions remain future work.

## Part-based organization and compatibility checks

2026-10-28, Windows Python 3.12, after the rename/cleanup:

- CPU checks: **13 passed, 2 skipped**; skips require Linux process lifecycle support.
- Config migration preserves legacy runtime values/hash and leaves existing new configs intact. The smoke checker emits the new Part 2 label/path.
- The existing stale-config check now toggles the actual renderer value, including when OSMesa is the default.
- Retained artifact manifests match all seven hashed dummy/smoke files; canonical config hash is unchanged. The raw registry is preserved.
- Full raw evidence (98 files) was copied and SHA-256 verified before removing the redundant `docs/evidence/` tree. The current result package has 14 files plus the shared registry. See [ARTIFACTS.md](ARTIFACTS.md).
- Ruff lint/format and generated literature checks pass. The external renamed notebook passes strict marimo structure checks; sandbox logging about an inaccessible user-level marimo config does not affect its successful exit.
- Local links in the active plan, handoff, reports and notebook guides resolve. No new GPU experiment was run; the completed smoke remains Part 2 setup validation, not E0.

The historical Linux CI evidence below predates this reorganization. No latest-commit remote CI pass is claimed here.

## Earlier implementation validation

2026-10-28, Windows Python 3.12, private project venv (before the part-based rename):

- Ruff lint and format checks pass.
- CPU tests: **12 passed, 2 skipped**. Skips are Linux process lifecycle tests;
  no GPU/model inference is simulated as an experiment result.
- Matrix Markdown matches the reviewed CSV (`render_literature.py --check`).
- External notebook passes `marimo check --strict` with marimo 0.25.2.
- External notebook was executed/exported from a clean local clone; bootstrap,
  runtime config and initial status display worked without launching GPU jobs.
- openpi tracked sources have no local modifications.

Tests cover action chunk validation, settling/replanning/early termination,
artifact/config integrity for the smoke check, registry idempotency/provenance, config
preservation, Python 3.8 grammar and a dependency-free orchestration import.
Linux process tests also cover detached job completion, duplicate protection,
restart/PID reuse detection and signalling only a managed process group.

Linux tests **passed: 14 tests on each of Python 3.8, 3.11 and 3.13**, including
the two Linux lifecycle tests. Ruff checks and matrix rendering checks also passed.
Evidence: [GitHub Actions run 40647871663](https://github.com/coutMinh/VLA-SceneGraph/actions/runs/40647871663),
implementation commit `23b23d2f8ce20aa6cbc088315d1d6d7e811820fe`.

WSL dependencies were prepared locally, but the WSL test execution command was
denied; no WSL test success is claimed.

Molab evidence collected in the live runtime on 2026-10-28 is archived under
[the Part 2 result package](../results/part2/smoke_2026-10-28/):

- RTX PRO 6000 Blackwell, driver 595.71.05; JAX 0.5.3 GPU JIT matrix check passed.
- Pinned pi0_libero checkpoint download completed and the policy server loaded
  its parameters and listened on `127.0.0.1:8000`.
- EGL device initialization failed. After installing OpenGL/OSMesa dependencies,
  the five-control-step render probe completed under OSMesa and both camera MP4s
  decoded.
- Dummy action-chunk inference passed (`dummy-d575feeb4253`).
- The Part 2 smoke check passed for config SHA-256
  `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
  Smoke run `smoke-a081cbb0b0ab` completed LIBERO Spatial task 0/init state 0
  successfully after 74 control steps. Both videos decoded; the run manifest,
  trajectory, metrics, logs, registry row and gate record are preserved.

This single episode is a smoke check, not a benchmark estimate or reproduction
of the full 50-initial-state evaluation. The server was stopped after the run.
