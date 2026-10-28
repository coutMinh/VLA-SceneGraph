# Validation · D1

2026-10-28, Windows Python 3.12, private project venv:

- Ruff lint and format checks pass.
- CPU tests: **12 passed, 2 skipped**. Skips are Linux process lifecycle tests;
  no GPU/model inference is simulated as an experiment result.
- Matrix Markdown matches the reviewed CSV (`render_literature.py --check`).
- External notebook passes `marimo check --strict` with marimo 0.25.2.
- External notebook was executed/exported from a clean local clone; bootstrap,
  runtime config and initial status display worked without launching GPU jobs.
- openpi tracked sources have no local modifications.

Tests cover action chunk validation, settling/replanning/early termination,
artifact/config integrity for G1, registry idempotency/provenance, config
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
[`evidence/molab/`](evidence/molab/):

- RTX PRO 6000 Blackwell, driver 595.71.05; JAX 0.5.3 GPU JIT matrix check passed.
- Pinned pi0_libero checkpoint download completed and the policy server loaded
  its parameters and listened on `127.0.0.1:8000`.
- EGL device initialization failed. After installing OpenGL/OSMesa dependencies,
  the five-control-step render probe completed under OSMesa and both camera MP4s
  decoded.
- Dummy action-chunk inference passed (`dummy-d575feeb4253`).
- G1 passed for config SHA-256
  `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
  Smoke run `smoke-a081cbb0b0ab` completed LIBERO Spatial task 0/init state 0
  successfully after 74 control steps. Both videos decoded; the run manifest,
  trajectory, metrics, logs, registry row and gate record are preserved.

This single episode is a smoke check, not a benchmark estimate or reproduction
of the full 50-initial-state evaluation. The server was stopped after the run.
