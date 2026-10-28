# Local validation · D1

2026-10-28, Windows Python 3.12, private project venv:

- Ruff lint and format checks pass.
- CPU tests: **12 passed, 2 skipped**. Skips are Linux process lifecycle tests;
  no GPU/model inference is simulated as a claimed experiment result.
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

Molab probe, environment installation, checkpoint inference, rendering and G1
remain **not run** from this session. There is no connected molab/document
execution session. Use the external D1 notebook to collect those artifacts.
