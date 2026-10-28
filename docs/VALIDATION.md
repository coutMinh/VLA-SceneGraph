# Local validation · D1

2026-10-28, Windows Python 3.12, private project venv:

- Ruff lint and format checks pass.
- CPU tests: **12 passed, 2 skipped**. Skips are Linux process lifecycle tests;
  no GPU/model inference is simulated as a claimed experiment result.
- Matrix Markdown matches the reviewed CSV (`render_literature.py --check`).
- External notebook passes `marimo check --strict` with marimo 0.25.2.
- openpi tracked sources have no local modifications.

Tests cover action chunk validation, settling/replanning/early termination,
artifact/config integrity for G1, registry idempotency/provenance, config
preservation, Python 3.8 grammar and a dependency-free orchestration import.
Linux process tests also cover detached job completion, duplicate protection,
restart/PID reuse detection and signalling only a managed process group.

Linux tests are configured in GitHub Actions for Python 3.8, 3.11 and 3.13.
WSL dependencies were prepared locally, but the WSL test execution command was
denied; no WSL test success is claimed.

Molab probe, environment installation, checkpoint inference, rendering and G1
remain **not run** from this session. There is no connected molab/document
execution session. Use the external D1 notebook to collect those artifacts.
