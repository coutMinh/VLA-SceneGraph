# VLA-SceneGraph

Assignment 3: **Part 1 Study → Part 2 Reproduce → Part 3 Analyse → Part 4 Improve**.
Implementation and progress are organized by Part and experiment ID, independent of dates.

| Part | Status | Entry point |
|---|---|---|
| 1 — Study | Four verified 2D/video scene-graph sources and notes; study incomplete | [Literature matrix](study/literature_matrix.md) |
| 2 — Reproduction | Official pi0_libero setup/smoke passed on molab; E0 full suite pending | [Smoke report and videos](results/part2/smoke_2026-10-28/README.md), [setup](SETUP.md) |
| 3 — Analysis | E1–E5 and optional E6 not run | [Part status](docs/PART_STATUS.md) |
| 4 — Improvement | Proposal/prototype not implemented | [Part status](docs/PART_STATUS.md) |

The smoke succeeded on one LIBERO Spatial task 0/init state 0 episode after 74 steps. It validates execution and is not a benchmark estimate. The full evaluator with per-episode resume, object poses and official-client comparison is the next Part 2 implementation.

## Layout

```text
src/sgvla/baselines/openpi/      # official baseline setup/server/smoke adapter
src/sgvla/                      # CLI, jobs, probe, registry, smoke check
configs/part2_smoke.json        # pinned model and tested OSMesa config
study/                         # Part 1 literature and reading notes
results/registry.csv            # original collected run index
results/part2/                  # retained Part 2 result package
docs/                          # Part status, verified facts, artifact map
tests/                         # CPU protocol and orchestration checks
```

Molab notebooks are outside this repo in `Homework/NganLe/molab/vla_scenegraph/part2_reproduce.py`. They clone/pull GitHub and invoke the CLI. The Python 3.13 kernel orchestrates separate Python 3.11 server and Python 3.8 client environments. All installed environments, model weights and live job outputs stay under `$SGVLA_WORK` (default `~/vla_scenegraph_work`), outside the source checkout.

Reviewed results stay in `results/part2/`; repetitive raw runtime/JSON state is archived outside the codebase. See [ARTIFACTS.md](docs/ARTIFACTS.md) for provenance and historical path mappings. Historical result IDs and hashes are preserved.

Read [PART_STATUS.md](docs/PART_STATUS.md), [SETUP.md](SETUP.md) and [verified findings](docs/SETUP_NOTES.md) before continuing. The parent workspace's `PLAN_CODEX.md` and dated `SESSION_HANDOFF.md` define the remaining work by Part.

## Local CPU development

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e '.[dev]' numpy
ruff check .
ruff format --check .
pytest -q
python scripts/render_literature.py --check
```

The control package has no ML dependencies and works through `PYTHONPATH=src`. Run GPU/simulation jobs on molab, not the Windows notebook/kernel. Linux lifecycle checks run in CI; Windows skips them. See [VALIDATION.md](docs/VALIDATION.md) for actual check evidence and its limits.
