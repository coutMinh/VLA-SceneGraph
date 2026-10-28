# VLA-SceneGraph

Official VLA reproduction and scene graph experiments for Assignment 3.
The D1 baseline is **pi0_libero**, using a pinned, unmodified openpi checkout.

**Status:** source verification and D1 tooling are implemented. Molab GPU
inference and G1 are **not yet run**. No benchmark success rate is claimed.

## Layout

```text
VLA-SceneGraph/                    # this GitHub repository
  src/sgvla/baselines/openpi/      # baseline setup, policy server and D1 smoke
  src/sgvla/                      # CLI, process management, probe, registry, G1
  configs/d1.json                 # verified upstream pins and smoke settings
  tests/                         # CPU protocol and orchestration checks
  study/                         # literature matrix and reading notes
  docs/                          # verified setup facts and blockers
```

Molab notebooks are delivered **outside this repository**, in the homework
workspace at `Homework/NganLe/molab/vla_scenegraph/d1.py`. They clone this repo
and invoke its CLI; they do not embed baseline implementations.

Runtime files are also outside the repository:

```text
$SGVLA_WORK/                       # default ~/vla_scenegraph_work
  VLA-SceneGraph/                  # cloned implementation repo
  third_party/openpi/              # pinned upstream + LIBERO submodule
  envs/server/                    # Python 3.11, official ML stack
  envs/client/                    # Python 3.8, official simulator stack
  configs/ docs/ jobs/             # runtime config, observed evidence, job state
  cache/ checkpoints/ results/    # weights, videos, metrics and registry
```

See [SETUP.md](SETUP.md) for the exact D1 run sequence, fallback and checks.
See [verified findings](docs/SETUP_NOTES.md) before changing any dependency.

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

The control package has no ML dependencies; it also works through `PYTHONPATH=src`.
Do not install the GPU/simulation stack into the molab kernel or Windows.
Linux-only lifecycle checks run on Linux and in CI; Windows skips those checks.

D2's resumable full-suite evaluator, analysis sweeps and scene graph prototype
remain later milestones. D1 evaluates a single episode only.
