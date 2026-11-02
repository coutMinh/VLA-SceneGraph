# Reproduce on molab

All experiments were run in a [molab](https://molab.marimo.io) notebook runtime
with an NVIDIA RTX PRO 6000 Blackwell GPU, using the official `pi0_libero` checkpoint
(openpi `215abfb`, LIBERO `f78abd6`, JAX backend, OSMesa rendering; pinned in
[configs/part2_smoke.json](../../configs/part2_smoke.json)). Differences from the official
evaluation are listed in [DEVIATIONS.md](DEVIATIONS.md). The
pipeline from inputs to metric is described in [PIPELINE.md](PIPELINE.md).

## 1. Check the results locally (CPU only)

The tests and the result tables need no GPU, simulator or checkpoint.

```sh
git clone https://github.com/coutMinh/VLA-SceneGraph.git
cd VLA-SceneGraph
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,media]"
pytest                                               # CPU tests
python -m sgvla.cross_suite.report results/cross_suite/episodes results/cross_suite
python scripts/plot_results.py                       # rebuilds media/success_overview.png
```

The report command rebuilds `results/cross_suite/summary.{md,json}` from the committed
per-episode `metrics.json` files.

## 2. Set up the GPU runtime on molab

Open a molab notebook with a GPU attached. Run the commands below in molab's terminal,
or from a notebook cell with `subprocess.run(["bash", "-lc", ...])`. The molab kernel
(Python 3.13) only orchestrates. The CLI builds two private environments under
`$SGVLA_WORK/envs/`: `server` (Python 3.11, openpi + JAX, serves π0) and `client`
(Python 3.8, LIBERO + MuJoCo). The two talk over a websocket on port 8000.

```sh
export SGVLA_WORK="$HOME/vla_scenegraph_work"
git clone https://github.com/coutMinh/VLA-SceneGraph.git "$SGVLA_WORK/VLA-SceneGraph"
cd "$SGVLA_WORK/VLA-SceneGraph"
export SGVLA_CONFIG="$SGVLA_WORK/configs/part2_smoke.json"
export PYTHONPATH="$PWD/src:$SGVLA_WORK/third_party/openpi/third_party/libero:$SGVLA_WORK/tools"
export LIBERO_CONFIG_PATH="$SGVLA_WORK/libero_config" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

sg() { python -m sgvla --work "$SGVLA_WORK" --config "$SGVLA_CONFIG" "$@"; }
p3() { "$SGVLA_WORK/envs/client/bin/python" -m sgvla.analysis --work "$SGVLA_WORK" --config "$SGVLA_CONFIG" "$@"; }

python -m sgvla --work "$SGVLA_WORK" init-config   # copies the pinned config to $SGVLA_CONFIG
sg run probe              # records GPU, driver, graphics libraries, storage
sg run install-server     # openpi at the pinned commit + JAX
sg run install-client     # LIBERO at the pinned commit
sg run repair-rendering   # only if OSMesa/OpenGL are missing (installs libopengl0, libosmesa6)
sg run repair-client      # fixes "cannot enable executable stack" on newer glibc
sg run backend-check      # real GPU matmul + JAX JIT
sg run download           # gs://openpi-assets/checkpoints/pi0_libero
sg run render             # one headless LIBERO frame
```

`sg status` shows every step with its log path. `sg start <step>` runs a step in the
background instead of `sg run`. If rendering or JAX fails, see
[SETUP.md](SETUP.md#runtime-failures).

## 3. Start the policy server

```sh
sg start server
until python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 1)" 2>/dev/null; do sleep 5; done
sg run dummy              # one inference call; expects a finite [50, 7] action chunk
```

The server keeps running for steps 4–6. Stop it with `sg stop server`.

## 4. E0: LIBERO-Spatial reproduction (50 episodes)

```sh
sg run eval-plan          # freezes $SGVLA_WORK/configs/e0_manifest.json (10 tasks × states 0–4)
sg run eval-parity        # our loop vs the official eval_libero on task 0, states 0–2
sg run eval-batch         # first 5 episodes
sg run eval-review        # checks the first batch; watch its videos before continuing
sg run eval-all           # remaining episodes
sg eval-status            # success rate and per-task counts
```

Episodes are written to `$SGVLA_WORK/results/part2/e0/<manifest-hash>/episodes/`, each
with both camera MP4s and `trajectory.npz`. Details: [E0_PROCEDURE.md](E0_PROCEDURE.md).

## 5. Spatial probes: S1, S2, T1 and controls

```sh
p3 --action verify             # S1 relations are unique in the scene
p3 --action verify-temporal    # T1 swap is stable and the inputs match
p3 --action verify-landmarks   # S2 landmark moves are valid
for c in S1 T1 T1-ctrl S2 S2-ctrl; do p3 --action plan --condition $c; done
for c in S1 T1 T1-ctrl S2 S2-ctrl; do
  p3 --action batch  --condition $c      # first 5 episodes
  p3 --action review --condition $c      # required before "all"
  p3 --action all    --condition $c
done
p3 --action report             # -> $SGVLA_WORK/results/part3/report.json
```

The plan step excludes initial states that fail verification. The committed copy is
[results/spatial/report.json](../../results/spatial/report.json). Protocol:
[SPATIAL.md](../protocols/SPATIAL.md).

## 6. Cross-suite: Object, Goal, Long (90 episodes)

```sh
"$SGVLA_WORK/envs/client/bin/python" -m sgvla.cross_suite --work "$SGVLA_WORK" --config "$SGVLA_CONFIG"
# optional: --suite libero_object | libero_goal | libero_10, or --only <episode-id> ...
python -m sgvla.cross_suite.report "$SGVLA_WORK/results/xsuite/268949d38229b7e2" "$SGVLA_WORK/results/xsuite-report"
```

Episodes go to `$SGVLA_WORK/results/xsuite/<manifest-hash>/<episode-id>/`. The manifest
and per-episode seeds are fixed in [spec.py](../../src/sgvla/cross_suite/spec.py), so the run
should land in the same folder as ours (`268949d38229b7e2`). If the folder name
differs, the manifest changed and the run is not the one reported here. One process per suite
(`--suite`) can run in parallel against the same server. Protocol:
[CROSS_SUITE.md](../protocols/CROSS_SUITE.md).

## After a molab restart

A restart stops the server and all background jobs. The work directory, environments
and checkpoint survive if molab keeps the disk. Re-run the `export` lines and the two
shell functions from step 2, then `sg start server`, and repeat the interrupted command.
Every runner skips completed episodes. For the cross-suite runner, first delete the
`claim` file in any episode folder that has no `complete.json`. Otherwise that episode
is skipped.

Policy noise is fixed per episode, so a rerun repeats the same requested actions. We have
not tested whether the outcomes match exactly on a different GPU or driver.
