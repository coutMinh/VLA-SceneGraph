# D1 execution on molab

Use the separately delivered `molab/vla_scenegraph/d1.py` notebook, or the Linux
terminal commands below. Source verification is in [SETUP_NOTES](docs/SETUP_NOTES.md).
GPU availability on molab must be measured there; local CPU checks do not pass G1.

## Bootstrap (Linux terminal alternative)

```sh
export SGVLA_WORK="$HOME/vla_scenegraph_work"
mkdir -p "$SGVLA_WORK"
git clone https://github.com/coutMinh/VLA-SceneGraph.git "$SGVLA_WORK/VLA-SceneGraph"
cd "$SGVLA_WORK/VLA-SceneGraph"
export PYTHONPATH="$PWD/src"
python -m sgvla init-config
export SGVLA_CONFIG="$SGVLA_WORK/configs/d1.json"
```

The commands below use the runtime config explicitly. The notebook does this
automatically. Run one step at a time, inspect `status` and its named log, and
wait for completion before proceeding. `server` stays running.

```sh
python -m sgvla --config "$SGVLA_CONFIG" start probe
python -m sgvla --config "$SGVLA_CONFIG" status
python -m sgvla --config "$SGVLA_CONFIG" start install-server
python -m sgvla --config "$SGVLA_CONFIG" start install-client
python -m sgvla --config "$SGVLA_CONFIG" start backend-check
python -m sgvla --config "$SGVLA_CONFIG" start download
python -m sgvla --config "$SGVLA_CONFIG" start render
python -m sgvla --config "$SGVLA_CONFIG" start server
# Wait for the server log to show that websocket serving is ready.
python -m sgvla --config "$SGVLA_CONFIG" start dummy
python -m sgvla --config "$SGVLA_CONFIG" start smoke
python -m sgvla --config "$SGVLA_CONFIG" gate
```

`run <action>` runs a step in the foreground instead. `start` detaches the job,
stores the PID with its Linux boot/start identity, and prevents duplicate starts.
Install/conversion jobs are serialized against other jobs. Config values are
snapshotted and hashed; results for a different config do not pass G1.

## Evidence and acceptance

- `docs/molab_probe.json`: observed GPU/driver, graphics libraries, storage and
  public GCS/HF access. Graphics library discovery is not proof of working rendering.
  Reported disk capacity does not verify molab's quota.
- `docs/server_freeze.txt`, `client_freeze.txt`: actual installed package versions.
- `docs/backend_check.json`: actual GPU matrix multiplication, including JAX JIT.
- `docs/checkpoint.json`: downloaded checkpoint path, not just a remote URL.
- `results/runs/<run_id>/`: metrics, provenance manifest, action/state traces and
  agentview/wrist MP4s. Dummy/render/smoke runs append to `results/registry.csv`,
  including failed attempts that reach the run stage. Setup/probe jobs use job logs.
- `docs/g1.json`: must say `passed: true`, with GPU, dummy and smoke checks true.
  A valid completed episode can be a **task failure**; D1 tests execution, not quality.

The smoke uses Spatial task 0, fixed init state 0, simulator seed 7, 10 settling
steps, 256 render -> 224 policy images, 180-degree rotation of both images,
8-dimensional state, a 50-action pi0 chunk, 5 actions per replan and at most 220
control steps. Video is encoded and decoded before it is accepted. It is not a
suite-level success estimate or a parity check against official evaluation.
The dummy call consumes policy randomness before smoke; restart the policy server
for D2's controlled comparison. The official server default is pi0.5; our explicit
pi0 factory avoids silently switching models.

## Runtime failures

If client import fails with `cannot enable executable stack`, run the explicit
`repair-client` action, then retry `render`. This observed glibc compatibility
repair and its recorded binary hashes are explained in
[RUNTIME_COMPATIBILITY.md](docs/RUNTIME_COMPATIBILITY.md).

If EGL fails, preserve the log, change `render_backend` in the runtime config to
`osmesa`, then repeat backend-check/render and restart the server with that config.
OSMesa must actually be installed on the host. The installer does not assume
root access or silently run apt; probe findings determine the required system fix.
On the observed root-enabled molab image, the explicit `repair-rendering` action
installs `libopengl0` and `libosmesa6` and records their versions before retrying.

If JAX fails but PyTorch GPU kernels work, change `backend` to `pytorch`, rerun
backend-check and download, then run `convert`. This calls the official converter,
patches transformers only in the private environment (`UV_LINK_MODE=copy`), and
copies normalization assets into the converted output. The asset path correction
is recorded in `docs/conversion.json`. Then start server -> dummy -> smoke -> gate.
This backend is prepared, **not yet validated on molab**.

Stop an old server before changing its config/backend. Use the notebook stop
control or `python -m sgvla stop server`; only the recorded, matching process
group is signalled. Restarting molab also stops background jobs. On restart,
dead jobs show interrupted; rerun the step. Completed checkpoint caches and
venvs persist if molab retains the work directory. D1 smoke creates a fresh
single-episode run; per-episode full-evaluation resumability belongs to D2.

If both backends fail, preserve diagnostics and use the same pinned stack on an
already available Colab/Kaggle GPU. Verify alternative model/checkpoint availability
before switching. Do not launch paid resources without authorization.
