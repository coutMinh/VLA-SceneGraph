# Part 2 · Verified upstream facts

Verified on 2026-10-28 against openpi commit
`215abfb217dbac7d5f1273282331b9b1866c0479` (not a floating `main`).
Source checkout: sibling `../third_party/openpi`; never edited by this project.
Runtime environments, checkpoints and results live outside this repository.

## Model decision

Primary: **pi0_libero**, explicit config and checkpoint
`gs://openpi-assets/checkpoints/pi0_libero`.
`src/openpi/training/config.py:653` defines the config; public GCS object metadata
was checked separately (archived source listing; see `ARTIFACTS.md`). An object listing
does not demonstrate a complete download or working inference.

Important: `scripts/serve_policy.py:82` defaults LIBERO to **pi05_libero**.
Using `--env LIBERO` alone would change our selected model. Our server loads the
explicit pi0 config through the official policy factory.

`pi0_fast_libero` and `pi05_libero` configs also exist (config.py:700, :744).
FAST weights have not been downloaded or verified here; the Part 2 smoke does not need them.
The official example's current results table is for **pi0.5**, not pi0. Do not
attribute those success rates to pi0.

## Environments

- Server: Python **3.11** (`.python-version:1`), upstream `uv.lock`;
  `jax[cuda12]==0.5.3`, `torch==2.7.1`, `numpy<2`, `transformers==4.53.2`
  (`pyproject.toml`). Use an external `UV_PROJECT_ENVIRONMENT` and `uv sync --frozen`.
- Client: Python **3.8**, as specified by `examples/libero/README.md:32` and
  its compiled requirements. Install both `examples/libero/requirements.txt`
  and the pinned LIBERO submodule requirements, then the official editable
  client and LIBERO packages. CUDA 11.3 torch wheels here are for a legacy
  simulation dependency; policy GPU inference runs only in the server env.
- LIBERO submodule: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` from the openpi git tree.
- Molab's Python 3.13 kernel only orchestrates subprocesses. Do not install the
  old numpy/JAX/LIBERO stack into that kernel.
- Upstream supports Ubuntu (tested on 22.04), not Windows. README specifies
  inference VRAM **>8 GB**; this is not a guarantee for every GPU or backend.
- Observed molab GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver
  595.71.05, 97,887 MiB reported memory and SM 12.0. Pinned JAX 0.5.3 passed a
  GPU JIT matrix operation; the selected pi0_libero checkpoint loaded and its
  server listened on localhost:8000. One separate Spatial smoke episode passed;
  this does not verify the full benchmark.
- Molab's usable storage quota remains unknown. GCS public metadata and Hugging
  Face public API access returned HTTP 200. The download log records completion
  of all 12,014,131,888 checkpoint bytes (11.2 GiB).
- EGL device initialization did not work. Installing the observed missing
  OpenGL/OSMesa packages and selecting OSMesa allowed a five-control-step render
  probe whose agentview and wrist MP4s both decoded. The Part 2 smoke config now defaults
  to this observed working renderer. See `RUNTIME_COMPATIBILITY.md`.
- Pairing briefly returned HTTP 403 after the dummy job was launched; the
  execution channel recovered. Dummy inference and the Spatial task 0/init state
  0 smoke episode passed, the Part 2 smoke check passed for the recorded config hash, and the server
  was stopped. The retained report/videos are under `../results/part2/smoke_2026-10-28/`; `ARTIFACTS.md` maps the full archive.

Runtime follow-up: molab's installed `uv venv --help` exposes `--allow-existing`.
The client installer uses it to reuse a partially installed environment when
retrying after an interrupted job, rather than recreating the environment.

## PyTorch fallback (available, not yet run)

README.md:190 documents pi0/pi0.5 PyTorch support. The official converter is
`examples/convert_jax_model_to_pytorch.py` (not under `scripts/`). It accepts
`--checkpoint-dir <download-root> --config-name pi0_libero --output-path <new-dir>`.
The converter restores `<root>/params`. At this pin its asset copy looks for
`<root>/../assets`, which differs from the checkpoint layout. Our wrapper copies
`<root>/assets` into the converted output and records this deviation.
`policies/policy_config.py:47` detects `model.safetensors` automatically.

Upstream requires its transformers replacement files for this backend. Copy
them only into our isolated server environment, using `UV_LINK_MODE=copy` so
shared uv cache files are not overwritten. Record conversion/backend as a
deviation. Conversion and LIBERO equivalence are **not established** until run.
Do not silently upgrade JAX if Blackwell fails: preserve diagnostics, then try
the explicit PyTorch path, or the same pinned setup on another available GPU.

## Official client protocol to preserve

From `examples/libero/main.py` at this commit:

| Setting | Value / source line |
|---|---|
| Seed | 7 (:45) |
| Trials per task | 50 fixed initial states (:37, :90, :108) |
| Render size | 256 (:19) |
| Policy image size | 224 (:28) |
| Orientation | Both images rotated 180 degrees (:128) |
| State | Position + quaternion-to-axis-angle + gripper qpos, 8 values (:147) |
| Executed actions before replanning | 5 (:29) |
| Settling steps | 10, action `[0]*6+[-1]` (:18, :36, :120) |
| Step limits | Spatial 220; Object 280; Goal 300; Long 520 (:64) |
| Action dimensions | 7 (`policies/libero_policy.py:110`) |
| pi0 chunk horizon | 50 (`models/pi0_config.py:27`) |
| pi0 FM steps | 10 (`models/pi0.py`, `sample_actions`) |

The completed Part 2 smoke runs only Spatial/task 0/init 0. It is a **smoke test**, never a benchmark
estimate. The smoke check requires a valid dummy action chunk, actual GPU evidence and a
complete episode with decodable MP4; the episode does not have to succeed.
The smoke check has passed. The resumable evaluator and official-client comparison have
also run for the reduced E0 pilot; see [the Molab run record](../results/part2/e0/molab-2026-10-29/README.md).

## Flow convention correction for the plan

`models/pi0.py`, `compute_loss`, uses
`x_t = t * noise + (1-t) * actions`, target `noise-actions`,
`t ~ Beta(1.5,1)*0.999+0.001`; inference integrates from t=1 to t=0
with `dt=-1/num_steps`. PLAN_CODEX.md has been corrected to this convention; use it consistently in Part 1b and the Part 2 pipeline document.

## Source links

- [Pinned README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)
- [Pinned LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)
- [Pinned model configs](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)
- [Pinned converter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/convert_jax_model_to_pytorch.py)

Source claims above refer to the pinned upstream. Observed setup/smoke results
are reported under `results/part2/`; no full-suite reproduction is claimed.
