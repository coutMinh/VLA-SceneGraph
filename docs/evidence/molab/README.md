# Molab D1 evidence

Collected from `$SGVLA_WORK` on 2026-10-28 after the smoke run. The `work/`
directory preserves the runtime-relative paths for configs, job records, logs,
metrics and videos. `collection.json` records the runtime root and code commit.
Checkpoint weights and installed environments are not copied into this repo.

## Gate and run

- G1 passed at `2026-10-28T08:09:09.845159Z`; all checks (`gpu_kernel`, `dummy`,
  `smoke`) are true in [`work/docs/g1.json`](work/docs/g1.json).
- Configuration SHA-256:
  `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
- Code commit: `c6ff824f928f0d61fc197e1f66a823878354e645`; openpi commit:
  `215abfb217dbac7d5f1273282331b9b1866c0479`; LIBERO commit:
  `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.71.05,
  97,887 MiB reported memory, SM 12.0. JAX 0.5.3 GPU JIT check passed.
- Dummy run `dummy-d575feeb4253` returned finite actions of shape `[50, 7]`;
  server metadata identified `pi0_libero`, JAX and 10 flow steps.
- Smoke run `smoke-a081cbb0b0ab` used LIBERO Spatial task 0, initial state 0,
  seed 7 and OSMesa rendering. The task succeeded after 74 control steps; both
  camera videos decoded to 74 frames. Its metrics explicitly set
  `benchmark_estimate` to false.
- The policy server was stopped after G1 passed to release the GPU.

Videos: [agentview](work/results/runs/smoke-a081cbb0b0ab/agentview.mp4) ·
[wrist camera](work/results/runs/smoke-a081cbb0b0ab/wrist.mp4)

## Artifact integrity

The smoke run manifest SHA-256 values were checked against each downloaded file:
both videos, `trajectory.npz`, `metrics.json` and
`initial_state_manifest.json` match. The manifest is
[`work/results/runs/smoke-a081cbb0b0ab/manifest.json`](work/results/runs/smoke-a081cbb0b0ab/manifest.json).
The complete registry and raw logs are under `work/results/`.

The archive records compatibility findings too: the server PyTorch 2.7.1 CUDA
build did not support this GPU, so the pinned JAX backend was used; the client
PyTorch shared libraries needed the targeted executable-stack repair; EGL device
rendering failed, so OSMesa was installed and used. The failed attempts and
repair reports are preserved in `work/results/logs/`, `work/jobs/` and
`work/docs/`.

This is one successful smoke episode and a passed execution gate. It is not the
full 50-initial-state benchmark and does not establish a success-rate estimate.
