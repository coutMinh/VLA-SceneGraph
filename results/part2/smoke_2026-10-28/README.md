# Part 2 — π0 setup and smoke result

Experiment classification: **setup/smoke validation**. E0 full reproduction has not run. Data was collected on 2026-10-28 using source commit `c6ff824f928f0d61fc197e1f66a823878354e645`; the subsequent organization/config changes do not change the completed experiment.

| Evidence | Observed result |
|---|---|
| Selected policy | `pi0_libero`, JAX 0.5.3, 10 flow steps, checkpoint `gs://openpi-assets/checkpoints/pi0_libero` |
| Pinned source | openpi `215abfb217dbac7d5f1273282331b9b1866c0479`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Hardware | NVIDIA RTX PRO 6000 Blackwell Server Edition; driver 595.71.05; SM 12.0 |
| Rendering | OSMesa; EGL device initialization failed before fallback |
| Dummy `dummy-d575feeb4253` | Completed; finite action shape `[50, 7]` |
| Smoke `smoke-a081cbb0b0ab` | Spatial task 0/init state 0, seed 7; task success after 74 control steps; both videos decoded |
| Smoke acceptance | Passed at `2026-10-28T08:09:09.845159Z`; raw historical gate label `G1` |

Config SHA-256: `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`. The [original config](config.json) and current `configs/part2_smoke.json` have the same canonical values/hash.

## Retained files

- [GPU/backend check](checks/backend_check.json), [runtime probe](checks/molab_probe.json), [original smoke acceptance report](checks/g1.json).
- [Dummy metrics](runs/dummy-d575feeb4253/metrics.json), actions and manifest in the same directory.
- [Smoke metrics](runs/smoke-a081cbb0b0ab/metrics.json), [manifest](runs/smoke-a081cbb0b0ab/manifest.json), initial-state manifest and `trajectory.npz`.
- Videos: [agentview](runs/smoke-a081cbb0b0ab/agentview.mp4) and [wrist](runs/smoke-a081cbb0b0ab/wrist.mp4).
- [Original registry](../../registry.csv), including setup/render/dummy/smoke provenance rows. Failed render diagnostics referenced there remain in the full archive.

Artifact SHA-256 values match the original manifests. Metrics contain original molab absolute paths; use the local links above. [ARTIFACTS.md](../../../docs/ARTIFACTS.md) maps the full raw archive, environment freezes, compatibility reports and historical source listing.

## Limits and reuse

This is one episode and does not estimate benchmark success rate. The trace contains actions/proprio, not all object poses needed for the planned Part 3 taxonomy. The dummy call consumed policy RNG before smoke; restart/control policy RNG for the official-client parity check and E0. The server was stopped after completion. Reuse environments/checkpoint when available; no GPU rerun is needed to adopt the part-based file names.
