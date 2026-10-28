# Assignment status by Part

Updated 2026-10-28 (Asia/Saigon). Work is organized by assignment Part and experiment ID; dates identify observations and handoffs only.

| Part | Completed artifacts/evidence | Remaining deliverables |
|---|---|---|
| 1a — Scene graphs | [Four verified 2D/video sources](../study/literature_matrix.md); [reading notes](../study/reading_notes/part1_scene_graphs_2d.md) | 3D/4D/manipulation sources; comparative synthesis and open questions |
| 1b — Diffusion / flow matching | Pinned pi0 time/sign convention verified in [setup notes](SETUP_NOTES.md) | Unified objectives/samplers, model table and differences document |
| 2 — Reproduction | Pinned upstream, environments/checkpoint, GPU check, dummy and one smoke episode; [run report](../results/part2/smoke_2026-10-28/README.md) | Resumable evaluator, official-client comparison, E0 full suite, pipeline/deviations documentation |
| 3 — Analysis | No experiment result yet | E1–E5, optional E6, taxonomy/video review, ≥3 strengths and ≥3 limitations |
| 4 — Improvement | Research direction in the plan | Evidence-based limitation selection, graph/interface/hypothesis/evaluation plan; optional prototype |

The student's personal reading progress has not been confirmed. Prepared notes do not certify reading completion. No assignment Part is complete in full.

## Part 2 setup/smoke result

- Model: explicit `pi0_libero` on JAX; RTX PRO 6000 Blackwell; OSMesa renderer.
- Dummy `dummy-d575feeb4253`: finite `[50, 7]` chunk.
- Smoke `smoke-a081cbb0b0ab`: Spatial task 0/init 0, simulator seed 7; success at 74 steps, two decoded camera videos.
- Smoke check passed for config SHA-256 `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
- This is setup validation, not E0 or a benchmark estimate. The policy server was stopped after the run.

## Next unfinished implementation: Part 2 / E0

Separate policy/evaluation config compatibility; implement resumable per-episode execution and all-object-pose logging; verify resume behavior and parity on three initial states; then time the first 20 episodes of Spatial's 500-episode evaluation. The full dependency order is in the parent workspace's `PLAN_CODEX.md`.

The external notebook is `Homework/NganLe/molab/vla_scenegraph/part2_reproduce.py`. It reuses runtime environments/checkpoints. [ARTIFACTS.md](ARTIFACTS.md) maps retained results and the complete external archive. Historical raw `G1`/`d1.json` names are preserved identifiers, not current stage names.
