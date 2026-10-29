# Assignment status by Part

Updated 2026-10-29 (Asia/Saigon). Work is organized by assignment Part and experiment ID; dates identify observations and handoffs only.

| Part | Completed artifacts/evidence | Remaining deliverables |
|---|---|---|
| 1a — Scene graphs | [Four verified 2D/video sources](../study/literature_matrix.md); [reading notes](../study/reading_notes/part1_scene_graphs_2d.md) | 3D/4D/manipulation sources; comparative synthesis and open questions |
| 1b — Diffusion / flow matching | Pinned pi0 time/sign convention verified in [setup notes](SETUP_NOTES.md) | Unified objectives/samplers, model table and differences document |
| 2 — Reproduction | Pinned setup/smoke; [CPU scene verification](SCENE_VERIFICATION.md); manifest-driven evaluator, trace, parity harness and Molab E0 notebook implemented | Run GPU parity and first-batch inspection, finish reduced E0 pilot (50 Spatial episodes), pipeline/deviations documentation |
| 3 — Analysis | No experiment result; [source protocol](SCENE_PROTOCOL.md) verifies identical bowl assets and object-specific goals | Simulator checks for relation uniqueness/pose interventions/swap stability; implement and run S1, T1/T1-ctrl, S2/S2-ctrl; labels and trajectory review |
| 4 — Improvement | Research direction in the plan | Evidence-based limitation selection, graph/interface/hypothesis/evaluation plan; optional prototype |

The student's personal reading progress has not been confirmed. Prepared notes do not certify reading completion. No assignment Part is complete in full.

## Part 2 setup/smoke result

- Model: explicit `pi0_libero` on JAX; RTX PRO 6000 Blackwell; OSMesa renderer.
- Dummy `dummy-d575feeb4253`: finite `[50, 7]` chunk.
- Smoke `smoke-a081cbb0b0ab`: Spatial task 0/init 0, simulator seed 7; success at 74 steps, two decoded camera videos.
- Smoke check passed for config SHA-256 `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
- This is setup validation, not E0 or a benchmark estimate. The policy server was stopped after the run.

## Next runtime validation: Part 2 / E0 reduced pilot

Policy/workload separation, resumable attempts, atomic validated completions, registry reconciliation, object/fixture pose traces, explicit per-episode noise, official-loop parity harness and batch inspection are implemented. [E0 operating procedure](../repro/E0.md) specifies the run order. CPU tests exercise real MP4 encoding/decoding and corrupt/interrupted/error resume. They do not establish GPU/model/simulator parity. Molab is exhausted; the user explicitly requested implementation only. No new policy evaluation was launched.

When resources return: sync code, reuse environments/checkpoint, verify backend, restart server with the new seeded adapter, run parity on task 0/states 0–2, then the first five E0 episodes and inspection before continuing to 50. Part 3 pose/swap interventions remain disabled until simulator verification; they were not part of this implementation request. Keep the paired subset and declare fresh-environment/episode-noise deviations.

External notebooks: `Homework/NganLe/molab/vla_scenegraph/part2_reproduce.py` (setup) and `part2_e0.py` (parity/E0). Both reuse runtime environments/checkpoints. [ARTIFACTS.md](ARTIFACTS.md) maps retained results and the complete external archive. Historical raw `G1`/`d1.json` names are preserved identifiers, not current stage names.
