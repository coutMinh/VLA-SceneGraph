# D1 status · 2026-10-28

| Planned task | Evidence / status |
|---|---|
| Verify official model/code/checkpoints/protocol | Done: SETUP_NOTES.md and evidence/pi0_libero_objects.json |
| Separate baseline implementation and external notebook | Done: src/sgvla/baselines/openpi; notebook delivered separately in Homework/NganLe/molab/vla_scenegraph |
| Prepare runtime probe, two environments, download and smoke | Implemented; actual molab installation and GPU execution pending |
| 2D/video study matrix | Four verified sources; study/literature_matrix.csv and reading_notes/d1_2d.md |
| Student reading (2–3 hours) | Student activity; not inferred from agent-prepared notes |
| Publication and CPU verification | Published; 14 tests passed on each Linux Python 3.8/3.11/3.13; external notebook passed structure and clean-clone startup checks |
| G1 | **Not run on molab**; no completed episode or benchmark results claimed |

The codebase is named `VLA-SceneGraph/` per the user's repository choice; this
replaces the plan's provisional `A3/` directory. The notebook remains outside
the implementation repository and clones GitHub code. No D2 full-suite evaluation
or Part 4 prototype has been started.

Runtime evidence is written under `$SGVLA_WORK/docs/` and `$SGVLA_WORK/results/`.
It does not overwrite source-verification notes. Bring back `g1.json`, probe,
freeze files, registry, run manifests, metrics and videos after running on molab.
