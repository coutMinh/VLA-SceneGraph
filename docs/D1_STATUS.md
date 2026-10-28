# D1 status · 2026-10-28

| Planned task | Evidence / status |
|---|---|
| Verify official model/code/checkpoints/protocol | Done: `SETUP_NOTES.md` and `evidence/pi0_libero_objects.json` |
| Separate baseline implementation and external notebook | Done: `src/sgvla/baselines/openpi`; notebook delivered separately in `Homework/NganLe/molab/vla_scenegraph` |
| Runtime probe and pinned environments | Done on molab; details in `RUNTIME_COMPATIBILITY.md` and `$SGVLA_WORK/docs/` |
| GPU backend | Pinned JAX 0.5.3 GPU JIT matrix check passed on RTX PRO 6000 Blackwell; server loaded the pinned pi0_libero checkpoint and listened on localhost:8000 |
| Checkpoint | Download completed: 11.2 GiB under `$SGVLA_WORK/cache/openpi/openpi-assets/checkpoints/pi0_libero` |
| Rendering | EGL device initialization failed; OSMesa fallback rendered and decoded agentview/wrist video for a five-control-step probe (`render-ab0d4dc4eb83`) |
| Dummy policy inference | Passed for the current config; metrics recorded for `dummy-d575feeb4253` |
| G1 | **Passed** at 2026-10-28 08:09:09 UTC; GPU kernel, dummy and smoke checks true for config hash `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658` |
| LIBERO smoke | `smoke-a081cbb0b0ab`: Spatial task 0, init state 0, π0 success after 74 steps; both camera videos decoded. This is one smoke episode, not a benchmark estimate. |
| 2D/video study matrix | Four verified sources; `study/literature_matrix.csv` and `reading_notes/d1_2d.md` |
| Student reading (2–3 hours) | Student activity; not inferred from agent-prepared notes |
| Publication and CPU verification | Published; 14 tests passed on each Linux Python 3.8/3.11/3.13; external notebook passed structure and clean-clone startup checks |

The codebase is named `VLA-SceneGraph/` per the user's repository choice; this
replaces the plan's provisional `A3/` directory. The notebook remains outside
the implementation repository and clones GitHub code. No D2 full-suite evaluation
or Part 4 prototype has been started.

Runtime results, environment snapshots and logs live under `$SGVLA_WORK/`,
outside the implementation repository. A reviewed copy of D1 evidence is stored
in `docs/evidence/molab/`; see its README for provenance and artifact hashes.
The server was stopped after G1 to release the GPU. D2 full-suite evaluation,
Part 4 prototype and the student's planned reading session have not been started.
