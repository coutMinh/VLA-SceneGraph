# D1 execution status

Date: 2026-10-28 (Asia/Saigon).

| Item | Status | Follow-up |
|---|---|---|
| Official source and checkpoint metadata | Verified; see `SETUP_NOTES.md` | None |
| Molab GPU/runtime | RTX PRO 6000 Blackwell; JAX GPU JIT and pi0_libero checkpoint load passed | Server stopped after smoke run |
| Rendering | EGL device initialization failed; OSMesa rendered and decoded both camera videos | OSMesa is configured for the current runtime |
| G1 dummy inference | Passed | Evidence in `docs/evidence/molab/work/results/runs/dummy-d575feeb4253/` |
| G1 smoke episode | Passed: Spatial task 0/init state 0, policy success after 74 steps | One smoke run only; not a benchmark estimate |
| GitHub publication | Baseline published to coutMinh/VLA-SceneGraph; Linux CI passed | This G1 run's evidence is archived in this repository |
| Student reading | Study matrix and notes are prepared; the planned 2–3 hour reading is student activity | Complete the reading and record personal notes separately |
| D2 full suite / Part 4 prototype | Not started | Continue in plan order after D1 review |

The pairing endpoint briefly returned HTTP 403 while the detached jobs continued.
The same execution channel recovered, both jobs were inspected, evidence was
collected, G1 passed, and the server was stopped. No D1 execution blocker remains.

If a later runtime fails: pinned openpi JAX -> explicit openpi PyTorch conversion
-> same setup on an already available Colab/Kaggle GPU -> verify SmolVLA checkpoint
-> OpenVLA-OFT. Paid resources require user approval. Preserve failing logs and
record backend/environment deviations; do not invent a result.
