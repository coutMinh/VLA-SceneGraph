# D1 execution status

Date: 2026-10-28 (Asia/Saigon).

| Item | Status | Next action |
|---|---|---|
| Official source and checkpoint metadata | Verified; see SETUP_NOTES.md | Run actual checkpoint download on molab |
| Molab GPU/runtime access from this session | No connected molab execution tool available | Open the external notebook on molab; run probe, install and G1 buttons |
| G1: dummy inference + rendered episode/video | **Not run** | Collect machine-generated evidence before marking G1 passed |
| Local GPU reproduction | Unavailable on the Windows machine described by Homework/AGENTS.md | Use molab; local checks cover CPU orchestration only |
| GitHub publication | Published to coutMinh/VLA-SceneGraph; Linux CI passed | Notebook can clone the implementation repo |
| Submission deadline | 2026-11-03 22:00 VNT is an assumption in the plan | Student confirms with supervisor |

Fallback order after an observed runtime failure: pinned openpi JAX -> explicit
openpi PyTorch conversion -> same setup on an already available Colab/Kaggle GPU
-> verify SmolVLA checkpoint -> OpenVLA-OFT. Paid resources require user approval.
Preserve the failing log and record backend/environment deviations; do not
invent a probe report or claim a smoke test passed on an unobserved machine.
