# Open work and runtime constraints by Part

Updated 2026-10-29 (Asia/Saigon).

| Scope | Current state | Required next work |
|---|---|---|
| Part 1 | Four verified 2D/video entries; personal reading unconfirmed | Complete the 3D/4D/manipulation comparison and diffusion/FM study |
| Part 2 setup | GPU, checkpoint, dummy and smoke passed | Reuse setup; inspect current molab status after reconnect/restart |
| Part 2 E0 | Evaluator/trace/parity harness implemented and CPU-checked; GPU runs pending | Molab exhausted; user requested implementation only. Later run parity, first-batch inspection, and 50-episode Spatial pilot |
| Part 3 | BDDL/initial-state source verification complete; no analysis experiment | Verify relations, interventions and swap stability in simulator; implement/run conditions and review trajectories |
| Part 4 | Not implemented | Select limitation after Part 3 evidence review |

There is no unresolved blocker to the completed smoke. JAX runs on the observed GPU; the pinned server PyTorch build does not. EGL failed; OSMesa works. See [runtime compatibility](RUNTIME_COMPATIBILITY.md). Molab storage quota remains unknown, and restarts stop detached jobs. Archived PID/job-state files are historical records, not current processes.

The selected E0 scope is a 50-episode Spatial pilot (10 tasks × 5 fixed initial states), not a full official suite. Molab restarts and pairing instability make short resumable batches important. No E0 or Part 3 evaluation has run yet. Paid compute and missing secrets require explicit user input; ordinary authorized implementation does not need repeated confirmation. Submission deadline 2026-11-03 22:00 VNT remains an assumption to verify with the supervisor.
