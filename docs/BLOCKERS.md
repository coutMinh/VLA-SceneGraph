# Open work and runtime constraints by Part

Updated 2026-10-28 (Asia/Saigon).

| Scope | Current state | Required next work |
|---|---|---|
| Part 1 | Four verified 2D/video entries; personal reading unconfirmed | Complete the 3D/4D/manipulation comparison and diffusion/FM study |
| Part 2 setup | GPU, checkpoint, dummy and smoke passed | Reuse setup; inspect current molab status after reconnect/restart |
| Part 2 E0 | Not implemented or run | Resumable evaluator, object-pose traces, controlled official-client comparison, then full Spatial suite |
| Part 3 | No analysis experiments have run | Reuse E0 baseline, implement E1–E5 and review trajectories |
| Part 4 | Not implemented | Select limitation after Part 3 evidence review |

There is no unresolved blocker to the completed smoke. JAX runs on the observed GPU; the pinned server PyTorch build does not. EGL failed; OSMesa works. See [runtime compatibility](RUNTIME_COMPATIBILITY.md). Molab storage quota remains unknown, and restarts stop detached jobs. Archived PID/job-state files are historical records, not current processes.

The current request reorganizes documentation, names and stored artifacts; no new E0/E1–E6 evaluation was requested or run. Further implementation should follow the requested Part. Paid compute and missing secrets require explicit user input; ordinary authorized implementation does not need repeated confirmation. Submission deadline 2026-11-03 22:00 VNT remains an assumption to verify with the supervisor.
