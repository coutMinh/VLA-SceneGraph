# Open work and runtime constraints by Part

Updated 2026-10-29 (Asia/Saigon).

| Scope | Current state | Required next work |
|---|---|---|
| Part 1 | Four verified 2D/video entries; personal reading unconfirmed | Complete the 3D/4D/manipulation comparison and diffusion/FM study |
| Part 2 setup | GPU, checkpoint, dummy, smoke, parity and E0 pilot passed | Preserve Molab artifacts; inspect the two task failures |
| Part 2 E0 | Fixed 50-episode Spatial pilot complete: 48 success, 2 task failures, 0 infrastructure errors | Manually review successful/failed trajectories; record failure causes before Part 3 analysis |
| Part 3 | S1, T1 and T1-ctrl completed; S2 paused before inference on a strict endpoint hash mismatch | Apply and sync tolerant S2 observation comparison; resume from the first-batch review gate |
| Part 4 | Not implemented | Select limitation after Part 3 evidence review |

There is no unresolved blocker to the completed smoke. JAX runs on the observed GPU; the pinned server PyTorch build does not. EGL failed; OSMesa works. See [runtime compatibility](RUNTIME_COMPATIBILITY.md). Molab storage quota remains unknown, and restarts stop detached jobs. Archived PID/job-state files are historical records, not current processes.

The selected E0 scope was a 50-episode Spatial pilot (10 tasks × 5 fixed initial states), not a full official suite. Molab restarts and pairing instability make short resumable batches important. E0 and parity are complete; S1, T1 and T1-ctrl have results, while S2 awaits Molab reconnection and the tolerant endpoint check. Paid compute and missing secrets require explicit user input; ordinary authorized implementation does not need repeated confirmation. Submission deadline 2026-11-03 22:00 VNT remains an assumption to verify with the supervisor. See the [E0 run record](../results/part2/e0/molab-2026-10-29/README.md).

## Current Molab dependency

On 2026-10-29, the paired bridge began returning `Failed to connect to marimo server` for the existing Molab URL. The tolerant S2 endpoint fix and visible notebook cells are implemented locally and pass CPU checks, but have not been synced or executed remotely. No new Molab runtime was started.
