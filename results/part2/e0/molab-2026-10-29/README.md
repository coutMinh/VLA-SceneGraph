# Reduced E0 Molab pilot — 2026-10-29
## Run identity
- Source commit: `1c98fd3efece9194330a038b2c62b7cb2d6a1046`, verified against the latest
  pushed commit before the Molab checkout was advanced and the run started.
- Manifest: 50 fixed LIBERO Spatial episodes, tasks 0–9 × initial states 0–4, seed 7.
  SHA-256: `fe252dd6412e9b88c1ae02e7356826a983bf6c335404013cc4b8f96e0897bd82`.
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition. Policy inference used JAX
  0.5.3 on CUDA; the pinned pi0_libero checkpoint was loaded.
- Parity gate: passed using the official client on task 0, initial states 0, 1, and 2.
  Policy inputs were byte-identical, and actions, chunks, observations, steps, and
  success results matched. Action comparison used absolute tolerance 1e-5 and zero
  relative tolerance; policy RNG was controlled.
## E0 outcome
| Task | Successes |
|---:|---:|
| 0 | 5/5 |
| 1 | 4/5 |
| 2 | 5/5 |
| 3 | 5/5 |
| 4 | 5/5 |
| 5 | 5/5 |
| 6 | 5/5 |
| 7 | 4/5 |
| 8 | 5/5 |
| 9 | 5/5 |
| **Total** | **48/50** |
All 50 episode artifacts passed final validation; 48 episodes succeeded, two ended at
the 220-step cap, and there were no infrastructure errors or pending/invalid episodes.
The task failures were `E0-t01-s01` and `E0-t07-s03`. The overall success estimate is
0.96 with a Wilson 95% interval of [0.865399, 0.988961]. The interval does not adjust
for dependence among initial states from the same task.
The first batch of five passed the artifact review before the remaining episodes ran.
Trace arrays and both camera videos passed integrity/decode checks. Manual visual
trajectory review remains pending; the two failures have not been classified by cause.
## Raw artifacts
The complete manifest, traces, actions, observations, videos, logs, parity evidence,
and summary remain in the Molab work directory:
`/home/marimo/vla_scenegraph_work/results/part2/e0/fe252dd6412e9b88c1ae02e7356826a983bf6c335404013cc4b8f96e0897bd82/`
The final summary is `summary.json` in that directory; episode artifacts are under
`episodes/`. The Molab checkout ran the latest pushed code. This is a reduced Spatial
pilot, not a full official-suite reproduction and not evidence of scene-graph gains.
