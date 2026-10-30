# S2 paired landmark result

The verified subset contains six matched task/state pairs: Spatial tasks 1 and 6,
with initial states 0, 1 and 2. In S2, the same relational instruction selects bowl 2
after a landmark move; in S2-ctrl, landmarks move while bowl 1 remains the target.

| Condition | Completed | Success | Wrong first object | Endpoint checks |
|---|---:|---:|---:|---:|
| S2 | 6/6 | 0/6 | 6/6 (bowl 1 before target bowl 2) | 6/6 passed |
| S2-ctrl | 6/6 | 6/6 | 0/6 | 6/6 passed |

The direct matched comparison has six control-only successes and no S2-only success;
its episode-level exact two-sided p-value is 0.03125. The E0-paired comparisons and
the per-condition uncertainty are in the [combined report](../../../results/part3/combined/summary.md).
Treat this as exploratory evidence consistent with static-position reliance when a
spatial relation changes. It covers only two tasks and three states per task. The
landmark displacements were not matched between conditions, the episode p-value does
not account for task clustering, and the result does not establish a general VLA
limitation or show that a scene graph improves the policy.

All 12 final runs used implementation SHA-256
`5e06c967a012ac494b1682447bd00ea8d3c04770ea739e7314a20f507031c1c3`. The first five
episodes of each condition passed the artifact storage gate and were visually checked
using both cameras. Their contact sheets, hashes, and review receipts are in
[`first_batch-source-5e06c967/`](first_batch-source-5e06c967/). No obvious video or
scene discontinuity was observed. Pose traces provide object identity; sampled frames
do not isolate exact contact and release, so failure-cause descriptions remain
provisional.

Two pre-inference S2-ctrl endpoint attempts from the earlier narrower tolerance are
retained as infrastructure failures and excluded from the final episode counts. The
final rerun has no endpoint mismatches; see
[endpoint replay diagnostics](endpoint_replay_diagnostics.md).
