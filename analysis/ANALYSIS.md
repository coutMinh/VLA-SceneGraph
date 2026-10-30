# Fresh Spatial rerun findings - 2026-10-30

The cross-suite follow-up (Object / Goal / Long) is in [CROSS_SUITE.md](CROSS_SUITE.md).

This is a preliminary evidence summary, not the complete assignment claim matrix. A formal Part 3 report still needs the required strength/limitation table and final reviewed-trajectory counts.

## Run scope

The new Molab runtime completed 105 episodes: E0 covers 10 Spatial tasks x 5 official
initial states (50); S1 covers tasks 1-9 x states 0-2 (27); T1 and T1-ctrl each cover
tasks 0, 2 and 8 with the predeclared selected state subset (8 each); S2 and S2-ctrl
each cover tasks 1 and 6 x states 0-2 (6 each). No infrastructure-error attempts
remain. This is a reduced Spatial study, not the full LIBERO suite.

| Condition | Successes | Failures |
|---|---:|---:|
| E0 | 49/50 | 1 |
| S1 | 22/27 | 5 |
| T1 | 0/8 | 8 |
| T1-ctrl | 8/8 | 0 |
| S2 | 0/6 | 6 |
| S2-ctrl | 6/6 | 0 |

The fresh E0 result is distinct from the historical 48/50 result. It is not a before/
after estimate because the runs are separate stochastic samples.

## Evidence-supported observations

- **S1 relation sensitivity remains weak in this sample.** The condition has 22/27
  successes versus 26/27 on its matched E0 subset: five baseline-only and one
  condition-only success. The exploratory exact paired two-sided p-value is 0.21875.
  Failure review found four episodes without placement and one wrong first placement.
- **S2 repeatedly selects the bowl associated with the old scene.** All six S2 episodes
  first placed bowl 1 although the changed relation targeted bowl 2; all six controls
  placed bowl 1 successfully. This fits a stale position/identity heuristic, but does
  not identify the mechanism on its own.
- **The S2 visibility check narrows the claim.** Exact policy inputs show the moved
  ramekin at its changed position in agent view. Wrist view does not consistently
  contain the full ramekin. The landmark was not shown to disappear from both cameras.
  Treatment/control displacement magnitudes differ (task 1: 0.51-0.56 m vs
  0.17-0.19 m; task 6: 0.25-0.34 m vs 0.20-0.22 m), so the condition contrast is
  exploratory, not a clean causal relocation test.
- **The T1 task result does not test visual-history memory.** T1 failed 8/8 and control
  succeeded 8/8, but 0/8 pairs had equal policy inputs or action prefixes up to first
  placement. The scripted history frames were captured before client setup and were
  not passed to the baseline policy. The predeclared temporal bound is not applicable.
- **S1 failure modes differ.** Reviewed failures include no apparent target motion,
  one wrong-bowl placement, target lift without placement, and a non-target partial
  lift. Sampled contact sheets cannot resolve exact grasp/release timing.

## Limits

Task/state coverage is selected and small; episodes are clustered by task. The report's
p-values are exploratory and do not correct for task clustering or multiple comparisons.
The S2 displacement confound remains, and T1 lacks the matched input/action prefixes
needed for the temporal bound. No graph-augmented policy has been evaluated. Do not
claim these runs show that a scene graph improves VLA performance.

## Local evidence

The machine-readable final report and source snapshot are under
`../../artifacts/part2/2026-10-30-rerun/`; the report SHA-256 is
`57e662b315f4d58e481573be1fac45bce7c6f1c0ceaea7aaa5833e9c4b517bb5`. The complete local archive is
`../../artifacts/part2/2026-10-30-rerun/spatial-rerun-2026-10-30-evidence.zip` (68,431,677 bytes; SHA-256 `0dc29398ee65d255732c5fd3aa5ca099b0ba2d4dd1129e961b37d90f72824c86`). All 1,291 member hashes passed local validation. See [artifact map](../docs/ARTIFACTS.md).
