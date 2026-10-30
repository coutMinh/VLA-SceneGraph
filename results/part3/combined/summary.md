# Part 3 exploratory results

These results describe the verified subset of LIBERO Spatial episodes. They are exploratory; they do not establish that a scene graph improves the policy.

Policy fingerprint: `536822bf671fb5ba1392ca44eb0a2264c02a54bed981564bee143a7c0ff818e2`  
Baseline source commit: `not recorded`  
Baseline implementation SHA-256: `9f6866cf9662be91310fc6e549bd5491aae4137e9af78180a09ee247032bfd4e`

| Condition | Completed / planned | Success | First-target success | Wrong first object | No placement | Matched E0: E0-only / condition-only | Paired p |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 27 / 27 | 23 | 23 | 1 | 3 | 4 / 1 | 0.375 |
| S2 | 6 / 6 | 0 | 0 | 6 | 0 | 5 / 0 | 0.0625 |
| S2-ctrl | 6 / 6 | 6 | 6 | 0 | 0 | 0 / 1 | 1 |
| T1 | 8 / 8 | 0 | 0 | 8 | 0 | 8 / 0 | 0.007812 |
| T1-ctrl | 8 / 8 | 8 | 8 | 0 | 0 | 0 / 0 | 1 |

Implementation hash by condition:

- S1: `f2dfb19ade6eeafd87c977e41de348f77ede1df072364b10cc44dd016589c1b3`
- S2: `5e06c967a012ac494b1682447bd00ea8d3c04770ea739e7314a20f507031c1c3`
- S2-ctrl: `5e06c967a012ac494b1682447bd00ea8d3c04770ea739e7314a20f507031c1c3`
- T1: `f2dfb19ade6eeafd87c977e41de348f77ede1df072364b10cc44dd016589c1b3`
- T1-ctrl: `f2dfb19ade6eeafd87c977e41de348f77ede1df072364b10cc44dd016589c1b3`

## Direct landmark comparison

S2 versus S2-ctrl, matched on 6 initial states: S2-only successes 0, control-only successes 6, difference -1.000, exact two-sided p=0.03125.

## Temporal prefix check

0 of 8 pairs matched observations and actions through first placement; the exclusive first-target bound held in 0 and was violated in 0. 8 pairs diverged before that point.

## Limits

- Verification exclusions restrict task/state coverage
- Failure-cause labels remain provisional until visually validated
- Small, selected task/state sets; episode intervals do not adjust for task clustering.
- S2 landmark displacement magnitudes were not matched between treatment and control.
- Failure-cause descriptions remain coarse where sampled video cannot isolate contact or release.
