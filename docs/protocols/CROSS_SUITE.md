# Cross-suite follow-up protocol (Object / Goal / Long)

Declared 2026-10-30 before any cross-suite policy rollout. Code: `src/sgvla/cross_suite/` (named `xsuite` when the run was made, commit `2b9a76b`)
(`spec.py` holds every task ID, condition and threshold below; the manifest SHA-256
prefix is `268949d38229b7e2`).

## Why

The Spatial study found two failure signals for `pi0_libero`:

- **S2 (spatial):** when a landmark moved so the stated relation selected the *other*
  identical bowl, pi0 placed the bowl at the trained position in 6/6 episodes
  (control 6/6 success).
- **T1 (temporal):** when the bowls were swapped before the policy started and the
  instruction referred to the pre-swap layout, pi0 failed 8/8 (control 8/8 success).
  The policy receives one frame and no history, so this is expected by design. The
  input prefixes did not match, so the result is not a clean memory test.

This follow-up asks whether the same two weaknesses appear in the other three LIBERO
suites. It uses each suite's structure:

| Suite | Structure we exploit | Weakness tested |
|---|---|---|
| Object | The target object always starts in one of only **two canonical "target slots"** (BDDL `floor_target_object_region`). | Position shortcut: does pi0 fetch *what is named* or *what is at the trained slot*? |
| Goal | All 10 tasks share **one fixed layout**; only the instruction changes. | Does pi0 act on an object's *current* position or the trained one? |
| Long | Two-object tasks have an **implicit sub-goal order**. | Is progress read from the scene, or replayed as a learned timeline? |

## Conditions (3 tasks × initial states 0–2 per condition)

| ID | Tasks | Intervention (applied after 10 idle settle steps) | Success |
|---|---|---|---|
| OBJ-E0 | 0, 1, 4 | none | official `In(target, basket)` |
| OBJ-swap | 0, 1, 4 | swap target ↔ object in `other_object_region_0` (the other canonical slot), xy only | official |
| OBJ-swap-ctrl | 0, 1, 4 | swap objects in `other_object_region_1` ↔ `_2`; target untouched | official |
| OBJ-lang | 0, 1, 4 | none; prompt names the object in `other_object_region_0` | `In(named object, basket)` |
| GOAL-E0 | 8, 6, 1 | none | official |
| GOAL-move | 8, 6, 1 | relocate the task-relevant object (plate / bowl / bowl) | official |
| GOAL-move-ctrl | 8, 6, 1 | relocate an irrelevant object (cream cheese / plate / cream cheese) | official |
| LONG-E0 | 0, 1, 7 | none | official (both objects in basket) |
| LONG-pre | 0, 1, 7 | first named object pre-placed in the basket | official |
| LONG-undo | 0, 1, 7 | the first object to rest in the basket for 20 steps is returned to its policy-start pose (once, mid-episode) | official |

Tasks: Object 0 alphabet soup, 1 cream cheese, 4 ketchup. Goal 8 bowl→plate,
6 cream cheese→bowl, 1 bowl→stove. Long 0 alphabet soup + tomato sauce,
1 cream cheese + butter (same scene as 0), 7 alphabet soup + cream cheese.

## Fixed settings

- Policy: same pinned `pi0_libero` server (JAX, 10 Euler steps, 5 actions per replan,
  224 px, 180° image rotation). Official per-suite step limits: Object 280, Goal 300,
  Long 520.
- Every condition, including each suite's E0, uses the same idle prelude: 10 settle
  steps, intervention (or none), then 20 more idle steps before the first policy
  call. This differs from the official 10-step wait and is the same for all conditions.
- Paired policy noise: the seed depends on (suite, task, state) only, so all
  conditions of a task/state pair share the same noise sequence.
- Relocation search (GOAL-move/ctrl): radii 0.12, 0.15, 0.10 m × 12 angles. The first
  candidate wins if it stays inside the bounding box of the movable objects
  (+3 cm), keeps ≥7 cm clearance and penetrates ≤4 mm. Displacement is recorded.
  Treatment and control move different objects, so their displacements are not matched.
- Stability gate before the policy starts: drift ≤1 cm and tilt ≤0.25 rad for every
  movable object after 20 idle steps. Failing episodes are **excluded**, not scored.
  LONG-pre instead requires the object to remain `In` the basket.
- Infrastructure errors (e.g. websocket handshake timeout) are logged as `error-*.json`.
  They never count as task failures; the episode is rerun.

## Outcome signals

All are computed from logged simulator state:

- `success`: the condition's success predicate before the step limit.
- `first_lifted`: the first movable object raised >5 cm above its policy-start
  height. A wrong `first_lifted` means a **wrong-object** choice.
- `watch_final`: final values of the goal predicates, plus `In(decoy, basket)` for
  Object.
- `undo`: for LONG-undo, whether and when the reversal fired and which object it
  returned.

Coarse labels (`success`, `wrong_object_first`, `target_lifted_not_placed`,
`no_lift`) are pose-derived. They are checked against the videos for the examples
shown in the README.

## Predictions

- If pi0 uses a position shortcut, OBJ-swap and GOAL-move fail with a wrong
  `first_lifted` or an action at the old location. Their controls stay near E0.
- If OBJ-lang works like Spatial S1, language mostly selects the object.
- A reactive single-frame policy should handle LONG-pre and LONG-undo: the current
  frame shows what remains to be done. Failure would point to a learned sub-goal
  timeline rather than missing memory.

## Scope

The study has 90 episodes: 3 tasks × 3 states per condition. Results are reported per
suite and are never pooled across suites. Exact McNemar p-values on pairs with E0 are
exploratory and do not correct for task clustering or multiple comparisons.
