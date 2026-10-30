# Spatial scene and evaluation protocol verification

Source inspection: 2026-10-29, pinned openpi `215abfb217dbac7d5f1273282331b9b1866c0479`
and its LIBERO submodule `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
This document records source evidence before evaluator implementation. Runtime
verification is separate; no swap stability or new GPU result is claimed here.

## Verified source facts

`libero/libero/benchmark/libero_suite_task_map.py` defines the task order below;
`benchmark/__init__.py` uses order index 0 (identity), filename-derived language
(not the BDDL `:language` string), and `*.pruned_init` loaded with `torch.load`.
The initial-state files are real ZIP/Pickle archives, not Git LFS pointers.
Their pickle globals are NumPy array reconstruction, dtype, ndarray and codecs.
The CPU verifier will read only these whitelisted globals, check all arrays, and
hash both each source file and every selected state. State vector positions alone
cannot identify object poses without the task's MuJoCo model.

All ten BDDL files declare `akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl`.
All goals are `(On akita_black_bowl_1 plate_1)`. The shared object class establishes
the same object asset, not pixel-identical renderings after arbitrary motion.

| Official task ID | Original relation of bowl 1 | Bowl 2 relation / fixed S1 wording fragment | S2 landmark proposal | Swap feasibility from source only |
|---|---|---|---|---|
| 0 | between plate and ramekin | next to the ramekin | plate + ramekin | both on table; candidate |
| 1 | next to ramekin | next to the cookie box | ramekin | both on table; candidate |
| 2 | table center | next to the plate | table center is fixed: no landmark-only S2 | both on table; candidate |
| 3 | on cookie box | on wooden cabinet | support relocation would disturb supported bowl; deferred | different supports; unverified |
| 4 | in top drawer | on wooden cabinet | cabinet is a fixture; no simple landmark-only S2 | drawer collision/access; unverified |
| 5 | on ramekin | on cookie box | support relocation would disturb supported bowl; deferred | different supports; unverified |
| 6 | next to cookie box | on stove | cookie box; validate relation uniqueness | different heights; unverified |
| 7 | on stove | on wooden cabinet | stove is a fixture; no simple landmark-only S2 | different supports; unverified |
| 8 | next to plate | next to the ramekin | plate (also destination): placement confound must be checked | both on table; candidate |
| 9 | on wooden cabinet | on stove | cabinet is a fixture; no simple landmark-only S2 | different supports; unverified |

S1 uses `pick up the black bowl <fragment> and place it on the plate`, exactly
the vocabulary/style of the filename-derived official instructions. E0 target
is bowl 1; S1 target is bowl 2. BDDL regions/ranges and source checksums belong in
the generated scene report. Proposed S2 coordinates must be computed and validated
in simulator per state; no hard-coded coordinates are approved by this table.
No S2 or T1 condition is enabled by the E0 manifest.

## Success and stopping

For E0, preserve upstream `done` (task predicate) and its early stopping/220 control
step limit after 10 settling steps. Record both bowl-on-plate predicates separately.
For later retargeted conditions, compute `On(selected_target, plate_1)` using
LIBERO's own `eval_predicate_fn` and object state dictionary; do not use the original
task's `done` to stop a bowl-2 episode. Also record distractor placement and first
placed bowl. The goal permits both bowls on the plate: a temporal indistinguishability
bound requires an exclusive target-selection metric, not unrestricted goal success.
Actual support uses `check_ontop`; a simple center-distance threshold is insufficient.

## Runtime protocol and parity acceptance (fixed before runs)

- Two 256px simulator cameras, rotated 180 degrees and resized/padded to 224px;
  8-D proprioception, filename-derived prompt, 50x7 returned actions, replan every 5,
  10 Euler sampling steps, simulator seed 7. Follow the pinned official client.
- Each resumable episode has a fresh environment and explicit simulator seed.
  This differs from the official client's environment reuse and continuous RNG stream;
  do not claim full official stochastic-sequence reproduction.
- Policy noise is controlled without changing upstream: our server adapter accepts
  a namespaced seed/call counter and passes explicit noise to the public
  `Policy.infer(..., noise=...)` API. Noise has the model's padded 50x32 shape.
  Retries use the same per-task/state seed, independent of batch order or dummy calls.
- Run unmodified `eval_libero` via temporary in-process boundary adapters on task 0,
  initial states 0, 1, 2, one selected episode per invocation. This preserves its
  actual preprocessing/action loop. Both clients use the same controlled noise.
- Require byte-identical policy inputs at every policy call, equal success and control
  step count, and allclose action chunks/executed actions (`rtol=0`, `atol=1e-5`).
  Record failures, source hashes and raw evidence. No relaxed fallback is automatic.
- E0 requires a compatible passing parity report. First batch is 5 episodes; inspect
  complete artifact validation, failures, latency, elapsed time, storage/free space
  before continuing in batches of 5-10. Report Wilson intervals and per-task N;
  infrastructure errors are excluded from success denominators and remain pending.

## Unfinished runtime checks

Rendering/physics, S1 relation uniqueness in each sampled state, collision-free S2
poses, support stability and swaps need the Linux Molab simulator. A CPU source
report must mark these pending. Source inspection is sufficient to build E0;
it is not authorization to report S2/T1 feasibility or execute unverified interventions.
