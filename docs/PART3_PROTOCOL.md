# Part 3 protocol: spatial grounding and temporal identity

Source inspection precedes simulator verification. The pinned LIBERO wrapper exposes
regenerate_obs_from_state, free-joint qpos/qvel, and object_states_dict. Its On predicate
calls support.check_ontop(bowl), including actual contact. We use that predicate for
placement and support; no distance-only replacement is used for success.

The E0 checkout and its 50 completions remain frozen. Part 3 uses a separate source
checkout with the same work directory, existing client/server venvs and checkpoint.
Part 3 never rewrites E0 completions or adopts their implementation fingerprint.

Predeclared state subset: task IDs 0–9, initial states 0–2; seed/noise pairing is E0's.
Excluded task/state pairs must have explicit verification reasons before policy runs.
S1 keeps the scene and changes the prompt to the source-verified bowl-2 relation.
Its stopping predicate is On(bowl 2, plate); original bowl-1 done is not used.

Pre-policy amendment: the initial verification-only run used 0.12/0.15 m. The
official scene's intended near placements measured 0.10–0.15 m, so those cutoffs
excluded valid source placements. Before any Part 3 policy inference, revise to
0.16/0.18 m and retain both verification reports. Task 0's alternative instruction
is ambiguous: both bowls are near the ramekin. Exclude its states from S1; do not
change its instruction after observing outcomes. No policy result informed this change.

Relation verification uses LIBERO predicates for source support/containment. For next
to, use planar distance <= 0.16 m for the named bowl and >= 0.18 m for the other,
with vertical separation <= 0.12 m. These are declared geometric operationalizations,
not official semantic annotations. Between uses projection in [0.15, 0.85] along the
landmark segment and perpendicular distance <= 0.05 m. Ambiguous states are excluded.

Temporal verification starts with table-supported task candidates 0, 1, 2, 8.
Other supports/drawers are excluded until separately verified. T1 and T1-ctrl use
the same history-phrased instruction, equal idle intervals and deterministic endpoints.
The swap is a scripted simulator intervention, not robot-executed manipulation.
Before policy launch require byte-identical processed two-camera images, prompt and
8-D proprioception for the pair. Record all comparison hashes and robot state checks.
The policy receives no prelude history. No temporal indistinguishability claim is
allowed if endpoint inputs differ; failed checks remain verification exclusions.

Metrics include target placement, distractor placement, first placed identity (including
ties), exclusive first-target success, steps and latency. The exclusive first-placement
metric is required for the temporal paired bound: unrestricted success can accept both
bowls on the plate. Pose-derived reach/lift/placement labels are provisional until
reviewed against video; a timeout alone cannot establish wrong-object behavior.

Run order: verify scenes, freeze manifests, S1 small batch and review, finish eligible
S1 states, then verified T1/T1-ctrl pairs. S2/S2-ctrl require an independently validated
landmark intervention and novelty control before enabling policy evaluation.

S2 verification searches only movable landmark tasks 1 (ramekin) and 6 (cookie box).
Task 0/8 require moving the destination plate and are deferred to avoid that confound;
other tasks involve support or fixture changes. For treatment/control require the
same movable landmark, a displacement of at least 4 cm, unchanged plate (<=2 mm),
other-object drift <=5 mm, stable placement and a unique intended relation. Search
radii are 0.13, 0.15, 0.11 m, with twelve fixed angles. Record unsuccessful candidates;
exclude a pair if either branch has no valid placement. Displacement magnitudes are
reported and are not assumed matched. Stabilize for 20 idle steps before policy input.

S2 endpoint replay stores both the exact verification hash and the processed observation
arrays. Before policy inference, compare against those arrays: prompt, shapes, and dtypes
must be exact, proprioception max absolute error <=1e-5, and each camera must have mean
absolute pixel error <=0.02 intensity levels, changed-value fraction <=0.01, and max
absolute pixel error <=64. These image bounds were calibrated on all six S2-ctrl replay
checks: the deterministic `t06-s02` replay differed by 0.01414 mean intensity and 0.00822
changed-value fraction on the wrist camera, while state max error was 7.87e-6 and the
other camera passed. Its second process replay reproduced the same differences. The prior
0.01/0.005 image bounds rejected this reproducible endpoint twice before policy inference.
The revised 0.02/0.01 bounds include that observed renderer drift with headroom; hashes
remain diagnostics. Record all component errors in `endpoint_check.json`; exceeding any
threshold is an infrastructure/protocol error and must not be scored as task failure.

Temporal intervention uses 40 scripted history steps, followed by ten stability steps.
The endpoint is assigned from the common settled simulator state, exchanging only
bowl free-joint qpos/qvel for the swap. Controller update/reset_goal is called for both
branches at the assigned endpoint (verified from the installed robosuite OSC source).
Full state assignments and per-frame qpos overrides are retained. This synthetic
endpoint construction is a declared deviation, not a naturally executed robot swap.
The endpoint comparison is checked before policy; recorded post-start inputs/actions
must also match through the first placement for a temporal paired-bound claim.
