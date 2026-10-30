# Assignment status by Part

Updated 2026-10-30 (Asia/Saigon). Work is organized by assignment Part and experiment ID; dates identify observations and handoffs only.

| Part | Completed artifacts/evidence | Remaining deliverables |
|---|---|---|
| 1a — Scene graphs | [14-row literature matrix](../study/literature_matrix.md) and [comparison synthesis](../study/scene_graph_synthesis.md); scope notes identify abstract/project-level checks | Student reading status; primary full-text follow-up for sources marked with limited inspection scope |
| 1b — Diffusion / flow matching | [Unified objective, architecture and sampler comparison](../study/diffusion_vs_flow_matching.md), including DDPM/DDIM, Diffusion Policy, Octo, CogACT, pinned pi0 and FAST contrast | Student review; current-run client-call timing is summarized in the study note, without a cross-model speed claim |
| 2 - Reproduction | Pinned setup/smoke and GPU official-client parity; fresh E0 49/50 successes, 50/50 valid artifacts; source snapshot and local review retained | Archive and all 1,291 member hashes validated locally; registry updated with fresh runtime records; final interpretation is documented |
| 3 - Analysis | Fresh run: S1 22/27, T1 0/8 vs control 8/8, S2 0/6 vs control 6/6; exact S2 policy-input audit and failure reviews retained locally | Complete the formal strength/limitation claim matrix and review-count summary; redesign temporal/relocation tests before causal claims |
| 3+ — Cross-suite follow-up | 90 predeclared episodes across Object/Goal/Long: OBJ-swap 0/9 (ctrl 7/9, lang 8/9), GOAL-move 4/9 (ctrl 8/8), LONG-pre 3/9, LONG-undo 0/9, all E0 9/9 ([analysis](../analysis/CROSS_SUITE.md)) | Small clustered samples; video review limited to README examples |
| 4 — Improvement | Research direction in the plan | Evidence-based limitation selection, graph/interface/hypothesis/evaluation plan; optional prototype |

The student's personal reading progress has not been confirmed. Prepared notes do not certify reading completion. No assignment Part is complete in full.

## Part 3 run update - 2026-10-29

- **S1:** 27/27 completed; 23 successes, 1 wrong-first placement and 3 episodes without placement by the 220-step limit; no infrastructure errors.
- **T1-ctrl:** 8/8 target successes. **T1:** 0/8 target successes; all 8 first placed bowl 2 instead of the history-selected bowl 1. All pairs matched through the first action chunk, then diverged at the second policy call; 0/8 pairs qualify for the temporal bound.
- **S2-ctrl:** 6/6 target successes. **S2:** 0/6 target successes; all 6 first placed bowl 1 while the relation-selected target was bowl 2. Each episode reached the 220-step limit; all 12 S2/S2-ctrl endpoint checks matched.
- First five episodes in both S2 conditions passed the storage gate and received two-camera contact-sheet review. No obvious trajectory/render discontinuity was seen. Identity labels come from object-pose traces; exact contact/release timing remains unresolved.
- The first endpoint tolerance caused two repeated pre-inference infrastructure attempts on `S2-ctrl-t06-s02`. Revised image bounds were frozen before the final paired run; both attempts remain archived and are not task failures. All final S2/S2-ctrl episodes use implementation hash `5e06c967a012ac494b1682447bd00ea8d3c04770ea739e7314a20f507031c1c3`.

## Part 2 setup/smoke result

- Model: explicit `pi0_libero` on JAX; RTX PRO 6000 Blackwell; OSMesa renderer.
- Dummy `dummy-d575feeb4253`: finite `[50, 7]` chunk.
- Smoke `smoke-a081cbb0b0ab`: Spatial task 0/init 0, simulator seed 7; success at 74 steps, two decoded camera videos.
- Smoke check passed for config SHA-256 `3974be2ae6ea751c4f679ac154b8844b6e0e353697851e9fd7a1f6bd0613d658`.
- This is setup validation, not E0 or a benchmark estimate. The policy server was stopped after the run.

## Part 2 / E0 reduced pilot result and next work

The live Molab run used commit `1c98fd3efece9194330a038b2c62b7cb2d6a1046`. Official-client parity passed on task 0, initial states 0–2, with controlled policy noise. The first five episodes passed artifact inspection before continuation. Final E0 validation found 50/50 complete episodes, 48 successes, two task failures at the 220-step cap, zero infrastructure errors and zero pending/invalid episodes. See the [run record](../results/part2/e0/molab-2026-10-29/README.md) and [operating procedure](../repro/E0.md).

The failures are `E0-t01-s01` and `E0-t07-s03`. The later [sampled-frame review](../analysis/trajectories/e0_review/README.md) inspected both failures and state-0 successes of the same tasks: the failures lift and transport the intended bowl, but do not reach the placement predicate. Precise contact/release/alignment causes remain unresolved. The 96% success rate has an episode-level Wilson 95% interval of approximately [86.54%, 98.90%], without adjustment for within-task dependence. This supports using E0 as the baseline for controlled comparisons; it does not establish a relational-grounding or memory limitation. The predeclared states 0–2 paired subset contains 29/30 E0 successes, derived from the complete run's failure IDs.

The direct paired S2 result is 0/6 S2-only successes versus 6/6 S2-ctrl-only successes (episode-level exact two-sided p=0.03125). This is consistent with relying on a trained bowl position instead of recomputing the relation after a landmark move; the control suggests landmark novelty alone does not explain the gap. It remains exploratory: only tasks 1 and 6 with three states each were eligible; landmark displacements were not matched, and the p-value does not adjust for task clustering. See the [combined report](../results/part3/combined/summary.md) and [endpoint diagnosis](../analysis/trajectories/S2/endpoint_replay_diagnostics.md).

A subsequent discussion identified landmark visibility as an unresolved S2 confound:
in the retained S2-t01-s00 contact sheet the moved ramekin is near the image boundary.
The exact processed inputs of both cameras need inspection before attributing the
wrong-object result specifically to relation computation. This is not a finding that
the landmark is absent from both cameras; that has not been established.

## Fresh rerun complete - 2026-10-30

The new Molab runtime completed all 105 planned episodes (50 E0 + 55 Part 3), with
zero infrastructure-error attempts. Fresh E0 is 49/50, separate from the historical
48/50 run. E0 failure: `E0-t04-s02` (no target placement by 220 steps).

| Condition | Episodes | Target successes | Failure diagnostics |
|---|---:|---:|---|
| E0 | 50 | 49 | One no-placement failure |
| S1 | 27 | 22 | Four no-placement; one wrong first placement |
| T1 | 8 | 0 | All first placed bowl 2 while history-selected target was bowl 1 |
| T1-ctrl | 8 | 8 | All target successes |
| S2 | 6 | 0 | All first placed bowl 1 while relation-selected target was bowl 2 |
| S2-ctrl | 6 | 6 | All target successes |

S1's paired E0 comparison has five baseline-only successes and one condition-only
success (difference -4/27; exploratory paired exact two-sided p=0.21875). T1's 8/8
pairs and S2's 6/6 pairs each have one target success only in the control condition.
Treat tests as exploratory: task coverage is selected, episodes cluster within task,
and S2 relocation distances differ. The T1 temporal bound is not applicable: 0/8
pairs have equal policy inputs or action prefixes up to first placement, and the
scripted history frames were not supplied to the policy. Thus T1 does not isolate
memory use.

The S2 exact-input audit confirms that the moved ramekin is visible at changed table
positions in the agent view; the wrist view does not consistently contain the full
ramekin. The evidence does not establish that the landmark disappeared from both
cameras. Relocation magnitudes are unmatched (task 1: about 0.51-0.56 m vs
0.17-0.19 m; task 6: about 0.25-0.34 m vs 0.20-0.22 m). S2 is consistent with
choosing bowl 1 despite the changed relation, but the contrast is exploratory.

The final report is `../../artifacts/part2/2026-10-30-rerun/final-part3-report.json`
(SHA-256 `57e662b315f4d58e481573be1fac45bce7c6f1c0ceaea7aaa5833e9c4b517bb5`). The
source snapshot fingerprint is `e872f49ed1f9c434dd90c05e95eca4bfce549a530973561bc723b42bedde5894`
(base commit `5aaefdb3d38751bfbdd50a2effab876e66a2c458`). The archive is locally validated: 68,431,677 bytes, SHA-256 `0dc29398ee65d255732c5fd3aa5ca099b0ba2d4dd1129e961b37d90f72824c86`, 1,291 member hashes verified, 105 unique episode rows. See [ARTIFACTS.md](ARTIFACTS.md).
External notebooks: `Homework/NganLe/molab/vla_scenegraph/part2_reproduce.py` (setup) and `part2_e0.py` (parity/E0). Both reuse runtime environments/checkpoints. [ARTIFACTS.md](ARTIFACTS.md) maps retained results and the complete external archive. Historical raw `G1`/`d1.json` names are preserved identifiers, not current stage names.
