# VLA-SceneGraph — where does π0 stop looking at the scene?

Assignment 3 (Scene Graphs × Vision-Language-Action models). We reproduce the official
**π0 (`pi0_libero`, openpi)** checkpoint on LIBERO. We then probe it with controlled
spatial and temporal interventions across all four LIBERO suites, to find a
limitation an explicit scene graph could address.

**Main finding.** π0 is near-perfect on the official scenes: 49/50 on Spatial and
9/9 on the Object, Goal and Long baselines. It fails whenever the *meaning* of the
scene and its *trained layout* disagree:

- In the Object suite, it fetches whatever sits at the trained position (0/9).
- In the Goal suite, it carries objects to where the destination used to be (1/6).
- In the Long suite, it does not re-read sub-goal progress from the scene (0/9).

Matched controls that change the layout without changing the meaning stay high:
S2-ctrl 6/6, T1-ctrl 8/8, OBJ-swap-ctrl 7/9 and GOAL-move-ctrl 8/8.

![Success rate per condition](media/success_overview.png)

## Results

### LIBERO-Spatial: reproduction and first probes

| ID | Intervention | Success | Reading |
|---|---|---:|---|
| E0 | Official scene and instruction (10 tasks × 5 initial states) | **49/50** | Reduced reproduction; official-client parity passed |
| S1 | The instruction names the *other* identical bowl by its relation | 22/27 | Language mostly followed (26/27 on the matched E0 subset) |
| S2 | A landmark is moved so the stated relation selects the other bowl | **0/6** | The bowl at the trained position is placed every time |
| S2-ctrl | The same landmark is moved; the target is unchanged | 6/6 | Layout novelty alone does not explain S2 |
| T1 | The bowls are swapped before the start; the instruction refers to the pre-swap layout | **0/8** | Expected for a policy that receives a single frame and no history |
| T1-ctrl | Same instruction and idle period, no swap | 8/8 | |

### Cross-suite follow-up: Object / Goal / Long

The protocol was declared before any rollout ([XSUITE_PROTOCOL](docs/XSUITE_PROTOCOL.md)).
Each condition runs 3 tasks × initial states 0–2, with policy noise paired to the
suite's E0.

| Suite | Condition | Intervention | Success |
|---|---|---|---:|
| Object | OBJ-E0 | official | 9/9 |
| | **OBJ-swap** | target ↔ the object in the *other* canonical target slot | **0/9** |
| | OBJ-swap-ctrl | two distractors swap places | 7/9 |
| | OBJ-lang | the instruction names the object in the other slot | 8/9 |
| Goal | GOAL-E0 | official | 9/9 |
| | **GOAL-move** | the task-relevant object is relocated (~12 cm) | **4/9** |
| | GOAL-move-ctrl | an irrelevant object is relocated | 8/8 |
| Long | LONG-E0 | official two-object tasks | 9/9 |
| | **LONG-pre** | the first sub-goal is already done at the start | **3/9** |
| | **LONG-undo** | the first placed object is returned to its start pose mid-episode | **0/9** |

Full per-episode tables: [results/xsuite/summary.md](results/xsuite/summary.md) ·
analysis: [analysis/CROSS_SUITE.md](analysis/CROSS_SUITE.md) ·
Spatial analysis: [analysis/ANALYSIS.md](analysis/ANALYSIS.md).

## Insights

1. **Position beats identity.** Swapping the target with the object in the other
   canonical slot fails 9/9. In 7 of those episodes the first object lifted is the
   decoy now sitting at the trained position. Swapping two unrelated objects costs
   only 2/9. Spatial S2 shows the same pattern with identical bowls; OBJ-swap
   shows it with visually distinct objects.
2. **Language selects a memorised (object, position) routine.** Naming a different
   object works (OBJ-lang 8/9, S1 22/27) when that object sits where its own
   training task puts it. Once identity and position disagree (OBJ-swap, S2), the
   position wins.
3. **pi0 is closed-loop for the object it picks up and open-loop for where it puts
   it.** When the object to pick up moved, pi0 reached it at its new position (3/3
   success). When the destination moved, it succeeded 1/6, and in 3 of the 5
   failures the carried object came within 1 cm of the destination's *old*
   position (measured from logged poses).
4. **Progress is a learned timeline, not something read from the scene.** If the
   first sub-goal is undone after it was achieved, pi0 places the second object and
   never returns to the first (0/9). This happens even though the current frame
   shows what is missing. Pre-completing the first sub-goal also breaks the
   routine (3/9; a wrong object was lifted first in 4/9).
5. **What this means for a scene graph (Part 4).** The failures share one gap: the
   policy has no explicit, current binding between *instruction entities*,
   *object identities/positions* and *task progress*. A graph that
   - resolves the instruction to a target node and destination node with live 3D
     poses, and
   - stores sub-goal predicates with an event history

   addresses exactly the variables these interventions perturb. The OBJ-lang
   result suggests that rewriting the instruction in text alone will not fix
   OBJ-swap or GOAL-move: language is followed only when it agrees with the trained
   layout. A spatial interface (target/destination masks or graph tokens) is the
   stronger candidate.

## Example episodes

Each clip shows agent view and wrist view, as the policy sees them (rotated 180°).

| Baseline | Intervention |
|---|---|
| ![](media/xsuite/OBJ-E0-t00-s00.gif) | ![](media/xsuite/OBJ-swap-t00-s00.gif) |
| ![](media/xsuite/GOAL-move-t01-s00.gif) | ![](media/xsuite/GOAL-move-t06-s01.gif) |
| ![](media/spatial/S2-ctrl-t01-s00.gif) | ![](media/spatial/S2-t01-s00.gif) |
| ![](media/spatial/T1-ctrl-t00-s00.gif) | ![](media/spatial/T1-t00-s00.gif) |
| ![](media/xsuite/LONG-pre-t01-s01.gif) | ![](media/xsuite/LONG-undo-t00-s00.gif) |
| ![](media/spatial/S1-t01-s00.gif) | ![](media/spatial/S1-t08-s00.gif) |

More examples: [media/](media/) (Spatial E0 success `E0-t01-s00`, the only E0
failure `E0-t04-s02`).

## Caveats

Each condition is small (6–27 episodes, 2–9 tasks) and clustered by task. All
p-values in the analysis are exploratory. Interventions are scripted simulator
state edits, not robot actions. Displacements differ between treatment and control.
T1 does not test memory cleanly, because the input prefixes diverge
([ANALYSIS](analysis/ANALYSIS.md)). E0 is a reduced 50-episode pilot, not a
full-suite reproduction. No graph-augmented policy has been evaluated yet.

## Repository layout

```text
src/sgvla/
  baselines/openpi/   setup, policy server, seeded-noise adapter, smoke episode
  eval/               scene verification, manifests, resumable E0 evaluator, traces, parity
  analysis/           Spatial S1/S2/T1 interventions, verification, outcome labels, reports
  xsuite/             cross-suite conditions (spec), simulator interventions, runner, report
configs/              pinned runtime config and frozen E0 manifest
docs/                 protocols (scene, Part 3, cross-suite), setup notes, status, artifacts
study/                Part 1: literature matrix, scene-graph synthesis, diffusion vs flow matching
repro/                Part 2: pipeline explanation, deviations, E0 procedure
analysis/             Part 3: ANALYSIS.md (Spatial), CROSS_SUITE.md, reviewed contact sheets
results/              registry.csv, Part 2/3 summaries, spatial_rerun_2026-10-30/, xsuite/
media/                README figure and GIFs (scripts/plot_results.py, scripts/make_gif.py)
tests/                CPU tests for manifests, resume/corruption, reports
```

Raw videos and traces are kept outside the repository: the Spatial rerun archive is
68 MB and the cross-suite archive 61 MB, both with SHA-256 manifests. See
[docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Reproduce

GPU work runs on a Linux GPU host (we used molab, RTX PRO 6000 Blackwell). Setup is
described in [SETUP.md](SETUP.md). In short: install the pinned openpi server
(Python 3.11) and LIBERO client (Python 3.8) environments, download `pi0_libero`,
and start the policy server.

```sh
export W=$HOME/vla_scenegraph_work C=$W/configs/part2_smoke.json
python -m sgvla --work $W --config $C start server
# Spatial E0 / Part 3: see repro/E0.md and docs/PART3_PROTOCOL.md
# Cross-suite (90 episodes; resumable; several workers per suite can share the claim files)
$W/envs/client/bin/python -m sgvla.xsuite --work $W --config $C --dry   # interventions only
$W/envs/client/bin/python -m sgvla.xsuite --work $W --config $C --suite libero_object
python -m sgvla.xsuite.report <results-dir> results/xsuite              # summary.{md,json}
python scripts/plot_results.py                                          # media/success_overview.png
```

Local CPU checks: `pip install -e '.[dev]'`, then `pytest -q` and `ruff check .`.

Study notes (Part 1): [literature matrix](study/literature_matrix.md),
[scene-graph synthesis](study/scene_graph_synthesis.md),
[diffusion vs flow matching](study/diffusion_vs_flow_matching.md).
Reproduction (Part 2): [pipeline](repro/PIPELINE.md), [deviations](repro/DEVIATIONS.md).
Status by Part: [docs/PART_STATUS.md](docs/PART_STATUS.md).
