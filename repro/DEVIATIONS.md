# Scope and deviations from the official reference

These experiments reproduce inference from a released checkpoint, not upstream
training and not the complete LIBERO benchmark. See [SETUP.md](../SETUP.md),
[PIPELINE.md](PIPELINE.md) and the manifests inside each run archive.

| Item | This project | Interpretation |
|---|---|---|
| Model | Explicit `pi0_libero`, JAX backend, released `gs://openpi-assets/checkpoints/pi0_libero` | Do not substitute the example's pi0.5 default or compare its published numbers as pi0 results |
| Source | openpi `215abfb217dbac7d5f1273282331b9b1866c0479`; LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` | Upstream sources stay unmodified; wrapper implementation hashes are recorded separately |
| Hardware | Observed NVIDIA RTX PRO 6000 Blackwell Server Edition | Recheck kernels and rendering on a replacement runtime |
| Environments | Python 3.11 policy server and Python 3.8 simulator, outside the notebook kernel | Freeze files and compatibility reports identify the actual runtime stack |
| Rendering | OSMesa; official 256-pixel rendering and 224-pixel processed policy inputs | Renderer change is declared; matching preprocessing is checked by parity |
| E0 scope | Spatial only; ten tasks, official initial states 0–4, seed 7: 50 episodes | Reduced pilot, not the official four-suite reference workload |
| Randomness | Explicit per-episode policy noise, paired by task/state | Differs from relying on one continuously advancing server RNG; controlled official-client parity tests the implemented contract |
| Execution loop | Ten settling steps, five actions per replan, ten flow steps, 50-action chunk, 220-action limit | Preserves the declared Spatial client settings |
| Part 3 | S1 27; S2/S2-ctrl 6 each; T1/T1-ctrl 8 each after verification | Selected task/state subsets; exclusions are recorded, not counted as policy failures |
| T1 intervention | Simulator-scripted bowl free-joint swap, stability checks and controller reset | Not a robot-executed swap; the baseline receives current observations rather than the prelude history |
| S2 endpoint replay | Saved processed arrays compared with predeclared state/image tolerances | Hash inequality alone is diagnostic; tolerance breaches are protocol errors before inference |
| Added logging | Both cameras, object identities/poses, simulator states, actions, timing and manifests | Simulator object state is evaluation evidence, not a baseline input |

The observed legacy client torch executable-stack repair and system OpenGL/OSMesa
installation are described in [RUNTIME_COMPATIBILITY.md](../docs/RUNTIME_COMPATIBILITY.md).
They affect the private environments/system loaders, not upstream source. Binary
repair hashes, actual library versions and freezes belong to each evidence archive.

## Historical completed run

The historical E0 source commit is `1c98fd3efece9194330a038b2c62b7cb2d6a1046`.
The [retained run record](../results/part2/e0/molab-2026-10-29/README.md) reports
48/50 successes, zero infrastructure errors, and an episode-level Wilson interval
of approximately 86.54%–98.90%. This interval does not adjust for task clustering.
Do not treat it as a full-suite benchmark estimate.

The [later E0 review](../analysis/trajectories/e0_review/README.md) supersedes the
run record's old statement that visual review was entirely pending. Both failures
handle the intended bowl without satisfying final placement. Exact low-level
contact/release causes remain unresolved.

## Fresh runtime and rerun - 2026-10-30

The replacement Molab runtime had no prior `~/vla_scenegraph_work`; the 105 fresh
rollouts are a separate run and do not recover prior videos. The run completed all
planned episodes with zero infrastructure-error attempts: E0 49/50; S1 22/27; T1
0/8 vs T1-ctrl 8/8; S2 0/6 vs S2-ctrl 6/6. The final report SHA-256 is
`57e662b315f4d58e481573be1fac45bce7c6f1c0ceaea7aaa5833e9c4b517bb5`; the local
source snapshot fingerprint is `e872f49ed1f9c434dd90c05e95eca4bfce549a530973561bc723b42bedde5894`
(base commit `5aaefdb3d38751bfbdd50a2effab876e66a2c458`). The deployed source had
uncommitted changes and has not been committed or pushed.

The runtime applied a provenance-only guard adjustment in
`runtime_adjustments/src/sgvla/analysis/baseline.py` after E0, so Part 3 could verify
the frozen deployed source snapshot instead of requiring the older source Git commit.
It does not change policy inference. E0 completions retain implementation fingerprint
`dd5705f2662b99446a281cc2ef4006cf5c70d4c80ad0acd45252b4372ce00a7d`; Part 3 uses
`9a3111d486b949dd54028a3f43ae50bd23afd46b5885acee78320471972f3bd1`. The archive
validator checks each condition against its own recorded source fingerprint.

S2's agent-view policy inputs show the relocated ramekin at its changed position;
the wrist view does not consistently show the full ramekin. The trial does not show
the landmark disappearing from both cameras. Relocation magnitudes are not matched,
so the 0/6 versus 6/6 outcome is an exploratory association, not a clean causal
estimate. For T1, none of eight pairs matched observations and actions through first
placement, and the scripted history frames are not policy inputs; the temporal bound
is not applicable. See [ARTIFACTS.md](../docs/ARTIFACTS.md). The downloaded archive passed ZIP CRC and all 1,291 member-hash checks; the episode index contains 105 unique rows. The archive SHA-256 is `0dc29398ee65d255732c5fd3aa5ca099b0ba2d4dd1129e961b37d90f72824c86`.
## Limits of inference

- S2 covers two tasks; treatment/control landmark displacement magnitudes were not
  matched. Landmark visibility is audited in exact policy inputs: the moved landmark remains visible in agent view, while wrist coverage is incomplete; relocation distances remain unmatched.
- In the fresh temporal comparison, none of eight pairs matched observations
  and actions through first placement. The predeclared temporal bound cannot be
  claimed as confirmed or refuted using those divergent pairs.
- Contact sheets use sampled frames. They do not establish exact grasp/contact or
  release timing when those moments fall between samples.
- All intervals/tests are exploratory at episode level; within-task dependence,
  selected verification coverage and multiple comparisons limit generalization.
- No graph-augmented policy has been evaluated by these baseline runs.
