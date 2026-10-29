# CPU source verification, not a policy evaluation

`scene_report.json` was generated from the read-only pinned openpi/LIBERO checkout
with `sgvla eval-verify` on 2026-10-29. It contains each task's real benchmark index,
BDDL definitions, official and alternate instructions, object-specific goal, landmark
placement regions, and initial-state hashes. See the generated table in
[`docs/SCENE_VERIFICATION.md`](../../../docs/SCENE_VERIFICATION.md) and the interpretation
in [`SCENE_PROTOCOL.md`](../../../docs/SCENE_PROTOCOL.md).

The report is sufficient to freeze [`configs/e0_manifest.json`](../../../configs/e0_manifest.json).
It does not establish measured object poses, relation uniqueness, collision-free landmark
interventions or swap stability. Those fields are marked pending simulator. No GPU/model
result, policy success rate or parity pass is inferred from this verification.

Reproduce from the code-repo root:

```sh
python -m sgvla eval-verify --upstream ../third_party/openpi \
  --output results/part2/scene_verification/scene_report.json \
  --markdown docs/SCENE_VERIFICATION.md
```
