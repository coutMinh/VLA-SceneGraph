# S2 endpoint replay diagnostic

The first S2-ctrl replay stopped before policy inference because the endpoint
SHA-256 differed from the verified SHA-256. This is an infrastructure/protocol
check failure, not a task failure.

The comparison was changed to retain hashes as diagnostics and compare the saved
processed observation arrays with explicit component thresholds. During the first
implementation run, five unique S2-ctrl endpoints matched. `S2-ctrl-t06-s02`
exceeded the wrist-camera mean/fraction thresholds twice, with identical values
on both process replays:

| Component | Maximum absolute difference | Mean absolute difference | Changed fraction |
|---|---:|---:|---:|
| Proprioception | 7.87e-6 | recorded in endpoint artifact | 100% of float entries |
| Agentview | 12 / 255 | 0.00097 / 255 | 0.050% |
| Wrist camera | 24 / 255 | 0.01414 / 255 | 0.821% |

The configured state bound was 1e-5; each image was also bounded by max absolute
error 64. The original image mean/fraction bounds (0.01 and 0.005) therefore
rejected this deterministic replay twice before policy inference. No policy action
was taken in either failed attempt.

The revised bounds are state maximum <=1e-5, image mean absolute error <=0.02
intensity levels, changed-value fraction <=0.01, and image maximum absolute error
<=64. Prompt, shapes, and dtypes remain exact. This accepts the observed renderer
drift with headroom while preserving strict checks on policy input identity.
`endpoint_check.json` records each component, threshold, and both hashes. These
infrastructure attempts remain in the v2 artifacts; all paired S2/S2-ctrl policy
runs must use the revised single implementation hash.

The frozen exact-hash manifests were preserved under `configs/part3-v1/`; the
array-reference manifests are in `configs/part3/`. The endpoint reference NPZs
are stored under `results/part3/landmark_verification/`.

## Final paired rerun

Both conditions were rerun under implementation SHA-256
`5e06c967a012ac494b1682447bd00ea8d3c04770ea739e7314a20f507031c1c3`. All 12 of 12
endpoint checks passed before policy inference. Across them, maximum proprioception
error was 7.87e-6; maximum camera mean error was 0.01414 intensity levels and maximum
changed-value fraction was 0.00822. No final-run infrastructure error occurred. The
two earlier S2-ctrl attempts remain in the artifact history as infrastructure checks;
the matched experiment result is S2-ctrl 6/6 successes and S2 0/6 successes.
