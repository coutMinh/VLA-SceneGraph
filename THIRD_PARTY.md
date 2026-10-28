# Upstream sources and attribution

| Project | Pin | Usage | License |
|---|---|---|---|
| [openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` | Official model, transforms, checkpoint loader, websocket protocol and LIBERO helper functions | Apache-2.0; see upstream LICENSE and LICENSE_GEMMA.txt |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` | Simulator, fixed initial states, tasks and assets through openpi's submodule | MIT; see also asset notices |

Upstream repositories are cloned outside this repository. No third-party source
files or weights are vendored here. Our D1 rollout follows the official client's
control ordering and imports its environment/quaternion helpers directly.
Its implementation has separate logging, unique video paths and explicit failure
handling. The PyTorch fallback copies upstream transformers replacement files
into an isolated environment only; upstream source remains unchanged.

Checkpoint/model access and usage remain subject to the original model terms.
Literature references link to the original papers; PDFs are not redistributed.
