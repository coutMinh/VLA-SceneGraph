# Observed molab compatibility issues

2026-10-28: RTX PRO 6000 Blackwell, driver 595.71.05, glibc 2.41.
The pinned JAX 0.5.3 passed actual GPU matrix multiplication. The pinned server
PyTorch 2.7.1 build failed on sm_120; it is not used by the selected JAX baseline.

The official Python 3.8 client installed successfully, but importing its
PyTorch 1.11.0+cu113 failed before rendering:

`libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument`

This matches the issue documented by the
[PyTorch package maintainers](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/350).
[patchelf's manual](https://github.com/NixOS/patchelf/blob/master/patchelf.1)
documents `--clear-execstack` as removing the executable GNU_STACK flag.

`repair-client` is an explicit compatibility step. It checks the observed import
error, installs the system `patchelf` utility only if necessary and root is
available, and clears the flag only on affected shared libraries in the isolated
client's `torch/lib` directory. It does not change upstream source or disable
the host's stack protections. Before/after hashes and the tool version are saved
to `$SGVLA_WORK/docs/client_compatibility.json`; a fresh torch import must pass.
Actual rendering and G1 are still required after that import check.
