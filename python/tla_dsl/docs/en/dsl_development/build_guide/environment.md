---
nav_order: 10
---

# Environment Preparation

This document describes the build environment requirements and environment check methods for CATLASS DSL.

## Quickly Preparing the Build Environment

You can choose either of the following quick preparation methods:

- [Conda Installation](conda.md): create a development environment with the repository `environment.yml`
  - You must install CANN manually and [build AscendNPU-IR](ascend_npu_ir.md)
- [Docker Installation](docker.md): build a development image with Docker
  - Includes the complete development dependencies

The following sections list all required dependency software; choose an installation method based on your actual environment.

## Build Environment Requirements

CATLASS DSL supports only Ascend950 series products.

| Component | Version or requirement |
| --- | --- |
| CANN | `>=9.1.0` |
| Python | `>=3.10,<3.14` |
| CMake | `>=3.28,<4.0` |
| Ninja | `>=1.12` |
| clang/lld | `>=10` (`19.1.7` recommended) |
| AscendNPU-IR | `feature/regbase@a07821269ede7a5e683ac02c8a2d291608083741` |
| numpy | `>=2` |
| pybind11 | `2.13.6` |

- To run host-only test cases, you also need to install `pytest`.
- To run end-to-end test cases, you also need to install `torch` and the matching `torch_npu`; see [PyTorch Ascend Installation and Deployment](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download).

## Environment Check

Basic environment check:

```bash
python --version
clang --version
clang++ --version
lld --version
cmake --version
ninja --version
test -n "${ASCEND_HOME_PATH}"
test -f "python/tla_dsl/3rdparty/AscendNPU-IR/build/install/lib/cmake/mlir/MLIRConfig.cmake" || \
  test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
python -c "import numpy"
```

If all of the commands above produce no output and exit with code `0`, the environment is ready; otherwise, error messages are printed.

Running end-to-end examples additionally requires:

```bash
npu-smi info
python -c "import torch; import torch_npu"
```

If NPU information is printed with no Python-related errors, the dependencies are ready.
