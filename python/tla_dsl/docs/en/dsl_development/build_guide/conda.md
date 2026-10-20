---
nav_order: 20
---

# Building a Development Environment with Conda

Using Conda to build a development environment does not include the following:

- CANN Toolkit
- AscendNPU-IR
- torch/torch_npu

## Creating the Environment

```bash
# Replace /path/to/catlass with the actual path of the CATLASS repository you cloned
cd /path/to/catlass/python/tla_dsl
conda env create -f environment.yml
conda activate ascend-catlass-dsl
```

`environment.yml` already installs Python, CMake, Ninja, Clang, lld, lit, and the project development dependencies, so `requirements.txt` does not need to be installed separately.

Confirm the tool versions:

```bash
python --version
cmake --version
ninja --version
clang --version
clang++ --version
lit --version
```

## Loading CANN

```bash
source /path/to/cann/set_env.sh
test -n "${ASCEND_HOME_PATH}"
```

The `ASCEND_HOME_PATH` obtained after loading CANN is also a prerequisite for building AscendNPU-IR (Section 3): its build depends on CANN by default (the BiShengIR part requires a CANN toolkit environment) unless `--disable-cann` is used.

## Building AscendNPU-IR

Generate the required artifacts as described in [Building AscendNPU-IR](ascend_npu_ir.md), and set:

```bash
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="/path/to/catlass/python/tla_dsl/3rdparty/AscendNPU-IR"
```

## Installing torch/torch_npu

To run NPU examples, install the `torch` and `torch_npu` versions matching your CANN and driver as described in [PyTorch Ascend Installation and Deployment](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download).
