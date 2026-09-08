---
nav_order: 20
---

# 使用 Conda 构建开发环境

使用Conda构建开发环境，不包括以下内容:
- CANN Toolkit
- AscendNPU-IR
- torch/torch_npu

## 创建环境

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl
conda env create -f environment.yml
conda activate ascend-catlass-dsl
```

`environment.yml` 已安装 Python、CMake、Ninja、Clang、lld、lit 和项目开发依赖，不需要再安装 `requirements.txt`。

确认工具版本：

```bash
python --version
cmake --version
ninja --version
clang --version
clang++ --version
lit --version
```

## 加载 CANN

```bash
source /path/to/cann/set_env.sh
test -n "${ASCEND_HOME_PATH}"
```

加载 CANN 后得到的 `ASCEND_HOME_PATH` 也是构建 AscendNPU-IR（第 3 节）的必要条件：其构建默认依赖 CANN（BiShengIR 部分需要 CANN 工具包环境），除非使用 `--disable-cann`。

## 构建 AscendNPU-IR

按照 [AscendNPU-IR 构建](ascend_npu_ir.md)生成所需产物，并设置：

```bash
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="/path/to/catlass/python/tla_dsl/3rdparty/AscendNPU-IR"
```

## 安装 torch/torch_npu
运行 NPU 示例时，按照 [PyTorch Ascend 安装部署](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)安装与 CANN 和驱动匹配的 `torch`、`torch-npu`。
