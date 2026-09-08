---
nav_order: 10
---

# 环境准备

本文档介绍 CATLASS DSL 的构建环境要求与环境检查方法。

## 快速准备构建环境

可选择下列两种快速准备方式，

- [Conda 安装](conda.md)：使用仓库的 `environment.yml` 创建开发环境
  - 需手动安装CANN和[构建 AscendNPU-IR](ascend_npu_ir.md)
- [Docker 安装](docker.md)：使用 Docker 构建开发镜像
  - 包含完整开发依赖

下文将列出所需的所有依赖软件，可根据实际环境自行选择安装方式。

## 构建环境要求

CATLASS DSL 仅支持 Ascend950 系列产品。

| 组件 | 版本或要求 |
| --- | --- |
| CANN | `>=9.1.0` |
| Python | `>=3.10,<3.14` |
| CMake | `>=3.28,<4.0` |
| Ninja | `>=1.12` |
| clang/lld | `>=10`(推荐 `19.1.7`) |
| AscendNPU-IR | `feature/regbase@a07821269ede7a5e683ac02c8a2d291608083741` |
| numpy | `>=2` |
| pybind11 | `2.13.6` |

- 若需执行纯Host测试用例，还需安装`pytest`
- 若需执行端到端测试用例，还需安装`torch`和配套的`torch_npu`，参考[PyTorch Ascend安装部署](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)

## 环境检查

基础环境检查：

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

以上命令均无任何输出、退出码为 `0`，即表示环境就绪；否则将输出错误信息。

运行端到端示例还需要：

```bash
npu-smi info
python -c "import torch; import torch_npu"
```

输出NPU信息，无Python相关报错，则代表依赖就绪。
