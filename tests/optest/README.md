# Torch Catlass

`tests/optest` 是 CATLASS 示例算子接入 PyTorch 的测试框架。框架将 AscendC/CATLASS kernel 通过 C++ extension 注册为 `torch.ops.catlass.*`，并由 Python 包 `torch_catlass` 提供调用入口。

详细设计见 [docs/design.md](docs/design.md)。

## 架构总览

框架按职责分为五层：

```text
Python API (`torch_catlass.ops.*`)
  -> Python loader (`torch_catlass/__init__.py`)
  -> PyTorch C++ extension (`src/`)
  -> Kernel adapter (`src/include/template/`)
  -> Kernel 实现 (`kernels/`)
```

- Python 层负责用户接口与动态库加载。
- C++ extension 层负责 `torch.ops.catlass.*` 注册和 NPU dispatch。
- adapter 层负责 Tensor 参数到 kernel ABI 参数的转换。
- kernel 层负责 prebuilt kernel 与 JIT kernel 的实现和执行。

## 目录结构

```text
tests/optest/
├── CMakeLists.txt
├── build.sh
├── pyproject.toml
├── docs/
│   └── design.md
├── include/
│   ├── catlass_kernel.h
│   ├── catlass_kernel_jit.h
│   └── catlass_kernel_prebuilt.h
├── src/
│   ├── catlass_torch.cpp
│   └── include/common/
├── kernels/
│   ├── 00_basic_matmul/
│   ├── include/
│   └── jit/
├── torch_catlass/
│   ├── __init__.py
│   └── ops/
├── tests/
│   └── test_00_basic_matmul.py
└── utils/
```

## 昇腾与架构识别说明

本项目面向昇腾 NPU 的 CANN/Ascend C 开发栈，相关术语和 SoC 信息以昇腾公开开发文档为准。

- Python loader 侧：通过 `torch_npu.npu.get_device_name()` 映射 arch id，用于加载 `torch_catlass/lib/<arch>/` 下的预构建库。
- JIT 编译侧：通过 `GetCurrentNPUArch()`（AscendC 平台 API）获取当前 SoC 对应架构，用于设置编译参数。
- 当前代码中的映射：
`Ascend910B.*` / `Ascend910_93` -> `2201`，`Ascend950` -> `3510`。

## 公共 ABI 与接口分层

`include/catlass_kernel.h` 仅作为聚合头，接口按模块拆分为：

- [catlass_kernel_jit.h](include/catlass_kernel_jit.h)：matmul/gemm/gemv 等 JIT 接口，参数语义为 `TParams + Params`。
- [catlass_kernel_prebuilt.h](include/catlass_kernel_prebuilt.h)：flash-attention/conv/mla 等 prebuilt 接口。

其中：
- `TParams` 表示编译期模板参数（dtype/layout/transpose 等）。
- `Params` 表示运行期参数（shape、buffer address 等）。

## 构建与测试

### 环境要求

- Python 3.11+
- PyTorch + torch-npu（与本机 CANN/驱动版本匹配）
- CMake 3.16+
- 可用昇腾 NPU 设备与环境

### 编译

```bash
bash build.sh
```

### 测试

```bash
pytest tests/ -v
```

## 使用示例

```python
import torch
import torch_catlass

a = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
b = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
c = torch_catlass.basic_matmul(a, b, outDType="float16")
```

## 开发指南

### 新增算子流程

1. 在 `include/catlass_kernel_jit.h` 或 `include/catlass_kernel_prebuilt.h` 预留 ABI（带 example 编号说明）。
2. 在 `kernels/` 实现 kernel（JIT template 或 prebuilt）。
3. 在 `src/` 增加 C++ op 适配和注册。
4. 在 `torch_catlass/ops/` 增加 Python wrapper。
5. 在 `tests/` 增加 pytest 用例。
