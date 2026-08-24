---
nav_order: 10
---

# 环境准备与构建

本文档介绍如何准备开发环境、构建 CATLASS DSL 并运行测试，包括安装依赖、配置环境变量等。

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
test -n "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"
test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
python -c "import mlir; import numpy"
```

以上命令均无任何输出、退出码为 `0`，即表示环境就绪；否则将输出错误信息。

运行端到端示例还需要：

```bash
npu-smi info
python -c "import torch; import torch_npu"
```

输出NPU信息，无Python相关报错，则代表依赖就绪。

## 构建 CATLASS DSL

除特别说明外，命令均在 `/path/to/catlass/python/tla_dsl` 执行，其中 `/path/to/catlass` 需替换为你 clone 的 CATLASS 仓库根目录。

### Development 模式构建（开发态）

```bash
cd /path/to/catlass/python/tla_dsl
./build.sh
# 强制清理并重新构建
./build.sh --clean
```

`build.sh` 默认执行 Development 模式构建，主要步骤为：

1. 检查 `ASCEND_HOME_PATH` 和 AscendNPU-IR 构建产物。
2. 从 AscendNPU-IR 的 `build/install` 设置 MLIR、LLVM 和 mlir Python 包路径。
3. 生成 TLA Python op 绑定。
4. 在 `csrc/mlir/build` 中构建 DSL 的二进制扩展 `_tla_type_bridge_native*.so` 和调试时使用的 `TlaCompile` 降级编译程序。
5. 以 editable 模式安装 `ascend-catlass-dsl`，且不重复安装依赖。

- `--clean` 会删除 DSL 子项目中的 `build/`、`csrc/mlir/build/`、`dist/`、egg-info、pytest 缓存和二进制库（`_tla_type_bridge_native*.so`）后，再执行 Development 模式构建。

构建成功后可检查关键产物：

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
python -c "import catlass"
```

以上命令均无任何输出、退出码为 `0`，即表示检查通过、产物就绪。

### Release模式

```bash
./build.sh --release
ls dist/*.whl
```

Release 模式在 `dist/` 生成 wheel。

## 运行测试用例

### pytest：前端降级到TLA IR

```bash
cd /path/to/catlass/python/tla_dsl
python -m pytest -q tests
```

运行单个测试文件时使用相同入口，例如：

```bash
python -m pytest -q tests/test_frontend_lowering.py
```

### lit test：TLA IR 降级到 NPUIR

```bash
lit -sv csrc/mlir/build/tests/lit
```

### e2e test：端到端验证

- 该测试额外依赖NPU环境，并且需要安装`torch` / `torch-npu`。

```bash
cd /path/to/catlass/python/tla_dsl
python examples/end_to_end/basic_mmad/basic_matmul.py --device 0
```

指定矩阵shape、layout和数据类型：

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py \
    --device 0 \
    --m 256 --n 512 --k 128 \
    --layout-a row --layout-b col \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
```

成功时输出包含：

```text
passed=True cache_key=<CACHE_KEY>
kernel.o=<CACHE_DIR>/<CACHE_KEY>/kernel.o
```

可用参数以脚本帮助为准：

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --help
```

### 完整的端到端测试

在 CATLASS 仓库根目录执行：

```bash
cd /path/to/catlass
bash tests/run_dsl_test.sh --device 0
```

该脚本会激活 Conda 环境、加载 CANN、检查 AscendNPU-IR、构建 DSL，然后以单个 pytest 进程运行 `tests/dsl_battery`。这是一组上板回归测试，不应与无需 NPU 的 DSL 单元测试混淆。

常用变量如下：

| 变量                                | 含义                                                     |
| ----------------------------------- | -------------------------------------------------------- |
| `ASCEND_HOME_PATH`                  | CANN Toolkit 根目录；脚本也会尝试定位并加载 `set_env.sh` |
| `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` | 已构建的 AscendNPU-IR 源码根目录                         |
| `CATLASS_DSL_DIR`                   | DSL 子项目路径；默认从脚本位置推导                       |
| `DEVICE_ID`                         | NPU device id；默认 `1`，可由 `--device` 覆盖            |
| `CATLASS_DSL_FORCE_RECOMPILE`       | 是否强制重新编译运行时产物；脚本默认设为 `1`             |

查看脚本当前支持的用例和路径解析规则：

```bash
bash tests/run_dsl_test.sh --help
```
