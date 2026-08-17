# 编译 CATLASS DSL 与运行测试

本文档说明项目构建、编译器回归测试和 NPU 端到端验证。环境和版本要求统一见[环境准备](00_environment_setup.md)。除特别说明外，命令均在 `/path/to/catlass/python/tla_dsl` 执行，其中 `/path/to/catlass` 需替换为你 clone 的 CATLASS 仓库根目录。


## 0. 前置条件

CATLASS DSL 的构建依赖已构建完成的 AscendNPU-IR，`./build.sh` 本身**不会**构建它。请先按[构建 AscendNPU-IR](advanced/ascend_npu_ir.md)完成构建（产物位于 `build/install`），再将 AscendNPU-IR 源码根目录通过环境变量 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 指定（设置命令见第 1 节），然后回到本节继续。

NPU 端到端示例另需 `torch` / `torch-npu`，已由 `environment.yml` 一并安装（见[环境准备](00_environment_setup.md#3-环境检查)）。

## 1. 构建前检查

先加载 CANN，并确认 AscendNPU-IR 已构建：

```bash
source /path/to/cann/set_env.sh
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="/path/to/catlass/python/tla_dsl/3rdparty/AscendNPU-IR"
export PYTHONPATH="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}"
test -n "${ASCEND_HOME_PATH}"
test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
```

`CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 的取值是 AscendNPU-IR 源码根目录。变量未设置时，`build.sh` 使用 `3rdparty/AscendNPU-IR`。

## 2. 构建

### 2.1 Debug 开发构建

```bash
cd /path/to/catlass/python/tla_dsl
./build.sh
```

`build.sh` 默认执行 Debug 开发构建，主要步骤为：

1. 检查 `ASCEND_HOME_PATH` 和 AscendNPU-IR 构建产物。
2. 从 AscendNPU-IR 的 `build/install` 设置 MLIR、LLVM 和 Python 包路径。
3. 生成 TLA Python op 绑定。
4. 在 `csrc/mlir/build` 中构建 `TlaCompile` 与类型桥接动态库（`_tla_type_bridge_native*.so`）。
5. 以 editable 模式安装 `ascend-catlass-dsl`，且不重复安装依赖。

构建成功后可检查关键产物：

```bash
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
test -n "$(find csrc/mlir/build/python/catlass -name '_tla_type_bridge_native*.so' -print -quit)"
python -c "import catlass"
```

以上命令均无任何输出、退出码为 `0`，即表示检查通过、产物就绪；若某条命令报错或返回非零退出码，则对应产物缺失。

### 2.2 Release wheel

```bash
./build.sh --release
ls dist/*.whl
```

Release 模式在 `dist/` 生成 wheel，不执行 editable 安装。

### 2.3 清理并重新构建

```bash
./build.sh --clean
```

`--clean` 会删除 DSL 子项目中的 `build/`、`csrc/mlir/build/`、`dist/`、egg-info、pytest 缓存和二进制库（`_tla_type_bridge_native*.so`）后，再执行 Debug 构建。

## 3. pytest

完成 Debug 构建后执行：

```bash
cd /path/to/catlass/python/tla_dsl
python -m pytest -q tests
```

运行单个测试文件时使用相同入口，例如：

```bash
python -m pytest -q tests/test_frontend_lowering.py
```

## 4. lit 编译器回归测试

```bash
lit -sv csrc/mlir/build/tests/lit
```

## 5. NPU 端到端示例

端到端示例会编译并启动核函数，需要可用 NPU，以及与 CANN 和驱动匹配的 `torch`、`torch-npu`。运行基础 MMAD 示例：

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

## 6. 仓库级端到端回归

在 CATLASS 仓库根目录执行：

```bash
cd /path/to/catlass
bash tests/run_dsl_test.sh --device 0
```

该脚本会激活 Conda 环境、加载 CANN、检查 AscendNPU-IR、构建 DSL，然后以单个 pytest 进程运行 `tests/dsl_battery`。这是一组上板回归测试，不应与无需 NPU 的 DSL 单元测试混淆。

常用变量如下：

| 变量 | 含义 |
| --- | --- |
| `ASCEND_HOME_PATH` | CANN Toolkit 根目录；脚本也会尝试定位并加载 `set_env.sh` |
| `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` | 已构建的 AscendNPU-IR 源码根目录 |
| `CATLASS_DSL_DIR` | DSL 子项目路径；默认从脚本位置推导 |
| `DEVICE_ID` | NPU device id；默认 `1`，可由 `--device` 覆盖 |
| `CATLASS_DSL_FORCE_RECOMPILE` | 是否强制重新编译运行时产物；脚本默认设为 `1` |

查看脚本当前支持的用例和路径解析规则：

```bash
bash tests/run_dsl_test.sh --help
```
