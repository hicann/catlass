# 编译 TLA DSL 与运行测试

## 1. 编译

确认 `ASCEND_HOME_PATH` 和 AscendNPU-IR 源码根目录后执行：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
export TLA_DSL_PREBUILT_ASCENDNPU_IR="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
./build.sh
```

默认构建为 Debug。`build.sh` 会：

1. 检查 `ASCEND_HOME_PATH`。
2. 从 `${TLA_DSL_PREBUILT_ASCENDNPU_IR}/build/install` 设置 MLIR/LLVM 路径。
3. 调用 `setup.py build_ext --inplace` 生成 Python op 绑定并编译 `tla-compiler`。
4. 以可编辑模式安装 Python 包，但不重复安装运行时依赖。

构建 Release wheel：

```bash
./build.sh --release
```

清理构建产物后重新构建：

```bash
./build.sh --clean
```

## 2. pytest

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m pytest -q tests
```

测试会检查已有编译产物；缺少产物时，`tests/conftest.py` 会尝试在
`csrc/mlir/build` 下配置并构建测试所需目标。

## 3. lit

先完成 pytest 的构建准备，再执行：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
lit -sv csrc/mlir/build/tests/lit
```

## 4. 端到端示例

仅编译 basic MMAD 示例：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python examples/end_to_end/basic_mmad/basic_matmul.py --build-only
```

上板执行：

```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 --all-layouts --m 1 --n 2 --k 3
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 --use-mutex
```

## 5. 端到端回归脚本

回到 CATLASS 仓库根目录执行：

```bash
cd "${CATLASS_ROOT}"
bash tests/run_dsl_test.sh --device 0
```

该脚本会激活 `ascend-catlass-dsl` Conda 环境、加载 CANN、检查 AscendNPU-IR，执行 `python/tla_dsl/build.sh`，再运行端到端用例。

脚本读取的主要变量如下：

| 变量 | 含义 |
| --- | --- |
| `ASCEND_HOME_PATH` | CANN toolkit 根目录 |
| `TLA_DSL_PREBUILT_ASCENDNPU_IR` | 已构建的 AscendNPU-IR 源码根目录 |
| `CONDA_ENV` | Conda 环境名，默认 `ascend-catlass-dsl` |
| `DEVICE_ID` | NPU device id，默认 `1`，可由 `--device` 覆盖 |
