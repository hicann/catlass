# 手动 CMake 编译

正常构建请使用 `build.sh`。本章只说明如何直接配置 `csrc/mlir`。

## 1. 前置条件

当前 shell 必须满足以下条件：

- `ASCEND_HOME_PATH` 指向 CANN Toolkit 根目录。
- `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 指向已构建的 AscendNPU-IR 源码根目录。
- 当前 Python 可以导入 `pybind11`。
- `PYTHONPATH` 中包含 AscendNPU-IR 的 `mlir_core`。

```bash
export ASCEND_HOME_PATH="/path/to/ascend-toolkit"
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="/path/to/AscendNPU-IR"
export PYTHONPATH="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import pybind11; import mlir"
```

## 2. 配置和编译

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"

cmake -S csrc/mlir -B build/cmake/manual \
  -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/cmake/manual --target tla-compiler
```

主要产物位于：

```text
build/cmake/manual/python/catlass/
build/cmake/manual/tools/tla-compile/TlaCompile
build/cmake/manual/tests/lit/
```

## 3. CMake 选项

以下选项由 `csrc/mlir/CMakeLists.txt` 定义：

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `ENABLE_CPU_TRACE_INTRINSIC` | `OFF` | 启用 CPU trace intrinsic |
| `BISHENGIR_BUILD_TEMPLATE` | `ON` | 构建 HIVM template bitcode |
| `CATLASS_INCLUDE_DIR` | 仓库的 `include/` | CATLASS 公共头文件目录 |

示例：

```bash
cmake -S csrc/mlir -B build/cmake/manual \
  -G Ninja \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release
```

`MLIR_DIR`、`LLVM_DIR` 和 `MLIR_TBLGEN_INCLUDE_DIR` 会根据 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 自动设置。

## 4. 常见错误

### 找不到 AscendNPU-IR 头文件或库

确认 `CATLASS_DSL_PREBUILT_ASCENDNPU_IR` 指向源码根目录，而不是 `build/install`，并检查[AscendNPU-IR 文档](ascend_npu_ir.md)列出的构建产物。

### 找不到 `pybind11`

```bash
python -c "import pybind11; print(pybind11.get_include())"
```

手动 CMake 构建不使用 PEP 517 构建隔离，因此 `pybind11` 必须能在当前 Python 环境中导入。

### 找不到 `libMLIRPythonCAPI.so`

```bash
python -c "import mlir._mlir_libs as libs; print(libs.__file__)"
```

若导入失败，检查 `PYTHONPATH` 是否包含 `${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core`。
