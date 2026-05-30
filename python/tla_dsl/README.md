# TLA_DSL beta version

> **实验性 Beta**：API、目录布局、构建选项与测试集均可能变更，不建议在生产环境依赖当前行为；升级前请阅读变更说明并自行回归。

---

## 1. TLA DSL 是什么

**TLA DSL** 是面向昇腾 **AIC / Cube** 等执行单元的声明式内核 DSL：在 Python 中描述张量视图、`tla.mmad` / `tla.copy`、同步与管线屏障等，经 **`tla` MLIR 方言**（`!tla.*` 类型与 `tla.*` 操作）降低为中间 MLIR，再与 **AscendNPU-IR / HIVM** 等工具链对接，生成可加载的 **`kernel.o`** 等产物。


| 路径 | 说明 |
|------|------|
| `catlass/` | DSL 前端、运行时、与 MLIR Python 绑定的桥接逻辑 |
| `csrc/mlir/` | `TlaCompile` 编译器、CMake 工程 `tla`、LLVM/MLIR Pass 与 lit 用例 |
| `csrc/mlir/lib/Tools/CMakeLists.txt` | `TlaCompilePipeline`、`_tla_type_bridge_native`（Python 类型桥接）及可选 runtime wrapper 目标 |
| `3rdparty/AscendNPU-IR` | AscendNPU-IR 子模块（`git submodule`）；TLA 需其 **`build/`** 下 TableGen 生成头与多份 `libBiShengIR*` |
| `tests/` | `pytest` 与 `lit` |
| `tools/generate_tla_python_bindings.py` | 由 TableGen 结果生成 `catlass/_mlir_bindings/tla_ops_gen.py` |
| `build.sh`、`scripts/build_wheels.sh` | 一键配置 **`csrc/mlir`**、`ninja tla-compiler`、可选 **`pip install -e .`** 与 **`hatch build`**（需 **§2.2–2.4** 与 **`ASCEND_HOME_PATH`**） |
| `examples/` | 端到端示例（如 `end_to_end/basic_mmad`） |

---

## 2. 从零开始：环境与完整指令

```bash
export CATLASS_ROOT="/path/to/catlass"   # 改成你的本机catlass代码仓库路径
cd "${CATLASS_ROOT}"
```

### 2.1 安装 Miniconda（若尚未安装）

从 [Miniconda 官方文档](https://docs.conda.io/en/latest/miniconda.html) 下载并安装后，**重新打开终端**，确保 `conda` 可用。

### 2.2 创建 Conda 环境并安装依赖

依赖由 **`python/tla_dsl/environment.yml`** 统一安装，下表为与 **TLA / MLIR 19.1.x** 对齐的版本摘要；**请勿**在未改 YAML 的情况下将 **`mlir` / `lit`** 等升到 **22+** 主版本，否则缺少 **`PybindAdaptors.h`**，`csrc/mlir` 配置会失败。

#### 2.2.1 依赖版本

| 组件 | 版本 / 约束 |
|------|----------------|
| Python | **3.11** |
| pip | **26.0.1** |
| CMake | **4.2.3** |
| Ninja | **1.13.2** |
| LLVM / MLIR / TableGen / `lit` | **19.1.7** |
| MLIR Python 绑定 | **19.1.7** |
| Clang 工具链（可选） | **19.1.7** |
| GCC / G++ | **11.3** |
| NumPy | **2.4.4** |
| pytest | **9.0.2** |
| pybind11 | **3.0.3–3.0.4** |
| setuptools / wheel | **82.0.1** / **0.46.3–0.47.0** |
| hatchling / hatch-vcs | **1.29.0** / **0.5.0** |
| hatch（开发 CLI） | **1.16.5** |
| ml_dtypes | **0.5.4**（建议） |

**推荐**：在 Catlass 仓库内可直接使用 **`python/tla_dsl/environment.yml`** 一键创建环境：

```bash
cd "${CATLASS_ROOT}"
conda env create -f python/tla_dsl/environment.yml
conda activate ascend-catlass-dsl
```

`lit` 用于运行 `tests/lit`；若你更倾向使用 **AscendNPU-IR / llvm-project** 源码树内的 **`llvm-lit`**，须保证其 **LLVM 主版本与 conda 中 MLIR 19.1.x 一致**，否则 FileCheck / 测试前端行为可能不匹配。

将 **MLIR 的 CMake 包** 与 **可执行文件** 暴露到当前环境（与 `catlass/test_bootstrap.py` 的探测逻辑一致）：

```bash
export MLIR_DIR="${CONDA_PREFIX}/lib/cmake/mlir"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
```

**校验 MLIR 与 Python 绑定**（构建 `tla` 工程前必须能通过）：

```bash
python -c "import mlir; import mlir.ir as ir; print('MLIR python OK:', ir.Context())"
test -f "${CONDA_PREFIX}/include/mlir/Bindings/Python/PybindAdaptors.h" \
  && echo "C++ PybindAdaptors.h OK (required by tla mlir CMake)" \
  || echo "FAIL: missing PybindAdaptors.h — your MLIR is too new (nanobind-only); pin llvm/mlir/mlir-python-bindings/lit to 19.1.7 as §2.2.1"
```

### 2.3 昇腾 CANN / 工具链

`python/tla_dsl/csrc/mlir` 的 CMake **要求**环境变量 **`ASCEND_HOME_PATH`** 指向已安装的 **CANN / ascend-toolkit**（需存在 `include/acl/acl.h` 或安装布局与 CMake 中的探测一致）。

```bash
# 示例：按本机 CANN 实际路径修改（常见为 toolkit 的 `latest` 目录）
export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
# 若安装脚本提供 set_env.sh，建议 source 一次，将运行时库路径写入当前 shell：
source "${ASCEND_HOME_PATH}/../ascend-toolkit/set_env.sh"
```

### 2.4 拉取并构建 AscendNPU-IR

`python/tla_dsl/csrc/mlir/CMakeLists.txt`（及 **`csrc/mlir/lib/Tools/CMakeLists.txt`** 中的 `_tla_type_bridge_native`）默认链接 **DSL 树内** 的 AscendNPU-IR：

```text
${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR
```

需 **CMake ≥ 3.28**、**Ninja ≥ 1.12**、**clang/clang++**，且 **§2.3** 中 CANN / **`ASCEND_HOME_PATH`** 已就绪。

```bash
cd "${CATLASS_ROOT}"
git submodule update --init python/tla_dsl/3rdparty/AscendNPU-IR

cd python/tla_dsl/3rdparty/AscendNPU-IR
git submodule update --init
mkdir -p build
cd build
cmake ../third-party/llvm-project/llvm -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="-Wno-c2y-extensions" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=../ \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DBISHENGIR_BUILD_STANDALONE_IR_ONLY=ON \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF
ninja -j"$(nproc)"
```

**校验 TableGen / 生成物是否就绪**（路径以本机构建树为准）：

```bash
IR="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
test -f "$IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h" && echo "HIVM.h OK"
test -f "$IR/build/tools/bishengir/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc" && echo "TableGen inc OK"
ls "$IR/build/lib"/libBiShengIRHIVMDialect.so 2>/dev/null || ls "$IR/build/lib"/libBiShengIRHIVMDialect.a 2>/dev/null && echo "HIVM lib OK"
```

上述检查通过后，**无需**再设 `TLA_DSL_PREBUILT`（默认即使用该路径）；或在其它机器上把 **`TLA_DSL_PREBUILT_ASCENDNPU_IR`** 指到同一套**已构建根目录**。

#### 使用已构建的 AscendNPU-IR（不重复编译）

```bash
export TLA_DSL_PREBUILT_ASCENDNPU_IR="/path/to/built/AscendNPU-IR"
```

### 2.5 配置并编译 `tla`

**推荐一键**（须已 **`conda activate`**、已设 **`ASCEND_HOME_PATH`** / **`MLIR_DIR`**（或 **`CONDA_PREFIX`** 可解析 MLIR），且 **§2.4** AscendNPU-IR 已就绪或已设 **`TLA_DSL_PREBUILT_ASCENDNPU_IR`**）：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
./build.sh
```

**手动**

```bash
cd "${CATLASS_ROOT}"

cmake -G Ninja \
  -S python/tla_dsl/csrc/mlir \
  -B python/tla_dsl/csrc/mlir/build \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DMLIR_TBLGEN_INCLUDE_DIR="${CONDA_PREFIX}/include" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C python/tla_dsl/csrc/mlir/build tla-compiler
```

完成后应存在可执行文件（路径以构建树为准）：

```text
python/tla_dsl/csrc/mlir/build/tools/tla-compile/TlaCompile
```

### 2.6 安装 Python 包（可编辑模式）

若 **未** 使用 **`./build.sh`**（脚本内已尝试 **`pip install -e .`**），仍在 **`environment.yml` 创建的环境**（默认名 **`ascend-catlass-dsl`**）中手动安装：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
pip install -U pip
pip install -e .
```

### 2.7 运行测试（可选）

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m pytest -q tests
```

`lit` 需系统或 conda 提供 **`lit`/`llvm-lit`** 可执行文件；若已安装：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
llvm-lit -sv csrc/mlir/build/tests/lit
```

---

## 3. 版本状态说明

- 当前 DSL、编译器管线与运行时处于 **Beta / 实验阶段**：行为、错误信息、默认 CMake 选项与示例均可能调整。
- **不保证**跨小版本的 IR 文本、缓存目录结构或 Python API 完全兼容；升级后请重新跑通你的用例与 CI。
- 生产或对外发布场景，请等待项目方发布的 **稳定版** 说明与支持策略。

---

## 4. 快速索引

| 主题 | 位置 |
|------|------|
| DLPack → TLA `Tensor` 字段来源与 layout 转换 | `docs/dlpack_to_tla_tensor.md` |
| MMAD 端到端示例与运行参数 | `examples/end_to_end/basic_mmad/README.md` |
| AscendNPU-IR 子模块、`build.sh`、TableGen 校验与 `TLA_DSL_PREBUILT` | 上文 **2.4** |
| 依赖版本表、`python/tla_dsl/environment.yml`、`lit` 与 MLIR 19.1.7 同栈 | 上文 **2.2.1** |
| 一键构建（`build.sh` / `hatch build`） | 上文 **2.5** |
| 仅配置 MLIR 子目录（不跑完整 Catlass） | 见上文 **2.5** 的 **`./build.sh`** 或 `cmake` / `ninja` |
