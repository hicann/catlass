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
| `3rdparty/AscendNPU-IR-Dev` | AscendNPU-IR-Dev 子模块（`git submodule`）；TLA 需其 **`build/`** / **`build/install`** 下 TableGen 生成头、MLIR/LLVM CMake 包与多份 `libMLIR*` / `libBiShengIR*` |
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

依赖由 **`python/tla_dsl/environment.yml`** 统一安装。该 conda 环境只负责 Python、CMake/Ninja、编译器和 Python 打包/测试工具；**LLVM / MLIR 由 AscendNPU-IR-Dev 构建产物提供**。

#### 2.2.1 依赖版本

| 组件 | 版本 / 约束 |
|------|----------------|
| Python | **3.11** |
| pip | **26.0.1** |
| CMake | **4.2.3** |
| Ninja | **1.13.2** |
| `lit` 测试运行器 | **19.1.7** |
| Clang 工具链（可选） | **19.1.7** |
| GCC / G++ | **11.3** |
| NumPy | **2.4.4** |
| pytest | **9.0.2** |
| PyTorch / torch-npu | 手动安装，见 **§2.2.2**（`--run` 上板端到端示例需要；`--build-only` 不需要） |
| pybind11 | **2.13.6** |
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

不要从 conda 安装或使用 MLIR / LLVM 作为 TLA DSL 的构建运行时依赖；后续步骤会显式导出 AscendNPU-IR-Dev 的 `MLIR_DIR`、`LLVM_DIR`、`PYTHONPATH` 与 `LD_LIBRARY_PATH`。

#### 2.2.2 手动安装 PyTorch / torch-npu（仅上板运行需要）

`torch` 与 `torch-npu` 都是开源软件包。请按照下表选择适合您的架构的版本，并注意要选择适配当前 Python版本(3.11)的wheel，也就是说wheel的文件名中应该有`cp311`的字样：

| 架构 | 推荐版本 |
|------|----------|
| `x86_64` | `torch==2.9.0+cpu`，`torch-npu==2.9.0.post2` |
| `aarch64` / ARM | `torch==2.9.0`，`torch-npu==2.9.0.post2` |

### 2.3 昇腾 CANN / 工具链

请 source 一个 `CANN 9.1.0` 的 `set_env.sh`


```bash
# 示例：按本机 CANN 9.1.0 实际路径修改
source /usr/local/cann_9.1.B106/Ascend/ascend-toolkit/set_env.sh
```

以上命令会自动把环境变量 **`ASCEND_HOME_PATH`** 指向已安装的 **CANN / ascend-toolkit**

### 2.4 拉取并构建 AscendNPU-IR-Dev

`python/tla_dsl/csrc/mlir/CMakeLists.txt`（及 **`csrc/mlir/lib/Tools/CMakeLists.txt`** 中的 `_tla_type_bridge_native`）默认链接 **DSL 树内** 的 AscendNPU-IR-Dev：

```text
${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR-Dev
```

需 **CMake ≥ 3.28**、**Ninja ≥ 1.12**、**clang/clang++**，且 **§2.3** 中 CANN / **`ASCEND_HOME_PATH`** 已就绪。若在共享测试环境中工作，可以复用已经构建好的 `AscendNPU-IR-Dev`（配置方式见 [设置 AscendNPU-IR-Dev 构建根目录](#设置-ascendnpu-ir-dev-构建根目录)）。
```bash
cd "${CATLASS_ROOT}"
git submodule sync --recursive
# 可选：如果你的环境有本地的AscendNPU-IR-Dev，或需要改用 SSH URL，可在 sync 之后覆盖本地 URL。
# git config --local submodule.python/tla_dsl/3rdparty/AscendNPU-IR-Dev.url /path/to/local/AscendNPU-IR-Dev
git submodule update --init python/tla_dsl/3rdparty/AscendNPU-IR-Dev

cd python/tla_dsl/3rdparty/AscendNPU-IR-Dev
git submodule update --init
./build-tools/build.sh \
  --c-compiler clang \
  --cxx-compiler clang++ \
  '--add-cmake-options=-DCMAKE_SYSROOT=/' \
  '--add-cmake-options=-DLLVM_ENABLE_ZSTD=OFF' \
  '--add-cmake-options=-DLLVM_ENABLE_RTTI=ON' \
  --build-type Release \
  -j 128 \
  --enable-assertion \
  --disable-werror \
  --disable-mlir-werror \
  --disable-bishengir-werror \
  --build-triton \
  --enable-lld \
  --build ./build \
  --apply-patches \
  --python-binding
```

#### 设置 AscendNPU-IR-Dev 构建根目录

默认使用仓库内submodule的路径：
```bash
export TLA_DSL_PREBUILT_ASCENDNPU_IR="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR-Dev"
```
如果复用已经构建好的AscendNPU-IR：
```bash
# export TLA_DSL_PREBUILT_ASCENDNPU_IR="/path/to/prebuilt/AscendNPU-IR-Dev"
```

**校验 TableGen / 生成物是否就绪**（路径以本机构建树为准）：

```bash
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h" && echo "HIVM.h OK"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/tools/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc" && echo "TableGen inc OK"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/lib/cmake/mlir/MLIRConfig.cmake" && echo "Ascend MLIR CMake package OK"
ls "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/lib"/libMLIRHIVMDialect.so 2>/dev/null || ls "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/lib"/libMLIRHIVMDialect.a 2>/dev/null && echo "HIVM lib OK"
```

#### 暴露 AscendNPU-IR-Dev 的 MLIR / LLVM 运行环境

构建和运行 TLA DSL 时，应使用 AscendNPU-IR-Dev 构建出的 MLIR Python 包与动态库，不要使用 conda 的 MLIR binding：

```bash
export MLIR_TBLGEN_INCLUDE_DIR="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/include"
export PYTHONPATH="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/python_packages/mlir_core:${PYTHONPATH:-}"
```

> `./build.sh` 会在配置 CMake 前调用 `tools/generate_tla_python_bindings.py`，脚本会使用$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/bin/mlir-tblgen，根据 `csrc/mlir/include/Dialect/Tla/IR/Tla.td` 重新生成 `catlass/_mlir_bindings/tla_ops_gen.py`，避免手动修改生成文件后与 TD 定义不一致。
### 2.5 配置并编译 `tla`

**推荐一键**（须已 **`conda activate`**、已设 **`ASCEND_HOME_PATH`**，且 **§2.4** AscendNPU-IR-Dev 已就绪或已设 **`TLA_DSL_PREBUILT_ASCENDNPU_IR`**；建议同时导出上文 AscendNPU-IR-Dev 的 **`PYTHONPATH`** / **`LD_LIBRARY_PATH`**）：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
./build.sh
```

**手动**

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"

cmake -G Ninja \
  -S csrc/mlir \
  -B csrc/mlir/build \
  -DCMAKE_C_COMPILER="$(which clang)" \
  -DCMAKE_CXX_COMPILER="$(which clang++)" \
  -DCMAKE_SYSROOT=/ \
  -DCMAKE_SUPPRESS_REGENERATION=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_TBLGEN_INCLUDE_DIR="${MLIR_TBLGEN_INCLUDE_DIR}" \
  -DBISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH}"

ninja -C csrc/mlir/build tla-compiler
```

完成后应存在可执行文件：

```text
csrc/mlir/build/tools/tla-compile/TlaCompile
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

`lit` 测试使用环境中的 **`lit`/`llvm-lit`** 可执行文件：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
llvm-lit -sv csrc/mlir/build/tests/lit
```
**端到端示例**
```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python examples/end_to_end/basic_mmad/basic_matmul.py --build-only
```
上板执行
```bash
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 --all-layouts --m 1 --n 2 --k 3
python examples/end_to_end/basic_mmad/basic_matmul.py --run --device 0 --use-mutex
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
| AscendNPU-IR-Dev 子模块、`build.sh`、TableGen 校验与 `TLA_DSL_PREBUILT` | 上文 **2.4** |
| 依赖版本表、`python/tla_dsl/environment.yml`、`lit` 与 MLIR 19.1.7 同栈 | 上文 **2.2.1** |
| 一键构建（`build.sh` / `hatch build`） | 上文 **2.5** |
| 仅配置 MLIR 子目录（不跑完整 Catlass） | 见上文 **2.5** 的 **`./build.sh`** 或 `cmake` / `ninja` |
