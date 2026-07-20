# TLA_DSL beta version

> **实验性 Beta**：API、目录布局、构建选项与测试集均可能变更，不建议在生产环境依赖当前行为；升级前请阅读变更说明并自行回归。

---

## 1. TLA DSL 是什么

**TLA DSL** 是面向昇腾 NPU 的声明式内核 DSL。在 Python 中描述张量视图、`tla.mmad` / `tla.copy`、同步与管线屏障等，经 **`tla` MLIR 方言**（`!tla.*` 类型与 `tla.*` 操作）降低为中间 MLIR，再与 **AscendNPU-IR / HIVM** 等工具链对接，生成可加载的 **`kernel.o`** 等产物。

| 路径 | 说明 |
|------|------|
| `catlass/` | DSL 前端、运行时、与 MLIR Python 绑定的桥接逻辑 |
| `csrc/mlir/` | `TlaCompile` 编译器、CMake 工程 `tla`、LLVM/MLIR Pass 与 lit 用例 |
| `csrc/mlir/lib/Tools/CMakeLists.txt` | `TlaCompilePipeline`、`_tla_type_bridge_native`（Python 类型桥接）及可选 runtime wrapper 目标 |
| `3rdparty/AscendNPU-IR` | AscendNPU-IR 子模块（`git submodule`）；TLA 需其 **`build/`** / **`build/install`** 下 TableGen 生成头、MLIR/LLVM CMake 包与多份 `libMLIR*` / `libBiShengIR*` |
| `tests/` | `pytest` 与 `lit` |
| `tools/generate_tla_python_bindings.py` | 由 TableGen 结果生成 `catlass/_mlir_bindings/tla_ops_gen.py` |
| `build.sh`、`scripts/build_wheels.sh` | 一键配置 **`csrc/mlir`**、`ninja tla-compiler`、可选 **`pip install -e .`** 与 **`hatch build`**（需 **§2.2–2.4** 与 **`ASCEND_HOME_PATH`**） |
| `examples/` | 端到端示例（如 `end_to_end/basic_mmad`） |

---

## 2. 从零开始：环境与完整指令

### 2.1 安装 Miniconda（若尚未安装）

从 [Miniconda 官方文档](https://docs.conda.io/en/latest/miniconda.html) 下载并安装后，**重新打开终端**，确保 `conda` 可用。

### 2.2 创建 Conda 环境并安装依赖

依赖由 **`python/tla_dsl/environment.yml`** 统一安装。该 conda 环境只负责 Python、CMake/Ninja、编译器和 Python 打包/测试工具。

- LLVM / MLIR 由 AscendNPU-IR 构建产物提供，不在下列依赖之内

#### 2.2.1 依赖版本

| 组件 | 版本 / 约束 |
|------|----------------|
| Python | `>=3.10,<3.14` |
| pip |  |
| CMake | `>=3.28`|
| Ninja | `>=1.12` |
| `lit` 测试运行器 | `19.1.7` |
| Clang | `19.1.7` |
| NumPy | `<2` |
| pytest | |
| PyTorch / torch-npu | `--run`上板运行时需手动安装，见 [2.2.2 安装pytorch](#222-安装pytorch)|
| pybind11 | **2.13.6** |
| setuptools / wheel | |
| hatchling / hatch-vcs |  |
| hatch（开发 CLI） |  |
| ml_dtypes | ） |

**推荐**：在 Catlass 仓库内可直接使用 **`python/tla_dsl/environment.yml`** 一键创建环境：

```bash
cd "${CATLASS_ROOT}"
conda env create -f python/tla_dsl/environment.yml
conda activate ascend-catlass-dsl
```

#### 2.2.2 安装pytorch

参考[PyTorch Ascend安装部署](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)安装pytorch及其NPU插件。

### 2.3 昇腾 CANN / 工具链

参考[CANN安装部署](https://www.hiascend.com/cann/download)安装CANN。

```bash
# 替换为您环境的实际路径
source /usr/local/Ascend/cann/set_env.sh
```

### 2.4 拉取并构建 AscendNPU-IR

`python/tla_dsl/csrc/mlir/CMakeLists.txt`（及 **`csrc/mlir/lib/Tools/CMakeLists.txt`** 中的 `_tla_type_bridge_native`）默认链接 **DSL 树内** 的 AscendNPU-IR：

```text
${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR
```

```bash
cd "${CATLASS_ROOT}"
git submodule sync --recursive
git submodule update --init python/tla_dsl/3rdparty/AscendNPU-IR

cd python/tla_dsl/3rdparty/AscendNPU-IR
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

#### 设置 AscendNPU-IR 构建根目录

默认使用仓库内submodule的路径：

```bash
export TLA_DSL_PREBUILT_ASCENDNPU_IR="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
```

可根据环境实际情况，修改该路径。

- 检查`AscendNPU-IR`产物是否就绪：

```bash
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h" && echo "HIVM.h OK"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/tools/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc" && echo "TableGen inc OK"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/lib/cmake/mlir/MLIRConfig.cmake" && echo "Ascend MLIR CMake package OK"
ls "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/lib"/libMLIRHIVMDialect.so 2>/dev/null || ls "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/lib"/libMLIRHIVMDialect.a 2>/dev/null && echo "HIVM lib OK"
```

#### 暴露 AscendNPU-IR 的 MLIR / LLVM 运行环境

构建和运行 TLA DSL 时，应使用 AscendNPU-IR 构建出的 MLIR Python 包与动态库，不要使用 conda 的 MLIR binding：

```bash
export MLIR_TBLGEN_INCLUDE_DIR="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/include"
export PYTHONPATH="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/python_packages/mlir_core:${PYTHONPATH:-}"
```

- `./build.sh` 会在配置 CMake 前调用 `tools/generate_tla_python_bindings.py`，脚本会使用`$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/bin/mlir-tblgen`，根据 `csrc/mlir/include/Dialect/Tla/IR/Tla.td` 重新生成 `catlass/_mlir_bindings/tla_ops_gen.py`，避免手动修改生成文件后与 TD 定义不一致。

### 2.5 配置并编译 `tla`

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

### 2.7 运行测试

#### IR测试（pytest）

该测试用于测试tla前端代码是否正确转换为TlaIR。

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python -m pytest -q tests
```

#### lit测试

该测试用于测试TlaIR是否正确降级到预期的NPUIR。
`lit` 测试使用环境中的 **`lit`/`llvm-lit`** 可执行文件：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
llvm-lit -sv csrc/mlir/build/tests/lit
```

#### 端到端测试

该测试是运行几个使用Tla DSL编写的完整算子的端到端验证。

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

#### 一键端到端回归：`tests/run_dsl_test.sh`

```bash
bash tests/run_dsl_test.sh --device 0
```

脚本会激活 conda、source CANN、导出 AscendNPU-IR 环境，**强制执行** `./build.sh` 后再跑上板用例。

需要设置的环境变量：

- `ASCEND_HOME_PATH`（指向 CANN toolkit 根目录）。在运行前执行 `source /path/to/ascend-toolkit/set_env.sh`，会**自动设置** `ASCEND_HOME_PATH`
- `TLA_DSL_PREBUILT_ASCENDNPU_IR`（已构建的 AscendNPU-IR 根目录）

可选：

- `DEVICE_ID`（默认 `1`，也可用 `--device`）
- `CONDA_ENV`（默认 `ascend-catlass-dsl`）

```bash
source /path/to/ascend-toolkit/set_env.sh
export TLA_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR
bash tests/run_dsl_test.sh --device 0
```

### 2.8 构建 API 文档（可选）

TLA DSL API 文档位于 `python/tla_dsl/docs/`，由脚本动态解析 `catlass.core_api` 生成 Markdown，再通过 MkDocs 构建为带左侧导航栏的静态 HTML。

#### 2.8.1 安装文档构建依赖

若已经按 **§2.2** 激活环境，可直接安装项目提供的 `docs` 可选依赖：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
pip install -e ".[docs]"
```

#### 2.8.2 生成 API Reference Markdown

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python3 tools/generate_api_reference.py
```

生成结果：

```text
python/tla_dsl/docs/api-reference.md
```

#### 2.8.3 构建带左侧导航栏的静态 HTML

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python3 -m mkdocs build --strict
```

构建结果：

```text
python/tla_dsl/site/index.html
```

打开 `site/index.html` 即可在浏览器查看静态文档站。`site/` 是构建产物，已通过仓库 `.gitignore` 忽略。

#### 2.8.4 本地实时预览

开发文档时可以启动本地服务：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
python3 -m mkdocs serve
```

根据终端输出打开对应地址，通常为：

```text
http://127.0.0.1:8000/
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
| 一键上板端到端回归（`tests/run_dsl_test.sh`） | 上文 **2.7**「一键端到端回归」 |
| API 文档生成与 MkDocs 静态站点构建 | 上文 **2.8** |
