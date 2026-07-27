# 环境准备

本文档介绍 CATLASS/TLA DSL 开发环境的准备方式。若无特殊说明，以下命令均在 DSL 子项目根目录（`${CATLASS_ROOT}/python/tla_dsl`）执行。

## 依赖列表

### 必需基础环境

支持的昇腾产品：Ascend950PR、Ascend950DT。

| 组件 | 版本或要求 |
| --- | --- |
| 操作系统 | CANN 支持的 Linux |
| CANN Toolkit | `>=9.1.0` |
| Python | `>=3.10,<3.14` |
| CMake | `>=3.28,<4.0` |
| Ninja | `>=1.12` |
| Clang / Clang++ | 构建 AscendNPU-IR 时 `>=10`；有条件时建议 `19` |
| lld | 与所用 Clang 配套 |
| lit、FileCheck | 与所用 LLVM 配套 |
| AscendNPU-IR | `feature/regbase@a07821269ede7a5e683ac02c8a2d291608083741` |

### Python 环境

| 用途 | 依赖 |
| --- | --- |
| 运行时 | `numpy` |
| 构建 | `setuptools`、`wheel`、`setuptools-scm`、`pybind11` |
| 开发与测试 | `pytest`、`ruff`、`black`、`mypy` |
| 文档 | `mkdocs` |
| 上板运行 | `torch`、`torch-npu`（与当前 CANN 和驱动匹配） |

## 依赖安装

### Docker 一键部署（推荐）

TLA DSL 的依赖准备步骤较多，推荐使用 Docker 镜像统一准备开发环境。

仓库提供 `Dockerfile` 和配套构建脚本。构建出的镜像包含 `clang-19`、`cmake`、`ninja` 和 `AscendNPU-IR` 等依赖。

`build_docker_image.sh` 用于调用 `docker build`，并支持按网络环境替换软件源。

镜像构建依赖基于 Debian 的 CANN 基础镜像。推荐使用 [AscendHub 上的 CANN 镜像](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)，以复用已安装的 CANN。确认基础镜像 tag 后，将其作为构建脚本的第一个参数传入：

```bash
bash build_docker_image.sh cann:9.1.0-beta.3-950-ubuntu22.04-py3.12
```

输出镜像名称为 `ascend-catlass-dsl`，tag 与基础镜像相同。可使用如下命令启动开发容器：

```bash
docker run \
    --rm \
    --name ascend-catlass-dsl-dev \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${CATLASS_ROOT}:/workspace/catlass" \
    -it ascend-catlass-dsl:9.1.0-beta.3-950-ubuntu22.04-py3.12 bash
```

镜像不包含 CATLASS 源码；上述命令会将完整 CATLASS 仓库挂载到 `/workspace/catlass`，并默认进入 `/workspace/catlass/python/tla_dsl`。镜像已配置 `TLA_DSL_PREBUILT_ASCENDNPU_IR`、`MLIR_TBLGEN_INCLUDE_DIR` 等构建所需环境变量，可直接编译。

### 手动部署

若不使用 Docker，可按以下步骤手动配置依赖。

另外，若您使用conda，则可使用代码仓中的配置文件，一次性安装除AscendNPU-IR之外的所有依赖：

```bash
conda env create -f environment.yml
conda activate ascend-catlass-dsl
```

#### CANN

参考[CANN安装部署](https://www.hiascend.com/cann/download)，安装CANN。

#### clang

TLA DSL的构建不强制要求使用clang。若您需要手动构建AscendNPU-IR，则需要安装clang。AscendNPU-IR的构建只需满足clang>=10，但建议在条件允许的情况下，使用clang 19，以便在生态上更加亲和。

#### CMake 与 Ninja

CMake 和 Ninja 是构建 TLA DSL 的基础工具，已包含在下文的 `requirements.txt` 中。

#### AscendNPU-IR

AscendNPU-IR 的编译安装见[AscendNPU-IR 文档](advanced/ascend_npu_ir.md)。

### Python 依赖

安装开发与测试依赖：

```bash
python -m pip install -r requirements.txt
```

部分 CANN Python 组件依赖额外的 Python 包。需要时可安装 `cann` 依赖组：

```bash
python -m pip install --group cann
```

## 初始化 CANN

在构建或运行 TLA DSL 前，加载实际安装的 CANN 环境：

```bash
source /path/to/cann/set_env.sh
test -n "${ASCEND_HOME_PATH}"
```

## 可选的上板运行依赖

编译和前端测试不要求 `torch` 或 `torch-npu`。运行 NPU 端到端示例时，需要安装与当前 CANN 和驱动匹配的版本。版本对应关系见
[PyTorch Ascend 安装部署](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)。
