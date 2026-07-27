# TLA DSL

TLA DSL 是 CATLASS 的 Python 前端。它在 AscendNPU-IR 的基础上构建 TLA Dialect，将 Python 中描述的 TLA 操作降级为 MLIR，并由 AscendNPU-IR 工具链编译为可执行的核函数产物。

## 文档

- [环境准备](docs/dev_guide/00_environment_setup.md)：Docker 开发环境、手动依赖安装和 CANN 初始化。
- [编译与测试](docs/dev_guide/01_build_and_test.md)：构建、pytest、lit 和端到端用例。
- [API 文档](docs/dev_guide/02_api_docs.md)：生成并预览 API 文档。
- [AscendNPU-IR 构建](docs/dev_guide/advanced/ascend_npu_ir.md)：手动构建子模块。
- [手动 CMake 构建](docs/dev_guide/advanced/manual_cmake_build.md)：直接配置 `csrc/mlir` 的进阶用法。

## 快速开始

推荐先构建开发环境镜像。镜像只包含工具链和依赖；启动容器时将完整 CATLASS 仓库挂载到容器中。

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
bash build_docker_image.sh cann:9.1.0-beta.3-950-ubuntu22.04-py3.12
```

具体的容器启动命令、设备挂载和手动环境准备方式见[环境准备](docs/dev_guide/00_environment_setup.md)。

完成依赖准备并构建 AscendNPU-IR 后，执行：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl"
export TLA_DSL_PREBUILT_ASCENDNPU_IR="${CATLASS_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
./build.sh
python -m pytest -q tests
```

## 兼容性

- 支持产品：Ascend950PR、Ascend950DT。
- 操作系统：CANN 支持的 Linux 发行版。
- CANN Toolkit：`>=9.1.0`。
- Python：`>=3.10,<3.14`。
- 手动构建 AscendNPU-IR 时需要 Clang/Clang++ `>=10`；如环境允许，建议使用 Clang 19。

## 目录概览

```text
python/tla_dsl/
├── 3rdparty/AscendNPU-IR/  # AscendNPU-IR 子模块
├── catlass/                # Python 前端与运行时
├── csrc/mlir/              # TLA Dialect 与编译器实现
├── docs/                   # 开发与 API 文档
├── examples/               # 端到端示例
├── tests/                  # 单元、lit 与端到端测试
├── Dockerfile              # 开发环境镜像定义
├── build_docker_image.sh   # 镜像构建入口
├── build.sh                # 项目构建入口
└── requirements.txt        # 非 CANN Python 依赖
```
