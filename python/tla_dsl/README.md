# CATLASS DSL

CATLASS DSL 是 CATLASS 的 Python 前端。它在 AscendNPU-IR 的基础上构建 TLA Dialect，将 Python 中描述的 TLA 操作降级为 MLIR，并由 AscendNPU-IR 工具链编译为可执行的核函数产物。

目前，CATLASS DSL 主要对 TLA 框架提供了DSL封装，可以使用`tla.make_tensor`等API来描述TLA操作，完成算子的实现。未来，我们将继续扩展CATLASS DSL，在全量支持TLA tensor抽象接口、矩阵向量计算等接口的基础之上，进行类似CATLASS 模板的封装，实现对齐CATLASS C++模板算子的开发体验。

## 文档

- [**快速开始**](docs/zh/quick_start.md)：安装与首个示例
- [**Kernel 开发指南**](docs/zh/kernel_development/index.md)：核心概念（语法约束、控制流、Layout、Tensor 接入、编译启动）与调测优化
- [**DSL 框架开发指南**](docs/zh/dsl_development/index.md)：环境搭建、构建测试、特性开发、进阶 CMake 调试
- [**API 文档**](docs/zh/api/index.md)：Kernel / Host API 参考与文档生成

## 快速开始

安装 CATLASS DSL 并运行首个算子示例，请参阅[快速开始](docs/zh/quick_start.md)。

## 兼容性

CATLASS DSL 各版本支持的硬件平台及所需的最低 CANN 版本如下表：

| CATLASS DSL 版本 | 最低支持 CANN 包版本 | 支持昇腾产品 |
| --- | --- | --- |
| 当前 | `>= 9.1.0` | Ascend 950PR / Ascend 950DT |

软硬件环境要求：

- CPU 架构：`aarch64` / `x86_64`
- 系统：CANN 支持的 Linux
- 软件依赖：Python `>= 3.10, < 3.14`、CMake `>= 3.28, < 4.0`、Ninja `>= 1.12`、Clang / Clang++ `>= 10`（构建 AscendNPU-IR 时；推荐 19）、lld、lit、FileCheck 与 LLVM 配套、AscendNPU-IR `feature/regbase@a07821269…`

完整的环境要求与安装方式见[环境准备](docs/zh/dsl_development/build_guide/environment.md)。

## 目录概览

```text
python/tla_dsl/
├── 3rdparty/AscendNPU-IR/  # AscendNPU-IR 子模块
├── catlass/                # Python 前端与运行时
├── csrc/mlir/              # TLA Dialect 与编译器实现
├── docs/                   # 文档
├── examples/               # 端到端示例
├── tests/                  # 单元、lit 与端到端测试
├── Dockerfile              # 开发环境镜像定义
├── build_docker_image.sh   # 镜像构建入口
├── build.sh                # 项目构建入口
└── requirements.txt        # 开发 Python 依赖
```
