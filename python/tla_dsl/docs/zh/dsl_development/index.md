---
nav_order: 30
---

# DSL 框架开发指南

本目录面向需要从源码配置构建环境、自行构建并深入开发 CATLASS DSL 的开发者。

> 当前版本暂不提供完整的产品包（一键安装即可运行），以下安装与构建步骤均为从源码的开发者流程。

## 环境搭建

| 文档 | 范围 |
|------|------|
| [Conda 安装](build_guide/conda.md) | 使用仓库 `environment.yml` 创建开发环境（不含 CANN 与 AscendNPU-IR）。 |
| [Docker 安装](build_guide/docker.md) | 使用仓库 `Dockerfile` 构建开发镜像，包含任何开发依赖。 |
| [环境准备](build_guide/index.md) | 环境要求总表、安装方式入口与环境检查。 |

## 构建与测试

| 文档 | 范围 |
|------|------|
| [编译 CATLASS DSL 与运行测试](build_guide/index.md) | `./build.sh` 构建、pytest、lit 与 NPU 端到端示例。 |
| [构建 AscendNPU-IR](build_guide/ascend_npu_ir.md) | 手动构建 DSL 所依赖的 AscendNPU-IR。 |
| [手动 CMake 构建](advanced/manual_cmake_build.md) | 直接配置 `csrc/mlir` 的进阶用法。 |

## 深入开发

| 文档 | 范围 |
|------|------|
| [特性开发](feature_development/index.md) | DSL 后端特性扩展，含 [bc 后端集成](feature_development/bc_backend_integration.md)（以 `tla.copy` 为例）。 |
