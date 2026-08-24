---
nav_order: 0
---

# TLA DSL 文档

## 文档索引

| 文档 | 范围 |
|-----|--------|
| [快速开始](quick_start.md) | 兼容性要求、安装方式与首个示例的最短路径。 |
| [环境准备](dsl_development/build_guide/index.md) | 环境要求、安装方式（Conda / Docker）与最短上手路径。 |
| [编译与测试](dsl_development/build_guide/index.md) | `./build.sh` 构建、pytest、lit 与 NPU 端到端示例。 |
| [Kernel API 参考](api/kernel_api_reference.md) | Kernel 侧 Core API（`tla.copy`、`tla.mmad`、Vector 运算、同步等）。 |
| [Host Tensor 接入](kernel_development/core_concepts/tensor_binding.md) | Host 侧 `from_dlpack` / `make_fake_tensor`，供 `tla.compile` / 启动使用。 |
| [静态与动态 Layout](kernel_development/core_concepts/layout.md) | 静态 / 动态 layout；在 Host tensor 上标记动态；在 kernel 中编程。 |
| [DSL 语法约束](kernel_development/core_concepts/syntax_guide.md) | `@tla.kernel` 内允许的 Python 写法。 |
| [环境变量](kernel_development/core_concepts/env_vars.md) | `CATLASS_DSL_*` 及相关环境变量。 |
| [构建 API 文档](api/generate_api_docs.md) | 如何从英文 docstring 重新生成 `docs/en/api/kernel_api_reference.md`。 |

英文 Kernel API 由脚本生成（`python tools/generate_api_reference.py`）；中文 Kernel API 为手工维护，英文稿变更后请同步更新。
