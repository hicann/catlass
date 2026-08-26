---
nav_order: 10
---

# 核心概念

CATLASS DSL 的核心概念与编程模型，包括 DSL 语法约束、控制流、Layout 与 Tensor 接入。

| 文档 | 范围 |
|------|------|
| [DSL 语法约束](syntax_guide.md) | `@tla.kernel` 内允许的 Python 写法与限制。 |
| [DSL 控制流](control_flow.md) | Python staging 与运行时控制流的边界。 |
| [DSL Layout](layout.md) | 静态 / 动态 layout 的含义与 kernel 侧编程。 |
| [DSL Tensor 接入](tensor_binding.md) | Host 侧 `from_dlpack` / `make_fake_tensor`。 |
| [Host API 参考](../../api/host_api_reference.md) | Host 侧 `@tla.kernel`、`tla.compile` / 启动、Host tensor。 |
| [DSL 环境变量](env_vars.md) | `CATLASS_DSL_*` 及相关环境变量。 |
