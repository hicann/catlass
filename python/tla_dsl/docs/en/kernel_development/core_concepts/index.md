---
nav_order: 10
---

# Core Concepts

The core concepts and programming model of CATLASS DSL, including DSL syntax constraints, control flow, Layout, Tensor integration,
and Host-side compile and launch.

| Document | Scope |
|------|------|
| [DSL Syntax Constraints](syntax_guide.md) | Python constructs and restrictions allowed inside `@tla.kernel`. |
| [DSL Control Flow](control_flow.md) | The boundary between Python staging and runtime control flow. |
| [DSL Layout](layout.md) | The meaning of static / dynamic layout and kernel-side programming. |
| [DSL Tensor Integration](tensor_binding.md) | Host-side `from_dlpack` / `make_fake_tensor`. |
| [Compile and Launch](compile_and_launch.md) | Compile samples, caching, repeated launch, and stream. |
| [Structured Kernel Arguments](structured_arguments.md) | How to pass tuple/list, structured objects, static fields, and custom adapters. |
| [Host API Reference](../../api/host_api_reference.md) | Host-side `@tla.kernel`, `tla.compile` / launch, Host tensor. |
| [DSL Environment Variables](env_vars.md) | `CATLASS_DSL_*` and related environment variables. |
