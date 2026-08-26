---
nav_order: 0
---

# TLA DSL Docs

Documentation for CATLASS DSL (`python/tla_dsl`) lives in this tree and is
rendered as part of the repository-wide documentation site (the DSL tab).
Chinese is the default locale; this English tree currently covers the pages
that have been translated.

## Document map

| Doc | Scope |
|-----|--------|
| [Kernel API Reference](api/kernel_api_reference.md) | Kernel-side Core APIs (`tla.copy`, `tla.mmad`, vector ops, sync, …). |
| [Host API Reference](api/host_api_reference.md) | Host-side `@tla.kernel`, `tla.compile` / launch, Host tensors. |
| [DSL Syntax Constraints](core_concepts/syntax_guide.md) | What Python is legal inside `@tla.kernel`. |

English Kernel / Host API Markdown is **generated**
(`python tools/generate_kernel_api_reference.py`,
`python tools/generate_host_api_reference.py`);
the Chinese pages are **hand-maintained** and should be synced when the
English pages change.
