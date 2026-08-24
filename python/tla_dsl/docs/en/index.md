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
| [DSL Syntax Constraints](core_concepts/syntax_guide.md) | What Python is legal inside `@tla.kernel`. |

English Kernel API Markdown is **generated** (`python tools/generate_api_reference.py`);
the Chinese Kernel API page is **hand-maintained** and should be synced when the
English page changes.
