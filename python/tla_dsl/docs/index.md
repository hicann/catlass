# TLA DSL Docs

This documentation set is a lightweight MkDocs site for the TLA DSL
(`python/tla_dsl`). Prefer the MkDocs site for reading; checked-in Markdown is
the source.

## Document map

| Doc | Scope |
|-----|--------|
| [Kernel API Reference](kernel-api-reference.md) / [中文](kernel-api-reference.zh.md) | Kernel-side Core APIs (`tla.copy`, `tla.mmad`, vector ops, sync, …). Host launch is out of scope. |
| [Host Tensor Binding](framework_integration.md) | Host `from_dlpack` / `make_fake_tensor` for `tla.compile` / launch. |
| [Dynamic Layout](dsl_dynamic_layout.md) | Marking Host tensors dynamic; using dynamic layout in kernels. |
| [DSL Syntax Constraints](dsl_python_syntax_guide.md) / [English](dsl_python_syntax_guide_en.md) | What Python is legal inside `@tla.kernel`. |
| [Environment Variables](environment_variables.md) | `CATLASS_DSL_*` and related env knobs. |
| [Building API docs](dev_guide/02_api_docs.md) | How to regenerate `kernel-api-reference.md` from English docstrings. |

English Kernel API Markdown is **generated** (`python tools/generate_api_reference.py`);
the Chinese Kernel API page is **hand-maintained** and should be synced when the
English page changes.

## Local Preview

From `python/tla_dsl`:

```bash
python3 -m mkdocs serve
```

To build static HTML:

```bash
python3 -m mkdocs build
```

Then open `site/index.html` in a browser.
