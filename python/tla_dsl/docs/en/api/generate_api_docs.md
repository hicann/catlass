---
nav_order: 20
---

# Generating API Documentation

CATLASS DSL API documentation is generated as Markdown by scripts that statically parse the source with AST analysis, and is then built into a static site by the MkDocs configuration at the repository root.

- Kernel API: `core_api.py`, `tla/tensor.py` → `docs/en/api/kernel_api_reference.md`
- Host API: Host source files → `docs/en/api/host_api_reference.md`

The shared parsing / Markdown rendering lives in `tools/common.py`.

## Prerequisites

The generation scripts only perform AST parsing. Install the documentation dependencies before building the MkDocs site:

```bash
# Replace /path/to/catlass with the actual path of the CATLASS repository you cloned
cd /path/to/catlass/python/tla_dsl
python -m pip install -r requirements-docs.txt
```

## Generating the API Reference Markdown

Source docstrings use English paragraph headings with directory tags:

- `Directory:`: the path of the API in the reference document TOC (nesting can be expressed with `/`)
- `Description:` / `Parameters:` / `Constraints:` / `Example:`

The generator builds the directory tree from `Directory:`; the section order and section introductions are written in the script's `DIRECTORY_SECTIONS` (Kernel) and `HOST_DIRECTORY_SECTIONS` (Host). Within a section, the API order follows the source definition order.

The generator produces the English reference documents (auto-generated, do not edit manually):

- `docs/en/api/kernel_api_reference.md` (`tools/generate_kernel_api_reference.py`)
- `docs/en/api/host_api_reference.md` (`tools/generate_host_api_reference.py`)

The Chinese versions `docs/zh/api/kernel_api_reference.md` and `docs/zh/api/host_api_reference.md` are **hand-maintained**; after the English draft changes, translate and update the Chinese draft in sync.

Environment variables are described in [Environment Variables](../kernel_development/core_concepts/env_vars.md) and are **not** scanned by the Host API generator.

```bash
cd /path/to/catlass/python/tla_dsl
python tools/generate_kernel_api_reference.py
python tools/generate_host_api_reference.py
```

Check whether the generated files match the current code without rewriting them:

```bash
python tools/generate_kernel_api_reference.py --check
python tools/generate_host_api_reference.py --check
```

When a file is out of date, `--check` prints a diff and returns a non-zero status, which is suitable for a pre-commit check.
