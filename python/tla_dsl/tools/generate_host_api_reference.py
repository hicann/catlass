#!/usr/bin/env python3
"""Generate TLA DSL Host API reference docs from Host sources (AST only).

API docstrings carry ``Directory:`` plus Description / Parameters / Constraints /
Example. Environment variables are documented separately in
``docs/zh/kernel_development/core_concepts/env_vars.md`` (not scanned here).
Writes English Markdown to ``docs/en/api/host_api_reference.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from common import (
    APIEntry,
    PACKAGE_ROOT,
    ParamInfo,
    directory_path,
    function_entry,
    render_reference,
    run_cli,
)


OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "api" / "host_api_reference.md"
GENERATED_BY = "python/tla_dsl/tools/generate_host_api_reference.py"
DEFAULT_SOURCE_PATH = PACKAGE_ROOT / "catlass" / "dsl.py"

# ``@tla.jit`` is intentionally omitted: helper semantics are not frozen.
# Do not add a ``Directory:`` docstring on ``catlass.dsl.jit`` until they are.
HOST_DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Decorators",
        "Host-side `@tla.kernel` entry, plus Host `@dataclass` packing. "
        "The decorated kernel body is not executed on the Host.",
    ),
    (
        "Compile and Launch",
        "Compile a decorated kernel and launch it on the NPU. Use "
        "`tla.compile` to obtain a callable `JitCompiledFunction`; call it "
        "directly to lazily create and then reuse its executor. Cache / arch / IR-dump "
        "knobs that are not function arguments are in "
        "`docs/zh/kernel_development/core_concepts/env_vars.md`.",
    ),
    (
        "Compile and Launch / Compile",
        "Build a device binary. Primary entry: `tla.compile`. "
        "`TlaJitFunction.compile` is a lower-level helper on the decorated "
        "function.",
    ),
    (
        "Compile and Launch / Launch",
        "Run a compiled kernel on the NPU by calling the `JitCompiledFunction` "
        "returned by `tla.compile`.",
    ),
    (
        "Compile and Launch / Inspect",
        "Dump frontend TLA IR without building a device binary or launching. "
        "See `TlaJitFunction.dump_mlir`.",
    ),
    (
        "Host Tensor",
        "Build Host `tla.Tensor` objects and mark layout extents dynamic so one "
        "artifact can run at different shapes. See also "
        "`docs/zh/kernel_development/core_concepts/layout.md`.",
    ),
    (
        "Host Tensor / Binding",
        "Bind a real NPU buffer with `from_dlpack`, or a metadata-only sample "
        "with `make_fake_tensor`.",
    ),
    (
        "Host Tensor / Dynamic Layout",
        "Mark static layout extents dynamic. See also "
        "`docs/zh/kernel_development/core_concepts/layout.md`.",
    ),
]
DIRECTORY_ORDER = [path for path, _ in HOST_DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(HOST_DIRECTORY_SECTIONS)

HOST_SOURCE_PATHS = (
    PACKAGE_ROOT / "catlass" / "dsl.py",
    PACKAGE_ROOT / "catlass" / "base_dsl" / "compiler.py",
    PACKAGE_ROOT / "catlass" / "base_dsl" / "jit_executor.py",
    PACKAGE_ROOT / "catlass" / "execution_lowering.py",
    PACKAGE_ROOT / "catlass" / "tla" / "runtime.py",
)

# Display names in the generated reference (source qualified names stay unchanged).
HOST_DISPLAY_NAMES = {
    "CompileCallable.__call__": "compile",
    "JitCompiledFunction.__call__": "JitCompiledFunction.__call__",
    "_Tensor.mark_layout_dynamic": "Tensor.mark_layout_dynamic",
    "_Tensor.mark_compact_shape_dynamic": "Tensor.mark_compact_shape_dynamic",
    # Stdlib ``@dataclass`` packing rules live on the struct-arg validator.
    "_validate_dataclass_kernel_arg": "dataclass",
}

# Signature / qualified-name overrides when the documented public face differs
# from the source helper (e.g. stdlib ``@dataclass``).
HOST_ENTRY_OVERRIDES: dict[str, dict[str, object]] = {
    "_validate_dataclass_kernel_arg": {
        "qualified_name": "dataclasses.dataclass",
        "is_class": False,
        "params": [
            ParamInfo("cls", "type", "positional", None),
            ParamInfo("frozen", "bool", "keyword_only", "False"),
            ParamInfo("kw_only", "bool", "keyword_only", "False"),
        ],
        "returns": "type",
    },
}


def _should_collect(name: str) -> bool:
    """Public symbols, or private helpers remapped into the Host reference."""
    return (not name.startswith("_")) or name in HOST_DISPLAY_NAMES


def _apply_overrides(source_name: str, entry: APIEntry) -> APIEntry:
    meta = HOST_ENTRY_OVERRIDES.get(source_name)
    if not meta:
        return entry
    return APIEntry(
        name=entry.name,
        qualified_name=str(meta.get("qualified_name", entry.qualified_name)),
        source_line=entry.source_line,
        docstring=entry.docstring,
        source_path=entry.source_path,
        is_class=bool(meta.get("is_class", entry.is_class)),
        params=list(meta.get("params", entry.params)),  # type: ignore[arg-type]
        returns=str(meta.get("returns", entry.returns)),
    )


def _module_qualname(path: Path) -> str:
    path = path.resolve()
    catlass_root = (PACKAGE_ROOT / "catlass").resolve()
    try:
        rel = path.relative_to(catlass_root)
        return "catlass." + ".".join(rel.with_suffix("").parts)
    except ValueError:
        rel = path.relative_to(PACKAGE_ROOT.resolve())
        return ".".join(rel.with_suffix("").parts)


def _collect_host_file(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mod = _module_qualname(path)
    src = path.resolve()
    entries: dict[str, APIEntry] = {}

    def maybe_add(source_name: str, entry: APIEntry) -> None:
        entry = _apply_overrides(source_name, entry)
        display = HOST_DISPLAY_NAMES.get(source_name, source_name)
        entry.name = display
        entries[display] = entry

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not _should_collect(node.name):
                continue
            doc = ast.get_docstring(node) or ""
            if directory_path(doc) is None:
                continue
            maybe_add(
                node.name,
                function_entry(
                    node.name,
                    node,
                    qualified_name=f"{mod}.{node.name}",
                    source_path=src,
                ),
            )
        elif isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or ""
            if directory_path(class_doc) is not None and _should_collect(node.name):
                maybe_add(
                    node.name,
                    APIEntry(
                        name=node.name,
                        qualified_name=f"{mod}.{node.name}",
                        source_line=node.lineno,
                        docstring=class_doc,
                        is_class=True,
                        source_path=src,
                    ),
                )
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                raw = f"{node.name}.{child.name}"
                if not _should_collect(child.name) and raw not in HOST_DISPLAY_NAMES:
                    continue
                doc = ast.get_docstring(child) or ""
                if directory_path(doc) is None:
                    continue
                maybe_add(
                    raw,
                    function_entry(
                        raw,
                        child,
                        qualified_name=f"{mod}.{node.name}.{child.name}",
                        source_path=src,
                        drop_self=True,
                    ),
                )
    return entries


def parse_host_apis() -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for path in HOST_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"host API source not found: {path}")
        entries.update(_collect_host_file(path))
    return entries


def generate(*, docs_dir: Path | None = None) -> str:
    docs_dir = docs_dir or OUTPUT_PATH.parent
    return render_reference(
        parse_host_apis(),
        docs_dir=docs_dir,
        title="TLA DSL Host API Reference",
        intro=[
            "This document describes the **TLA DSL Host-side APIs** "
            "(typically imported as `import catlass.tla as tla`). It covers the "
            "`@tla.kernel` decorator, Host `@dataclass` packing, "
            "`tla.compile` / `JitCompiledFunction` launch, and Host tensors. "
            "Environment variables are in "
            "`docs/zh/kernel_development/core_concepts/env_vars.md`. "
            "Kernel-side ops live in `docs/en/api/kernel_api_reference.md`.",
            "Interface descriptions and examples come from each API's source docstring "
            "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
            "These APIs are called from Python Host scripts, **outside** a `@tla.kernel` "
            "function body.",
        ],
        header_sources=(
            "Do not edit manually. Update Host docstrings in catlass/dsl.py,",
            "catlass/base_dsl/compiler.py, catlass/base_dsl/jit_executor.py,",
            "catlass/execution_lowering.py, and catlass/tla/runtime.py.",
        ),
        leftovers_title="Other Host APIs",
        leftovers_blurb=(
            "APIs still collected from Host sources but not yet filed under "
            "the directory tree above."
        ),
        directory_order=DIRECTORY_ORDER,
        directory_intros=DIRECTORY_INTROS,
        default_source_path=DEFAULT_SOURCE_PATH,
        generated_by=GENERATED_BY,
    )


def main() -> int:
    return run_cli(
        description=__doc__,
        default_output=OUTPUT_PATH,
        generate_fn=generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
