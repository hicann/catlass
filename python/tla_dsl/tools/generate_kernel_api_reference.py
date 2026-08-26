#!/usr/bin/env python3
"""Generate TLA DSL Kernel API reference docs from ``core_api.py`` (AST only).

API docstrings carry ``Directory:`` plus Description / Parameters / Constraints /
Example. Section order and blurbs live in ``DIRECTORY_SECTIONS`` below.
Writes English Markdown to ``docs/en/api/kernel_api_reference.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from common import (
    APIEntry,
    PACKAGE_ROOT,
    const_str,
    directory_path,
    function_entry,
    has_dsl_user_op,
    parse_all_names,
    render_reference,
    run_cli,
)


CORE_API_PATH = PACKAGE_ROOT / "catlass" / "core_api.py"
TENSOR_API_PATH = PACKAGE_ROOT / "catlass" / "tla" / "tensor.py"
OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "api" / "kernel_api_reference.md"
GENERATED_BY = "python/tla_dsl/tools/generate_kernel_api_reference.py"

DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Basic Data Types and Operations",
        "Construction and views for front-end structured values such as "
        "Shape / Coord / Stride / Layout / Tensor, plus pointer helpers.",
    ),
    (
        "Data Movement",
        "Tensor copies between on-chip and global memory, and UB register load/store.",
    ),
    ("Matrix Compute", "Cube-side matrix multiply-accumulate (`tla.mmad`)."),
    (
        "Vector Compute",
        "Compute and mask ops on the register-vector path; usually must be "
        "called inside `tla.vec.func()`.",
    ),
    ("Vector Compute / Mask Compute", "Mask creation and tail-mask updates."),
    (
        "Vector Compute / Basic Arithmetic",
        "Element-wise arithmetic and unary math ops. `VectorSSA` overloads "
        "`+` / `-` / `*` / `/` for `add` / `sub` / `mul` / `div` when no "
        "`mask=` is needed.",
    ),
    ("Vector Compute / Logical Compute", "Bitwise and logical ops on Mask / Vector."),
    (
        "Vector Compute / Compare and Select",
        "Vector compares that produce masks, and masked select.",
    ),
    (
        "Vector Compute / Data Fill",
        "Constant fill and lane-index sequence construction.",
    ),
    (
        "Vector Compute / Discrete and Aggregate",
        "Gather elements from a UB tensor by index.",
    ),
    (
        "Vector Compute / Data Rearrange",
        "Interleave / deinterleave and related lane reshuffles.",
    ),
    ("Vector Compute / Data Compress", "Compress valid lanes under a mask."),
    (
        "Sync Control",
        "In-core / cross-core flags, pipe barriers, mutexes, and local-memory "
        "barriers.",
    ),
    (
        "System Variable Access",
        "Architecture attributes on `tla.arch` (layout tags, pipe identifiers, "
        "block helpers, etc.).",
    ),
    ("Resource Management", "On-chip scratch allocation via `allocate`."),
    (
        "Debug APIs",
        "In-kernel scalar / tensor debug printing.",
    ),
    (
        "Scopes and Control Flow",
        "Cube / Vector / `vec.func` regions and kernel-side loop ranges.",
    ),
]
DIRECTORY_ORDER = [path for path, _ in DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(DIRECTORY_SECTIONS)


def _unary_op_template(tree: ast.Module) -> APIEntry | None:
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_make_unary_op":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "_unary"
                and has_dsl_user_op(child.decorator_list)
            ):
                return function_entry("_unary", child)
    return None


def _doc_from_make_unary_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "doc":
            return const_str(kw.value) or ""
    if len(call.args) >= 2:
        return const_str(call.args[1]) or ""
    return ""


def _collect_core_functions(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_vec_func" and has_dsl_user_op(node.decorator_list):
            entry = function_entry("vec.func", node)
            entry.qualified_name = "catlass.core_api._vec_func"
            entries["vec.func"] = entry
            continue
        if node.name not in exported:
            continue
        if node.name == "print" or has_dsl_user_op(node.decorator_list):
            entries[node.name] = function_entry(node.name, node)
    return entries


def _collect_unary_aliases(
    tree: ast.Module, exported: set[str], template: APIEntry | None
) -> dict[str, APIEntry]:
    if template is None:
        return {}
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_make_unary_op"
        ):
            continue
        name = target.id
        entries[name] = APIEntry(
            name=name,
            qualified_name=f"catlass.core_api.{name}",
            source_line=node.lineno,
            params=list(template.params),
            returns=template.returns,
            docstring=_doc_from_make_unary_call(value),
        )
    return entries


def _collect_arch_namespace(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    if "arch" not in exported:
        return {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "arch"
            and target.attr == "__doc__"
        ):
            return {
                "arch": APIEntry(
                    name="arch",
                    qualified_name="catlass.core_api.arch",
                    source_line=node.lineno,
                    docstring=const_str(value) or "",
                    is_namespace=True,
                )
            }
    return {
        "arch": APIEntry(
            name="arch",
            qualified_name="catlass.core_api.arch",
            source_line=None,
            docstring="",
            is_namespace=True,
        )
    }


def _collect_tensor_methods(path: Path) -> dict[str, APIEntry]:
    """Document Tensor methods that are ``@dsl_user_op`` and have ``Directory:``."""
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tensor_cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in {"Tensor", "_Tensor"}
        ),
        None,
    )
    if tensor_cls is None:
        return {}

    entries: dict[str, APIEntry] = {}
    for node in tensor_cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not has_dsl_user_op(node.decorator_list):
            continue
        doc = ast.get_docstring(node) or ""
        if directory_path(doc) is None:
            continue
        entries[f"Tensor.{node.name}"] = function_entry(
            f"Tensor.{node.name}",
            node,
            qualified_name=f"catlass.tla.tensor.{tensor_cls.name}.{node.name}",
            source_path=path.resolve(),
            drop_self=True,
        )
    return entries


def parse_core_api(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exported = parse_all_names(tree)
    entries: dict[str, APIEntry] = {}
    entries.update(_collect_core_functions(tree, exported))
    entries.update(_collect_unary_aliases(tree, exported, _unary_op_template(tree)))
    entries.update(_collect_arch_namespace(tree, exported))
    entries.update(_collect_tensor_methods(TENSOR_API_PATH))
    return entries


def generate(*, docs_dir: Path | None = None) -> str:
    if not CORE_API_PATH.is_file():
        raise FileNotFoundError(f"core_api not found: {CORE_API_PATH}")

    docs_dir = docs_dir or OUTPUT_PATH.parent
    return render_reference(
        parse_core_api(CORE_API_PATH),
        docs_dir=docs_dir,
        title="TLA DSL Kernel API Reference",
        intro=[
            "This document describes the **TLA DSL kernel-side Core APIs** "
            "(typically imported as `import catlass.tla as tla`). "
            "It covers data structures, compute / sync helpers, on-chip resources, "
            "and debug printing. Host-side compile / launch / tensor binding are in "
            "`docs/en/api/host_api_reference.md`; environment variables are in "
            "`docs/zh/kernel_development/core_concepts/env_vars.md`.",
            "Interface descriptions and examples come from each op's source docstring "
            "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
            "All APIs must be called inside a `@tla.kernel`-decorated kernel function body.",
        ],
        header_sources=(
            "Do not edit manually. Update docstrings/Examples in catlass/core_api.py",
            "(or the defining module for imported types) instead.",
        ),
        leftovers_title="Other Core APIs",
        leftovers_blurb=(
            "APIs still exported by the current source but not yet filed under "
            "the directory tree above."
        ),
        directory_order=DIRECTORY_ORDER,
        directory_intros=DIRECTORY_INTROS,
        default_source_path=CORE_API_PATH,
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
