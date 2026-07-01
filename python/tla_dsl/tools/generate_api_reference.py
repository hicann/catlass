#!/usr/bin/env python3
"""Generate the TLA DSL core API reference from live Python objects."""

from __future__ import annotations

import argparse
import difflib
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "python" / "tla_dsl"
OUTPUT_PATH = PACKAGE_ROOT / "docs" / "api-reference.md"
HIDDEN_PARAMETERS = {"loc"}
CATLASS_PACKAGE_DIR = PACKAGE_ROOT / "catlass"
CATLASS_INIT = CATLASS_PACKAGE_DIR / "__init__.py"


@dataclass(frozen=True)
class APIEntry:
    name: str
    qualified_name: str
    source_path: Path | None
    source_line: int | None
    signature: str
    summary: str

    @property
    def source_link(self) -> str | None:
        if self.source_path is None or self.source_line is None:
            return None
        rel = os.path.relpath(self.source_path, OUTPUT_PATH.parent).replace(os.sep, "/")
        return f"{rel}#L{self.source_line}"


def import_core_api() -> Any:
    existing = sys.modules.get("catlass")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != CATLASS_INIT:
            raise RuntimeError(
                "Refusing to generate docs with a preloaded catlass module from "
                f"{existing_file!r}; expected {CATLASS_INIT}."
            )
    else:
        spec = importlib.util.spec_from_file_location(
            "catlass",
            CATLASS_INIT,
            submodule_search_locations=[str(CATLASS_PACKAGE_DIR)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load package spec for {CATLASS_INIT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["catlass"] = module
        spec.loader.exec_module(module)

    return importlib.import_module("catlass.core_api")


def public_core_op_names(core_api: Any) -> list[str]:
    names: list[str] = []
    for name in getattr(core_api, "__all__", []):
        obj = getattr(core_api, name, None)
        if callable(obj) and getattr(obj, "__wrapped__", None) is not None:
            names.append(name)
    return names


def support_names(core_api: Any) -> list[str]:
    candidates = ["arch", "LocalmemAllocator", "TlaCoreAPIError"]
    return [name for name in candidates if hasattr(core_api, name)]


def source_location(obj: Any) -> tuple[Path | None, int | None]:
    target = inspect.unwrap(obj) if callable(obj) else obj
    try:
        source_file = inspect.getsourcefile(target)
        _, line = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None, None
    if source_file is None:
        return None, None
    return Path(source_file).resolve(), line


def filtered_signature(obj: Any) -> str:
    if inspect.isclass(obj):
        bases = [
            base.__name__
            for base in obj.__bases__
            if base is not object and base.__name__ != "ABC"
        ]
        return f"class {obj.__name__}({', '.join(bases)})" if bases else f"class {obj.__name__}"

    if not callable(obj):
        return f"{type(obj).__name__} object"

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{getattr(obj, '__name__', type(obj).__name__)}(...)"

    params = [
        param
        for param in sig.parameters.values()
        if param.name not in HIDDEN_PARAMETERS
    ]
    sig = sig.replace(parameters=params)
    text = str(sig)
    text = text.replace("'", "")
    return f"{getattr(obj, '__name__', type(obj).__name__)}{text}"


def first_paragraph(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc and callable(obj):
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            doc = inspect.getdoc(wrapped) or ""
    if not doc:
        return "> TODO: 补充 API 说明。该占位由生成器发现 docstring 缺失后自动生成。"
    return dedent(doc).strip().split("\n\n")[0].replace("\n", " ")


def namespace_summary(obj: Any) -> str:
    members = getattr(obj, "_members", None)
    if not isinstance(members, dict):
        return first_paragraph(obj)
    names = ", ".join(f"`{name}`" for name in sorted(members))
    return f"Core namespace exported by `catlass.core_api`; available members: {names}."


def build_entry(name: str, obj: Any) -> APIEntry:
    path, line = source_location(obj)
    module_name = getattr(obj, "__module__", "catlass.core_api")
    object_name = getattr(obj, "__name__", name)
    if module_name == "catlass.core_api" or inspect.isclass(obj):
        qualified_name = f"{module_name}.{object_name}"
    else:
        qualified_name = f"catlass.core_api.{name}"
    summary = namespace_summary(obj) if name == "arch" else first_paragraph(obj)
    return APIEntry(
        name=name,
        qualified_name=qualified_name,
        source_path=path,
        source_line=line,
        signature=filtered_signature(obj),
        summary=summary,
    )


def render_entry(entry: APIEntry) -> str:
    lines = [f"### `{entry.name}`", ""]
    if entry.source_link is not None:
        lines.extend([f"Source: [`{entry.qualified_name}`]({entry.source_link})", ""])
    else:
        lines.extend([f"Source: `{entry.qualified_name}`", ""])
    lines.extend(["```python", entry.signature, "```", "", entry.summary, ""])
    return "\n".join(lines)


def render_section(title: str, entries: list[APIEntry]) -> str:
    lines = [f"## {title}", ""]
    for entry in entries:
        lines.append(render_entry(entry))
    return "\n".join(lines).rstrip() + "\n"


def generate() -> str:
    core_api = import_core_api()
    core_entries = [
        build_entry(name, getattr(core_api, name))
        for name in public_core_op_names(core_api)
    ]
    core_entries.sort(key=lambda entry: entry.source_line or 0)
    support_entries = [
        build_entry(name, getattr(core_api, name))
        for name in support_names(core_api)
    ]

    header = [
        "<!--",
        "This file is generated by python/tla_dsl/tools/generate_api_reference.py.",
        "Do not edit manually. Update catlass.core_api public exports, type annotations, or docstrings instead.",
        "-->",
        "",
        "# TLA DSL Core API Reference",
        "",
        "本文档由 `catlass.core_api` 的运行时对象动态生成，当前只覆盖 Core API，暂不包含 launch / runtime / execution 等接口。",
        "",
        "生成依据包括 `catlass.core_api.__all__`、`@dsl_user_op` 包装关系、`inspect.signature`、docstring 和源码位置。",
        "",
    ]
    sections = [
        render_section("Core DSL Operations", core_entries),
        render_section("Core Support Objects", support_entries),
    ]
    return "\n".join(header + sections).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"output Markdown path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    content = generate()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if args.check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != content:
            print(
                "\n".join(
                    difflib.unified_diff(
                        existing.splitlines(),
                        content.splitlines(),
                        fromfile=str(output),
                        tofile="generated",
                        lineterm="",
                    )
                )
            )
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"generated {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
