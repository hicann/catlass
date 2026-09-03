"""Shared AST parsing and Markdown rendering for TLA DSL API references."""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # python/tla_dsl
HIDDEN_PARAMETERS = frozenset({"loc"})
SECTION_LABELS = ("Description", "Parameters", "Constraints", "Example")
ROOT_TITLE = "API Reference"


@dataclass(frozen=True)
class DocNode:
    title: str
    apis: tuple[str, ...] = ()
    children: tuple["DocNode", ...] = ()
    intro: str = ""


@dataclass
class ParamInfo:
    name: str
    annotation: str
    kind: str  # positional | var_positional | keyword_only | var_keyword
    default: str | None


@dataclass
class APIEntry:
    name: str
    qualified_name: str
    source_line: int | None
    params: list[ParamInfo] = field(default_factory=list)
    returns: str = ""
    docstring: str = ""
    is_class: bool = False
    is_namespace: bool = False
    is_env: bool = False
    # Language-boundary topic with no callable prototype (Host reference only).
    is_concept: bool = False
    source_path: Path | None = None


class MissingAPIDocError(RuntimeError):
    """Public API is missing required English docstring sections."""


def unparse(node: ast.AST | None, *, missing: str = "Any") -> str:
    if node is None:
        return missing
    try:
        return ast.unparse(node).replace("typing.", "")
    except Exception:
        return missing


def const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def decorator_basename(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def has_dsl_user_op(decorators: list[ast.expr]) -> bool:
    return any(decorator_basename(d) == "dsl_user_op" for d in decorators)


def params_from_args(args: ast.arguments) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    pos_args = list(args.posonlyargs) + list(args.args)
    default_offset = len(pos_args) - len(args.defaults)

    for index, arg in enumerate(pos_args):
        default = None
        if index >= default_offset:
            default = unparse(args.defaults[index - default_offset], missing="...")
        params.append(
            ParamInfo(arg.arg, unparse(arg.annotation), "positional", default)
        )

    if args.vararg is not None:
        params.append(
            ParamInfo(
                args.vararg.arg,
                unparse(args.vararg.annotation),
                "var_positional",
                None,
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        default = unparse(default_node, missing="...") if default_node else None
        params.append(
            ParamInfo(arg.arg, unparse(arg.annotation), "keyword_only", default)
        )

    if args.kwarg is not None:
        params.append(
            ParamInfo(
                args.kwarg.arg,
                unparse(args.kwarg.annotation),
                "var_keyword",
                None,
            )
        )

    return [p for p in params if p.name not in HIDDEN_PARAMETERS]


def function_entry(
    name: str,
    node: ast.FunctionDef,
    *,
    qualified_name: str | None = None,
    source_path: Path | None = None,
    drop_self: bool = False,
) -> APIEntry:
    params = params_from_args(node.args)
    if drop_self:
        params = [p for p in params if p.name != "self"]
    return APIEntry(
        name=name,
        qualified_name=qualified_name or f"catlass.core_api.{name}",
        source_line=node.lineno,
        params=params,
        returns=unparse(node.returns, missing="") if node.returns else "",
        docstring=ast.get_docstring(node) or "",
        source_path=source_path,
    )


def parse_all_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


def directory_path(doc: str) -> str | None:
    """Return the ``Directory:`` path, or None if missing."""
    if not doc:
        return None
    match = re.search(r"(?m)^Directory:\s*(.+?)\s*$", dedent(doc).strip())
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split("/") if p.strip()]
    return " / ".join(parts) if parts else None


def docstring_sections(doc: str) -> dict[str, str]:
    """Split a docstring into the four fixed English sections."""
    sections = {label: "" for label in SECTION_LABELS}
    if not doc:
        return sections
    text = re.sub(r"(?m)^\s*Examples\s*:", "Example:", dedent(doc).strip())
    hits = [
        (label, m.start(), m.end())
        for label in SECTION_LABELS
        for m in [re.search(rf"(?m)^[ \t]*{label}[ \t]*:", text)]
        if m
    ]
    hits.sort(key=lambda item: item[1])
    for i, (label, _start, content_start) in enumerate(hits):
        content_end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        body = dedent(text[content_start:content_end]).strip()
        if label == "Example":
            code = re.search(r"```(?:python)?\n(.*?)```", body, flags=re.S)
            sections[label] = dedent(code.group(1)).strip() if code else body
        else:
            sections[label] = body
    return sections


def require_docs(entries: dict[str, APIEntry]) -> None:
    problems: list[str] = []
    for name in sorted(entries):
        entry = entries[name]
        loc = entry.qualified_name + (
            f" (approx. L{entry.source_line})" if entry.source_line else ""
        )
        if directory_path(entry.docstring) is None:
            problems.append(f"- `{name}` @ {loc}: missing `Directory:`")
        sections = docstring_sections(entry.docstring)
        for label in SECTION_LABELS:
            if not sections[label]:
                problems.append(f"- `{name}` @ {loc}: missing or empty `{label}:`")
    if problems:
        raise MissingAPIDocError(
            "Incomplete API docs. Add `Directory:` plus docstring sections "
            "`Description:` / `Parameters:` / `Constraints:` / `Example:` "
            "in source:\n" + "\n".join(problems)
        )


def _path_prefixes(path: str) -> list[str]:
    parts = path.split(" / ")
    return [" / ".join(parts[:i]) for i in range(1, len(parts) + 1)]


def build_doc_tree(
    entries: dict[str, APIEntry],
    *,
    directory_order: list[str],
    directory_intros: dict[str, str],
    default_source_path: Path,
) -> DocNode:
    """Group APIs by ``Directory:`` and order sections via ``directory_order``."""
    leaf_apis: dict[str, list[str]] = {}
    first_line: dict[str, int] = {}
    ranked = {path: idx for idx, path in enumerate(directory_order)}

    pending = sorted(
        (
            (
                str(entry.source_path or default_source_path),
                entry.source_line or 0,
                name,
                directory_path(entry.docstring),
            )
            for name, entry in entries.items()
        )
    )
    for _path, line, name, directory in pending:
        if not directory:
            continue
        leaf_apis.setdefault(directory, []).append(name)
        first_line.setdefault(directory, line)

    path_set: set[str] = set()
    for leaf in leaf_apis:
        path_set.update(_path_prefixes(leaf))
    for path in directory_intros:
        path_set.update(_path_prefixes(path))

    def order_key(path: str) -> tuple[int, int, str]:
        if path in ranked:
            return (0, ranked[path], path)
        prefix = f"{path} / "
        child_ranks = [ranked[p] for p in ranked if p.startswith(prefix)]
        if child_ranks:
            return (0, min(child_ranks), path)
        return (1, first_line.get(path, 10**9), path)

    def build_node(path: str) -> DocNode:
        title = path.rsplit(" / ", 1)[-1]
        prefix = f"{path} / "
        child_titles = sorted(
            {
                candidate[len(prefix) :]
                for candidate in path_set
                if candidate.startswith(prefix)
                and " / " not in candidate[len(prefix) :]
            },
            key=lambda t: order_key(prefix + t),
        )
        children = tuple(build_node(prefix + t) for t in child_titles)
        return DocNode(
            title=title,
            apis=() if children else tuple(leaf_apis.get(path, ())),
            children=children,
            intro=directory_intros.get(path, ""),
        )

    top = sorted({p for p in path_set if " / " not in p}, key=order_key)
    return DocNode(title=ROOT_TITLE, children=tuple(build_node(p) for p in top))


def format_signature(entry: APIEntry, *, default_source_path: Path) -> str:
    if entry.is_env:
        return entry.name
    if entry.is_namespace:
        return f"tla.{entry.name}"
    if entry.is_class:
        return f"class tla.{entry.name}"

    parts: list[str] = []
    saw_kwonly = False
    for param in entry.params:
        if param.kind == "keyword_only" and not saw_kwonly:
            parts.append("*")
            saw_kwonly = True
        if param.kind == "var_positional":
            piece = f"*{param.name}: {param.annotation}"
        elif param.kind == "var_keyword":
            piece = f"**{param.name}: {param.annotation}"
        else:
            piece = f"{param.name}: {param.annotation}"
            if param.default is not None:
                piece += f" = {param.default}"
        parts.append(piece)

    if entry.qualified_name.startswith("dataclasses."):
        callee = entry.qualified_name
    elif entry.name.startswith("Tensor."):
        method = entry.name.split(".", 1)[1]
        src = entry.source_path or default_source_path
        if src.name == "runtime.py" and "tla" in src.parts:
            callee = f"tensor.{method}"
        else:
            callee = f"tile.{method}"
    elif "." in entry.name:
        head, _sep, _rest = entry.name.partition(".")
        # PascalCase head => Class.method (Host JIT types); keep bare.
        # lowercase head => module attr (e.g. vec.func) => call as tla.vec.func.
        if head[:1].isupper():
            callee = entry.name
        else:
            callee = f"tla.{entry.name}"
    else:
        callee = f"tla.{entry.name}"
    sig = f"{callee}({', '.join(parts)})"
    return f"{sig} -> {entry.returns}" if entry.returns else sig


def source_markdown(
    entry: APIEntry, docs_dir: Path, *, default_source_path: Path
) -> str:
    if entry.is_concept:
        return "**Source:** Host language boundary (not a callable API)"
    if entry.source_line is None:
        if entry.is_env:
            return f"**Source:** environment variable `{entry.name}`"
        return f"**Source:** `{entry.qualified_name}`"
    src = entry.source_path or default_source_path
    rel = Path(os.path.relpath(src, docs_dir))
    link = f"[`{entry.qualified_name}`]({rel.as_posix()}#L{entry.source_line})"
    if entry.is_env:
        return f"**Source:** environment variable {link}"
    return f"**Source:** {link}"


def _anchor(title: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\- ]", "", title).strip().replace(" ", "-").lower()


def numbered_title(number: str | None, title: str) -> str:
    if not number:
        return title
    return f"{number} {title}" if "." in number else f"{number}. {title}"


def render_entry(
    entry: APIEntry,
    heading_level: int,
    docs_dir: Path,
    *,
    default_source_path: Path,
) -> str:
    sections = docstring_sections(entry.docstring)
    for label in SECTION_LABELS:
        if not sections[label]:
            raise MissingAPIDocError(
                f"`{entry.name}` missing {label}:; please update {entry.qualified_name}"
            )
    h = "#" * heading_level
    parts = [
        f"{h} `{entry.name}`",
        "",
        source_markdown(entry, docs_dir, default_source_path=default_source_path),
        "",
        "Description:",
        "",
        sections["Description"],
        "",
    ]
    if not entry.is_concept:
        parts.extend(
            [
                "Prototype:",
                "",
                "```python",
                format_signature(entry, default_source_path=default_source_path),
                "```",
                "",
                "Parameters:",
                "",
                sections["Parameters"],
                "",
            ]
        )
    parts.extend(
        [
            "Constraints:",
            "",
            sections["Constraints"],
            "",
            "Example:",
            "",
            "```python",
            sections["Example"],
            "```",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(parts)


def iter_apis(node: DocNode) -> list[str]:
    names = list(node.apis)
    for child in node.children:
        names.extend(iter_apis(child))
    return names


def render_toc(tree: DocNode) -> str:
    lines = ["## Table of Contents", ""]

    def walk(node: DocNode, number: str) -> None:
        heading = numbered_title(number, node.title)
        indent = "  " * number.count(".")
        lines.append(f"{indent}- [{heading}](#{_anchor(heading)})")
        for idx, child in enumerate(node.children, start=1):
            walk(child, f"{number}.{idx}")

    for idx, child in enumerate(tree.children, start=1):
        walk(child, str(idx))
    lines.append("")
    return "\n".join(lines)


def render_doc_node(
    node: DocNode,
    entries: dict[str, APIEntry],
    heading_level: int,
    docs_dir: Path,
    section_number: str,
    *,
    default_source_path: Path,
) -> str:
    parts = [f"{'#' * heading_level} {numbered_title(section_number, node.title)}", ""]
    if node.intro:
        parts.extend([node.intro, ""])
    for name in node.apis:
        if name in entries:
            parts.append(
                render_entry(
                    entries[name],
                    heading_level + 1,
                    docs_dir,
                    default_source_path=default_source_path,
                )
            )
    for idx, child in enumerate(node.children, start=1):
        parts.append(
            render_doc_node(
                child,
                entries,
                heading_level + 1,
                docs_dir,
                f"{section_number}.{idx}",
                default_source_path=default_source_path,
            )
        )
    return "\n".join(parts).rstrip() + "\n"


def render_reference(
    entries: dict[str, APIEntry],
    *,
    docs_dir: Path,
    title: str,
    intro: list[str],
    header_sources: tuple[str, ...],
    leftovers_title: str,
    leftovers_blurb: str,
    directory_order: list[str],
    directory_intros: dict[str, str],
    default_source_path: Path,
    generated_by: str,
) -> str:
    require_docs(entries)
    tree = build_doc_tree(
        entries,
        directory_order=directory_order,
        directory_intros=directory_intros,
        default_source_path=default_source_path,
    )
    used = set(iter_apis(tree))
    leftovers = sorted(
        (e for name, e in entries.items() if name not in used),
        key=lambda e: (str(e.source_path or ""), e.source_line or 0, e.name),
    )

    header = [
        "<!--",
        f"This file is generated by {generated_by}.",
        *header_sources,
        "-->",
        "",
        f"# {title}",
        "",
        *[paragraph + "\n" for paragraph in intro],
        "---",
        "",
        render_toc(tree),
        "---",
        "",
    ]

    body = [
        render_doc_node(
            child,
            entries,
            2,
            docs_dir,
            str(idx),
            default_source_path=default_source_path,
        )
        for idx, child in enumerate(tree.children, start=1)
    ]
    if leftovers:
        body.append(f"## {leftovers_title}\n")
        body.append(leftovers_blurb + "\n")
        body.extend(
            render_entry(e, 3, docs_dir, default_source_path=default_source_path)
            for e in leftovers
        )

    return "\n".join(header + body).rstrip() + "\n"


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"generated {path.relative_to(PACKAGE_ROOT)}")


def check_output(path: Path, content: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing == content:
        print(f"up to date {path.relative_to(PACKAGE_ROOT)}")
        return True
    diff = "\n".join(
        difflib.unified_diff(
            existing.splitlines(),
            content.splitlines(),
            fromfile=str(path),
            tofile="generated",
            lineterm="",
        )
    )
    print(diff)
    print(f"out of date {path.relative_to(PACKAGE_ROOT)}")
    return False


def run_cli(
    *,
    description: str,
    default_output: Path,
    generate_fn: Callable[..., str],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output path (default: {default_output.relative_to(PACKAGE_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if generated Markdown is stale.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PACKAGE_ROOT / args.output
    try:
        content = generate_fn(docs_dir=output.parent)
    except MissingAPIDocError as exc:
        print(exc)
        return 1
    if args.check:
        return 0 if check_output(output, content) else 1
    write_output(output, content)
    return 0
