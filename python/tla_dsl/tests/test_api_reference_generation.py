from __future__ import annotations

import difflib
import importlib
import pathlib
import sys


def _assert_generated_doc_is_current(
    relative: str, module_name: str, generate_fn_name: str, regen_hint: str
) -> None:
    package_root = pathlib.Path(__file__).resolve().parents[1]
    doc = package_root / relative
    assert doc.is_file(), f"missing checked-in API reference: {doc}"

    sys.path.insert(0, str(package_root / "tools"))
    gen = importlib.import_module(module_name)

    expected = getattr(gen, generate_fn_name)(docs_dir=doc.parent)
    existing = doc.read_text(encoding="utf-8")
    if existing == expected:
        return

    diff = "\n".join(
        difflib.unified_diff(
            existing.splitlines(),
            expected.splitlines(),
            fromfile=str(doc),
            tofile="generated",
            lineterm="",
        )
    )
    raise AssertionError(
        f"{relative} is out of date. Regenerate with:\n"
        f"  cd python/tla_dsl && {regen_hint}\n"
        f"{diff}"
    )


def test_generated_api_reference_is_current() -> None:
    """Fail when docs/en/api/kernel_api_reference.md is stale."""
    _assert_generated_doc_is_current(
        "docs/en/api/kernel_api_reference.md",
        "generate_kernel_api_reference",
        "generate",
        "python3 tools/generate_kernel_api_reference.py",
    )


def test_generated_host_api_reference_is_current() -> None:
    """Fail when docs/en/api/host_api_reference.md is stale."""
    _assert_generated_doc_is_current(
        "docs/en/api/host_api_reference.md",
        "generate_host_api_reference",
        "generate",
        "python3 tools/generate_host_api_reference.py",
    )
