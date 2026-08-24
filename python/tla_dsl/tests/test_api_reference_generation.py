from __future__ import annotations

import difflib
import pathlib
import sys


def test_generated_api_reference_is_current() -> None:
    """Fail when docs/en/api/kernel_api_reference.md is stale relative to core_api.py / generator."""
    package_root = pathlib.Path(__file__).resolve().parents[1]
    doc = package_root / "docs" / "en" / "api" / "kernel_api_reference.md"
    assert doc.is_file(), f"missing checked-in API reference: {doc}"

    sys.path.insert(0, str(package_root / "tools"))
    from generate_api_reference import generate  # noqa: E402

    expected = generate(docs_dir=doc.parent)
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
        "docs/en/api/kernel_api_reference.md is out of date. Regenerate with:\n"
        "  cd python/tla_dsl && python3 tools/generate_api_reference.py\n"
        f"{diff}"
    )
