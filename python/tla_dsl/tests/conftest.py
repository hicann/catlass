from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _load_pretest_bootstrap(repo_root: Path):  # type: ignore[no-untyped-def]
    module_path = repo_root / "tests" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("catlass_test_bootstrap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load pretest bootstrap module from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = _load_pretest_bootstrap(repo_root)
    bootstrap.ensure_pretest_mlir_build(repo_root)


def _compiler_tlair(kernel: Any, *, type_args: tuple[Any, ...] | None = None) -> str:
    """Return the generic Tla IR text consumed by the runtime compiler path."""
    from catlass.base_dsl import BaseDSL

    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=type_args,
        location=kernel.decorator_location,
    )
    return lowered.asm(generic=True)


@pytest.fixture
def compiler_tlair() -> Any:
    return _compiler_tlair
