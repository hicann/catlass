"""Pre-test bootstrap utilities for verifying MLIR build artifacts.

This module does NOT trigger a build — it only checks whether pre-built
binaries exist and warns if they are missing.  Run ``build.sh`` first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class PretestBuildError(RuntimeError):
    """Raised when the pre-test MLIR binary check fails."""


def _resolve_mlir_core_python_path(*, env: Mapping[str, str]) -> str | None:
    """Derive the mlir_core Python package path from ``CATLASS_DSL_PREBUILT_ASCENDNPU_IR``.

    Returns a path suitable for prepending to ``PYTHONPATH``, or ``None``
    if the path does not exist on disk.
    """
    prebuilt = env.get("CATLASS_DSL_PREBUILT_ASCENDNPU_IR")
    if prebuilt:
        mlir_core = (
            Path(prebuilt) / "build" / "install" / "python_packages" / "mlir_core"
        )
        if mlir_core.is_dir():
            return str(mlir_core)
    return None


def _prebuilt_binaries_exist(build_dir: Path) -> bool:
    type_bridge_glob = list(
        (build_dir / "python" / "catlass").glob("_tla_type_bridge_native*.so")
    )
    tla_compile = build_dir / "tools" / "tla-compile" / "TlaCompile"
    return len(type_bridge_glob) > 0 and tla_compile.is_file()


def ensure_pretest_mlir_build(repo_root: Path) -> None:
    """Verify that pre-built MLIR build artifacts are available for tests.

    This function does **not** trigger a build.  If binaries are missing
    it raises ``PretestBuildError`` with instructions to run ``build.sh``.

    Regardless of the binary state, the function always sets up
    ``PYTHONPATH`` (and ``sys.path``) so that ``import mlir`` works during
    test collection.
    """
    env = dict(os.environ)

    # Always set up PYTHONPATH for mlir_core, regardless of whether binaries
    # exist, so that ``import mlir`` works during test collection.
    mlir_core_path = _resolve_mlir_core_python_path(env=env)
    if mlir_core_path:
        current = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{mlir_core_path}{os.pathsep}{current}" if current else mlir_core_path
        )
        # PYTHONPATH is only read at interpreter startup; also update
        # sys.path directly so the change takes effect immediately.
        import sys

        if mlir_core_path not in sys.path:
            sys.path.insert(0, mlir_core_path)

    build_dir = repo_root / "csrc" / "mlir" / "build"

    if not _prebuilt_binaries_exist(build_dir):
        raise PretestBuildError(
            "Pre-built MLIR binaries not found. "
            "Run `build.sh` (or `build.sh --debug`) first."
        )
