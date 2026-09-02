"""Pre-test bootstrap utilities for verifying MLIR build artifacts.

This module does NOT trigger a build — it only checks whether pre-built
binaries exist and warns if they are missing.  Run ``build.sh`` first.
"""

from __future__ import annotations

from pathlib import Path


class PretestBuildError(RuntimeError):
    """Raised when the pre-test MLIR binary check fails."""


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
    """
    build_dir = repo_root / "csrc" / "mlir" / "build"

    if not _prebuilt_binaries_exist(build_dir):
        raise PretestBuildError(
            "Pre-built MLIR binaries not found. "
            "Run `./build.sh` first."
        )
