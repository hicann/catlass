from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_bootstrap(repo_root: Path):  # type: ignore[no-untyped-def]
    module_path = repo_root / "tests" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location(
        "catlass_test_bootstrap", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load pretest bootstrap module from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ensure_pretest_mlir_build_runs_cmake_and_ninja_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without pre-built binaries, ensure_pretest_mlir_build raises."""
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "include"
    include_dir.mkdir(parents=True)
    mlir_dir = tmp_path / "cmake" / "mlir"
    mlir_dir.mkdir(parents=True)

    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(include_dir))
    monkeypatch.setenv("MLIR_DIR", str(mlir_dir))
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)

    with pytest.raises(bootstrap.PretestBuildError, match="Pre-built MLIR"):
        bootstrap.ensure_pretest_mlir_build(tmp_path)


def test_ensure_pretest_mlir_build_uses_llvm_config_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without any MLIR env vars and without binaries, it raises."""
    bootstrap = _load_bootstrap(REPO_ROOT)
    monkeypatch.delenv("MLIR_TBLGEN_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("MLIR_DIR", raising=False)
    monkeypatch.delenv("CATLASS_DSL_PREBUILT_ASCENDNPU_IR", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)

    with pytest.raises(bootstrap.PretestBuildError, match="Pre-built MLIR"):
        bootstrap.ensure_pretest_mlir_build(tmp_path)


def test_resolve_include_dir_rejects_invalid_env_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bad MLIR_TBLGEN_INCLUDE_DIR doesn't change the binary-missing error."""
    bootstrap = _load_bootstrap(REPO_ROOT)
    bad_include = tmp_path / "bad-include"
    bad_include.mkdir()
    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(bad_include))

    with pytest.raises(bootstrap.PretestBuildError, match="Pre-built MLIR"):
        bootstrap.ensure_pretest_mlir_build(tmp_path)


def test_ensure_pretest_mlir_build_resets_stale_compiler_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale CMakeCache.txt is left intact — no build logic runs."""
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "include"
    include_dir.mkdir(parents=True)
    mlir_dir = tmp_path / "cmake" / "mlir"
    mlir_dir.mkdir(parents=True)
    build_dir = tmp_path / "csrc" / "mlir" / "build"
    build_dir.mkdir(parents=True)
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        "CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-11\n"
        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-11\n"
    )

    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(include_dir))
    monkeypatch.setenv("MLIR_DIR", str(mlir_dir))
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)

    with pytest.raises(bootstrap.PretestBuildError, match="Pre-built MLIR"):
        bootstrap.ensure_pretest_mlir_build(tmp_path)

    # The cache file must NOT be removed — no build logic runs.
    assert cache.is_file()
