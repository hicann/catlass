from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_bootstrap(repo_root: Path):  # type: ignore[no-untyped-def]
    module_path = repo_root / "tests" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("catlass_test_bootstrap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load pretest bootstrap module from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _touch_opbase(include_dir: Path) -> None:
    opbase = include_dir / "mlir" / "IR" / "OpBase.td"
    opbase.parent.mkdir(parents=True, exist_ok=True)
    opbase.write_text("// test\n")


def test_ensure_pretest_mlir_build_respects_skip_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    monkeypatch.setenv("TLA_DSL_SKIP_PRETEST_BUILD", "1")
    called = False

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args=["noop"], returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ensure_pretest_mlir_build(tmp_path)
    assert called is False


def test_ensure_pretest_mlir_build_runs_cmake_and_ninja_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "include"
    _touch_opbase(include_dir)
    mlir_dir = tmp_path / "cmake" / "mlir"
    mlir_dir.mkdir(parents=True)

    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(include_dir))
    monkeypatch.setenv("MLIR_DIR", str(mlir_dir))
    monkeypatch.delenv("TLA_DSL_SKIP_PRETEST_BUILD", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name, path=None: f"/toolchain/bin/{name}",
    )

    seen: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):  # type: ignore[no-untyped-def]
        del check, kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ensure_pretest_mlir_build(tmp_path)

    assert seen == [
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            "csrc/mlir",
            "-B",
            "csrc/mlir/build",
            f"-DMLIR_TBLGEN_INCLUDE_DIR={include_dir}",
            "-DCMAKE_C_COMPILER=/toolchain/bin/gcc",
            "-DCMAKE_CXX_COMPILER=/toolchain/bin/g++",
            f"-DMLIR_DIR={mlir_dir}",
        ],
        ["ninja", "-C", "csrc/mlir/build", "tla-compiler"],
    ]


def test_ensure_pretest_mlir_build_uses_llvm_config_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "llvm-include"
    _touch_opbase(include_dir)
    monkeypatch.delenv("MLIR_TBLGEN_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("MLIR_DIR", raising=False)
    monkeypatch.delenv("TLA_DSL_SKIP_PRETEST_BUILD", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name, path=None: (
            f"/toolchain/bin/{name}" if name in {"gcc", "g++"} else None
        ),
    )

    seen: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):  # type: ignore[no-untyped-def]
        del check, kwargs
        seen.append(list(cmd))
        if cmd[:2] == ["llvm-config", "--includedir"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=str(include_dir), stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ensure_pretest_mlir_build(tmp_path)

    assert seen[0] == ["llvm-config", "--includedir"]
    assert seen[1][:6] == ["cmake", "-G", "Ninja", "-S", "csrc/mlir", "-B"]
    assert seen[2] == ["ninja", "-C", "csrc/mlir/build", "tla-compiler"]


def test_resolve_include_dir_rejects_invalid_env_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    monkeypatch.delenv("TLA_DSL_SKIP_PRETEST_BUILD", raising=False)
    bad_include = tmp_path / "bad-include"
    bad_include.mkdir()
    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(bad_include))

    with pytest.raises(bootstrap.PretestBuildError, match="does not contain"):
        bootstrap.ensure_pretest_mlir_build(tmp_path)


def test_ensure_pretest_mlir_build_resets_stale_compiler_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "include"
    _touch_opbase(include_dir)
    mlir_dir = tmp_path / "cmake" / "mlir"
    mlir_dir.mkdir(parents=True)
    build_dir = tmp_path / "csrc" / "mlir" / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-11\n"
        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-11\n"
    )

    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(include_dir))
    monkeypatch.setenv("MLIR_DIR", str(mlir_dir))
    monkeypatch.delenv("TLA_DSL_SKIP_PRETEST_BUILD", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name, path=None: f"/conda/bin/{name}",
    )

    seen: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):  # type: ignore[no-untyped-def]
        del check, kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ensure_pretest_mlir_build(tmp_path)

    assert (build_dir / "CMakeCache.txt").exists() is False
    assert seen[0] == [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        "csrc/mlir",
        "-B",
        "csrc/mlir/build",
        f"-DMLIR_TBLGEN_INCLUDE_DIR={include_dir}",
        "-DCMAKE_C_COMPILER=/conda/bin/gcc",
        "-DCMAKE_CXX_COMPILER=/conda/bin/g++",
        f"-DMLIR_DIR={mlir_dir}",
    ]


def test_ensure_pretest_mlir_build_honors_runtime_wrapper_env_toggle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = _load_bootstrap(REPO_ROOT)
    include_dir = tmp_path / "include"
    _touch_opbase(include_dir)
    mlir_dir = tmp_path / "cmake" / "mlir"
    mlir_dir.mkdir(parents=True)

    monkeypatch.setenv("MLIR_TBLGEN_INCLUDE_DIR", str(include_dir))
    monkeypatch.setenv("MLIR_DIR", str(mlir_dir))
    monkeypatch.setenv("TLA_DSL_BUILD_RUNTIME_WRAPPER", "ON")
    monkeypatch.delenv("TLA_DSL_SKIP_PRETEST_BUILD", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        bootstrap.shutil,
        "which",
        lambda name, path=None: None,
    )

    seen: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):  # type: ignore[no-untyped-def]
        del check, kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap.ensure_pretest_mlir_build(tmp_path)

    assert "-DTLA_DSL_BUILD_RUNTIME_WRAPPER=ON" in seen[0]
