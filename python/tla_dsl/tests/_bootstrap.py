"""Pre-test bootstrap utilities for refreshing MLIR build artifacts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


class PretestBuildError(RuntimeError):
    """Raised when the pre-test MLIR refresh cannot be completed."""


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _has_opbase(include_dir: Path) -> bool:
    return (include_dir / "mlir" / "IR" / "OpBase.td").is_file()


def _resolve_mlir_include_dir(
    *, env: Mapping[str, str], repo_root: Path, runner: "_Runner"
) -> Path:
    configured = env.get("MLIR_TBLGEN_INCLUDE_DIR")
    if configured:
        configured_path = Path(configured)
        if _has_opbase(configured_path):
            return configured_path
        raise PretestBuildError(
            "MLIR_TBLGEN_INCLUDE_DIR is set but does not contain "
            f"`mlir/IR/OpBase.td`: {configured_path}"
        )

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_include = Path(conda_prefix) / "include"
        if _has_opbase(conda_include):
            return conda_include

    try:
        llvm_config = runner(
            ["llvm-config", "--includedir"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
    except OSError:
        llvm_config = None
    if llvm_config is not None and llvm_config.returncode == 0:
        llvm_include = Path(llvm_config.stdout.strip())
        if llvm_include and _has_opbase(llvm_include):
            return llvm_include

    raise PretestBuildError(
        "Unable to locate MLIR includes containing `mlir/IR/OpBase.td`. "
        "Set `MLIR_TBLGEN_INCLUDE_DIR`, `CONDA_PREFIX`, or ensure `llvm-config` "
        "is available with MLIR headers."
    )


def _resolve_mlir_dir(*, env: Mapping[str, str]) -> str | None:
    configured = env.get("MLIR_DIR")
    if configured:
        return configured

    conda_prefix = env.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    conda_mlir_dir = Path(conda_prefix) / "lib" / "cmake" / "mlir"
    if conda_mlir_dir.is_dir():
        return str(conda_mlir_dir)
    return None


def _resolve_compiler(
    *, env: Mapping[str, str], var_name: str, fallback: str
) -> str | None:
    configured = env.get(var_name)
    if configured:
        configured_path = Path(configured)
        if configured_path.is_file():
            return str(configured_path)
        discovered = shutil.which(configured, path=env.get("PATH"))
        if discovered:
            return discovered
        raise PretestBuildError(
            f"{var_name} is set but does not resolve to an executable: {configured}"
        )
    return shutil.which(fallback, path=env.get("PATH"))


def _cached_compiler(cache_file: Path, key: str) -> str | None:
    if not cache_file.is_file():
        return None
    prefix = f"{key}:FILEPATH="
    for line in cache_file.read_text().splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value or None
    return None


def _reset_build_dir_if_compiler_drift(
    *,
    build_dir: Path,
    c_compiler: str | None,
    cxx_compiler: str | None,
) -> None:
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        return

    cached_c = _cached_compiler(cache_file, "CMAKE_C_COMPILER")
    cached_cxx = _cached_compiler(cache_file, "CMAKE_CXX_COMPILER")
    should_reset = False

    for cached, desired in ((cached_c, c_compiler), (cached_cxx, cxx_compiler)):
        if cached and not Path(cached).exists():
            should_reset = True
            break
        if cached and desired and Path(cached) != Path(desired):
            should_reset = True
            break

    if should_reset:
        shutil.rmtree(build_dir)


def _run_checked(
    cmd: Sequence[str], *, cwd: Path, env: Mapping[str, str], runner: "_Runner"
) -> None:
    try:
        proc = runner(cmd, cwd=cwd, env=env)
    except OSError as exc:
        rendered = " ".join(cmd)
        raise PretestBuildError(f"Failed to start command: {rendered}") from exc
    if proc.returncode != 0:
        rendered = " ".join(cmd)
        raise PretestBuildError(f"Command failed ({proc.returncode}): {rendered}")


class _Runner:
    def __call__(self, *args, **kwargs):
        return subprocess.run(*args, check=False, **kwargs)


def _prebuilt_binaries_exist(build_dir: Path) -> bool:
    type_bridge_glob = list(
        (build_dir / "python" / "catlass").glob("_tla_type_bridge_native*.so")
    )
    tla_compile = build_dir / "tools" / "tla-compile" / "TlaCompile"
    return len(type_bridge_glob) > 0 and tla_compile.is_file()


def ensure_pretest_mlir_build(repo_root: Path) -> None:
    """Ensure Tla MLIR build artifacts are available for tests.

    By default, checks whether pre-built binaries already exist on disk
    (from a prior ``build.sh`` invocation).  If they do, the build is
    skipped.  If they are missing, a fresh cmake + ninja build is
    triggered automatically.

    Environment variables
    ---------------------
    TLA_DSL_SKIP_PRETEST_BUILD : str
        If truthy (1/true/on/yes), unconditionally skip the build.
    TLA_DSL_FORCE_PRETEST_BUILD : str
        If truthy, unconditionally trigger a full cmake + ninja build,
        even when pre-built binaries already exist.
    """
    env = dict(os.environ)

    if _is_truthy(env.get("TLA_DSL_SKIP_PRETEST_BUILD")):
        return

    build_dir = repo_root / "csrc" / "mlir" / "build"

    if not _is_truthy(env.get("TLA_DSL_FORCE_PRETEST_BUILD")):
        if _prebuilt_binaries_exist(build_dir):
            return

    runner = _Runner()
    include_dir = _resolve_mlir_include_dir(env=env, repo_root=repo_root, runner=runner)
    mlir_dir = _resolve_mlir_dir(env=env)
    c_compiler = _resolve_compiler(env=env, var_name="CC", fallback="gcc")
    cxx_compiler = _resolve_compiler(env=env, var_name="CXX", fallback="g++")
    _reset_build_dir_if_compiler_drift(
        build_dir=build_dir, c_compiler=c_compiler, cxx_compiler=cxx_compiler
    )

    build_env = dict(env)
    build_env["MLIR_TBLGEN_INCLUDE_DIR"] = str(include_dir)
    if mlir_dir:
        build_env["MLIR_DIR"] = mlir_dir
    runtime_wrapper = env.get("TLA_DSL_BUILD_RUNTIME_WRAPPER")
    if runtime_wrapper:
        build_env["TLA_DSL_BUILD_RUNTIME_WRAPPER"] = runtime_wrapper
    if c_compiler:
        build_env["CC"] = c_compiler
    if cxx_compiler:
        build_env["CXX"] = cxx_compiler

    cmake_cmd = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        "csrc/mlir",
        "-B",
        "csrc/mlir/build",
        f"-DMLIR_TBLGEN_INCLUDE_DIR={include_dir}",
    ]
    if c_compiler:
        cmake_cmd.append(f"-DCMAKE_C_COMPILER={c_compiler}")
    if cxx_compiler:
        cmake_cmd.append(f"-DCMAKE_CXX_COMPILER={cxx_compiler}")
    if mlir_dir:
        cmake_cmd.append(f"-DMLIR_DIR={mlir_dir}")
    if runtime_wrapper:
        cmake_cmd.append(f"-DTLA_DSL_BUILD_RUNTIME_WRAPPER={runtime_wrapper}")
    _run_checked(cmake_cmd, cwd=repo_root, env=build_env, runner=runner)
    _run_checked(
        ["ninja", "-C", "csrc/mlir/build", "tla-compiler"],
        cwd=repo_root,
        env=build_env,
        runner=runner,
    )
