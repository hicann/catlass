"""Compilation and execution support for Tla DSL kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base_dsl.arch import (
    TlaKernelTarget,
    arch_scope_for_target as _arch_scope_for_target_impl,
    default_target_arch,
    get_kernel_target as _get_kernel_target,
    parse_arch_scope as _parse_arch_scope_impl,
)
from .base_dsl import BaseDSL, DSLLocation
from .compiler_bridge import (
    BridgeUnavailableError,
    lower_tlair_module_to_mlir,
    resolve_bridge_extension_path,
)
from . import execution_triton

DEFAULT_ARCH_SCOPE = "aiv.c310"
SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")


class TlaExecutionError(RuntimeError):
    """Base class for execution and toolchain failures."""


class TlaCompilerBridgeUnavailableError(TlaExecutionError):
    """Raised when the typed Tla compiler bridge cannot be resolved."""


class TlaBackendCompilerNotFoundError(TlaExecutionError):
    """Raised when the backend compiler cannot be resolved."""


class TlaKernelCompileError(TlaExecutionError):
    """Raised when kernel lowering or backend compilation fails."""


class TlaRuntimeUnavailableError(TlaExecutionError):
    """Raised when Ascend runtime dependencies are unavailable."""


class TlaUnsupportedAbiError(TlaExecutionError):
    """Raised when attempting to execute an unsupported ABI shape."""


@dataclass(frozen=True)
class TlaKernelArtifact:
    cache_key: str
    cache_dir: Path
    tlair_mlir: str
    lowered_llvm: str
    entrypoint: str
    compiler_bridge_path: Path | None
    hivmc_path: Path
    kernel_binary_path: Path
    runtime: "TlaRuntimeOptions | None" = None
    pass_ir_dump: str = ""


@dataclass(frozen=True)
class TlaExecutionResult:
    artifact: TlaKernelArtifact
    module_handle: int
    function_handle: int
    device: int


@dataclass(frozen=True)
class TlaRuntimeOptions:
    backend: str = "ascend"
    cache_enabled: bool = True
    cache_dir: Path | None = None
    force_recompile: bool = False
    hivmc: str | None = None
    hivmc_args: tuple[str, ...] = ()
    target_arch: str = "c310"
    core_type: str = "aiv"
    kernel_mode: str = "aiv"
    shared: int = 1
    arch_scope: str = DEFAULT_ARCH_SCOPE
    mlir_print_ir_before: tuple[str, ...] = ()
    mlir_print_ir_after: tuple[str, ...] = ()
    mlir_print_ir_before_all: bool = False
    mlir_print_ir_after_all: bool = False


_MEMORY_COMPILE_CACHE_LOCK = threading.RLock()
_MEMORY_COMPILE_CACHE: dict[str, TlaKernelArtifact] = {}


def compile_kernel(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any],
    runtime: TlaRuntimeOptions,
    type_args: Sequence[Any] | None = None,
    decorator_location: DSLLocation | None = None,
) -> TlaKernelArtifact:
    if runtime.backend != "ascend":
        raise TlaExecutionError(f"Unsupported backend: {runtime.backend}")
    if kind == "ascendnpuir_kernel":
        return _compile_ascendnpuir_kernel(
            options=options,
            runtime=runtime,
            decorator_location=decorator_location,
        )

    lowered = BaseDSL()._lower(
        fn,
        kind=kind,
        options=dict(options),
        type_args=type_args,
        location=decorator_location,
    )
    tlair_mlir = lowered.asm(generic=True)
    entrypoint = _extract_entrypoint(tlair_mlir)
    compiler_bridge_path = resolve_bridge_extension_path()
    hivmc = _resolve_hivmc_a5(runtime.hivmc)
    target = _resolve_kernel_target(runtime)
    cache_dir = runtime.cache_dir or _default_cache_dir()
    cache_key = _cache_key(
        tlair_mlir=tlair_mlir,
        entrypoint=entrypoint,
        runtime=runtime,
        compiler_bridge_path=compiler_bridge_path,
        hivmc=hivmc,
        target=target,
    )
    artifact_dir = cache_dir / cache_key
    manifest = artifact_dir / "manifest.json"

    if runtime.cache_enabled and not runtime.force_recompile:
        cached_memory = _get_memory_cached_artifact(cache_key)
        if cached_memory is not None:
            return cached_memory

    if runtime.cache_enabled and not runtime.force_recompile and manifest.exists():
        cached = _load_manifest(manifest)
        kernel_path = artifact_dir / str(cached["kernel_binary"])
        mlir_path = artifact_dir / str(cached["lowered_mlir"])
        cached_pass_dump = cached.get("pass_ir_dump")
        pass_dump_path = (
            artifact_dir / str(cached_pass_dump) if cached_pass_dump else None
        )
        if kernel_path.exists() and mlir_path.exists():
            artifact = TlaKernelArtifact(
                cache_key=cache_key,
                cache_dir=artifact_dir,
                tlair_mlir=tlair_mlir,
                lowered_llvm=mlir_path.read_text(),
                entrypoint=entrypoint,
                compiler_bridge_path=compiler_bridge_path,
                hivmc_path=hivmc,
                kernel_binary_path=kernel_path,
                runtime=runtime,
                pass_ir_dump=pass_dump_path.read_text()
                if pass_dump_path and pass_dump_path.exists()
                else "",
            )
            _set_memory_cached_artifact(artifact)
            return artifact

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = artifact_dir / "lowered.mlir"
    pass_dump_path = artifact_dir / "pass-ir-dump.mlir"
    kernel_path = artifact_dir / "kernel.o"

    lowering_result = _run_tla_lowering_to_mlir(
        lowered_module=lowered.module,
        tlair_mlir=tlair_mlir,
        mlir_path=mlir_path,
        runtime=runtime,
    )
    if lowering_result.pass_ir_dump:
        pass_dump_path.write_text(lowering_result.pass_ir_dump)
    runtime_for_hivmc = _runtime_options_for_offline_ascendnpuir_mlir(
        runtime, mlir_path.read_text()
    )
    _run_checked(
        _build_hivmc_a5_command(
            compiler=hivmc,
            mlir_path=mlir_path,
            kernel_path=kernel_path,
            runtime=runtime_for_hivmc,
        ),
        label="hivmc-a5",
        cwd=artifact_dir,
    )
    if not kernel_path.exists():
        raise TlaKernelCompileError(
            "hivmc-a5 completed but output kernel artifact was not "
            f"created at {kernel_path}"
        )

    manifest.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "entrypoint": entrypoint,
                "kernel_binary": kernel_path.name,
                "lowered_mlir": mlir_path.name,
                "pass_ir_dump": pass_dump_path.name
                if lowering_result.pass_ir_dump
                else None,
                "compiler_bridge": (
                    str(compiler_bridge_path) if compiler_bridge_path else None
                ),
                "hivmc": str(hivmc),
                "arch_scope": runtime_for_hivmc.arch_scope,
                "target_arch": runtime_for_hivmc.target_arch,
                "core_type": runtime_for_hivmc.core_type,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifact = TlaKernelArtifact(
        cache_key=cache_key,
        cache_dir=artifact_dir,
        tlair_mlir=tlair_mlir,
        lowered_llvm=mlir_path.read_text(),
        entrypoint=entrypoint,
        compiler_bridge_path=compiler_bridge_path,
        hivmc_path=hivmc,
        kernel_binary_path=kernel_path,
        pass_ir_dump=lowering_result.pass_ir_dump,
        runtime=runtime_for_hivmc,
    )
    if runtime.cache_enabled and not runtime.force_recompile:
        _set_memory_cached_artifact(artifact)
    return artifact


def _compile_ascendnpuir_kernel(
    *,
    options: Mapping[str, Any],
    runtime: TlaRuntimeOptions,
    decorator_location: DSLLocation | None,
) -> TlaKernelArtifact:
    mlir_file_opt = options.get("mlir_file")
    if mlir_file_opt is None:
        raise TlaExecutionError("ascendnpuir kernel options are missing `mlir_file`.")
    mlir_path = Path(str(mlir_file_opt)).expanduser().resolve()
    if not mlir_path.exists():
        raise TlaExecutionError(f"AscendNPU-IR MLIR file does not exist: {mlir_path}")
    mlir_text = mlir_path.read_text()
    runtime = _runtime_options_for_offline_ascendnpuir_mlir(runtime, mlir_text)
    decorated_entrypoint = str(options.get("entrypoint") or "").strip() or None
    mlir_entrypoint = _try_extract_entrypoint(mlir_text)
    if (
        decorated_entrypoint is not None
        and mlir_entrypoint is not None
        and decorated_entrypoint != mlir_entrypoint
    ):
        location_text = ""
        if decorator_location is not None:
            location_text = f" Decorator: {decorator_location.filename}:{decorator_location.lineno}."
        raise TlaExecutionError(
            "Decorated function name does not match the MLIR entrypoint. "
            f"decorated={decorated_entrypoint!r} mlir={mlir_entrypoint!r}.{location_text}"
        )
    entrypoint = decorated_entrypoint or mlir_entrypoint
    if entrypoint is None:
        raise TlaExecutionError(
            f"Could not infer kernel entrypoint from offline MLIR file: {mlir_path}"
        )
    hivmc = _resolve_hivmc_a5(runtime.hivmc)
    cache_dir = runtime.cache_dir or _default_cache_dir()
    cache_key = _offline_mlir_cache_key(
        mlir_text=mlir_text,
        entrypoint=entrypoint,
        runtime=runtime,
        hivmc=hivmc,
    )
    artifact_dir = cache_dir / cache_key
    manifest = artifact_dir / "manifest.json"

    if runtime.cache_enabled and not runtime.force_recompile:
        cached_memory = _get_memory_cached_artifact(cache_key)
        if cached_memory is not None:
            return cached_memory

    if runtime.cache_enabled and not runtime.force_recompile and manifest.exists():
        cached = _load_manifest(manifest)
        kernel_path = artifact_dir / str(cached["kernel_binary"])
        mlir_copy_path = artifact_dir / str(cached["llvm_ir"])
        if kernel_path.exists() and mlir_copy_path.exists():
            artifact = TlaKernelArtifact(
                cache_key=cache_key,
                cache_dir=artifact_dir,
                tlair_mlir=mlir_text,
                lowered_llvm=mlir_copy_path.read_text(),
                entrypoint=entrypoint,
                compiler_bridge_path=None,
                hivmc_path=hivmc,
                kernel_binary_path=kernel_path,
                runtime=runtime,
                pass_ir_dump="",
            )
            _set_memory_cached_artifact(artifact)
            return artifact

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlir_copy_path = artifact_dir / "input.mlir"
    mlir_copy_path.write_text(mlir_text)
    kernel_path = artifact_dir / "kernel.o"

    _run_checked(
        _build_hivmc_a5_command(
            compiler=hivmc,
            mlir_path=mlir_copy_path,
            kernel_path=kernel_path,
            runtime=runtime,
        ),
        label="hivmc-a5",
        cwd=artifact_dir,
    )
    if not kernel_path.exists():
        raise TlaKernelCompileError(
            "hivmc-a5 completed but output kernel artifact was not "
            f"created at {kernel_path}"
        )

    manifest.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "entrypoint": entrypoint,
                "kernel_binary": kernel_path.name,
                "llvm_ir": mlir_copy_path.name,
                "compiler_bridge": None,
                "hivmc": str(hivmc),
                "arch_scope": runtime.arch_scope,
                "target_arch": runtime.target_arch,
                "core_type": runtime.core_type,
                "source_mlir": str(mlir_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifact = TlaKernelArtifact(
        cache_key=cache_key,
        cache_dir=artifact_dir,
        tlair_mlir=mlir_text,
        lowered_llvm=mlir_text,
        entrypoint=entrypoint,
        compiler_bridge_path=None,
        hivmc_path=hivmc,
        kernel_binary_path=kernel_path,
        runtime=runtime,
        pass_ir_dump="",
    )
    if runtime.cache_enabled and not runtime.force_recompile:
        _set_memory_cached_artifact(artifact)
    return artifact


def _get_memory_cached_artifact(cache_key: str) -> TlaKernelArtifact | None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        artifact = _MEMORY_COMPILE_CACHE.get(cache_key)
    if artifact is None:
        return None
    if not artifact.kernel_binary_path.exists():
        _drop_memory_cached_artifact(cache_key)
        return None
    return artifact


def _set_memory_cached_artifact(artifact: TlaKernelArtifact) -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE[artifact.cache_key] = artifact


def _drop_memory_cached_artifact(cache_key: str) -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE.pop(cache_key, None)


def execute_kernel(
    artifact: TlaKernelArtifact,
    *,
    runtime: TlaRuntimeOptions,
    launch_args: Sequence[Any],
    launch_kwargs: Mapping[str, Any],
) -> TlaExecutionResult:
    loader = _AscendLoader()
    grid = launch_kwargs.get("grid", (1, 1, 1))
    if isinstance(grid, int):
        grid = (int(grid), 1, 1)
    if len(grid) != 3:
        raise TlaUnsupportedAbiError("`grid` must be an int or a 3-tuple.")
    device = int(launch_kwargs.get("device", loader.get_current_device()))
    stream = launch_kwargs.get("stream")
    if stream is None:
        stream = loader.get_current_stream(device)
    module_handle, function_handle = loader.load_binary(
        name=f"{artifact.entrypoint} {runtime.kernel_mode}",
        kernel_path=artifact.kernel_binary_path,
        shared=runtime.shared,
        device=device,
    )
    if launch_args:
        _mark_tensor_launch_args_uploaded(launch_args)
        packed_args = execution_triton.try_build_packed_launch_args(
            artifact=artifact,
            launch_args=launch_args,
            grid=(int(grid[0]), int(grid[1]), int(grid[2])),
            device=device,
            try_extract_entrypoint=_try_extract_entrypoint,
            unsupported_abi_error=TlaUnsupportedAbiError,
            runtime_unavailable_error=TlaRuntimeUnavailableError,
        )
        if packed_args is not None:
            loader.launch_with_packed_args(
                function=function_handle,
                stream=int(stream),
                grid_x=int(grid[0]),
                grid_y=int(grid[1]),
                grid_z=int(grid[2]),
                args=packed_args,
            )
        else:
            flat_args = _flatten_launch_args(launch_args)
            loader.launch_with_args(
                function=function_handle,
                stream=int(stream),
                grid_x=int(grid[0]),
                grid_y=int(grid[1]),
                grid_z=int(grid[2]),
                args=flat_args,
            )
    else:
        loader.launch_zero_arg(
            function=function_handle,
            stream=int(stream),
            grid_x=int(grid[0]),
            grid_y=int(grid[1]),
            grid_z=int(grid[2]),
        )
    return TlaExecutionResult(
        artifact=artifact,
        module_handle=module_handle,
        function_handle=function_handle,
        device=device,
    )


def _mark_tensor_launch_args_uploaded(args: Sequence[Any]) -> None:
    for arg in args:
        if hasattr(arg, "prepare_for_launch") and callable(arg.prepare_for_launch):
            arg.prepare_for_launch()
        elif hasattr(arg, "upload_data") and callable(arg.upload_data):
            arg.upload_data()


def _flatten_launch_args(args: Sequence[Any]) -> list[int]:
    flattened: list[int] = []
    for arg in args:
        if hasattr(arg, "__c_pointers__"):
            ptrs = arg.__c_pointers__()
            flattened.extend(int(ptr) for ptr in ptrs)
            continue
        if hasattr(arg, "data_ptr") and callable(arg.data_ptr):
            flattened.append(int(arg.data_ptr()))
            continue
        if isinstance(arg, bool):
            flattened.append(int(arg))
            continue
        if isinstance(arg, int):
            flattened.append(int(arg))
            continue
        if isinstance(arg, float):
            import struct

            packed = struct.unpack("Q", struct.pack("d", arg))[0]
            flattened.append(int(packed))
            continue
        raise TlaUnsupportedAbiError(
            "Launch arguments must provide __c_pointers__(), data_ptr(), or be int/float."
        )
    return flattened


def runtime_options_from_kwargs(kwargs: Mapping[str, Any]) -> TlaRuntimeOptions:
    explicit_arch_scope = kwargs.get("arch_scope")
    target_arch = str(
        kwargs.get(
            "target_arch",
            default_target_arch().value,
        )
    ).lower()
    core_type = str(
        kwargs.get(
            "core_type",
            "aiv",
        )
    ).lower()
    if explicit_arch_scope is not None:
        arch_scope = str(explicit_arch_scope).lower()
        target_arch, core_type = _parse_arch_scope(arch_scope)
    else:
        arch_scope = _arch_scope_for_target(
            target_arch=target_arch, core_type=core_type
        )
    kernel_mode = str(kwargs.get("kernel_mode", core_type)).lower()
    if kernel_mode != core_type:
        raise TlaExecutionError(
            f"kernel_mode={kernel_mode!r} does not match core_type={core_type!r}."
        )
    return TlaRuntimeOptions(
        backend=str(kwargs.get("backend", "ascend")),
        cache_enabled=bool(
            kwargs.get("cache", _env_truthy("TLA_DSL_CACHE", default="1"))
        ),
        cache_dir=(
            Path(str(kwargs["cache_dir"])).expanduser().resolve()
            if kwargs.get("cache_dir")
            else None
        ),
        force_recompile=bool(
            kwargs.get(
                "force_recompile", _env_truthy("TLA_DSL_FORCE_RECOMPILE", default="0")
            )
        ),
        hivmc=(str(kwargs["hivmc"]) if kwargs.get("hivmc") is not None else None),
        hivmc_args=tuple(kwargs.get("hivmc_args", ()) or ()),
        target_arch=target_arch,
        core_type=core_type,
        kernel_mode=kernel_mode,
        shared=int(kwargs.get("shared", 1)),
        arch_scope=arch_scope,
        mlir_print_ir_before=_string_tuple(kwargs.get("mlir_print_ir_before", ())),
        mlir_print_ir_after=_string_tuple(kwargs.get("mlir_print_ir_after", ())),
        mlir_print_ir_before_all=bool(kwargs.get("mlir_print_ir_before_all", False)),
        mlir_print_ir_after_all=bool(kwargs.get("mlir_print_ir_after_all", False)),
    )


def runtime_options_for_launch(runtime: TlaRuntimeOptions) -> TlaRuntimeOptions:
    if runtime.cache_dir is not None:
        return runtime
    temp_dir = Path(tempfile.mkdtemp(prefix="tla-dsl-kernel-")).resolve()
    return replace(runtime, cache_enabled=False, cache_dir=temp_dir)


def _runtime_options_for_offline_ascendnpuir_mlir(
    runtime: TlaRuntimeOptions, mlir_text: str
) -> TlaRuntimeOptions:
    target_arch = runtime.target_arch
    core_type = runtime.core_type
    hivmc_args = runtime.hivmc_args

    if (
        "hivm.module_core_type<AIC>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIC>" in mlir_text
    ):
        core_type = "aic"
    elif (
        "hivm.module_core_type<AIV>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir_text
    ):
        core_type = "aiv"

    if "dav-c310" in mlir_text or 'hacc.target<"Ascend950PR_9589">' in mlir_text:
        target_arch = "c310"

    hivmc_args = execution_triton.infer_hivmc_args(mlir_text, hivmc_args)

    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type=core_type)
    if (
        runtime.target_arch == target_arch
        and runtime.core_type == core_type
        and runtime.kernel_mode == core_type
        and runtime.arch_scope == arch_scope
        and runtime.hivmc_args == hivmc_args
    ):
        return runtime
    return replace(
        runtime,
        target_arch=target_arch,
        core_type=core_type,
        kernel_mode=core_type,
        arch_scope=arch_scope,
        hivmc_args=hivmc_args,
    )


def _extract_entrypoint(mlir_text: str) -> str:
    match = re.search(r"@([A-Za-z_][A-Za-z0-9_]*)\s*\(", mlir_text)
    if match:
        return match.group(1)
    sym_match = re.search(r'sym_name\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)"', mlir_text)
    if sym_match:
        return sym_match.group(1)
    raise TlaExecutionError("Could not infer kernel entrypoint from lowered MLIR.")


def _try_extract_entrypoint(mlir_text: str) -> str | None:
    func_matches = list(
        re.finditer(
            r"\bfunc\.func\s+(?:private\s+)?@([A-Za-z_][A-Za-z0-9_]*)\b",
            mlir_text,
        )
    )
    for index, match in enumerate(func_matches):
        end = (
            func_matches[index + 1].start()
            if index + 1 < len(func_matches)
            else len(mlir_text)
        )
        if "hacc.entry" in mlir_text[match.start() : end]:
            return match.group(1)
    for match in func_matches:
        declaration_prefix = mlir_text[max(0, match.start() - 32) : match.start()]
        if "private" not in declaration_prefix:
            return match.group(1)
    try:
        return _extract_entrypoint(mlir_text)
    except TlaExecutionError:
        return None


def _cache_key(
    *,
    tlair_mlir: str,
    entrypoint: str,
    runtime: TlaRuntimeOptions,
    compiler_bridge_path: Path | None,
    hivmc: Path,
    target: TlaKernelTarget,
) -> str:
    key_payload = {
        "entrypoint": entrypoint,
        "backend": runtime.backend,
        "kernel_mode": runtime.kernel_mode,
        "arch_scope": runtime.arch_scope,
        "target_arch": runtime.target_arch,
        "core_type": runtime.core_type,
        "cce_arch": target.cce_arch,
        "compiler_bridge": str(compiler_bridge_path) if compiler_bridge_path else None,
        "hivmc": str(hivmc),
        "compiler_bridge_fingerprint": _tool_fingerprint(compiler_bridge_path),
        "hivmc_version": _tool_version(hivmc),
        "hivmc_fingerprint": _tool_fingerprint(hivmc),
        "mlir": tlair_mlir,
        "hivmc_args": list(runtime.hivmc_args),
        "mlir_print_ir_before": list(runtime.mlir_print_ir_before),
        "mlir_print_ir_after": list(runtime.mlir_print_ir_after),
        "mlir_print_ir_before_all": runtime.mlir_print_ir_before_all,
        "mlir_print_ir_after_all": runtime.mlir_print_ir_after_all,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _offline_mlir_cache_key(
    *,
    mlir_text: str,
    entrypoint: str,
    runtime: TlaRuntimeOptions,
    hivmc: Path,
) -> str:
    key_payload = {
        "entrypoint": entrypoint,
        "backend": runtime.backend,
        "kernel_mode": runtime.kernel_mode,
        "target_arch": runtime.target_arch,
        "core_type": runtime.core_type,
        "compiler": str(hivmc),
        "compiler_version": _tool_version(hivmc),
        "compiler_fingerprint": _tool_fingerprint(hivmc),
        "compiler_args": list(runtime.hivmc_args),
        "mlir": mlir_text,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _tool_version(binary: Path) -> str:
    try:
        proc = subprocess.run(
            [str(binary), "--version"], check=False, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    text = (proc.stdout or "") + (proc.stderr or "")
    text = text.strip()
    if not text:
        return f"exit:{proc.returncode}"
    return text.splitlines()[0][:200]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _tool_fingerprint(binary: Path | None) -> str:
    if binary is None:
        return "unresolved"
    try:
        stat = binary.stat()
    except OSError:
        return "missing"
    digest = hashlib.sha256()
    try:
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    return f"{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


def _default_cache_dir() -> Path:
    cache = os.getenv("TLA_DSL_CACHE_DIR")
    if cache:
        return Path(cache).expanduser().resolve()
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg) / "catlass").expanduser().resolve()
    return (Path.home() / ".cache" / "catlass").resolve()


def _parse_arch_scope(arch_scope: str) -> tuple[str, str]:
    if arch_scope not in SUPPORTED_ARCH_SCOPES:
        raise TlaExecutionError(
            f"Unsupported arch_scope={arch_scope!r}. Supported: {', '.join(SUPPORTED_ARCH_SCOPES)}."
        )
    try:
        return _parse_arch_scope_impl(arch_scope)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _arch_scope_for_target(*, target_arch: str, core_type: str) -> str:
    try:
        return _arch_scope_for_target_impl(target_arch=target_arch, core_type=core_type)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_kernel_target(runtime: TlaRuntimeOptions) -> TlaKernelTarget:
    try:
        return _get_kernel_target(
            target_arch=runtime.target_arch,
            core_type=runtime.core_type,
            arch_scope=runtime.arch_scope,
        )
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_hivmc_a5(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
    env = os.getenv("TLA_DSL_HIVMC_A5")
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.exists():
            return candidate
    root = os.getenv("TLA_DSL_ASCENDNPU_IR_ROOT")
    candidates: list[Path] = []
    if root:
        root_path = Path(root).expanduser().resolve()
        candidates.extend(
            [
                root_path / "build" / "bin" / "hivmc-a5",
                root_path / "hivmc-a5",
            ]
        )
    which = shutil.which("hivmc-a5")
    if which:
        candidates.append(Path(which).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise TlaBackendCompilerNotFoundError(
        "hivmc-a5 not found. Set `TLA_DSL_HIVMC_A5`, set "
        "`TLA_DSL_ASCENDNPU_IR_ROOT`, pass `hivmc=...`, or add `hivmc-a5` to PATH."
    )


def _build_hivmc_a5_command(
    *,
    compiler: Path,
    mlir_path: Path,
    kernel_path: Path,
    runtime: TlaRuntimeOptions,
) -> list[str]:
    if execution_triton.should_use_triton_compile_mode(runtime.hivmc_args):
        return [
            str(compiler),
            str(mlir_path),
            "--target=Ascend950PR_9589",
            *runtime.hivmc_args,
            "-o",
            str(kernel_path),
        ]

    template_bitcode = _resolve_hivm_template_bitcode(runtime)
    return [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--disable-ffts",
        "--enable-hivm-compile=False",
        f"--link-aicore-bitcode={template_bitcode}",
        "-o",
        str(kernel_path),
        *runtime.hivmc_args,
    ]


def _mlir_build_dirs() -> list[Path]:
    # .../python/tla_dsl/catlass/execution.py -> .../python/tla_dsl/csrc/mlir/build
    dsl_root = Path(__file__).resolve().parents[1]
    nested = dsl_root / "csrc" / "mlir" / "build"
    # Legacy ascend-catlass-DSL: .../python/tla_dsl/execution.py -> repo/mlir/build
    legacy = Path(__file__).resolve().parents[2] / "mlir" / "build"
    return [nested, legacy]


def _resolve_hivm_template_bitcode(runtime: TlaRuntimeOptions) -> str:
    explicit = os.getenv("TLA_DSL_HIVM_TEMPLATE_BC")
    if explicit:
        paths = [Path(item).expanduser().resolve() for item in explicit.split(",")]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise TlaRuntimeUnavailableError(
                "TLA_DSL_HIVM_TEMPLATE_BC references missing files: "
                + ", ".join(missing)
            )
        return ",".join(str(path) for path in paths)

    candidates: list[Path] = []
    if runtime.core_type == "aic":
        for build_dir in _mlir_build_dirs():
            candidates.extend(
                [
                    build_dir / "meta_op.aic.c310.bc",
                    build_dir / "bc" / "meta_op.aic.c310.bc",
                ]
            )
    else:
        for build_dir in _mlir_build_dirs():
            candidates.append(build_dir / "bc" / "meta_op.aiv.c310.bc")
    existing = [path.resolve() for path in candidates if path.exists()]
    if existing:
        return ",".join(str(path) for path in existing)
    raise TlaRuntimeUnavailableError(
        "C310 HIVM template bitcode not found. Build Tla DSL templates or set "
        "`TLA_DSL_HIVM_TEMPLATE_BC`."
    )


def _run_checked(
    command: list[str], *, label: str, cwd: Path, stdin_text: str | None = None
) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            input=stdin_text,
        )
    except subprocess.CalledProcessError as exc:
        raise TlaKernelCompileError(
            f"{label} failed with exit code {exc.returncode}\n"
            f"cmd: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc


def _run_tla_lowering_to_mlir(
    *,
    lowered_module: Any | None,
    tlair_mlir: str,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    try:
        return _run_typed_bridge_to_mlir(
            lowered_module=lowered_module,
            mlir_path=mlir_path,
            runtime=runtime,
        )
    except (TlaCompilerBridgeUnavailableError, TlaKernelCompileError):
        tla_compile = _resolve_tla_compile()
        if tla_compile is None:
            raise
        return _run_tla_compile_cli_to_mlir(
            tla_compile=tla_compile,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            runtime=runtime,
        )


def _run_typed_bridge_to_mlir(
    *,
    lowered_module: Any | None,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    if lowered_module is None:
        raise TlaCompilerBridgeUnavailableError(
            "Python runtime compilation requires a live MLIR module. "
            "String TLA MLIR lowering is not supported."
        )
    try:
        result = lower_tlair_module_to_mlir(
            lowered_module,
            mlir_print_ir_before=runtime.mlir_print_ir_before if runtime else (),
            mlir_print_ir_after=runtime.mlir_print_ir_after if runtime else (),
            mlir_print_ir_before_all=runtime.mlir_print_ir_before_all
            if runtime
            else False,
            mlir_print_ir_after_all=runtime.mlir_print_ir_after_all
            if runtime
            else False,
        )
    except BridgeUnavailableError as exc:
        raise TlaCompilerBridgeUnavailableError(str(exc)) from exc
    except Exception as exc:
        raise TlaKernelCompileError(
            f"In-process Tla compiler bridge failed.\nerror:\n{exc}"
        ) from exc
    mlir_path.write_text(result.lowered_mlir)
    return result


def _resolve_tla_compile() -> Path | None:
    explicit = os.getenv("TLA_DSL_COMPILE")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    for build_dir in _mlir_build_dirs():
        candidates.extend(
            [
                build_dir / "TlaCompile",
                build_dir / "tools" / "tla-compile" / "TlaCompile",
            ]
        )
    for which_name in ("TlaCompile",):
        which = shutil.which(which_name)
        if which:
            candidates.append(Path(which).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _tla_compile_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dir = Path(conda_prefix) / "lib"
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    return env


def _run_tla_compile_cli_to_mlir(
    *,
    tla_compile: Path,
    tlair_mlir: str,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    del runtime
    input_path = mlir_path.with_suffix(".tlair.mlir")
    input_path.write_text(tlair_mlir)
    try:
        subprocess.run(
            [str(tla_compile), str(input_path), "-o", str(mlir_path)],
            check=True,
            capture_output=True,
            text=True,
            env=_tla_compile_env(),
        )
    except subprocess.CalledProcessError as exc:
        raise TlaKernelCompileError(
            f"TlaCompile CLI fallback failed with exit code {exc.returncode}\n"
            f"cmd: {tla_compile} {input_path} -o {mlir_path}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if not mlir_path.exists():
        raise TlaKernelCompileError(
            "TlaCompile CLI fallback completed but did not produce lowered MLIR at "
            f"{mlir_path}"
        )
    return type(
        "TlaLoweringResult",
        (),
        {
            "lowered_mlir": mlir_path.read_text(),
            "pass_ir_dump": "",
        },
    )()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise TlaExecutionError(f"Invalid cache manifest at {path}: {exc}") from exc


def _env_truthy(name: str, *, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def _resolve_runtime_wrapper_library(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.getenv("TLA_DSL_RUNTIME_WRAPPER")
    if env:
        candidates.append(Path(env))
    package_root = Path(__file__).resolve().parent
    candidates.append(package_root / "bin" / "libtla_dsl_runtime_wrapper.so")
    for build_dir in _mlir_build_dirs():
        candidates.extend(
            [
                build_dir / "libtla_dsl_runtime_wrapper.so",
                build_dir / "tla_dsl_runtime_wrapper.so",
            ]
        )
        for path in build_dir.rglob("*tla_dsl_runtime_wrapper*.so*"):
            candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise TlaRuntimeUnavailableError(
        "Failed to resolve the CMake-built runtime wrapper. "
        "Build `tla_dsl_runtime_wrapper` and/or set TLA_DSL_RUNTIME_WRAPPER."
    )


def _build_cpp_extension(module_name: str, src: str, *, ascend_path: Path) -> Path:
    cache_dir = _default_cache_dir() / "extensions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    key = hashlib.sha256((module_name + src + sys.version).encode("utf-8")).hexdigest()
    output = cache_dir / f"{module_name}_{key}{ext_suffix}"
    if output.exists():
        return output

    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise TlaRuntimeUnavailableError(
            "No C++ compiler found (set CXX or install clang++/g++)."
        )

    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:  # pragma: no cover
        scheme = "posix_prefix"
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include = Path(sysconfig.get_paths(scheme=scheme)["include"])
    asc_include = ascend_path / "include"
    asc_lib64 = ascend_path / "lib64"
    if not asc_include.exists() or not asc_lib64.exists():
        raise TlaRuntimeUnavailableError(
            f"Ascend toolkit not found under {ascend_path}. Missing include/ or lib64/."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        src_path = tmpdir_path / f"{module_name}.cpp"
        out_path = tmpdir_path / f"{module_name}{ext_suffix}"
        src_path.write_text(src)
        cmd = [
            cxx,
            str(src_path),
            "-shared",
            "-fPIC",
            "-O2",
            "-std=c++17",
            f"-I{py_include}",
            f"-I{asc_include}",
            f"-I{asc_include / 'experiment'}",
            "-L" + str(asc_lib64),
            "-lruntime",
            "-lascendcl",
            f"-Wl,-rpath,{asc_lib64}",
            "-o",
            str(out_path),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise TlaRuntimeUnavailableError(
                "Failed to build Ascend runtime extension.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        out_path.replace(output)
    return output


def _load_ext_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise TlaRuntimeUnavailableError(
            f"Failed to load extension module spec for {module_name}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ASCEND_EXT_SRC = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <string>
#include <vector>

#if __has_include("experiment/runtime/runtime/rt.h")
#include "experiment/runtime/runtime/rt.h"
#elif __has_include("runtime/rt.h")
#include "runtime/rt.h"
#else
#error "Cannot find Ascend runtime header (rt.h)"
#endif

static PyObject *loadKernelBinary(PyObject *, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  const char *kernel_mode;
  if (!PyArg_ParseTuple(args, "ss#iis", &name, &data, &data_size, &shared, &device, &kernel_mode)) {
    return NULL;
  }

  rtDevBinary_t devbin;
  devbin.data = data;
  devbin.length = data_size;
  std::string mode{kernel_mode};
  devbin.magic = mode == "aiv" ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
  devbin.version = 0;

  rtError_t ret = rtSetDevice(device);
  if (ret != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtSetDevice failed: 0x%x", ret);
    return NULL;
  }

  void *module = nullptr;
  ret = rtDevBinaryRegister(&devbin, &module);
  if (ret != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtDevBinaryRegister failed: 0x%x", ret);
    return NULL;
  }

  auto *stub = new uint64_t(0);
  void *stub_ptr = reinterpret_cast<void *>(stub);
  ret = rtFunctionRegister(module, stub_ptr, name, (void *)name, 0);
  if (ret != RT_ERROR_NONE) {
    delete stub;
    PyErr_Format(PyExc_RuntimeError, "rtFunctionRegister failed: 0x%x", ret);
    return NULL;
  }
  return Py_BuildValue("(KK)", reinterpret_cast<uint64_t>(module), reinterpret_cast<uint64_t>(stub_ptr));
}

static PyObject *launchZeroArg(PyObject *, PyObject *args) {
  unsigned long long function_u64;
  unsigned long long stream_u64;
  int gx, gy, gz;
  if (!PyArg_ParseTuple(args, "KKiii", &function_u64, &stream_u64, &gx, &gy, &gz)) {
    return NULL;
  }
  const void *function = reinterpret_cast<const void *>(function_u64);
  rtStream_t stream = reinterpret_cast<rtStream_t>(stream_u64);
  uint32_t block_num = static_cast<uint32_t>(gx) * static_cast<uint32_t>(gy) * static_cast<uint32_t>(gz);
  rtError_t ret = rtKernelLaunch(function, block_num, NULL, 0, NULL, stream);
  if (ret != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtKernelLaunch failed: 0x%x", ret);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *launchWithArgs(PyObject *, PyObject *args) {
  unsigned long long function_u64;
  unsigned long long stream_u64;
  int gx, gy, gz;
  PyObject *arg_seq = nullptr;
  if (!PyArg_ParseTuple(args, "KKiiiO", &function_u64, &stream_u64, &gx, &gy, &gz, &arg_seq)) {
    return NULL;
  }
  PyObject *seq = PySequence_Fast(arg_seq, "args must be a sequence");
  if (!seq) {
    return NULL;
  }
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  std::vector<uint64_t> values;
  values.reserve(static_cast<size_t>(n));
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
    unsigned long long val = PyLong_AsUnsignedLongLong(item);
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return NULL;
    }
    values.push_back(static_cast<uint64_t>(val));
  }
  Py_DECREF(seq);

  const void *function = reinterpret_cast<const void *>(function_u64);
  rtStream_t stream = reinterpret_cast<rtStream_t>(stream_u64);
  uint32_t block_num = static_cast<uint32_t>(gx) * static_cast<uint32_t>(gy) * static_cast<uint32_t>(gz);
  void *args_array = values.empty() ? NULL : static_cast<void *>(values.data());
  rtError_t ret = rtKernelLaunch(function, block_num, args_array,
                                 values.size() * sizeof(uint64_t), NULL, stream);
  if (ret != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtKernelLaunch failed: 0x%x", ret);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS, "Load kernel binary"},
    {"launch_zero_arg", launchZeroArg, METH_VARARGS, "Launch zero-arg kernel"},
    {"launch_with_args", launchWithArgs, METH_VARARGS, "Launch kernel with args"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "tla_dsl_ascend_rt",
    NULL,
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit_tla_dsl_ascend_rt(void) {
  return PyModule_Create(&ModuleDef);
}
"""


class _AscendLoader:
    def __init__(self) -> None:
        self._module = None

    def _ensure_loaded(self) -> None:
        if self._module is not None:
            return
        so_path = _resolve_runtime_wrapper_library()
        lib = ctypes.CDLL(str(so_path))
        lib.tla_runtime_last_error.restype = ctypes.c_char_p
        lib.tla_runtime_load_kernel.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.tla_runtime_load_kernel.restype = ctypes.c_int
        lib.tla_runtime_launch_kernel.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.tla_runtime_launch_kernel.restype = ctypes.c_int
        self._module = lib

    def _last_error(self) -> str:
        self._ensure_loaded()
        message = self._module.tla_runtime_last_error()
        if not message:
            return "unknown runtime wrapper error"
        return message.decode("utf-8", errors="replace")

    def get_current_device(self) -> int:
        try:
            from . import runtime as runtime_mod

            runtime_device = runtime_mod.current_device_id()
            if runtime_device is not None:
                return int(runtime_device)
        except Exception:
            pass
        try:
            import torch  # type: ignore
            import torch_npu  # noqa: F401

            return int(torch.npu.current_device())
        except Exception:
            return int(os.getenv("TLA_DSL_NPU_DEVICE", "0"))

    def get_current_stream(self, device: int) -> int:
        try:
            from . import runtime as runtime_mod

            runtime_stream = runtime_mod.current_stream()
            if runtime_stream is not None:
                return int(runtime_stream)
        except Exception:
            pass
        try:
            import torch  # type: ignore
            import torch_npu  # noqa: F401

            return int(torch.npu.current_stream(device).npu_stream)
        except Exception as exc:
            raise TlaRuntimeUnavailableError(
                "Failed to infer current NPU stream. Install torch_npu or pass "
                "`stream=<rtStream_t integer>`."
            ) from exc

    def load_binary(
        self, *, name: str, kernel_path: Path, shared: int, device: int
    ) -> tuple[int, int]:
        self._ensure_loaded()
        del shared, device
        fn_name, mode = name.split()
        module_handle = ctypes.c_uint64(0)
        function_handle = ctypes.c_uint64(0)
        ret = self._module.tla_runtime_load_kernel(
            os.fsencode(str(kernel_path)),
            os.fsencode(fn_name),
            os.fsencode(mode),
            ctypes.byref(module_handle),
            ctypes.byref(function_handle),
        )
        if ret != 0:
            raise TlaRuntimeUnavailableError(self._last_error())
        return int(module_handle.value), int(function_handle.value)

    def launch_zero_arg(
        self, *, function: int, stream: int, grid_x: int, grid_y: int, grid_z: int
    ) -> None:
        self._ensure_loaded()
        ret = self._module.tla_runtime_launch_kernel(
            ctypes.c_uint64(int(function)),
            ctypes.c_uint64(int(stream)),
            int(grid_x),
            int(grid_y),
            int(grid_z),
            ctypes.POINTER(ctypes.c_uint64)(),
            0,
        )
        if ret != 0:
            raise TlaRuntimeUnavailableError(self._last_error())

    def launch_with_args(
        self,
        *,
        function: int,
        stream: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        args: Sequence[int],
    ) -> None:
        self._ensure_loaded()
        arg_array = (ctypes.c_uint64 * len(args))(*[int(arg) for arg in args])
        ret = self._module.tla_runtime_launch_kernel(
            ctypes.c_uint64(int(function)),
            ctypes.c_uint64(int(stream)),
            int(grid_x),
            int(grid_y),
            int(grid_z),
            arg_array,
            len(args),
        )
        if ret != 0:
            raise TlaRuntimeUnavailableError(self._last_error())

    def launch_with_packed_args(
        self,
        *,
        function: int,
        stream: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        args: bytes,
    ) -> None:
        self._ensure_loaded()
        if len(args) % ctypes.sizeof(ctypes.c_uint64) != 0:
            raise TlaRuntimeUnavailableError(
                "Packed kernel arguments must be a multiple of 8 bytes."
            )
        arg_count = len(args) // ctypes.sizeof(ctypes.c_uint64)
        if arg_count == 0:
            raw_ptr = None
        else:
            arg_array = (ctypes.c_uint64 * arg_count).from_buffer_copy(args)
            raw_ptr = ctypes.cast(arg_array, ctypes.POINTER(ctypes.c_uint64))
        ret = self._module.tla_runtime_launch_kernel(
            ctypes.c_uint64(int(function)),
            ctypes.c_uint64(int(stream)),
            int(grid_x),
            int(grid_y),
            int(grid_z),
            raw_ptr,
            arg_count,
        )
        if ret != 0:
            raise TlaRuntimeUnavailableError(self._last_error())
