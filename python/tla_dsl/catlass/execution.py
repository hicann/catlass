"""Compilation and execution support for Tla DSL kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .base_dsl.arch import (
    TlaKernelTarget,
    arch_scope_for_target as _arch_scope_for_target_impl,
    default_target_arch,
    get_kernel_target as _get_kernel_target,
    parse_arch_scope as _parse_arch_scope_impl,
)
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.typing import Numeric
from .compiler_bridge import (
    BridgeLoweringError,
    BridgeUnavailableError,
    KernelAbiArgumentKind,
    KernelAbiIntegerSignedness,
    KernelAbiLayout,
    KernelAbiScalarCategory,
    KernelAbiScalarDescriptor,
    kernel_abi_from_dict,
    kernel_abi_to_dict,
    lower_tlair_module_to_mlir,
    resolve_bridge_extension_path,
)
from .types import dtype_size_bytes

DEFAULT_ARCH_SCOPE = "aiv.c310"
SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")
_POINTER_ABI_SIZE = 8
_DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT = b"TLA_PRNT"
_DEBUG_PRINT_WORKSPACE_SENTINEL = int.from_bytes(
    _DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
_PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT = b"TLA_TPRN"
_PRINT_TENSOR_WORKSPACE_SENTINEL = int.from_bytes(
    _PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
# Cache compatibility tokens; change only when the corresponding ABI changes.
_DEBUG_PRINT_WORKSPACE_ABI_REVISION = "debug-print-workspace-i64-v1"
_PRINT_TENSOR_WORKSPACE_ABI_REVISION = (
    "print-tensor-workspace-i64-dynamic-shape-typed-multirecord-mixed-subblock"
)
_PRINT_TENSOR_HELPER_ABI_MARKER = b"__tla_print_tensor_abi"
_PRINT_TENSOR_MAX_BLOCKS = 1 << 16  # uint16_t block_idx
_PRINT_TENSOR_CORE_RECORDS = 108  # RuntimeWrapper::kDebugCoreRecords
_PRINT_TENSOR_FIFO_BYTES = 1 << 20  # kRingBufferBytes
_PRINT_TENSOR_SHAPE_HEADER_BYTES = 48  # sizeof(PrintShapeTlv)
_PRINT_TENSOR_HEADER_BYTES = 72  # sizeof(PrintTensorTlv)
_PRINT_TENSOR_PAYLOAD_ALIGNMENT = 32  # tensor payload alignment
_PRINT_TENSOR_MAX_F32_ELEMENTS = 262_112
_NATIVE_PRINT_TENSOR_RECORD_RE = re.compile(
    r"^DumpTensor: call=(?P<call>\d+), block=(?P<block>\d+), "
    r"(?:subblock=(?P<subblock>[01]), )?"
    r"data_type=(?P<dtype>[A-Za-z0-9_]+), "
    r"position=(?P<position>[A-Za-z0-9_]+), "
    r"shape=\[(?P<shape>[\d,\s]+)\] dump_size=(?P<count>\d+) "
    r"\[(?P<values>[^\]]*)\]$"
)
_FLOAT_LITERAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)$",
    re.IGNORECASE,
)
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")
_HIVM_TEMPLATE_BITCODE_ATTRS = {
    "meta_op.aic.c310.bc": ("hivm.aic_bitcode", "hivm.aic_bitcode"),
    "meta_op.aiv.c310.bc": ("hivm.aiv_bitcode", "hivm.aiv_bitcode"),
}
_MAX_KERNEL_ABI_PAYLOAD_SIZE = 1 << 20
_ONLINE_CACHE_ABI_VERSION = 5


class TlaExecutionError(RuntimeError):
    """Base class for execution and toolchain failures."""


class TlaCompilerBridgeUnavailableError(TlaExecutionError):
    """Raised when the typed Tla compiler bridge cannot be resolved."""


class TlaBackendCompilerNotFoundError(TlaExecutionError):
    """Raised when the backend compiler cannot be resolved."""


class TlaKernelCompileError(TlaExecutionError):
    """Raised when kernel lowering or backend compilation fails."""

    def __init__(self, message: str, *, pass_ir_dump: str = "") -> None:
        super().__init__(message)
        self.pass_ir_dump = pass_ir_dump


class TlaRuntimeUnavailableError(TlaExecutionError):
    """Raised when Ascend runtime dependencies are unavailable."""


class TlaUnsupportedAbiError(TlaExecutionError):
    """Raised when attempting to execute an unsupported ABI shape."""


@dataclass(frozen=True)
class _PrintTensorDTypeSpec:
    mlir_token: str
    native_name: str
    numeric_kind: str
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class _PrintTensorMetadata:
    shape: tuple[int, ...]
    count: int | None
    dtype: str
    position: str
    call: int = 0


@dataclass(frozen=True)
class _NativePrintTensorRecord:
    native_dtype: str
    position: str
    declared_count: int
    values_text: str
    shape: tuple[int, ...]
    call: int
    block: int
    subblock: int | None


_PRINT_TENSOR_DTYPES = {
    "f16": _PrintTensorDTypeSpec("f16", "float16", "float"),
    "f32": _PrintTensorDTypeSpec("f32", "float32", "float"),
    "i8": _PrintTensorDTypeSpec("i8", "int8", "integer", -128, 127),
    "i16": _PrintTensorDTypeSpec("i16", "int16", "integer", -32768, 32767),
    "i32": _PrintTensorDTypeSpec("i32", "int32", "integer", -2147483648, 2147483647),
    "u8": _PrintTensorDTypeSpec("ui8", "uint8", "integer", 0, 255),
    "u16": _PrintTensorDTypeSpec("ui16", "uint16", "integer", 0, 65535),
    "u32": _PrintTensorDTypeSpec("ui32", "uint32", "integer", 0, 4294967295),
}
_PRINT_TENSOR_DTYPES_BY_MLIR = {
    spec.mlir_token: token
    for token, spec in _PRINT_TENSOR_DTYPES.items()
}
_PRINT_TENSOR_DTYPES_BY_NATIVE = {
    spec.native_name: token
    for token, spec in _PRINT_TENSOR_DTYPES.items()
}


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
    kernel_abi: KernelAbiLayout | None = None


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


@dataclass(frozen=True)
class _KernelLaunchPlan:
    entrypoint: str
    kernel_mode: str
    grid: tuple[int, int, int]
    payload: bytes
    expects_debug_fifo: bool
    # 1 moves the workspace to arg 0 for pure kernels; 2 replaces the
    # trailing marker in place for mixed split kernels.
    expects_print_tensor: bool | int


@dataclass(frozen=True)
class _LogicalMixedHandoff:
    entrypoint: str
    user_arg_types: tuple[str, ...]


_MEMORY_COMPILE_CACHE_LOCK = threading.RLock()
_MEMORY_COMPILE_CACHE: dict[str, TlaKernelArtifact] = {}
_NATIVE_PRINT_TENSOR_STDOUT_LOCK = threading.RLock()


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
        cached_kernel_abi = _kernel_abi_from_manifest(cached)
        kernel_path = artifact_dir / str(cached["kernel_binary"])
        mlir_path = artifact_dir / str(cached["lowered_mlir"])
        cached_pass_dump = cached.get("pass_ir_dump")
        pass_dump_path = (
            artifact_dir / str(cached_pass_dump) if cached_pass_dump else None
        )
        if (
            _cache_manifest_has_current_workspace_abis(cached)
            and kernel_path.exists()
            and mlir_path.exists()
        ):
            lowered_llvm = mlir_path.read_text()
            artifact = TlaKernelArtifact(
                cache_key=cache_key,
                cache_dir=artifact_dir,
                tlair_mlir=tlair_mlir,
                lowered_llvm=lowered_llvm,
                entrypoint=entrypoint,
                compiler_bridge_path=compiler_bridge_path,
                hivmc_path=hivmc,
                kernel_binary_path=kernel_path,
                runtime=_runtime_options_from_lowered_mlir(runtime, lowered_llvm),
                pass_ir_dump=pass_dump_path.read_text()
                if pass_dump_path and pass_dump_path.exists()
                else "",
                kernel_abi=cached_kernel_abi,
            )
            _set_memory_cached_artifact(artifact)
            return artifact

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = artifact_dir / "lowered.mlir"
    pass_dump_path = artifact_dir / "pass-ir-dump.mlir"
    kernel_path = artifact_dir / "kernel.o"

    try:
        lowering_result = _run_tla_lowering_to_mlir(
            lowered_module=lowered.module,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            runtime=runtime,
        )
    except TlaKernelCompileError as exc:
        if exc.pass_ir_dump:
            pass_dump_path.write_text(exc.pass_ir_dump)
            raise TlaKernelCompileError(
                f"{exc}\npass IR dump: {pass_dump_path}",
                pass_ir_dump=exc.pass_ir_dump,
            ) from exc
        raise
    if lowering_result.pass_ir_dump:
        pass_dump_path.write_text(lowering_result.pass_ir_dump)
    runtime_for_hivmc = _runtime_options_from_lowered_mlir(
        runtime, mlir_path.read_text()
    )
    hivmc_mlir_path, template_bitcode = _create_stamped_hivmc_input(
        mlir_path, runtime_for_hivmc
    )
    try:
        _run_checked(
            _build_hivmc_a5_command(
                compiler=hivmc,
                mlir_path=hivmc_mlir_path,
                kernel_path=kernel_path,
                runtime=runtime_for_hivmc,
                template_bitcode=template_bitcode,
            ),
            label="hivmc-a5",
            cwd=artifact_dir,
        )
    finally:
        if hivmc_mlir_path != mlir_path:
            hivmc_mlir_path.unlink(missing_ok=True)
    if not kernel_path.exists():
        raise TlaKernelCompileError(
            "hivmc-a5 completed but output kernel artifact was not "
            f"created at {kernel_path}"
        )

    manifest.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "debug_print_workspace_abi_revision": (
                    _DEBUG_PRINT_WORKSPACE_ABI_REVISION
                ),
                "print_tensor_workspace_abi_revision": (
                    _PRINT_TENSOR_WORKSPACE_ABI_REVISION
                ),
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
                "kernel_abi": kernel_abi_to_dict(
                    getattr(lowering_result, "kernel_abi", None)
                ),
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
        kernel_abi=getattr(lowering_result, "kernel_abi", None),
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
    grid = _normalize_launch_grid(launch_kwargs.get("grid", (1, 1, 1)))
    device, stream = _resolve_launch_context(loader, launch_kwargs)
    if launch_args:
        _mark_tensor_launch_args_uploaded(launch_args)
    plan = _build_kernel_launch_plan(
        artifact=artifact,
        runtime=runtime,
        launch_args=launch_args,
        grid=grid,
    )
    print_metadata = (
        _print_tensor_static_metadata_records(
            artifact.tlair_mlir, entrypoint=artifact.entrypoint
        )
        if plan.expects_print_tensor
        else None
    )

    module_handle, function_handle = loader.load_binary(
        name=f"{plan.entrypoint} {plan.kernel_mode}",
        kernel_path=artifact.kernel_binary_path,
        shared=runtime.shared,
        device=device,
    )

    def launch() -> None:
        loader.launch_with_args(
            function=function_handle,
            stream=int(stream),
            grid_x=int(plan.grid[0]),
            grid_y=int(plan.grid[1]),
            grid_z=int(plan.grid[2]),
            args=plan.payload,
            expects_debug_fifo=plan.expects_debug_fifo,
            expects_print_tensor=plan.expects_print_tensor,
        )

    if print_metadata is None:
        launch()
    else:
        native_output = _capture_c_stdout(launch)
        print_block_count = _checked_print_tensor_block_count(plan.grid)
        helper_core = (
            _mixed_print_tensor_helper_core(artifact.lowered_llvm)
            if plan.kernel_mode == "mix"
            else runtime.core_type
        )
        expected_subblocks: tuple[int | None, ...] = (
            (0, 1)
            if helper_core == "aiv" and plan.kernel_mode == "mix"
            else (0,)
            if helper_core == "aiv"
            else (None,)
        )
        decoded = _decode_native_print_tensor_records(
            native_output,
            metadata=print_metadata,
            block_count=print_block_count,
            expected_subblocks=expected_subblocks,
        )
        preserve_legacy_format = len(print_metadata) * print_block_count == 1
        for metadata, record, values in decoded:
            print(
                _format_print_tensor_record(
                    values,
                    shape=record.shape,
                    dtype=metadata.dtype,
                    call=None if preserve_legacy_format else metadata.call,
                    block=record.block,
                    position=(metadata.position if plan.kernel_mode == "mix" else None),
                    subblock=record.subblock,
                )
            )
    return TlaExecutionResult(
        artifact=artifact,
        module_handle=module_handle,
        function_handle=function_handle,
        device=device,
    )


def _normalize_launch_grid(grid: Any) -> tuple[int, int, int]:
    if isinstance(grid, int):
        return (int(grid), 1, 1)
    if len(grid) != 3:
        raise TlaUnsupportedAbiError("`grid` must be an int or a 3-tuple.")
    return tuple(int(item) for item in grid)


def _checked_print_tensor_block_count(grid: tuple[int, int, int]) -> int:
    block_count = math.prod(grid)
    if any(extent <= 0 for extent in grid) or block_count > _PRINT_TENSOR_MAX_BLOCKS:
        raise TlaExecutionError(
            "tla.print_tensor 16-bit block identity requires a positive "
            f"grid with at most {_PRINT_TENSOR_MAX_BLOCKS} blocks; "
            f"got {grid}"
        )
    return block_count


def _print_tensor_native_wire_bytes(metadata: _PrintTensorMetadata) -> int:
    count = _PRINT_TENSOR_MAX_F32_ELEMENTS if metadata.count is None else metadata.count
    element_bytes = dtype_size_bytes(metadata.dtype)
    if element_bytes <= 0:
        raise TlaExecutionError(
            "tla.print_tensor FIFO capacity calculation encountered "
            f"unsupported dtype {metadata.dtype!r}"
        )
    payload_bytes = count * element_bytes
    aligned_payload_bytes = (
        payload_bytes + _PRINT_TENSOR_PAYLOAD_ALIGNMENT - 1
    ) // _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    return (
        _PRINT_TENSOR_SHAPE_HEADER_BYTES
        + _PRINT_TENSOR_HEADER_BYTES
        + aligned_payload_bytes * _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    )


def _validate_print_tensor_fifo_capacity(
    artifact: TlaKernelArtifact,
    grid: tuple[int, int, int],
    *,
    helper_core: str,
    mixed: bool,
) -> None:
    block_count = _checked_print_tensor_block_count(grid)
    records_per_block = 2 if helper_core == "aiv" and mixed else 1
    core_records = block_count * records_per_block
    if core_records > _PRINT_TENSOR_CORE_RECORDS:
        raise TlaExecutionError(
            "tla.print_tensor fixed 1 MiB FIFO workspace supports at most "
            f"{_PRINT_TENSOR_CORE_RECORDS} core records; got {core_records} "
            f"from {block_count} blocks"
        )
    metadata = _print_tensor_static_metadata_records(
        artifact.tlair_mlir, entrypoint=artifact.entrypoint
    )
    record_bytes = [_print_tensor_native_wire_bytes(item) for item in metadata]
    per_block_bytes = sum(record_bytes)
    if per_block_bytes > _PRINT_TENSOR_FIFO_BYTES:
        raise TlaExecutionError(
            "tla.print_tensor launch exceeds the conservative per-core FIFO capacity: "
            f"{per_block_bytes} bytes/block, capacity is "
            f"{_PRINT_TENSOR_FIFO_BYTES} bytes"
        )


def _resolve_launch_context(
    loader: "_AscendLoader", launch_kwargs: Mapping[str, Any]
) -> tuple[int, int]:
    device = int(launch_kwargs.get("device", loader.get_current_device()))
    stream = launch_kwargs.get("stream")
    if stream is None:
        stream = loader.get_current_stream(device)
    return device, int(stream)


def _build_kernel_launch_plan(
    *,
    artifact: TlaKernelArtifact,
    runtime: TlaRuntimeOptions,
    launch_args: Sequence[Any],
    grid: tuple[int, int, int],
) -> _KernelLaunchPlan:
    expects_print_tensor = _has_print_tensor_workspace(artifact)
    if expects_print_tensor:
        helper_core = (
            _mixed_print_tensor_helper_core(artifact.lowered_llvm)
            if runtime.kernel_mode == "mix"
            else runtime.core_type
        )
        _validate_print_tensor_fifo_capacity(
            artifact,
            grid,
            helper_core=helper_core,
            mixed=runtime.kernel_mode == "mix",
        )
    logical_mixed_handoff = _extract_logical_mixed_handoff(artifact.lowered_llvm)
    expected_abi_entrypoint = (
        logical_mixed_handoff.entrypoint
        if logical_mixed_handoff is not None and runtime.kernel_mode == "mix"
        else artifact.entrypoint
    )
    _validate_kernel_abi_layout(
        artifact.kernel_abi, expected_entrypoint=expected_abi_entrypoint
    )
    if logical_mixed_handoff is not None and runtime.kernel_mode == "mix":
        payload, effective_grid = _build_logical_mixed_handoff_launch_args(
            launch_args,
            grid,
            logical_mixed_handoff.user_arg_types,
            artifact.kernel_abi,
        )
        payload = _append_debug_print_workspace_payload(payload, artifact)
        if expects_print_tensor:
            extension = bytearray(payload)
            _align_payload(extension, _POINTER_ABI_SIZE)
            extension.extend(
                _PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                    _POINTER_ABI_SIZE, byteorder="little", signed=False
                )
            )
            payload = bytes(extension)
        return _KernelLaunchPlan(
            entrypoint=logical_mixed_handoff.entrypoint,
            kernel_mode="mix",
            grid=effective_grid,
            payload=payload,
            expects_debug_fifo=_has_debug_print_workspace(artifact),
            expects_print_tensor=2 if expects_print_tensor else False,
        )
    payload = _pack_launch_args(launch_args, artifact.kernel_abi)
    payload = _append_debug_print_workspace_payload(payload, artifact)
    if expects_print_tensor:
        extension = bytearray(payload)
        _align_payload(extension, _POINTER_ABI_SIZE)
        extension.extend(
            _PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                _POINTER_ABI_SIZE, byteorder="little", signed=False
            )
        )
        payload = bytes(extension)
    return _KernelLaunchPlan(
        entrypoint=artifact.entrypoint,
        kernel_mode=runtime.kernel_mode,
        grid=grid,
        payload=payload,
        expects_debug_fifo=_has_debug_print_workspace(artifact),
        expects_print_tensor=expects_print_tensor,
    )


def _mark_tensor_launch_args_uploaded(args: Sequence[Any]) -> None:
    for arg in args:
        if hasattr(arg, "prepare_for_launch") and callable(arg.prepare_for_launch):
            arg.prepare_for_launch()


def _runtime_arg_values(arg: Any) -> list[int]:
    try:
        if hasattr(arg, "__c_pointers__"):
            return [int(ptr) for ptr in arg.__c_pointers__()]
        data_ptr = getattr(arg, "data_ptr", None)
        if callable(data_ptr):
            return [int(data_ptr())]
        if data_ptr is not None:
            return [int(data_ptr)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TlaUnsupportedAbiError(
            "Launch argument cannot provide exactly one concrete host value."
        ) from exc
    raise TlaUnsupportedAbiError(
        "Launch arguments must provide exactly one host value."
    )


def _align_payload(payload: bytearray, alignment: int) -> None:
    payload.extend(b"\0" * (-len(payload) % alignment))


def _scalar_storage_size(descriptor: KernelAbiScalarDescriptor) -> int:
    if descriptor.category is KernelAbiScalarCategory.INDEX:
        return 8
    return max(1, descriptor.bit_width // 8)


def _pack_scalar_argument(
    value: Any,
    descriptor: KernelAbiScalarDescriptor,
    mlir_type: str,
    storage_size: int,
) -> bytes:
    expected_size = _scalar_storage_size(descriptor)
    if storage_size != expected_size:
        raise TlaUnsupportedAbiError(
            f"kernel ABI scalar {mlir_type} declares unsupported storage size "
            f"{storage_size}; expected {expected_size}"
        )
    if not isinstance(value, Numeric):
        value_type = type(value)
        if (
            value_type is bool
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 1
        ):
            return bytes((int(value),))
        if (
            value_type is int
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 32
            and descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNLESS
        ):
            lower, upper = -(1 << 31), (1 << 31) - 1
            if value < lower or value > upper:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                )
            return value.to_bytes(storage_size, byteorder="little", signed=True)
        if (
            value_type is float
            and descriptor.category is KernelAbiScalarCategory.FLOAT
            and descriptor.float_format is not None
            and descriptor.float_format.value == "f32"
        ):
            try:
                return struct.pack("<f", value)
            except (OverflowError, struct.error) as exc:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                ) from exc
        raise TlaUnsupportedAbiError(
            f"plain Python {value_type.__name__} does not match kernel ABI type "
            f"{mlir_type}"
        )
    host_type = type(value)
    host_dtype = str(getattr(host_type, "dtype", "")).lower()
    host_is_float = bool(getattr(host_type, "is_float", False))
    type_matches = host_is_float == (
        descriptor.category is KernelAbiScalarCategory.FLOAT
    )
    if descriptor.category is KernelAbiScalarCategory.FLOAT:
        if descriptor.float_format is None:
            raise TlaUnsupportedAbiError(
                f"kernel ABI float scalar {mlir_type} has no format"
            )
        type_matches = type_matches and host_dtype == descriptor.float_format.value
    elif descriptor.category is KernelAbiScalarCategory.INDEX:
        type_matches = type_matches and host_dtype == "index"
    else:
        type_matches = (
            type_matches and int(getattr(host_type, "width", 0)) == descriptor.bit_width
        )
        if descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNED:
            type_matches = type_matches and bool(getattr(host_type, "signed", False))
        elif descriptor.integer_signedness is KernelAbiIntegerSignedness.UNSIGNED:
            type_matches = type_matches and not bool(getattr(host_type, "signed", True))
    if not type_matches:
        raise TlaUnsupportedAbiError(
            f"typed host scalar {host_dtype or type(value).__name__} does not match "
            f"kernel ABI type {mlir_type}"
        )
    if descriptor.category is not KernelAbiScalarCategory.FLOAT:
        host_value = getattr(value, "value", None)
        if not isinstance(host_value, (bool, int)):
            raise TlaUnsupportedAbiError(
                f"kernel ABI scalar {mlir_type} requires a concrete integer host value"
            )
        width = int(getattr(host_type, "width", storage_size * 8))
        if width == 1:
            lower, upper = 0, 1
        elif bool(getattr(host_type, "signed", False)):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        else:
            lower, upper = 0, (1 << width) - 1
        if int(host_value) < lower or int(host_value) > upper:
            raise TlaUnsupportedAbiError(
                f"scalar value for {mlir_type} does not fit in its declared type"
            )
    values = _runtime_arg_values(value)
    if len(values) != 1:
        raise TlaUnsupportedAbiError(
            "each logical launch argument must provide exactly one host value"
        )
    bits = int(values[0])
    if bits < 0 or bits >= (1 << (storage_size * 8)):
        raise TlaUnsupportedAbiError(
            f"scalar value for {mlir_type} does not fit in {storage_size} bytes"
        )
    return bits.to_bytes(storage_size, byteorder="little", signed=False)


def _logical_launch_arg_count(layout: KernelAbiLayout) -> int:
    if not layout.arguments:
        return 0
    return 1 + max(
        (
            argument.index
            if argument.logical_index is None
            else argument.logical_index
        )
        for argument in layout.arguments
    )


def _memref_launch_field_value(tensor: Any, field: str) -> int:
    builder = getattr(tensor, "build_memref_launch_fields", None)
    if not callable(builder):
        raise TlaUnsupportedAbiError(
            "dynamic GM memref launch requires Tensor.build_memref_launch_fields()"
        )
    fields = builder()
    if field not in fields:
        raise TlaUnsupportedAbiError(
            f"memref launch field {field!r} missing from tensor descriptor"
        )
    return int(fields[field])


def _pack_launch_args(
    args: Sequence[Any], layout: KernelAbiLayout | None
) -> bytes:
    _validate_kernel_abi_layout(layout)
    if layout is None:
        raise TlaUnsupportedAbiError("kernel ABI layout is missing")
    logical_count = _logical_launch_arg_count(layout)
    if len(args) != logical_count:
        raise TlaUnsupportedAbiError(
            "kernel launch argument count does not match ABI layout: "
            f"got {len(args)}, expected {logical_count}"
        )
    if layout.total_size < 0:
        raise TlaUnsupportedAbiError("kernel ABI layout has an invalid total size")
    host_offsets: list[int] = []
    host_size = 0
    for argument in layout.arguments:
        host_alignment = 8 if argument.storage_size == 8 else 4
        host_size = (
            (host_size + host_alignment - 1) // host_alignment
        ) * host_alignment
        host_offsets.append(host_size)
        host_size += argument.storage_size
    host_size = ((host_size + 7) // 8) * 8
    if host_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI host payload exceeds the supported maximum size"
        )
    payload = bytearray(host_size)
    for index, argument in enumerate(layout.arguments):
        if argument.index != index:
            raise TlaUnsupportedAbiError(
                "kernel ABI arguments must be ordered by contiguous index"
            )
        logical_index = (
            argument.index if argument.logical_index is None else argument.logical_index
        )
        if logical_index < 0 or logical_index >= len(args):
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} logical_index {logical_index} is out of range"
            )
        value = args[logical_index]
        start = host_offsets[index]
        end = start + argument.storage_size
        if start < 0 or argument.storage_size <= 0 or end < start or end > host_size:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage does not fit in payload"
            )
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if isinstance(value, Numeric) or type(value) in (bool, int, float):
                raise TlaUnsupportedAbiError(
                    f"kernel ABI argument {index} requires a pointer"
                )
            values = _runtime_arg_values(value)
            if len(values) != 1:
                raise TlaUnsupportedAbiError(
                    "each logical launch argument must provide exactly one host value"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
            pointer = int(values[0])
            if pointer < 0 or pointer >= (1 << (argument.storage_size * 8)):
                raise TlaUnsupportedAbiError(
                    f"pointer value does not fit in {argument.storage_size} bytes"
                )
            encoded = pointer.to_bytes(
                argument.storage_size, byteorder="little", signed=False
            )
        elif argument.kind is KernelAbiArgumentKind.MEMREF_FIELD:
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} has no field name"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported memref_field storage size {argument.storage_size}"
                )
            field_value = _memref_launch_field_value(value, argument.field)
            if field_value < 0 or field_value >= (1 << (argument.storage_size * 8)):
                raise TlaUnsupportedAbiError(
                    f"memref field {argument.field!r} does not fit in "
                    f"{argument.storage_size} bytes"
                )
            encoded = field_value.to_bytes(
                argument.storage_size, byteorder="little", signed=False
            )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {index} has no scalar descriptor"
                )
            encoded = _pack_scalar_argument(
                value,
                argument.scalar,
                argument.mlir_type,
                argument.storage_size,
            )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        payload[start:end] = encoded
    return bytes(payload)


def _append_debug_print_workspace_payload(
    payload: bytes, artifact: TlaKernelArtifact
) -> bytes:
    if not _has_debug_print_workspace(artifact):
        return payload
    extension = bytearray(payload)
    _align_payload(extension, _POINTER_ABI_SIZE)
    extension.extend(
        _DEBUG_PRINT_WORKSPACE_SENTINEL.to_bytes(
            _POINTER_ABI_SIZE, byteorder="little", signed=False
        )
    )
    return bytes(extension)


def _validate_kernel_abi_layout(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str | None = None
) -> None:
    if layout is None or layout.schema_version not in (3, 4):
        raise TlaUnsupportedAbiError(
            "A supported compiler-produced kernel ABI layout is required before launch."
        )
    if not layout.entrypoint.strip():
        raise TlaUnsupportedAbiError("kernel ABI layout entrypoint must be nonempty")
    if expected_entrypoint is not None and layout.entrypoint != expected_entrypoint:
        raise TlaUnsupportedAbiError(
            "kernel ABI layout entrypoint does not match launch artifact: "
            f"{layout.entrypoint!r} != {expected_entrypoint!r}"
        )
    if layout.total_size < 0:
        raise TlaUnsupportedAbiError("kernel ABI layout has an invalid total size")
    if layout.total_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI payload exceeds the supported maximum size"
        )
    if layout.total_size % 8 != 0:
        raise TlaUnsupportedAbiError("kernel ABI total size must be rounded to 8 bytes")
    previous_end = 0
    for index, argument in enumerate(layout.arguments):
        if argument.index != index:
            raise TlaUnsupportedAbiError(
                "kernel ABI arguments must be ordered by contiguous index"
            )
        if argument.alignment != 4:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} must declare 4-byte alignment"
            )
        if argument.offset < previous_end or argument.offset % 4 != 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} offset must be ordered, "
                "non-overlapping, and 4-byte aligned"
            )
        if argument.storage_size <= 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage size must be positive"
            )
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI pointer argument {index} cannot have a scalar descriptor"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.MEMREF_FIELD:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} cannot have a scalar descriptor"
                )
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} requires a field name"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported memref_field storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {index} requires a scalar descriptor"
                )
            expected_size = _scalar_storage_size(argument.scalar)
            if argument.storage_size != expected_size:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar {argument.mlir_type} declares unsupported "
                    f"storage size {argument.storage_size}; expected {expected_size}"
                )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        previous_end = argument.offset + argument.storage_size
        if previous_end > layout.total_size:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage does not fit in payload"
            )
    required_size = ((previous_end + 7) // 8) * 8
    if layout.total_size != required_size:
        raise TlaUnsupportedAbiError(
            "kernel ABI total size is not exactly sufficient for its arguments"
        )


def _split_top_level_csv(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    brace_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif char == "(" and angle_depth == 0:
            paren_depth += 1
        elif char == ")" and angle_depth == 0 and paren_depth > 0:
            paren_depth -= 1
        elif char == "{" and angle_depth == 0 and paren_depth == 0:
            brace_depth += 1
        elif char == "}" and angle_depth == 0 and paren_depth == 0 and brace_depth > 0:
            brace_depth -= 1
        elif char == "," and angle_depth == 0 and paren_depth == 0 and brace_depth == 0:
            item = text[start:index].strip()
            if item:
                result.append(item)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def _find_matching_function_type_paren(text: str, start: int) -> int:
    angle_depth = 0
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif char == "(" and angle_depth == 0:
            depth += 1
        elif char == ")" and angle_depth == 0:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _extract_named_func_args(mlir_text: str, func_name: str) -> list[str]:
    match = re.search(
        rf"(?:func\.func|\"func\.func\")\s+(?:private\s+)?@{re.escape(func_name)}\s*\(",
        mlir_text,
    )
    if not match:
        return []
    start = match.end() - 1
    end = _find_matching_function_type_paren(mlir_text, start)
    if end <= start:
        return []
    params = _split_top_level_csv(mlir_text[start + 1 : end])
    return params


def _mixed_handoff_user_arg_types(params: Sequence[str]) -> tuple[str, ...]:
    """Exclude compiler-owned print workspace operands from a mixed handoff.

    The split functions carry the same user ABI plus an internal workspace
    when scalar debug or tensor printing is present. That workspace is
    supplied by a marker-backed FIFO descriptor, never by a public launch
    argument.
    """

    user_arg_types: list[str] = []
    for param in params:
        if (
            "tla.debug_print.workspace" in param
            or "tla.print_tensor.workspace" in param
        ):
            continue
        if ":" in param:
            user_arg_types.append(param.split(":", 1)[1].strip())
        else:
            user_arg_types.append(param.strip())
    return tuple(user_arg_types)


def _extract_mixed_handoff_entrypoints(mlir_text: str) -> tuple[str, str] | None:
    names = re.findall(
        r"(?:func\.func|\"func\.func\")\s+@([A-Za-z_][A-Za-z0-9_]*)",
        mlir_text,
    )
    aic_names = [name for name in names if name.endswith("_mix_aic")]
    aiv_names = [name for name in names if name.endswith("_mix_aiv")]
    if len(aic_names) != 1 or len(aiv_names) != 1:
        return None
    return aic_names[0], aiv_names[0]


def _extract_logical_mixed_handoff(mlir_text: str) -> _LogicalMixedHandoff | None:
    entrypoints = _extract_mixed_handoff_entrypoints(mlir_text)
    if entrypoints is None:
        return None
    aic_name, aiv_name = entrypoints
    base_name = aic_name.removesuffix("_mix_aic")
    if aiv_name.removesuffix("_mix_aiv") != base_name:
        return None
    params = _extract_named_func_args(mlir_text, aic_name)
    if not params:
        raise TlaUnsupportedAbiError(
            "mixed handoff AIC split function must expose argument types"
        )
    return _LogicalMixedHandoff(base_name, _mixed_handoff_user_arg_types(params))


def _extract_logical_mixed_handoff_entrypoint(mlir_text: str) -> str | None:
    handoff = _extract_logical_mixed_handoff(mlir_text)
    return None if handoff is None else handoff.entrypoint


def _build_logical_mixed_handoff_launch_args(
    launch_args: Sequence[Any],
    grid: Sequence[int],
    arg_types: Sequence[str],
    kernel_abi: KernelAbiLayout | None,
) -> tuple[bytes, tuple[int, int, int]]:
    # Host still passes one object per logical kernel argument. Dynamic GM expands
    # each Tensor into many device params / memref_field slots in the ABI, so do
    # not compare against the raw split-function parameter count.
    expected = (
        _logical_launch_arg_count(kernel_abi)
        if kernel_abi is not None
        else len(arg_types)
    )
    if len(launch_args) != expected:
        raise TlaUnsupportedAbiError(
            "mixed handoff launch argument count does not match ABI layout: "
            f"got {len(launch_args)}, expected {expected}"
        )
    return _pack_launch_args(launch_args, kernel_abi), tuple(int(item) for item in grid)


def _has_debug_print_workspace(artifact: TlaKernelArtifact) -> bool:
    mlir_text = artifact.lowered_llvm or artifact.tlair_mlir
    return "tla.debug_print.workspace" in mlir_text


def _has_print_tensor_workspace(artifact: TlaKernelArtifact) -> bool:
    mlir_text = artifact.lowered_llvm or artifact.tlair_mlir
    return "tla.print_tensor.workspace" in mlir_text


def _capture_c_stdout(launch: Callable[[], None]) -> str:
    libc = ctypes.CDLL(None)
    with _NATIVE_PRINT_TENSOR_STDOUT_LOCK:
        sys.stdout.flush()
        libc.fflush(None)
        saved_stdout = os.dup(1)
        try:
            with tempfile.TemporaryFile(mode="w+b") as captured:
                os.dup2(captured.fileno(), 1)
                try:
                    launch()
                finally:
                    libc.fflush(None)
                    sys.stdout.flush()
                    os.dup2(saved_stdout, 1)
                captured.seek(0)
                return captured.read().decode("utf-8", errors="replace")
        finally:
            os.close(saved_stdout)


def _print_tensor_static_metadata_records(
    tlair_mlir: str, *, entrypoint: str | None = None
) -> tuple[_PrintTensorMetadata, ...]:
    function_text = (
        _mlir_entrypoint_text(tlair_mlir, entrypoint)
        if entrypoint is not None
        else tlair_mlir
    )
    op_matches = list(re.finditer(r'"tla\.print_tensor"', function_text))
    if not op_matches:
        raise TlaExecutionError(
            "tla.print_tensor requires static shape metadata; found 0 records"
        )
    constants = {
        match.group("result"): int(match.group("value"))
        for match in re.finditer(
            r'(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*"arith\.constant"\(\)'
            r"\s*<\{\s*value\s*=\s*(?P<value>\d+)\s*:\s*i64\s*\}>",
            function_text,
        )
    }
    op_ends = [match.start() for match in op_matches[1:]]
    op_ends.append(len(function_text))
    metadata = []
    for call, (match, end) in enumerate(zip(op_matches, op_ends)):
        op_text = function_text[match.start() : end]
        operand_match = re.match(
            r'"tla\.print_tensor"\(\s*[^,]+,\s*'
            r"(?P<length>%[A-Za-z0-9_.$-]+)\s*\)",
            op_text,
        )
        metadata.append(
            _parse_print_tensor_static_metadata(
                op_text,
                call=call,
                length=(
                    constants.get(operand_match.group("length"))
                    if operand_match is not None
                    else None
                ),
            )
        )
    return tuple(metadata)


def _mlir_entrypoint_text(mlir_text: str, entrypoint: str) -> str:
    function_matches = list(
        re.finditer(
            r'"(?:tla|func)\.func"\s*\('
            r"|(?:tla|func)\.func\s+@(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b",
            mlir_text,
        )
    )
    for index, match in enumerate(function_matches):
        end = (
            function_matches[index + 1].start()
            if index + 1 < len(function_matches)
            else len(mlir_text)
        )
        function_text = mlir_text[match.start() : end]
        name = match.group("name")
        if name is None:
            name_match = re.search(r'\bsym_name\s*=\s*"([^"]+)"', function_text)
            name = name_match.group(1) if name_match is not None else None
        if name == entrypoint:
            return function_text
    raise TlaExecutionError(
        f"tla.print_tensor entrypoint {entrypoint!r} is missing from the compiled artifact"
    )


def _parse_print_tensor_static_metadata(
    op_text: str, *, call: int, length: int | None = None
) -> _PrintTensorMetadata:
    shape_match = re.search(r"\bshape\s*=\s*array<i64:\s*([-\d,\s]+)>", op_text)
    if shape_match is None:
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata is missing from the compiled artifact"
        )
    operand_match = re.search(r"!tla\.ptr<\s*([^,\s>]+)\s*,\s*(gm|ub)\s*,", op_text)
    if operand_match is None:
        raise TlaExecutionError(
            "tla.print_tensor operand type metadata is missing or malformed"
        )
    mlir_dtype = operand_match.group(1)
    position = operand_match.group(2).upper()
    shape = tuple(int(item.strip()) for item in shape_match.group(1).split(","))
    if len(shape) not in (1, 2) or any(extent == 0 or extent < -1 for extent in shape):
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata contains an invalid extent"
        )
    length_match = re.search(r"\blength\s*=\s*(\d+)", op_text)
    if length is None and length_match is not None:
        length = int(length_match.group(1))
    if length is not None and (
        not 1 <= length <= _PRINT_TENSOR_MAX_F32_ELEMENTS
        or (all(extent > 0 for extent in shape) and length > math.prod(shape))
    ):
        raise TlaExecutionError(
            "tla.print_tensor static length metadata is invalid for the tensor shape"
        )
    dtype = _PRINT_TENSOR_DTYPES_BY_MLIR.get(mlir_dtype)
    if dtype is None:
        raise TlaExecutionError(
            f"tla.print_tensor runtime metadata has unsupported dtype {mlir_dtype!r}; "
            "supported: f16, f32, i8, i16, i32, ui8, ui16, ui32"
        )
    return _PrintTensorMetadata(
        shape=shape,
        count=length,
        dtype=dtype,
        position=position,
        call=call,
    )


def _print_tensor_decode_error(detail: str) -> TlaExecutionError:
    return TlaExecutionError(
        f"tla.print_tensor native initialization or decoding failed: {detail}"
    )


def _parse_native_print_tensor_records(output: str) -> list[_NativePrintTensorRecord]:
    records: list[_NativePrintTensorRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("DumpTensor:"):
            continue
        match = _NATIVE_PRINT_TENSOR_RECORD_RE.fullmatch(line)
        if match is None:
            raise _print_tensor_decode_error("malformed native record")
        shape = tuple(int(item.strip()) for item in match.group("shape").split(","))
        records.append(
            _NativePrintTensorRecord(
                native_dtype=match.group("dtype"),
                position=match.group("position"),
                declared_count=int(match.group("count")),
                values_text=match.group("values"),
                shape=shape,
                call=int(match.group("call")),
                block=int(match.group("block")),
                subblock=(
                    int(match.group("subblock"))
                    if match.group("subblock") is not None
                    else None
                ),
            )
        )
    return records


def _print_tensor_integer_bounds(
    dtype: str, spec: _PrintTensorDTypeSpec
) -> tuple[int, int]:
    if spec.minimum is None or spec.maximum is None:
        raise TlaExecutionError(
            f"tla.print tensor dtype metadata is missing integer bounds (dtype={dtype})"
        )
    return spec.minimum, spec.maximum


def _decode_print_tensor_record_values(
    record: _NativePrintTensorRecord, *, dtype: str
) -> list[float | int]:
    spec = _PRINT_TENSOR_DTYPES[dtype]
    items = (
        [item.strip() for item in record.values_text.split(",")]
        if record.values_text.strip()
        else []
    )
    if len(items) != record.declared_count:
        raise _print_tensor_decode_error(
            f"payload count is {len(items)}, record declares {record.declared_count}",
        )
    if spec.numeric_kind == "float":
        if any(_FLOAT_LITERAL_RE.fullmatch(item) is None for item in items):
            raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
        return [float(item) for item in items]
    if any(_INTEGER_LITERAL_RE.fullmatch(item) is None for item in items):
        raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
    values = [int(item, 10) for item in items]
    minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
    if any(value < minimum or value > maximum for value in values):
        raise _print_tensor_decode_error(
            f"{dtype} integer value outside [{minimum}, {maximum}]"
        )
    return values


def _decode_native_print_tensor_records(
    output: str,
    *,
    metadata: Sequence[_PrintTensorMetadata],
    block_count: int = 1,
    expected_subblocks: tuple[int | None, ...] = (None,),
) -> list[tuple[_PrintTensorMetadata, _NativePrintTensorRecord, list[float | int]]]:
    records = _parse_native_print_tensor_records(output)
    expected_metadata = {item.call: item for item in metadata}
    expected = {
        (item.call, block, subblock)
        for item in metadata
        for block in range(block_count)
        for subblock in expected_subblocks
    }
    record_values: dict[int, list[float | int]] = {}
    malformed_identities: set[tuple[int, int, int | None]] = set()
    for record in records:
        identity = (record.call, record.block, record.subblock)
        metadata_item = expected_metadata.get(record.call)
        dtype = (
            metadata_item.dtype
            if metadata_item is not None
            else _PRINT_TENSOR_DTYPES_BY_NATIVE.get(record.native_dtype)
        )
        if dtype is None:
            malformed_identities.add(identity)
            continue
        try:
            record_values[id(record)] = _decode_print_tensor_record_values(
                record, dtype=dtype
            )
        except TlaExecutionError:
            malformed_identities.add(identity)
    if malformed_identities:
        raise _print_tensor_decode_error(
            f"malformed records {_sort_print_tensor_identities(malformed_identities)}"
        )

    identities = [(record.call, record.block, record.subblock) for record in records]
    unexpected_identities = _sort_print_tensor_identities(set(identities) - expected)
    if unexpected_identities:
        raise _print_tensor_decode_error(
            f"unexpected call/block/subblock identities {unexpected_identities}",
        )

    mismatches: dict[str, set[tuple[int, int, int | None]]] = {
        "native dtype": set(),
        "position": set(),
        "declared count": set(),
        "shape": set(),
    }
    for record in records:
        identity = (record.call, record.block, record.subblock)
        item = expected_metadata[record.call]
        if record.native_dtype != _PRINT_TENSOR_DTYPES[item.dtype].native_name:
            mismatches["native dtype"].add(identity)
        if record.position != item.position:
            mismatches["position"].add(identity)
        if item.count is not None and record.declared_count != item.count:
            mismatches["declared count"].add(identity)
        if (
            len(record.shape) != len(item.shape)
            or any(extent < 1 for extent in record.shape)
            or math.prod(record.shape) < record.declared_count
            or any(
                expected != -1 and expected != actual
                for expected, actual in zip(item.shape, record.shape, strict=True)
            )
        ):
            mismatches["shape"].add(identity)
    for label, mismatch in mismatches.items():
        if mismatch:
            raise _print_tensor_decode_error(
                f"unexpected {label} records {_sort_print_tensor_identities(mismatch)}"
            )

    observed: set[tuple[int, int, int | None]] = set()
    duplicates = set()
    for identity in identities:
        if identity in observed:
            duplicates.add(identity)
        observed.add(identity)
    if duplicates:
        raise _print_tensor_decode_error(
            "duplicate call/block/subblock identities "
            f"{_sort_print_tensor_identities(duplicates)}"
        )
    missing = _sort_print_tensor_identities(expected - observed)
    if missing:
        raise _print_tensor_decode_error(
            f"missing call/block/subblock identities {missing}"
        )

    return [
        (expected_metadata[record.call], record, record_values[id(record)])
        for record in records
    ]


def _sort_print_tensor_identities(
    identities: Iterable[tuple[int, int, int | None]],
) -> list[tuple[int, int, int | None]]:
    return sorted(
        identities,
        key=lambda identity: (
            identity[0],
            identity[1],
            -1 if identity[2] is None else identity[2],
        ),
    )


def _format_print_tensor_record(
    values: Sequence[float | int],
    *,
    shape: Sequence[int],
    dtype: str = "f32",
    call: int | None = None,
    block: int | None = None,
    position: str | None = None,
    subblock: int | None = None,
) -> str:
    spec = _PRINT_TENSOR_DTYPES.get(dtype)
    if spec is None:
        raise TlaExecutionError(
            f"tla.print tensor formatting failed (dtype={dtype}): unsupported dtype"
        )
    count = len(values)
    rendered_shape = ",".join(str(int(extent)) for extent in shape)
    if spec.numeric_kind == "float":
        rendered_values = ", ".join(str(float(value)) for value in values)
    else:
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                "integer dtype requires integer values"
            )
        minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
        if any(value < minimum or value > maximum for value in values):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                f"integer value outside [{minimum}, {maximum}]"
            )
        rendered_values = ", ".join(str(value) for value in values)
    public_dtype = "float32" if dtype == "f32" else dtype
    identity = "" if call is None else f"call={call} block={block} "
    rendered_position = f"position={position} " if position is not None else ""
    rendered_subblock = f"subblock={subblock} " if subblock is not None else ""
    return (
        f"tla.print {identity}dtype={public_dtype} "
        f"{rendered_position}{rendered_subblock}"
        f"shape=[{rendered_shape}] count={count} "
        f"values=[{rendered_values}]"
    )


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
    if kernel_mode != "mix" and kernel_mode != core_type:
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


def _runtime_options_from_lowered_mlir(
    runtime: TlaRuntimeOptions, mlir_text: str
) -> TlaRuntimeOptions:
    target_arch = runtime.target_arch
    core_type = runtime.core_type
    kernel_mode = runtime.kernel_mode
    hivmc_args = runtime.hivmc_args

    if _extract_logical_mixed_handoff_entrypoint(mlir_text) is not None:
        core_type = "aic"
        kernel_mode = "mix"
    elif (
        "hivm.module_core_type<AIC>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIC>" in mlir_text
    ):
        core_type = "aic"
        kernel_mode = "aic"
    elif (
        "hivm.module_core_type<AIV>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir_text
    ):
        core_type = "aiv"
        kernel_mode = "aiv"

    if "dav-c310" in mlir_text or 'hacc.target<"Ascend950PR_9589">' in mlir_text:
        target_arch = "c310"

    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type=core_type)
    if (
        runtime.target_arch == target_arch
        and runtime.core_type == core_type
        and runtime.kernel_mode == kernel_mode
        and runtime.arch_scope == arch_scope
        and runtime.hivmc_args == hivmc_args
    ):
        return runtime
    return replace(
        runtime,
        target_arch=target_arch,
        core_type=core_type,
        kernel_mode=kernel_mode,
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
        "debug_print_workspace_abi_revision": _DEBUG_PRINT_WORKSPACE_ABI_REVISION,
        "print_tensor_workspace_abi_revision": _PRINT_TENSOR_WORKSPACE_ABI_REVISION,
        "cache_abi_version": _ONLINE_CACHE_ABI_VERSION,
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
    template_bitcode: str | None = None,
) -> list[str]:
    if template_bitcode is None:
        template_bitcode = _resolve_hivm_template_bitcode(runtime)
    command = [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
    ]
    if runtime.kernel_mode == "mix":
        command.extend(
            [
                "--disable-ffts",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_path),
            ]
        )
    else:
        command.extend(
            [
                "--disable-ffts",
                "--enable-hivm-compile=False",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_path),
            ]
        )
    command.extend(runtime.hivmc_args)
    return command


def _create_stamped_hivmc_input(
    mlir_path: Path, runtime: TlaRuntimeOptions
) -> tuple[Path, str | None]:
    """Stamp a private HIVMC input only when debug-print helpers are present."""
    compiler_text = mlir_path.read_text()
    if (
        "tla.debug_print.workspace" not in compiler_text
        and "tla.print_tensor.workspace" not in compiler_text
    ):
        return mlir_path, None
    compiler_input = mlir_path.with_name(f"{mlir_path.stem}.hivmc-input.mlir")
    compiler_input.write_text(mlir_path.read_text())
    template_bitcode = _resolve_hivm_template_bitcode(runtime)
    if "tla.print_tensor.workspace" in compiler_text:
        helper_core_type = (
            _mixed_print_tensor_helper_core(compiler_text)
            if runtime.kernel_mode == "mix"
            else runtime.core_type
        )
        helper = {
            "aic": ("Cube", "print_tensor.aic.c310.bc"),
            "aiv": ("Vector", "print_tensor.aiv.c310.bc"),
        }.get(helper_core_type)
        if helper is None:
            raise TlaRuntimeUnavailableError(
                "tla.print_tensor helper split could not be determined"
            )
        helper_dir, helper_name = helper
        probe_candidates = [
            build_dir / "bc" / helper_dir / helper_name
            for build_dir in _mlir_build_dirs()
        ]
        probe_bitcode = next(
            (path.resolve() for path in probe_candidates if path.exists()), None
        )
        if probe_bitcode is None:
            raise TlaRuntimeUnavailableError(
                f"C310 {helper_core_type} print tensor bitcode was not built"
            )
        helper_bytes = probe_bitcode.read_bytes()
        if _PRINT_TENSOR_HELPER_ABI_MARKER not in helper_bytes:
            raise TlaRuntimeUnavailableError(
                "C310 print tensor helper does not provide the required ABI marker"
            )
        template_bitcode = f"{template_bitcode},{probe_bitcode}"
    _stamp_hivm_template_bitcode_attrs(compiler_input, template_bitcode)
    return compiler_input, template_bitcode


def _mixed_print_tensor_helper_core(compiler_text: str) -> str:
    helper_call = re.search(r"\bcall\s+@_mlir_ciface_tla_print_tensor_", compiler_text)
    if helper_call is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is missing from lowered MLIR"
        )
    functions = list(
        re.finditer(
            r"func\.func(?:\s+private)?\s+@(?P<name>[^\s(]+)\s*\(",
            compiler_text,
        )
    )
    containing_function = [
        match
        for match in functions
        if match.start() < helper_call.start()
    ]
    if not containing_function:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper split could not be determined"
        )
    split = re.search(r"_mix_(aic|aiv)$", containing_function[-1].group("name"))
    if split is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is not inside a split function"
        )
    return split.group(1)


def _stamp_hivm_template_bitcode_attrs(mlir_path: Path, template_bitcode: str) -> None:
    text = mlir_path.read_text()
    additions: list[str] = []
    added_attr_names: set[str] = set()
    for raw_path in template_bitcode.split(","):
        path = Path(raw_path)
        attr = _HIVM_TEMPLATE_BITCODE_ATTRS.get(path.name)
        if attr is None or attr[0] in text or attr[0] in added_attr_names:
            continue
        added_attr_names.add(attr[0])
        additions.append(
            f"{attr[0]} = #hivm.{attr[1].removeprefix('hivm.')}<{json.dumps(str(path))}>"
        )
    if not additions:
        return
    add_text = ", ".join(additions)
    attributes_re = re.compile(
        r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?\s+attributes\s*\{)"
        r"(?P<body>.*?)"
        r"(?P<suffix>\}\s*\{)",
        re.DOTALL,
    )
    match = attributes_re.search(text)
    if match:
        body = match.group("body").strip()
        new_body = f"{add_text}, {body}" if body else add_text
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + new_body
            + match.group("suffix")
            + text[match.end() :]
        )
        return

    match = re.search(r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?)(?P<brace>\s*\{)", text)
    if match:
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + f" attributes {{{add_text}}}"
            + match.group("brace")
            + text[match.end() :]
        )


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
    if runtime.kernel_mode == "mix":
        repo_aic_candidates: list[Path] = []
        for build_dir in _mlir_build_dirs():
            repo_aic_candidates.append(build_dir / "bc" / "meta_op.aic.c310.bc")

        aiv_candidates: list[Path] = []
        for build_dir in _mlir_build_dirs():
            aiv_candidates.append(build_dir / "bc" / "meta_op.aiv.c310.bc")
        ascend_home = os.getenv("ASCEND_HOME_PATH")
        if ascend_home:
            cann_lib = Path(ascend_home) / "tools" / "bishengir" / "lib"
            aiv_candidates.append(cann_lib / "meta_op.aiv.c310.bc")
        repo_aic = next(
            (path.resolve() for path in repo_aic_candidates if path.exists()), None
        )
        aiv_bc = next(
            (path.resolve() for path in aiv_candidates if path.exists()), None
        )
        if repo_aic is not None and aiv_bc is not None:
            return f"{repo_aic},{aiv_bc}"
        raise TlaRuntimeUnavailableError(
            "C310 mixed HIVM template bitcode not found. Expected repo AIC bitcode "
            "and CANN AIV bitcode, or set `TLA_DSL_HIVM_TEMPLATE_BC`."
        )

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
        ascend_home = os.getenv("ASCEND_HOME_PATH")
        if ascend_home:
            candidates.append(
                Path(ascend_home)
                / "tools"
                / "bishengir"
                / "lib"
                / "meta_op.aiv.c310.bc"
            )
    existing = next((path.resolve() for path in candidates if path.exists()), None)
    if existing is not None:
        return str(existing)
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
    except BridgeLoweringError as exc:
        raise TlaKernelCompileError(
            f"In-process Tla compiler bridge failed.\nerror:\n{exc}",
            pass_ir_dump=exc.pass_ir_dump,
        ) from exc
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
    return os.environ.copy()


def _run_tla_compile_cli_to_mlir(
    *,
    tla_compile: Path,
    tlair_mlir: str,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    input_path = mlir_path.with_suffix(".tlair.mlir")
    input_path.write_text(tlair_mlir)
    cmd = [str(tla_compile), str(input_path), "-o", str(mlir_path)]
    print_requested = False
    if runtime is not None:
        for pass_name in runtime.mlir_print_ir_before:
            cmd.append(f"--mlir-print-ir-before={pass_name}")
        for pass_name in runtime.mlir_print_ir_after:
            cmd.append(f"--mlir-print-ir-after={pass_name}")
        if runtime.mlir_print_ir_before_all:
            cmd.append("--mlir-print-ir-before-all")
        if runtime.mlir_print_ir_after_all:
            cmd.append("--mlir-print-ir-after-all")
        print_requested = bool(
            runtime.mlir_print_ir_before
            or runtime.mlir_print_ir_after
            or runtime.mlir_print_ir_before_all
            or runtime.mlir_print_ir_after_all
        )
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=_tla_compile_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stderr_message = "<captured in pass IR dump>" if print_requested else stderr
        raise TlaKernelCompileError(
            f"TlaCompile CLI fallback failed with exit code {exc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{stderr_message}",
            pass_ir_dump=stderr if print_requested else "",
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
            "pass_ir_dump": completed.stderr if print_requested else "",
        },
    )()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise TlaExecutionError(f"Invalid cache manifest at {path}: {exc}") from exc


def _cache_manifest_has_current_debug_print_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("debug_print_workspace_abi_revision")
        == _DEBUG_PRINT_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_print_tensor_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("print_tensor_workspace_abi_revision")
        == _PRINT_TENSOR_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_workspace_abis(
    manifest: Mapping[str, Any],
) -> bool:
    return _cache_manifest_has_current_debug_print_workspace_abi(
        manifest
    ) and _cache_manifest_has_current_print_tensor_workspace_abi(manifest)


def _kernel_abi_from_manifest(manifest: Mapping[str, Any]) -> KernelAbiLayout:
    if "kernel_abi" not in manifest or manifest["kernel_abi"] is None:
        raise TlaKernelCompileError(
            "Cached artifact has no compiler-produced kernel_abi descriptor; "
            "the cache predates the current kernel ABI contract."
        )
    try:
        layout = kernel_abi_from_dict(manifest["kernel_abi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    if layout is None:
        raise TlaKernelCompileError(
            "Invalid cached kernel ABI descriptor: decoded to no layout"
        )
    try:
        _validate_kernel_abi_layout(
            layout, expected_entrypoint=str(manifest.get("entrypoint", ""))
        )
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    return layout


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

static PyObject *launchWithArgs(PyObject *, PyObject *args) {
  unsigned long long function_u64;
  unsigned long long stream_u64;
  int gx, gy, gz;
  const char *arg_data = nullptr;
  Py_ssize_t arg_size = 0;
  if (!PyArg_ParseTuple(args, "KKiiiy#", &function_u64, &stream_u64, &gx, &gy, &gz, &arg_data, &arg_size)) {
    return NULL;
  }

  const void *function = reinterpret_cast<const void *>(function_u64);
  rtStream_t stream = reinterpret_cast<rtStream_t>(stream_u64);
  uint32_t block_num = static_cast<uint32_t>(gx) * static_cast<uint32_t>(gy) * static_cast<uint32_t>(gz);
  void *args_array = arg_size == 0 ? NULL : const_cast<char *>(arg_data);
  rtError_t ret = rtKernelLaunch(function, block_num, args_array,
                                 static_cast<size_t>(arg_size), NULL, stream);
  if (ret != RT_ERROR_NONE) {
    PyErr_Format(PyExc_RuntimeError, "rtKernelLaunch failed: 0x%x", ret);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS, "Load kernel binary"},
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
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
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

    def launch_with_args(
        self,
        *,
        function: int,
        stream: int,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        args: bytes,
        expects_debug_fifo: bool,
        expects_print_tensor: bool | int,
    ) -> None:
        self._ensure_loaded()
        # The Ascend runtime rejects rtKernelLaunch with a zero-size argument
        # buffer (0x107000) -- verified that this is about arg_size == 0, not a
        # NULL pointer: passing a non-NULL pointer to an empty array fails the
        # same way. Pad a single zero slot so a no-argument kernel presents a
        # non-empty buffer (arg_size == 8); the kernel never reads it.
        if not args:
            args = b"\x00" * ctypes.sizeof(ctypes.c_uint64)
        arg_size = len(args)
        arg_array = (ctypes.c_uint8 * arg_size).from_buffer_copy(args)
        raw_ptr = ctypes.cast(arg_array, ctypes.POINTER(ctypes.c_uint8))
        ret = self._module.tla_runtime_launch_kernel(
            ctypes.c_uint64(int(function)),
            ctypes.c_uint64(int(stream)),
            int(grid_x),
            int(grid_y),
            int(grid_z),
            raw_ptr,
            arg_size,
            int(expects_debug_fifo),
            int(expects_print_tensor),
        )
        if ret != 0:
            raise TlaRuntimeUnavailableError(self._last_error())
