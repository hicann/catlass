from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from ..execution import TlaExecutionResult


@dataclass(frozen=True)
class JitFunctionArtifacts:
    """Side artifacts retained for inspection and diagnostics.

    These values are not inputs to ``JitModule`` and are never used to own
    runtime binary or function handles.
    """

    MLIR: str
    LLVM: str
    BINARY: Path
    cache_key: str
    cache_dir: Path
    hivmc_path: Path
    PASS_IR_DUMP: str | None = None
    compiler_bridge_path: Path | None = None


@dataclass(frozen=True)
class ExecutionArgs:
    """Runtime ABI binder for TLA kernel launch arguments.

    ``original_signature`` (when set) is the pre-Constexpr-strip kernel
    signature; ``signature`` is the filtered runtime copy. Packing always
    requires a compiler-produced ``kernel_abi`` layout so Dynamic-GM / memref
    ABI stays signature-driven.
    """

    original_signature: inspect.Signature | None = None
    kernel_abi: Any | None = None
    abi_packer: Any | None = None
    signature: inspect.Signature | None = None

    def __post_init__(self) -> None:
        if self.original_signature is not None and self.signature is None:
            object.__setattr__(
                self,
                "signature",
                self.filter_runtime_signature(self.original_signature),
            )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        kernel_abi: Any | None = None,
        abi_packer: Any | None = None,
    ) -> "ExecutionArgs":
        """Build a binder from a kernel callable's original signature."""
        from .core import BaseDSL

        return cls(
            original_signature=BaseDSL()._get_signature(fn),
            kernel_abi=kernel_abi,
            abi_packer=abi_packer,
        )

    def filter_runtime_signature(self, sig: inspect.Signature) -> inspect.Signature:
        """Drop Constexpr parameters from a signature."""
        from .runtime.jit_arg_adapters import is_arg_annotation_constexpr

        filtered_params = []
        for index, (name, param) in enumerate(sig.parameters.items()):
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                filtered_params.append(param)
                continue
            if is_arg_annotation_constexpr(param.annotation, name, index, None):
                continue
            filtered_params.append(param)
        return sig.replace(parameters=filtered_params)

    def get_rectified_args_from_original_args(
        self,
        full_args: Sequence[Any],
        full_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Strip Constexpr parameters from a full kernel call argument list."""
        from .runtime.jit_arg_adapters import is_arg_annotation_constexpr

        sig = self.original_signature
        runtime_sig = self.signature
        assert sig is not None and runtime_sig is not None

        runtime_arity = sum(
            1
            for param in runtime_sig.parameters.values()
            if param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        )
        # Already-stripped launch args: do not re-bind against the original
        # signature (Constexpr slots between runtime params would shift).
        if len(full_args) == runtime_arity and not full_kwargs:
            return tuple(full_args)

        bound = sig.bind_partial(*full_args, **dict(full_kwargs or {}))
        bound.apply_defaults()
        for index, (name, param) in enumerate(sig.parameters.items()):
            if is_arg_annotation_constexpr(param.annotation, name, index, None):
                bound.arguments.pop(name, None)

        runtime_bound = inspect.BoundArguments(runtime_sig, bound.arguments)
        return tuple(runtime_bound.args) + tuple(runtime_bound.kwargs.values())

    def get_rectified_args(
        self, launch_args: Sequence[Any], **_kwargs: Any
    ) -> tuple[Any, ...]:
        """Normalize call args before packing (adapters + passthrough)."""
        # Tensor/pointer providers are already in the representation consumed by
        # the prepared ABI packer. This is the common kernel-launch fast path.
        if all(hasattr(arg, "__c_pointers__") for arg in launch_args):
            return tuple(launch_args)

        from .runtime.jit_arg_adapters import (
            JitArgAdapterRegistry,
            _adapt_from_data_ptr,
        )
        from .typing import Numeric, as_numeric, is_constexpr_annotation

        rectified: list[Any] = []
        for arg in launch_args:
            if is_dataclass(arg) and not isinstance(arg, type):
                # Unpack a stdlib dataclass into one entry per **dynamic** field,
                # matching the frontend's scalar_group block args / ABI slots.
                # ``Constexpr`` fields are compile-time constants with no ABI slot.
                # Numerics and tensor-like fields (``__c_pointers__``) pass through;
                # plain scalars become Numerics via ``as_numeric``.
                for field in fields(arg):
                    if is_constexpr_annotation(field.type):
                        continue
                    value = getattr(arg, field.name)
                    if isinstance(value, Numeric) or hasattr(value, "__c_pointers__"):
                        rectified.append(value)
                    else:
                        rectified.append(as_numeric(value))
                continue
            if isinstance(arg, Numeric) or hasattr(arg, "__c_pointers__"):
                rectified.append(arg)
            else:
                adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                if adapter is not None:
                    rectified.append(adapter(arg))
                else:
                    rectified.append(_adapt_from_data_ptr(arg))
        return tuple(rectified)

    def generate_launch_payload(self, launch_args: Sequence[Any]) -> bytes:
        """Pack ``launch_args`` into the Ascend host launch byte buffer."""
        from .. import execution as execution_mod

        if self.kernel_abi is None:
            raise execution_mod.TlaUnsupportedAbiError(
                "A compiler-produced kernel ABI layout is required before packing "
                "launch arguments."
            )
        rectified = self.get_rectified_args(launch_args)
        if self.abi_packer is not None:
            return execution_mod._pack_launch_args_prepared(rectified, self.abi_packer)
        return execution_mod._pack_launch_args(rectified, self.kernel_abi)


@dataclass(frozen=True)
class JitModule:
    """Prepared TLA launch state shared by executors of one compiled function."""

    kernel_binary_path: Path
    entrypoint: str
    kernel_mode: str
    execution_args: ExecutionArgs
    uses_scalar_print: bool
    uses_tensor_print: bool
    is_mixed: bool
    print_metadata: tuple[Any, ...] | None
    print_helper_core: str | None


class JitExecutor:
    """Callable TLA executor owning one loaded binary/function pair."""

    def __init__(self, jit_module: JitModule) -> None:
        from .. import execution as execution_mod
        from .runtime.ascend_stream_adapter import current_device

        self.jit_module = jit_module
        self.device = int(current_device())
        execution_mod.load_acl()
        binary_handle, function_handle = execution_mod.load_binary(
            entrypoint=jit_module.entrypoint,
            kernel_mode=jit_module.kernel_mode,
            kernel_binary_path=jit_module.kernel_binary_path,
        )
        self.binary_handle = int(binary_handle)
        self.function_handle = int(function_handle)

    @property
    def kernel_binary_path(self) -> Path:
        return self.jit_module.kernel_binary_path

    def __call__(
        self,
        *launch_args: Any,
        block_num: int | None = None,
        args: Sequence[Any] | None = None,
        **launch_kwargs: Any,
    ) -> TlaExecutionResult:
        """Launch using this executor's prepared binary state."""
        from .. import execution as execution_mod

        if launch_args and args is not None:
            raise execution_mod.TlaUnsupportedAbiError(
                "Launch arguments specified multiple times."
            )
        if args is None:
            args = launch_args
        if block_num is None:
            block_num = 1
        if not isinstance(block_num, int):
            raise execution_mod.TlaUnsupportedAbiError("`block_num` must be an int.")

        for arg in args:
            prepare_for_launch = getattr(arg, "prepare_for_launch", None)
            if callable(prepare_for_launch):
                prepare_for_launch()

        module = self.jit_module
        uses_tensor_print = module.uses_tensor_print
        print_metadata = module.print_metadata
        if uses_tensor_print:
            assert print_metadata is not None
            assert module.print_helper_core is not None
            execution_mod._validate_print_tensor_fifo_capacity(
                print_metadata,
                block_num,
                helper_core=module.print_helper_core,
                mixed=module.is_mixed,
            )

        payload = module.execution_args.generate_launch_payload(args)
        payload = execution_mod._append_debug_print_workspace_payload(
            payload, enabled=module.uses_scalar_print
        )
        if uses_tensor_print:
            extension = bytearray(payload)
            execution_mod._align_payload(extension, execution_mod._POINTER_ABI_SIZE)
            extension.extend(
                execution_mod._PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                    execution_mod._POINTER_ABI_SIZE,
                    byteorder="little",
                    signed=False,
                )
            )
            payload = bytes(extension)

        raw_stream = launch_kwargs.get("stream")
        if raw_stream is None:
            from .runtime.ascend_stream_adapter import current_stream

            stream = int(current_stream(self.device))
        else:
            from .runtime.ascend_stream_adapter import as_stream

            stream = as_stream(raw_stream, device=self.device)

        def launch() -> None:
            execution_mod.launch_kernel(
                function_handle=self.function_handle,
                stream=int(stream),
                block_num=block_num,
                payload=payload,
                uses_scalar_print=module.uses_scalar_print,
                uses_tensor_print=uses_tensor_print,
                is_mixed=module.is_mixed,
            )

        if print_metadata is None:
            launch()
        else:
            native_output = execution_mod._capture_c_stdout(launch)
            print_block_count = execution_mod._checked_print_tensor_block_count(
                block_num
            )
            helper_core = module.print_helper_core
            assert helper_core is not None
            expected_subblocks: tuple[int | None, ...] = (
                (0, 1)
                if helper_core == "aiv" and module.is_mixed
                else (0,)
                if helper_core == "aiv"
                else (None,)
            )
            decoded = execution_mod._decode_native_print_tensor_records(
                native_output,
                metadata=print_metadata,
                block_count=print_block_count,
                expected_subblocks=expected_subblocks,
            )
            preserve_legacy_format = len(print_metadata) * print_block_count == 1
            for metadata, record, values in decoded:
                print(
                    execution_mod._format_print_tensor_record(
                        values,
                        shape=record.shape,
                        dtype=metadata.dtype,
                        call=None if preserve_legacy_format else metadata.call,
                        block=record.block,
                        position=metadata.position if module.is_mixed else None,
                        subblock=record.subblock,
                    )
                )

        return execution_mod.TlaExecutionResult(
            binary_handle=self.binary_handle,
            function_handle=self.function_handle,
            device=self.device,
        )


class JitCompiledFunction:
    """Own compiled launch state and one lazily materialized executor."""

    def __init__(
        self,
        *,
        jit_module: JitModule,
        artifacts: JitFunctionArtifacts,
    ) -> None:
        self.jit_module = jit_module
        self.artifacts = artifacts
        self._executor_lock = threading.Lock()
        self._executor: JitExecutor | None = None

    @property
    def __mlir__(self) -> str:
        return self.artifacts.MLIR

    @property
    def cache_key(self) -> str:
        return self.artifacts.cache_key

    @property
    def kernel_binary_path(self) -> Path:
        return self.jit_module.kernel_binary_path

    @property
    def entrypoint(self) -> str:
        return self.jit_module.entrypoint

    @property
    def kernel_mode(self) -> str:
        return self.jit_module.kernel_mode

    @property
    def execution_args(self) -> ExecutionArgs:
        return self.jit_module.execution_args

    def _get_executor(self) -> JitExecutor:
        executor = self._executor
        if executor is not None:
            return executor

        with self._executor_lock:
            if self._executor is None:
                self._executor = JitExecutor(self.jit_module)
            return self._executor

    def __call__(
        self,
        *launch_args: Any,
        block_num: int | None = None,
        args: Sequence[Any] | None = None,
        **launch_kwargs: Any,
    ) -> TlaExecutionResult:
        """Directory: Compile and Launch / Launch
        Description:
            Launch a compiled kernel on the NPU, passing runtime kernel arguments
            and launch options such as `block_num` and `stream`. The executor and
            binary are loaded lazily on the first call and reused thereafter.

        Parameters:
            - *`launch_args`* (`Any`): Positional runtime kernel arguments matching
              the `@tla.kernel` signature (bound Host tensors, scalars, or
              `@dataclass` instances). Mutually exclusive with `args=`.
            - *`block_num`* (`int | None`): Number of blocks to launch. Optional;
              default `1`. Must be an `int` when provided.
            - *`args`* (`Sequence[Any] | None`): Explicit runtime argument sequence.
              Optional; default `None`. Cannot be combined with non-empty
              `*launch_args`.
            - *`stream`* (`Any`, via `**launch_kwargs`): Optional ACL stream handle.
              When omitted, uses the current stream for the executor's device.

        Constraints:
            - `*launch_args` and `args=` must not both be non-empty
              (`TlaUnsupportedAbiError`).
            - Launch args must be bound NPU buffers for tensors (`from_dlpack`);
              `make_fake_tensor` samples are for compile only.
            - `block_num` must be an `int` (default `1`).

        Example:
        ```python
        compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
        compiled(tx, ty, block_num=1)
        compiled(args=(tx, ty), block_num=1)
        ```

        """
        return self._get_executor()(
            *launch_args,
            block_num=block_num,
            args=args,
            **launch_kwargs,
        )


__all__ = [
    "ExecutionArgs",
    "JitFunctionArtifacts",
    "JitModule",
    "JitCompiledFunction",
]
