"""Ascend Catlass DSL: Tla IR dialect, Python ``catlass`` front-end, and lowering/runtime entry points."""

from __future__ import annotations

from typing import Any

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for source tree
    __version__ = "0.0.0"

from . import dsl as _dsl
from . import tla
from . import types as types
from .tla.runtime import _Tensor
from . import core_api as core
from .address_space import AddressSpace
from .base_dsl.typing import Constexpr, JitArgument, Pointer
from . import runtime as _runtime
from . import core_api as _core_api  # noqa: F401
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.jit_executor import TlaJitExecutor as _TlaJitExecutor

PASSES = ()

# Stable explicit exports.
TlaIRNotExecutableError = _runtime.TlaIRNotExecutableError
TlaCoreAPIError = _runtime.TlaCoreAPIError
TlaExecutionError = _runtime.TlaExecutionError
TlaCompilerBridgeUnavailableError = _runtime.TlaCompilerBridgeUnavailableError
TlaBackendCompilerNotFoundError = _runtime.TlaBackendCompilerNotFoundError
TlaKernelCompileError = _runtime.TlaKernelCompileError
TlaRuntimeUnavailableError = _runtime.TlaRuntimeUnavailableError
TlaUnsupportedAbiError = _runtime.TlaUnsupportedAbiError
TlaKernelArtifact = _runtime.TlaKernelArtifact
TlaExecutionResult = _runtime.TlaExecutionResult
TlaJitExecutor = _TlaJitExecutor
TlaJitFunction = _dsl.TlaJitFunction
AscendNpuIrKernelFunction = _dsl.AscendNpuIrKernelFunction
compile = _dsl.compile
jit = _dsl.jit
kernel = _dsl.kernel
ascendnpuir_kernel = _dsl.ascendnpuir_kernel
Tensor = _Tensor
TypedTensor = tla.TypedTensor
Scalar = types.Scalar
Numeric = types.Numeric
Bool = types.Bool
Int8 = types.Int8
Int16 = types.Int16
Int32 = types.Int32
Int64 = types.Int64
UInt8 = types.UInt8
UInt16 = types.UInt16
UInt32 = types.UInt32
UInt64 = types.UInt64
Index = types.Index
Float32 = types.Float32
Float64 = types.Float64
Float16 = types.Float16
BFloat16 = types.BFloat16
const_expr = _runtime.const_expr
initialize = _runtime.initialize
finalize = _runtime.finalize
runtime_state = _runtime.runtime_state
current_device_id = _runtime.current_device_id
current_stream = _runtime.current_stream
utils = _runtime.utils
pipes = _runtime.pipes
cross_modes = _runtime.cross_modes
arch = _runtime.arch
fp16 = "f16"
bf16 = "bf16"
fp32 = "f32"


__all__ = [
    "__version__",
    "PASSES",
    "TlaIRNotExecutableError",
    "TlaCoreAPIError",
    "TlaExecutionError",
    "TlaCompilerBridgeUnavailableError",
    "TlaBackendCompilerNotFoundError",
    "TlaKernelCompileError",
    "TlaRuntimeUnavailableError",
    "TlaUnsupportedAbiError",
    "TlaKernelArtifact",
    "TlaExecutionResult",
    "TlaJitExecutor",
    "TlaJitFunction",
    "AscendNpuIrKernelFunction",
    "compile",
    "jit",
    "kernel",
    "ascendnpuir_kernel",
    "DSLLocation",
    "BaseDSL",
    "const_expr",
    "initialize",
    "finalize",
    "runtime_state",
    "current_device_id",
    "current_stream",
    "tla",
    "types",
    "core",
    "Tensor",
    "TypedTensor",
    "Pointer",
    "Scalar",
    "Numeric",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Index",
    "Float32",
    "Float64",
    "Float16",
    "BFloat16",
    "JitArgument",
    "AddressSpace",
    "Constexpr",
    "utils",
    "pipes",
    "cross_modes",
    "arch",
    "fp16",
    "bf16",
    "fp32",
    *getattr(_runtime, "_CORE_API_EXPORTS", []),
]


def __getattr__(name: str) -> Any:
    # Core API helpers are served from runtime so package exports remain stable.
    try:
        return getattr(_runtime, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_runtime)))
