"""Tla DSL runtime entry points and dynamic op helper registry."""

from __future__ import annotations

import contextvars
import importlib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field  # noqa: F401 — field used below
from typing import Any, Callable

from .base_dsl.runtime.dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLDeviceType,
)

from .execution import (
    TlaBackendCompilerNotFoundError,
    TlaKernelArtifact,
    TlaKernelCompileError,
    TlaCompilerBridgeUnavailableError,
    TlaExecutionError,
    TlaExecutionResult,
    TlaRuntimeUnavailableError,
    TlaUnsupportedAbiError,
    compile_kernel,
    execute_kernel,
    runtime_options_for_launch,
    runtime_options_from_kwargs,
)
from .types import RuntimeTensorError


class TlaIRNotExecutableError(RuntimeError):
    """Raised when attempting to execute APIs that only exist on the lowered TLA MLIR path."""


class TlaCoreAPIError(RuntimeError):
    """Raised when a user-facing Tla API call violates preconditions."""


@dataclass(frozen=True)
class TlaRuntimeState:
    """Global initialized runtime state."""

    device_id: int | None = None
    stream: Any | None = None
    device_ptrs: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class _FrontendEmitState:
    """Active frontend emission state."""

    arg_bindings: dict[int, Any]
    category_bindings: dict[int, str]
    module: Any | None = None
    #: ``mlir.Value`` -> host :class:`~catlass.tla.runtime._Tensor` for execution lowering.
    tensor_host_by_value: dict[Any, Any] = field(default_factory=dict)
    #: ``mlir.Value`` -> structured Tla tensor type descriptor.
    tensor_type_by_value: dict[Any, Any] = field(default_factory=dict)
    #: ``mlir.Value`` -> resolved tensor metadata fields (shape/stride/coord/origin_shape/...).
    tensor_metadata_by_value: dict[Any, dict[str, Any]] = field(default_factory=dict)
    mutex_guard_depth: int = 0
    #: Stack of enclosing region wrappers, each one of "cube" / "vector" /
    #: "vec.func" (the wrapper's own name).
    active_regions: list[str] = field(default_factory=list)


_FRONTEND_EMIT_STATE: contextvars.ContextVar[_FrontendEmitState | None] = (
    contextvars.ContextVar("tla_frontend_emit_state", default=None)
)
_GLOBAL_RUNTIME_STATE = TlaRuntimeState()


def _load_acl() -> Any:
    try:
        return importlib.import_module("acl")
    except ImportError as exc:
        raise TlaRuntimeUnavailableError(
            "Failed to import `acl`. Ensure the Ascend Python runtime is installed."
        ) from exc


def _normalize_runtime_device(device: int | str | None = None) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        device_id = device
    elif isinstance(device, str):
        stripped = device.strip()
        if stripped.isdigit():
            device_id = int(stripped)
        elif stripped.startswith("npu:") and stripped[4:].isdigit():
            device_id = int(stripped[4:])
        else:
            raise TlaCoreAPIError(
                "tla.initialize device must be an integer 0-7 or 'npu:0' through 'npu:7'."
            )
    else:
        raise TlaCoreAPIError(
            "tla.initialize device must be an integer 0-7 or 'npu:0' through 'npu:7'."
        )
    if device_id < 0 or device_id > 7:
        raise TlaCoreAPIError(
            "tla.initialize device must be an integer 0-7 or 'npu:0' through 'npu:7'."
        )
    return device_id


def _require_acl_success(ret: Any, op_name: str) -> None:
    if int(ret) != 0:
        raise TlaRuntimeUnavailableError(f"{op_name} failed with ret={int(ret)}")


def initialize(device: int | str | None = None) -> TlaRuntimeState:
    """Initialize the global Ascend runtime state and create a stream."""

    global _GLOBAL_RUNTIME_STATE
    if _GLOBAL_RUNTIME_STATE.device_id is not None:
        raise TlaExecutionError(
            "tla.initialize() was already called. Call tla.finalize() before reinitializing."
        )
    device_id = _normalize_runtime_device(device)
    acl = _load_acl()
    _require_acl_success(acl.init(), "acl.init")
    try:
        _require_acl_success(acl.rt.set_device(device_id), "acl.rt.set_device")
        stream, ret = acl.rt.create_stream()
        _require_acl_success(ret, "acl.rt.create_stream")
    except Exception:
        try:
            acl.rt.reset_device(device_id)
        except Exception:
            pass
        try:
            acl.finalize()
        except Exception:
            pass
        raise
    _GLOBAL_RUNTIME_STATE = TlaRuntimeState(
        device_id=device_id,
        stream=stream,
        device_ptrs=(),
    )
    return _GLOBAL_RUNTIME_STATE


def finalize() -> None:
    """Finalize the global Ascend runtime state."""

    global _GLOBAL_RUNTIME_STATE
    state = _GLOBAL_RUNTIME_STATE
    if state.device_id is None:
        raise TlaExecutionError(
            "tla.finalize() requires a prior call to tla.initialize()."
        )
    acl = _load_acl()
    try:
        from . import types as types_mod
    except Exception:
        types_mod = None
    for dev_ptr in state.device_ptrs:
        _require_acl_success(acl.rt.free(dev_ptr), "acl.rt.free")
    if types_mod is not None:
        types_mod.invalidate_runtime_allocations(device_ptrs=state.device_ptrs)
    _require_acl_success(acl.rt.reset_device(state.device_id), "acl.rt.reset_device")
    _require_acl_success(acl.finalize(), "acl.finalize")
    _GLOBAL_RUNTIME_STATE = TlaRuntimeState()


def runtime_state() -> TlaRuntimeState:
    """Return the current global runtime state snapshot."""

    return _GLOBAL_RUNTIME_STATE


def current_device_id() -> int | None:
    """Return the initialized device id, if any."""

    return _GLOBAL_RUNTIME_STATE.device_id


def current_stream() -> Any | None:
    """Return the initialized runtime stream, if any."""

    return _GLOBAL_RUNTIME_STATE.stream


def register_device_ptr(ptr: int) -> None:
    """Track a device allocation for cleanup during finalize."""

    global _GLOBAL_RUNTIME_STATE
    if ptr == 0 or ptr in _GLOBAL_RUNTIME_STATE.device_ptrs:
        return
    _GLOBAL_RUNTIME_STATE = TlaRuntimeState(
        device_id=_GLOBAL_RUNTIME_STATE.device_id,
        stream=_GLOBAL_RUNTIME_STATE.stream,
        device_ptrs=_GLOBAL_RUNTIME_STATE.device_ptrs + (int(ptr),),
    )


@contextmanager
def _frontend_emission(
    *,
    arg_bindings: dict[int, Any] | None = None,
    category_bindings: dict[int, str] | None = None,
    tensor_host_by_value: dict[Any, Any] | None = None,
    module: Any | None = None,
) -> Any:
    """Activate frontend direct-op emission context."""
    state = _FrontendEmitState(
        arg_bindings=dict(arg_bindings or {}),
        category_bindings=dict(category_bindings or {}),
        tensor_host_by_value=dict(tensor_host_by_value or {}),
        module=module,
    )
    token = _FRONTEND_EMIT_STATE.set(state)
    try:
        yield state
    finally:
        _FRONTEND_EMIT_STATE.reset(token)


@contextmanager
def _eager_capture() -> Any:
    """Compatibility context that provides a minimal direct-emission session."""

    from mlir import ir as mlir_ir  # type: ignore[assignment]

    with mlir_ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with mlir_ir.Location.unknown(ctx):
            module = mlir_ir.Module.create()
            with mlir_ir.InsertionPoint(module.body):
                with _frontend_emission(module=module) as state:
                    yield state


def _current_frontend_state() -> _FrontendEmitState | None:
    return _FRONTEND_EMIT_STATE.get()


def _resolve_frontend_bound_value(value: Any) -> Any | None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return None
    return state.arg_bindings.get(id(value))


def _bind_frontend_value(proxy: Any, value: Any) -> None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return
    state.arg_bindings[id(proxy)] = value


def _bind_frontend_category(value: Any, category: str) -> None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return
    state.category_bindings[id(value)] = category


def _has_enclosing_region(kind: str) -> bool:
    """True if some enclosing region is ``kind`` (``cube`` / ``vector`` / ``vec.func``).

    Walks all active region wrappers, so an op nested several levels deep (e.g.
    inside an ``scf.for`` inside a ``tla.vec.func``) still matches. A ``tla.vec.func``
    is always nested inside a ``tla.vector`` (enforced when it is entered), so a
    ``"vector"`` requirement stays satisfied from inside a ``vec.func`` via the
    enclosing region on the stack.
    """
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return True  # No frontend state to inspect; defer to the MLIR verifier.
    return kind in state.active_regions


def _require_enclosing_region(op_name: str, kind: str) -> None:
    """Require an enclosing ``tla.<kind>()`` region (cube / vector / vec.func)."""
    if not _has_enclosing_region(kind):
        raise TlaCoreAPIError(f"tla.{op_name} must be nested inside tla.{kind}()")


def _require_enclosing_cube_or_vector(op_name: str) -> None:
    """Require an enclosing tla.cube() or tla.vector() region (either core kind).

    Used by synchronization/mutex/barrier ops, which must sit inside a core
    region but not the bare tla.func scope. Mirrors the MLIR op verifier.
    """
    if not (_has_enclosing_region("cube") or _has_enclosing_region("vector")):
        raise TlaCoreAPIError(
            f"tla.{op_name} must be nested inside tla.cube() or tla.vector()"
        )


def _resolve_frontend_bound_category(value: Any) -> str | None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return None
    return state.category_bindings.get(id(value))


class _IndexExpr:
    """Frontend proxy that carries an SSA index value through transformed Python code."""

    def __init__(self, value: Any) -> None:
        self._value = value
        _bind_frontend_value(self, value)
        _bind_frontend_category(self, "index")
        _bind_frontend_category(value, "index")

    def _binary(self, other: Any, op_name: str) -> "_IndexExpr":
        from mlir import ir as mlir_ir  # type: ignore[assignment]

        lhs = _coerce_index_value(self)
        rhs = _coerce_index_value(other)
        op = mlir_ir.Operation.create(
            op_name, operands=[lhs, rhs], results=[mlir_ir.IndexType.get()]
        )
        return _IndexExpr(op.results[0])

    def _compare(self, other: Any, predicate: Any) -> "_BoolExpr":
        from mlir.dialects import arith  # type: ignore[import-not-found]

        lhs = _coerce_index_value(self)
        rhs = _coerce_index_value(other)
        return _BoolExpr(arith.CmpIOp(predicate, lhs, rhs).result)

    def __add__(self, other: Any) -> "_IndexExpr":
        return self._binary(other, "arith.addi")

    def __radd__(self, other: Any) -> "_IndexExpr":
        return _IndexExpr(other).__add__(self)

    def __sub__(self, other: Any) -> "_IndexExpr":
        return self._binary(other, "arith.subi")

    def __rsub__(self, other: Any) -> "_IndexExpr":
        return _IndexExpr(other).__sub__(self)

    def __mul__(self, other: Any) -> "_IndexExpr":
        return self._binary(other, "arith.muli")

    def __rmul__(self, other: Any) -> "_IndexExpr":
        return _IndexExpr(other).__mul__(self)

    def __floordiv__(self, other: Any) -> "_IndexExpr":
        return self._binary(other, "arith.divui")

    def __rfloordiv__(self, other: Any) -> "_IndexExpr":
        return _IndexExpr(other).__floordiv__(self)

    def __mod__(self, other: Any) -> "_IndexExpr":
        return self._binary(other, "arith.remui")

    def __rmod__(self, other: Any) -> "_IndexExpr":
        return _IndexExpr(other).__mod__(self)

    def __eq__(self, other: Any) -> "_BoolExpr":  # type: ignore[override]
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.eq)

    def __ne__(self, other: Any) -> "_BoolExpr":  # type: ignore[override]
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.ne)

    def __lt__(self, other: Any) -> "_BoolExpr":
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.slt)

    def __le__(self, other: Any) -> "_BoolExpr":
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.sle)

    def __gt__(self, other: Any) -> "_BoolExpr":
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.sgt)

    def __ge__(self, other: Any) -> "_BoolExpr":
        from mlir.dialects import arith  # type: ignore[import-not-found]

        return self._compare(other, arith.CmpIPredicate.sge)

    def __bool__(self) -> bool:
        raise TlaIRNotExecutableError(
            "SSA index values cannot be used as Python booleans"
        )


class _BoolExpr:
    """Frontend proxy for an SSA i1 value."""

    def __init__(self, value: Any) -> None:
        self._value = value
        _bind_frontend_value(self, value)
        _bind_frontend_category(self, "bool")
        _bind_frontend_category(value, "bool")

    @property
    def value(self) -> Any:
        return self._value

    def __bool__(self) -> bool:
        raise TlaIRNotExecutableError(
            "SSA bool values cannot be used as Python booleans"
        )


def _coerce_bool_value(value: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value):
        return resolved
    if isinstance(value, mlir_ir.Value):
        return value
    if isinstance(value, bool):
        return _const_i1(int(value))
    raise TlaCoreAPIError(f"Expected bool-like value, got {type(value).__name__}")


def _const_i1(value: int) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    i1 = mlir_ir.IntegerType.get_signless(1)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i1],
        attributes={"value": mlir_ir.IntegerAttr.get(i1, int(value))},
    )
    return op.results[0]


def _coerce_index_value(value: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    while isinstance(value, _IndexExpr):
        value = value._value
    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value):
        return resolved
    if isinstance(value, mlir_ir.Value):
        return value
    if isinstance(value, bool):
        return _const_index(int(value))
    if isinstance(value, int):
        return _const_index(int(value))
    raise TlaCoreAPIError(f"Expected index-like value, got {type(value).__name__}")


def _const_index(value: int) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.IndexType.get()],
        attributes={
            "value": mlir_ir.IntegerAttr.get(mlir_ir.IndexType.get(), int(value))
        },
    )
    return op.results[0]


class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _Utils:
    GM = _Sentinel("GM")
    L1 = _Sentinel("L1")
    L0A = _Sentinel("L0A")
    L0B = _Sentinel("L0B")
    L0C = _Sentinel("L0C")
    UB = _Sentinel("UB")


class _Pipes:
    SCALAR = _Sentinel("SCALAR")
    VECTOR = _Sentinel("VECTOR")
    CUBE = _Sentinel("CUBE")
    MTE1 = _Sentinel("MTE1")
    MTE2 = _Sentinel("MTE2")
    MTE3 = _Sentinel("MTE3")
    ALL = _Sentinel("ALL")
    MTE4 = _Sentinel("MTE4")
    MTE5 = _Sentinel("MTE5")
    V2 = _Sentinel("V2")
    FIX = _Sentinel("FIX")


class _CrossModes:
    NPU = _Sentinel("NPU")
    VECTORS_CORE = _Sentinel("VECTORS_CORE")
    SINGLE_CORE = _Sentinel("SINGLE_CORE")


class _RegionStub:
    def __init__(self, display_name: str) -> None:
        self._display_name = display_name

    def __enter__(self) -> None:
        raise TlaIRNotExecutableError(
            f"{self._display_name} is only available in lowered TLA MLIR"
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


def _capture_caller_location() -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    frame = inspect.currentframe()
    if frame is None:
        return mlir_ir.Location.unknown()
    try:
        frame = frame.f_back
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename != __file__:
                frame_info = inspect.getframeinfo(frame)
                lineno = int(getattr(frame_info, "lineno", 0) or 0)
                col_offset = 0
                positions = getattr(frame_info, "positions", None)
                if positions is not None:
                    col_offset = int(getattr(positions, "col_offset", 0) or 0)
                if lineno <= 0:
                    return mlir_ir.Location.unknown()
                file_loc = mlir_ir.Location.file(
                    frame_info.filename, lineno, col_offset
                )
                return mlir_ir.Location.name(frame_info.function, childLoc=file_loc)
            frame = frame.f_back
    finally:
        del frame
    return mlir_ir.Location.unknown()


def _internal_frontend_for(
    range_value: Any,
    body_fn: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
) -> Any:
    from . import tla_ast_decorators as _ast_decorators

    return _ast_decorators._internal_frontend_for(
        range_value, body_fn, *carried_values, carried_names=carried_names
    )


_VEC_FUNC_MODES = {"simd", "SIMD", "simt", "SIMT"}


def _validate_vec_func_mode(mode: Any) -> None:
    if not isinstance(mode, str):
        raise TlaCoreAPIError(
            f"tla.vec.func: mode must be a string; got {type(mode).__name__}"
        )
    if mode not in _VEC_FUNC_MODES:
        accepted = ", ".join(sorted(repr(value) for value in _VEC_FUNC_MODES))
        raise TlaCoreAPIError(
            f"tla.vec.func: mode must be one of {accepted}; got {mode!r}"
        )


def _internal_frontend_region(
    kind: str, body_fn: Callable[[], Any], *, mode: Any = None
) -> None:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if kind not in {"cube", "vector", "vec.func"}:
        raise TlaIRNotExecutableError(f"Unsupported TLA region wrapper: {kind}")
    if mode is not None:
        if kind != "vec.func":
            raise TlaCoreAPIError(f"tla.{kind}: unexpected mode argument")
        _validate_vec_func_mode(mode)
    if kind == "vec.func":
        _require_enclosing_region("vec.func", "vector")
    mlir_loc = _capture_caller_location()
    op = mlir_ir.Operation.create(f"tla.{kind}", regions=1, loc=mlir_loc)
    if kind == "vec.func":
        op.attributes["mode"] = mlir_ir.StringAttr.get("simd" if mode is None else mode)
    block = op.regions[0].blocks.append()
    state = _FRONTEND_EMIT_STATE.get()
    with mlir_ir.InsertionPoint(block):
        if state is not None:
            state.active_regions.append(kind)
        try:
            from . import tla_ast_decorators as _ast_decorators

            _ast_decorators._call_with_control_flow_source(body_fn)
        finally:
            if state is not None:
                state.active_regions.pop()


utils = _Utils()
pipes = _Pipes()
cross_modes = _CrossModes()
_CORE_API_EXPORTS = (
    "dsl_user_op",
    "arch",
    "vec",
    "mask",
    "create_mask",
    "update_mask",
    "tile_view",
    "make_tensor",
    "make_tensor_like",
    "copy",
    "flag",
    "cross_flag",
    "cross_core_set_flag",
    "cross_core_wait_flag",
    "set_flag",
    "wait_flag",
    "pipe_barrier",
    "local_mem_bar",
    "mutex",
    "mutex_guard",
    "range",
    "cube",
    "vector",
    "mmad",
    "full",
    "arange",
    "add",
    "sub",
    "mul",
    "max",
    "min",
    "div",
    "where",
    "not_",
    "and_",
    "or_",
    "xor",
    "exp",
    "log",
    "sqrt",
    "abs",
    "neg",
    "gather",
    "cmp",
    "make_ptr",
    "recast_ptr",
    "make_shape",
    "make_coord",
    "make_stride",
    "make_layout",
    "IndexTree",
    "range_constexpr",
)


def const_expr(value: Any) -> bool:
    """Return a Python bool for frontend-time control flow."""
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value) or isinstance(value, mlir_ir.Value):
        raise TlaCoreAPIError("tla.const_expr requires a compile-time Python value")
    return bool(value)


def constexpr(value: Any) -> Any:
    """Mark a value as constexpr for folding during lowering."""

    return value


def jit(fn: Callable[..., Any]) -> Any:
    """Compat wrapper for the Tla DSL jit decorator."""

    from .dsl import jit as _jit

    return _jit(fn)


def kernel(fn: Callable[..., Any]) -> Any:
    """Compat wrapper for the Tla DSL kernel decorator."""

    from .dsl import kernel as _kernel

    return _kernel(fn)


def ascendnpuir_kernel(filename: str) -> Any:
    """Compat wrapper for the file-backed AscendNPU-IR kernel decorator."""

    from .dsl import ascendnpuir_kernel as _ascendnpuir_kernel

    return _ascendnpuir_kernel(filename)


def __getattr__(name: str) -> Any:
    if name in _CORE_API_EXPORTS:
        from . import core_api as _core_api

        return getattr(_core_api, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CORE_API_EXPORTS))


from .tla.runtime import (  # noqa: E402
    DlpackBridgeError,
    export_dlpack_capsule,
    from_dlpack,
    make_fake_tensor,
    _Tensor,
)


__all__ = [
    "TlaCoreAPIError",
    "TlaIRNotExecutableError",
    "TlaExecutionError",
    "TlaCompilerBridgeUnavailableError",
    "TlaBackendCompilerNotFoundError",
    "TlaKernelCompileError",
    "TlaRuntimeUnavailableError",
    "TlaUnsupportedAbiError",
    "TlaKernelArtifact",
    "TlaExecutionResult",
    "TlaRuntimeState",
    "RuntimeTensorError",
    "DlpackBridgeError",
    "ASCEND_DEVICE_TYPES",
    "DLDeviceType",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
    "_Tensor",
    "arch",
    "const_expr",
    "constexpr",
    "current_device_id",
    "current_stream",
    "finalize",
    "initialize",
    "jit",
    "kernel",
    "ascendnpuir_kernel",
    "pipes",
    "runtime_state",
    "utils",
    *_CORE_API_EXPORTS,
]
