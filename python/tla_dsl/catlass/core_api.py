"""Explicit user-facing Tla DSL op helpers with inline preconditions."""

from __future__ import annotations

import builtins as _builtins
import inspect
import math
import warnings
from enum import Enum
from itertools import chain
from typing import Any, Callable, Iterable, Sequence, TypeAlias

from mlir import ir as mlir_ir  # type: ignore[assignment]
from mlir._mlir_libs._mlir import (  # type: ignore[import-not-found]
    register_value_caster as _register_value_caster,
)
from mlir.dialects import arith as _mlir_arith  # type: ignore[import-not-found]

from . import _tla_type_bridge
from .params import CastParams

# Element-type tokens the tla.cast lowering supports: signed ints and the AVE
# float set. Unsigned ints, Bool (i1) and Float64 (f64) are rejected by VectorSSA.to.
_CAST_SUPPORTED_DTYPES = frozenset(
    {"i8", "i16", "i32", "i64", "f16", "bf16", "f32"}
)
from ._mlir_bindings import tla_ops_gen as _tla_ops_gen
from .base_dsl import ast_helpers as _ast_helpers
from .base_dsl.op import dsl_user_op, _capture_user_loc
from .base_dsl.typing import Float32, Int8, Numeric, as_numeric
from .base_dsl.typing import Pointer as PointerABC
from .base_dsl.typing import Pointer as PointerTypeHint
from .tla.tensor import normalize_tile_view_coord
from .tla.typing import Tensor
from .utils.localmem_allocator import LocalmemAllocator
from . import runtime as _runtime
from .tla.tensor import _Tensor
from .runtime import (
    TlaCoreAPIError,
    TlaIRNotExecutableError,
    _RegionStub,
    _Sentinel,
)
from .execution_lowering import TlaLoweringError
from .types import (
    AddressSpace,
    TlaIndexTreeType,
    TlaLayoutDescriptor,
    TlaTensorTypeDescriptor,
    PtrType,
    LayoutType,
    TlaCoord,
    TlaCrossFlag,
    TlaFlag,
    TlaIndex,
    TlaLayout,
    TlaMutex,
    TlaRegion,
    TlaShape,
    TlaStride,
    TlaTensor,
    TlaTile,
    TlaValue,
    dtype_size_bytes,
)
from .params import CopyParams, CopyL0C2DstParams, QuantMode, L0C2UBMode, MemType


_PIPE_VALUES = {
    "scalar",
    "vector",
    "cube",
    "mte1",
    "mte2",
    "mte3",
    "all",
    "mte4",
    "mte5",
    "v2",
    "fix",
    "virtual_mte2_l1a",
    "virtual_mte2_l1b",
    "num",
}
_CROSS_MODE_VALUES = {"npu", "vectors_core", "single_core"}
_MISSING = object()
_SUPPORTED_COMPARE_ELEMENT_TYPES = frozenset({"f16", "f32", "i32", "u32"})
_MASK_CMP_MODES = ("lt", "le", "gt", "ge", "eq", "ne")


def _check_compare_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_COMPARE_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_COMPARE_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported compare element type {element_type}; "
        f"supported element types are {supported}",
    )


class ReductionOp(Enum):
    ADD = "add"
    MAX = "max"
    MIN = "min"


_SUPPORTED_REDUCTION_ELEMENT_TYPES = frozenset(
    {
        "f16",
        "f32",
        "i16",
        "i32",
        "u16",
        "u32",
    }
)


def _check_reduction_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_REDUCTION_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_REDUCTION_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported reduction element type {element_type}; "
        f"supported element types are {supported}",
    )


class _Shape:
    """SSA wrapper for the **result** of ``make_shape`` (``!tla.shape<...>``).

    Nested structure is expressed only via **tuple trees** in ``make_shape(*components)``;
    do not pass ``_Shape`` / ``_Coord`` / ``_Stride`` as inner components.
    """

    __slots__ = ("_shape_value", "_components")

    def __init__(
        self, *, shape_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._shape_value = shape_value
        self._components = components


class _Coord:
    """SSA wrapper for the **result** of ``make_coord`` (``!tla.coord<...>``). See ``_Shape`` re: tuple-only nesting."""

    __slots__ = ("_coord_value", "_components")

    def __init__(
        self, *, coord_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._coord_value = coord_value
        self._components = components


class _Stride:
    """SSA wrapper for ``make_stride`` (``!tla.stride<...>``).

    Like :class:`_Shape` / :class:`_Coord`, stores ``_components`` (the tuple tree passed to
    ``make_stride``) so Python code can recover the same tree as ``make_shape`` without
    reparsing the MLIR type string. :class:`_Layout` does **not** mirror this: layout is a
    single fused SSA value built from shape+stride (+ optional origin) operands.
    """

    __slots__ = ("_stride_value", "_components")

    def __init__(
        self, *, stride_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._stride_value = stride_value
        self._components = components


class _Layout:
    """SSA wrapper for ``make_layout`` (``!tla.layout<...>``).

    The fused layout SSA is the primary payload. The source ``_Shape`` / ``_Stride`` /
    origin ``_Shape`` wrappers and the resolved layout-tag token are also retained so
    ``make_tensor`` can rebuild the Python layout/tensor descriptor trees without
    re-parsing the MLIR type string (same component trees as MLIR: layout type is derived
    from shape+stride types).
    """

    __slots__ = ("_layout_value", "_shape", "_stride", "_origin_shape", "_layout_tag")

    def __init__(
        self,
        *,
        layout_value: mlir_ir.Value,
        shape: _Shape | None = None,
        stride: _Stride | None = None,
        origin_shape: _Shape | None = None,
        layout_tag: str | None = None,
    ) -> None:
        self._layout_value = layout_value
        self._shape = shape
        self._stride = stride
        self._origin_shape = origin_shape
        self._layout_tag = layout_tag


IndexLike: TypeAlias = int | mlir_ir.Value
IndexTree: TypeAlias = IndexLike | tuple["IndexTree", ...]
ShapeLike: TypeAlias = IndexTree
CoordLike: TypeAlias = IndexTree
StrideLike: TypeAlias = IndexTree
ValueLike: TypeAlias = Numeric | mlir_ir.Value
TileLike: TypeAlias = mlir_ir.Value
MemrefLike: TypeAlias = Tensor | mlir_ir.Value
FlagLike: TypeAlias = mlir_ir.Value
CrossFlagLike: TypeAlias = mlir_ir.Value
MutexLike: TypeAlias = mlir_ir.Value
PipeLike: TypeAlias = str | _Sentinel
CrossModeLike: TypeAlias = str | _Sentinel
AddressSpaceLike: TypeAlias = AddressSpace | _Sentinel
DTypeLike: TypeAlias = mlir_ir.Type | type[Numeric]
LiteralLike: TypeAlias = bool | int | float | str | mlir_ir.Type


class _LayoutTagSentinel(_Sentinel):
    """Marks ``tla.arch.*`` values that are valid ``Tensor.layout_tag`` / ``make_tensor_like`` tags."""


@_register_value_caster(PtrType.get_static_typeid(), replace=True)
class _Pointer(PointerABC):
    """Concrete JIT pointer for ``!tla.ptr<...>``."""

    __slots__ = ("value", "_ptr_ty", "_alloc_size_bytes")
    __tla_category__ = "pointer"

    def __new__(cls, value: Any, alloc_size_bytes: int | None = None) -> Any:
        # MLIR may assign one TypeID to all unregistered ``!tla.*`` types; the value
        # caster still dispatches on that id, so only wrap true ``!tla.ptr`` SSAs.
        if cls is _Pointer and isinstance(value, _Pointer):
            return value
        if cls is _Pointer and isinstance(value, mlir_ir.Value):
            if not PtrType.isinstance(value.type):
                return value
        return super().__new__(cls)

    def __init__(self, value: Any, alloc_size_bytes: int | None = None) -> None:
        if isinstance(value, _Pointer):
            if alloc_size_bytes is None:
                alloc_size_bytes = value._alloc_size_bytes
            value = value.value
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Pointer expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not PtrType.isinstance(value.type):
            raise TypeError(f"Pointer expects !tla.ptr<...>, got {value.type}")
        object.__setattr__(self, "value", value)
        pt = value.type
        if not PtrType.isinstance(pt):
            raise TypeError(f"Pointer expects !tla.ptr typeid, got {pt!r}")
        if not isinstance(pt, PtrType):
            pt = PtrType(pt)
        object.__setattr__(self, "_ptr_ty", pt)
        object.__setattr__(self, "_alloc_size_bytes", alloc_size_bytes)

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __str__(self) -> str:
        return str(self.value.type)

    def __repr__(self) -> str:
        return f"Pointer({self.value})"

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_Pointer":
        """Rebuild from MLIR SSA values.

        Accepts a single ``mlir.ir.Value`` (``!tla.ptr``) or an existing :class:`_Pointer`;
        ``_Pointer.__init__`` enforces ``!tla.ptr``.
        """
        if len(values) != 1:
            raise ValueError(f"Pointer expects 1 MLIR value, got {len(values)}")
        v0 = values[0]
        if isinstance(v0, _Pointer):
            inner = v0.value
        elif isinstance(v0, mlir_ir.Value):
            inner = v0
        else:
            raise TypeError(
                f"Expected _Pointer or mlir.ir.Value, but got {type(v0).__name__}"
            )
        # Do not register pointers in arg_bindings: they use id()-keyed dict entries that
        # can outlive ephemeral _Pointer wrappers after GC, so id reuse maps a new pointer
        # to a stale SSA value (wrong addrspace / wrong copy route).
        return _Pointer(inner)

    @property
    def dtype(self) -> type[Numeric]:
        return Numeric.from_mlir_type(self.pointee)

    @property
    def alignment(self) -> int:
        return self._ptr_ty.alignment

    @property
    def max_alignment(self) -> int:
        return self.alignment

    @property
    def memspace(self) -> AddressSpace:
        return AddressSpace.from_mlir_token(self._ptr_ty.addrspace)

    @property
    def type(self) -> Any:
        return self.value.type

    @property
    def pointee(self) -> mlir_ir.Type:
        return self._ptr_ty.pointee

    @property
    def addrspace(self) -> AddressSpace:
        """Address space of ``!tla.ptr`` (same as :attr:`memspace`)."""
        return AddressSpace.from_mlir_token(self._ptr_ty.addrspace)

    def __add__(self, other: Any) -> "_Pointer":
        """Offset this pointer by a scalar **element count** (``ptr + n`` / ``n + ptr``).

        Advances the pointer by ``other`` elements of its pointee type (not bytes),
        preserving the pointee type and address space. ``other`` may be a Python ``int``
        or an integer/index SSA value (e.g. a ``tla.range`` loop index).
        """
        _require_frontend_state("ptr_add")
        _require_category("ptr_add", "ptr", self, "pointer", 0)
        loc = _capture_user_loc()
        ctx = loc.context if loc is not None else mlir_ir.Context()
        alloc_size_bytes = getattr(self, "_alloc_size_bytes", None)
        p = _coerce_pointer_arg(self)
        src_ty = p._ptr_ty

        offset_value = _as_index_value(other)
        offset_ty = offset_value.type
        if not (
            isinstance(offset_ty, mlir_ir.IndexType)
            or isinstance(offset_ty, mlir_ir.IntegerType)
        ):
            _op_error(
                "ptr_add",
                f"offset must be an integer or index SSA value, got {offset_ty}",
            )

        out_ptr_ty = PtrType.get(
            src_ty.pointee, src_ty.addrspace, src_ty.alignment, context=ctx
        )
        op = mlir_ir.Operation.create(
            "tla.ptr_add",
            operands=[p.value, offset_value],
            results=[out_ptr_ty],
            loc=loc,
        )
        return _Pointer(op.results[0], alloc_size_bytes=alloc_size_bytes)

    def __radd__(self, other: Any) -> "_Pointer":
        return self.__add__(other)


class VectorSSA:
    """Frontend proxy for loaded vector SSA values."""

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"VectorSSA expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_tensor(value.type):
            raise TypeError(f"VectorSSA expects !tla.tensor<...>, got {value.type}")
        self.value = value
        self.__tla_category__ = "vector_ssa"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "vector_ssa")
        _runtime._bind_frontend_category(value, "vector_ssa")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __add__(self, other: Any) -> "VectorSSA":
        return add(self, other)

    def __radd__(self, other: Any) -> "VectorSSA":
        return add(other, self)

    def __sub__(self, other: Any) -> "VectorSSA":
        return sub(self, other)

    def __mul__(self, other: Any) -> "VectorSSA":
        return mul(self, other)

    def __rmul__(self, other: Any) -> "VectorSSA":
        return mul(other, self)

    def __truediv__(self, other: Any) -> "VectorSSA":
        return div(self, other)

    @dsl_user_op
    def reduce(
        self,
        kind: ReductionOp,
        *,
        mask: Any,
        init_value: Any | None = None,
        reduction_profile: Any | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        return _emit_vector_reduce(
            self,
            kind,
            mask=mask,
            init_value=init_value,
            reduction_profile=reduction_profile,
            loc=loc,
        )

    @dsl_user_op
    def to(
        self,
        dst_type: Any,
        params: CastParams,
        mask: Any | None = None,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Convert this register-resident vector to ``dst_type`` (element-type cast).

        ``dst_type`` is a concrete Numeric element type; only the types the AVE
        cast lowering supports are allowed: signed integers (``tla.Int8`` ..
        ``tla.Int64``) and floats ``tla.Float16`` / ``tla.BFloat16`` /
        ``tla.Float32``. Unsigned integers, ``tla.Bool`` (i1) and ``tla.Float64``
        are rejected. ``params`` is a required
        :class:`~catlass.params.CastParams` selecting rounding / saturation /
        register slot; ``mask`` optionally predicates which lanes convert. Lowers
        to ``tla.cast`` and must be used inside a ``tla.vec.func`` region.
        """
        _require_category("cast", "operand", self, "vector_ssa", 0)
        if not (
            isinstance(dst_type, type)
            and issubclass(dst_type, Numeric)
            and dst_type.dtype
        ):
            _op_error(
                "cast",
                f"invalid argument 'dst_type' (position 0): expected a concrete "
                f"Numeric element type, got {_type_name(dst_type)}",
            )
        # The lowering only emits signed-int and {f16,bf16,f32} AVE cast paths, so
        # reject unsigned ints, Bool (i1) and Float64 up front (rather than
        # emitting AVE IR the backend cannot legalize / would treat as signed).
        if dst_type.dtype not in _CAST_SUPPORTED_DTYPES:
            _op_error(
                "cast",
                f"unsupported cast target dtype '{dst_type.dtype}': tla.cast "
                f"supports signed integers (i8/i16/i32/i64) and floats "
                f"(f16/bf16/f32); unsigned, bool and f64 are not supported",
            )
        if not isinstance(params, CastParams):
            _op_error(
                "cast",
                f"invalid argument 'params' (position 1): expected CastParams, "
                f"got {_type_name(params)}",
            )
        if mask is not None:
            _require_category("cast", "mask", mask, "mask_ssa", 2)
        _require_frontend_state("cast")
        _runtime._require_enclosing_region("cast", "vec.func")
        operand_value = _as_value(self)
        context = operand_value.type.context
        src_desc = _tla_tensor_type_for_mlir_value(operand_value)
        result_desc = src_desc.with_updates(element_type=_dtype_to_str(dst_type))
        with context:
            trait_attr = mlir_ir.DenseI32ArrayAttr.get(params.codes())
        mask_value = _as_value(mask) if mask is not None else None
        result = _tla_ops_gen.cast(
            result_desc.to_mlir_type(context),
            operand_value,
            trait_attr,
            mask=mask_value,
            loc=loc,
        )
        _register_tla_tensor_type(result, result_desc)
        return VectorSSA(result)


class MaskSSA:
    """Frontend proxy for a vector mask SSA value (`!tla.mask`)."""

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"MaskSSA expects mlir.ir.Value, got {type(value).__name__}"
            )
        if str(value.type) != "!tla.mask":
            raise TypeError(f"MaskSSA expects !tla.mask, got {value.type}")
        self.value = value
        self.__tla_category__ = "mask_ssa"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "mask_ssa")
        _runtime._bind_frontend_category(value, "mask_ssa")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]


class _MutexValue:
    """Frontend proxy for an SSA ``!tla.mutex`` value."""

    def __init__(self, value: mlir_ir.Value, resource: str, mutex_id: int) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Mutex value expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_mutex(value.type):
            raise TypeError(f"Mutex value expects !tla.mutex, got {value.type}")
        self.value = value
        self.resource = resource
        self.id = mutex_id
        self.__tla_category__ = "mutex"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "mutex")
        _runtime._bind_frontend_category(value, "mutex")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_MutexValue":
        if len(values) != 1:
            raise ValueError(f"Mutex expects 1 MLIR value, got {len(values)}")
        v0 = values[0]
        if isinstance(v0, _MutexValue):
            inner = v0.value
        elif isinstance(v0, mlir_ir.Value):
            inner = v0
        else:
            raise TypeError(
                f"Expected _MutexValue or mlir.ir.Value, but got {type(v0).__name__}"
            )
        return _MutexValue(inner, self.resource, self.id)

    def lock(self, *, pipe: PipeLike, loc: mlir_ir.Location | None = None) -> None:
        return mutex_lock(self, pipe=pipe, loc=loc)

    def unlock(self, *, pipe: PipeLike, loc: mlir_ir.Location | None = None) -> None:
        return mutex_unlock(self, pipe=pipe, loc=loc)


class _MutexGuard:
    """Context manager that wraps a TLA op block with inferred mutex access."""

    def __init__(
        self, mutexes: tuple[Any, ...], loc: mlir_ir.Location | None = None
    ) -> None:
        self._mutexes = mutexes
        self._loc = loc
        self._state: Any | None = None
        self._block: Any | None = None
        self._start_op_count = 0
        self._entered = False

    def __enter__(self) -> "_MutexGuard":
        state = _runtime._current_frontend_state()
        if state is None:
            raise TlaIRNotExecutableError(
                "tla.mutex_guard is only available in lowered Tla IR"
            )
        for index, mutex_value in enumerate(self._mutexes):
            _require_category(
                "mutex_guard", f"mutex[{index}]", mutex_value, "mutex", index
            )
        try:
            block = mlir_ir.InsertionPoint.current.block
        except Exception as exc:
            raise TlaLoweringError(
                "tla.mutex_guard requires an active MLIR insertion point"
            ) from exc
        self._state = state
        self._block = block
        self._start_op_count = len(list(block.operations))
        state.mutex_guard_depth += 1
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc, tb
        if self._state is not None and self._entered:
            self._state.mutex_guard_depth = _builtins.max(0, self._state.mutex_guard_depth - 1)
        if exc_type is not None:
            return False
        if self._block is None:
            raise TlaLoweringError("tla.mutex_guard lost its MLIR insertion block")
        block_ops = list(self._block.operations)
        body_ops = block_ops[self._start_op_count :]
        if not body_ops:
            raise TlaLoweringError(
                "tla.mutex_guard body must emit at least one tla.copy or tla.mmad"
            )
        pipe = _infer_mutex_guard_pipe(body_ops)
        first_body_op = _raw_operation(body_ops[0])
        with mlir_ir.InsertionPoint(first_body_op):
            for mutex_value in self._mutexes:
                _emit_mutex_lock_op(mutex_value, pipe=pipe, loc=self._loc)
        for mutex_value in reversed(self._mutexes):
            _emit_mutex_unlock_op(mutex_value, pipe=pipe, loc=self._loc)
        return False


class _Namespace:
    def __init__(self) -> None:
        self._members: dict[str, Callable[..., Any]] = {}

    def _set(self, name: str, value: Callable[..., Any]) -> None:
        self._members[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _require_generated(symbol_name: str) -> None:
    if not hasattr(_tla_ops_gen, symbol_name):
        raise RuntimeError(
            f"Generated Tla binding is missing `{symbol_name}`; regenerate "
            "catlass/_mlir_bindings/tla_ops_gen.py"
        )


def _require_frontend_state(op_name: str) -> None:
    state = _runtime._current_frontend_state()
    if state is None:
        raise TlaIRNotExecutableError(
            f"tla.{op_name} is only available in lowered Tla IR"
        )


def _region_stub(op_name: str) -> _RegionStub:
    return _RegionStub(f"tla.{op_name}")


def _resolve_bound_value(value: Any) -> Any:
    """Resolve frontend proxy bindings.

    Keep ``Numeric`` / ``VectorSSA`` / ``MaskSSA`` wrappers intact (user-facing
    object stays typed; use ``_as_value`` / ``ir_value`` to obtain SSA).
    """
    if isinstance(value, mlir_ir.Value):
        return value
    if isinstance(value, (Numeric, VectorSSA, MaskSSA)):
        return value
    bound = _runtime._resolve_frontend_bound_value(value)
    if bound is not None:
        return bound
    return value


def _coerce_pointer_arg(x: Any) -> _Pointer:
    """Resolve frontend bindings, then same path as :meth:`_Pointer.__new_from_mlir_values__`."""
    return _Pointer.__new_from_mlir_values__(None, [_resolve_bound_value(x)])  # type: ignore[arg-type]


def _const_bool(value: bool) -> mlir_ir.Value:
    i1_type = mlir_ir.IntegerType.get_signless(1)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i1_type],
        attributes={
            "value": mlir_ir.IntegerAttr.get(i1_type, bool(value))
        },
    )
    return op.results[0]


def _as_i1_value(value: Any) ->mlir_ir.Value:
    if isinstance(value, bool):
        val = _const_bool(value)
    elif _category(value) == "bool":
        val = _as_value(value)
    else:
        raise TlaLoweringError(f"value expected to be a bool, got {type(value).__name__}")
    return val


def _const_index(value: int) -> mlir_ir.Value:
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.IndexType.get()],
        attributes={
            "value": mlir_ir.IntegerAttr.get(mlir_ir.IndexType.get(), int(value))
        },
    )
    return op.results[0]


def _const_i64(value: int, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    i64_type = mlir_ir.IntegerType.get_signless(64)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i64_type],
        attributes={"value": mlir_ir.IntegerAttr.get(i64_type, int(value))},
        loc=loc,
    )
    return op.results[0]


def _const_i32(value: int, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    i32_type = mlir_ir.IntegerType.get_signless(32)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i32_type],
        attributes={"value": mlir_ir.IntegerAttr.get(i32_type, value)},
        loc=loc,
    )
    return op.results[0]


def _const_f32(
    value: float, *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.F32Type.get()],
        attributes={"value": mlir_ir.FloatAttr.get_f32(float(value))},
        loc=loc,
    )
    return op.results[0]


_FULL_SUPPORTED_DTYPES = frozenset(
    ("i1", "i8", "i16", "i32", "i64", "bf16", "f16", "f32")
)

_ARANGE_SUPPORTED_DTYPES = frozenset(("i8", "i16", "i32", "i64"))
_ARANGE_ORDERS = frozenset(("increase", "decrease"))

# Width of one Ascend vector register tile in bytes. Must stay in sync with
# TlaVectorRegionPass::kVectorBytes in csrc/mlir/lib/Passes/TlaVectorRegionPass.cpp.
_VECTOR_REGISTER_BYTES = 256


def _vector_lane_count(element_bytes: int) -> int:
    """Return lane count for one vector register tile at the given element width."""
    if element_bytes <= 0:
        raise TlaCoreAPIError(
            f"element size must be positive for vector lane count, got {element_bytes}"
        )
    return _VECTOR_REGISTER_BYTES // element_bytes


def _as_index_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        # Index path: Int*/Bool/Index only (signless). Reject UInt* — use .to(Int*).
        if not (type(resolved).is_integer and type(resolved).signed):
            raise TlaLoweringError(
                f"Expected signed integer Numeric index, got {type(resolved).__name__}; "
                f"cast explicitly with .to(Int32) (or another Int*) before indexing"
            )
        resolved = resolved.ir_value()
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(resolved.type, mlir_ir.IndexType):
            return resolved
        if (
            mlir_ir.IntegerType.isinstance(resolved.type)
            and mlir_ir.IntegerType(resolved.type).is_signless
        ):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[mlir_ir.IndexType.get()],
            ).results[0]
        raise TlaLoweringError(
            f"Expected index-like operand, got SSA type {resolved.type}"
        )
    if isinstance(resolved, bool):
        return _const_index(int(resolved))
    if isinstance(resolved, int):
        return _const_index(resolved)
    raise TlaLoweringError(f"Expected index-like operand, got {type(value).__name__}")


def _as_i64_value(value: Any, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    i64_type = mlir_ir.IntegerType.get_signless(64)
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(resolved.type, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[i64_type],
                loc=loc,
            ).results[0]
        if mlir_ir.IntegerType.isinstance(resolved.type):
            int_type = mlir_ir.IntegerType(resolved.type)
            if int_type.width == 64:
                return resolved
            cast_name = "arith.extsi" if int_type.width < 64 else "arith.trunci"
            return mlir_ir.Operation.create(
                cast_name,
                operands=[resolved],
                results=[i64_type],
                loc=loc,
            ).results[0]
    if isinstance(resolved, bool):
        return _const_i64(int(resolved), loc=loc)
    if isinstance(resolved, int):
        return _const_i64(resolved, loc=loc)
    if isinstance(resolved, Numeric) and isinstance(
        resolved.value, (bool, int)
    ):
        dtype = type(resolved).dtype.lower()
        if dtype == "index" or (dtype.startswith("i") and dtype[1:].isdigit()):
            return _const_i64(int(resolved.value), loc=loc)
    raise TlaLoweringError(f"Expected i64-like operand, got {type(value).__name__}")


def _coerce_inttoptr_address(
    addr_token: str,
    value: int | mlir_ir.Value | Numeric,
    loc: mlir_ir.Location | None,
) -> mlir_ir.Value:
    """Integer SSA for ``tla.inttoptr`` (``gm`` / ``generic`` → i64, else i32)."""
    t = addr_token.strip().lower()
    target_ty = (
        mlir_ir.IntegerType.get_signless(64)
        if t in ("gm", "generic")
        else mlir_ir.IntegerType.get_signless(32)
    )
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        if not type(resolved).is_integer:
            _op_error(
                "make_ptr",
                f"address must be int or integer Numeric/SSA, got {_type_name(value)}",
            )
        resolved = resolved.ir_value(loc=loc)
    if isinstance(resolved, mlir_ir.Value):
        if PtrType.isinstance(resolved.type):
            _op_error(
                "make_ptr",
                "address must be integer or index SSA, not !tla.ptr",
            )
        vt = resolved.type
        if isinstance(vt, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[target_ty],
                loc=loc,
            ).results[0]
        if mlir_ir.IntegerType.isinstance(vt):
            int_type = mlir_ir.IntegerType(vt)
            if int_type.width == target_ty.width:
                return resolved
            cast_name = (
                "arith.extsi" if int_type.width < target_ty.width else "arith.trunci"
            )
            return mlir_ir.Operation.create(
                cast_name,
                operands=[resolved],
                results=[target_ty],
                loc=loc,
            ).results[0]
        _op_error(
            "make_ptr",
            f"address SSA must be integer or index, got {vt}",
        )
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        _op_error(
            "make_ptr",
            f"address must be int or integer SSA, got {_type_name(value)}",
        )
    return mlir_ir.Operation.create(
        "arith.constant",
        results=[target_ty],
        attributes={"value": mlir_ir.IntegerAttr.get(target_ty, int(resolved))},
        loc=loc,
    ).results[0]


def _as_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, _Pointer):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, _Tensor):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, VectorSSA):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, MaskSSA):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, Numeric):
        resolved = resolved.ir_value()
    if isinstance(resolved, _MutexValue):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, mlir_ir.Value):
        st = _runtime._current_frontend_state()
        if st is not None:
            host = st.tensor_host_by_value.get(resolved)
            if host is not None:
                st.tensor_host_by_value[resolved] = host
        return resolved
    if isinstance(resolved, bool):
        return _const_index(int(resolved))
    if isinstance(resolved, int):
        return _const_index(resolved)
    if isinstance(resolved, float):
        return _const_f32(resolved)
    raise TlaLoweringError(f"Expected SSA operand, got {type(value).__name__}")


def _as_branch_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        i1 = mlir_ir.IntegerType.get_signless(1)
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[i1],
            attributes={"value": mlir_ir.IntegerAttr.get(i1, int(resolved))},
        ).results[0]
    return _as_value(value)


def _wrap_frontend_value(value: mlir_ir.Value) -> Any:
    if PtrType.isinstance(value.type):
        return _Pointer(value)
    if _tla_type_bridge.type_is_tensor(value.type):
        return _Tensor(value)
    if _tla_type_bridge.type_is_mutex(value.type):
        return _MutexValue(value, "", -1)
    if isinstance(value.type, mlir_ir.IndexType):
        return _runtime._IndexExpr(value)
    if mlir_ir.IntegerType.isinstance(value.type):
        int_type = mlir_ir.IntegerType(value.type)
        if int_type.width == 1:
            return _runtime._BoolExpr(value)
    return value


def unpack_to_irvalue(
    mixed_values: list[Any] | tuple[Any, ...],
    body_name: str,
    full_write_args_count: int = 0,
    mixed_value_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[mlir_ir.Value], tuple[list[Any], list[str]]]:
    """Flatten frontend values into MLIR values for dynamic SCF regions."""
    del full_write_args_count
    from .base_dsl.utils import tree_utils

    names = tuple(
        mixed_value_names
        if mixed_value_names is not None
        else (str(index) for index in _builtins.range(len(mixed_values)))
    )
    specs: list[Any] = []
    ir_values: list[mlir_ir.Value] = []
    leaf_names: list[str] = []
    for index, value in enumerate(mixed_values):
        name = names[index] if index < len(names) else str(index)
        leaves, spec, names_for_value = tree_utils.flatten_frontend_if_tree(value, name)
        ir_values.extend(leaves)
        specs.append(spec)
        leaf_names.extend(names_for_value)
    if not all(isinstance(value, mlir_ir.Value) for value in ir_values):
        raise TlaCoreAPIError(
            f"Dynamic {body_name} values must flatten to MLIR SSA values"
        )
    return ir_values, (specs, leaf_names)


def _collect_tla_tensor_type_metadata(
    ir_values: list[mlir_ir.Value] | tuple[mlir_ir.Value, ...],
) -> list[TlaTensorTypeDescriptor | None]:
    return [
        _tla_tensor_type_for_mlir_value(value)
        if _tla_type_bridge.type_is_tensor(value.type)
        else None
        for value in ir_values
    ]


def pack_from_irvalue(
    ir_values: list[mlir_ir.Value] | tuple[mlir_ir.Value, ...],
    pytree_def: tuple[list[Any], list[str]],
    mixed_values: list[Any] | tuple[Any, ...],
    full_write_args_count: int = 0,
    tensor_type_metadata: list[TlaTensorTypeDescriptor | None] | None = None,
) -> list[Any]:
    """Rebuild frontend values from MLIR values produced by dynamic SCF ops."""
    del full_write_args_count
    from .base_dsl.utils import tree_utils

    specs, _ = pytree_def
    if tensor_type_metadata is None:
        source_values, _ = unpack_to_irvalue(
            mixed_values, "SCF tensor metadata propagation"
        )
        tensor_type_metadata = _collect_tla_tensor_type_metadata(source_values)
    if len(tensor_type_metadata) != len(ir_values):
        raise TlaCoreAPIError(
            "Dynamic SCF result count does not match its carried value metadata"
        )
    for result, tensor_type in zip(ir_values, tensor_type_metadata, strict=True):
        if not _tla_type_bridge.type_is_tensor(result.type):
            continue
        if tensor_type is None:
            raise TlaCoreAPIError(
                "Dynamic SCF tensor result is missing structured type metadata"
            )
        # SCF already requires every carried edge to have the same MLIR tensor
        # type. Preserve that structured type model on block arguments and
        # results so they remain valid inputs to tile_view/make_tensor_like.
        # The _Tensor tree reconstruction below restores runtime descriptor
        # leaves from the accompanying SCF-carried index SSA values.
        _register_tla_tensor_type(result, tensor_type)
    wrapped = [_wrap_frontend_value(value) for value in ir_values]
    return list(tree_utils.rebuild_frontend_if_carried_values(wrapped, specs))



def _const_attr(value: Any) -> mlir_ir.Attribute:
    if isinstance(value, Numeric) and isinstance(value.value, (bool, int, float)):
        value = value.value
    if isinstance(value, bool):
        return mlir_ir.BoolAttr.get(value)
    if isinstance(value, int):
        return mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(64), value)
    if isinstance(value, float):
        return mlir_ir.FloatAttr.get_f32(value)
    if isinstance(value, str):
        return mlir_ir.StringAttr.get(value)
    return mlir_ir.StringAttr.get(str(value))


def _coerce_type(type_like: Any) -> mlir_ir.Type:
    if isinstance(type_like, mlir_ir.Type):
        return type_like
    to_mlir_type = getattr(type_like, "to_mlir_type", None)
    if callable(to_mlir_type):
        return to_mlir_type()
    raise TypeError(
        "expected mlir.ir.Type or object with to_mlir_type(); "
        f"got {type(type_like).__name__}"
    )


def _const_int_value(value: Any) -> int | None:
    # Treat plain Python ints as static dimensions first so they are not confused with
    # execution-lowering arg_bindings: e.g. type_args=(4,) binds id(4) to a block arg, and
    # resolving before this check could make literals in make_shape(4, 8, 16) look dynamic.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return int(resolved)
    if isinstance(resolved, int):
        return resolved
    if isinstance(resolved, Numeric) and isinstance(resolved.value, (bool, int)):
        dtype = type(resolved).dtype.lower()
        if dtype.startswith("i") or dtype == "index":
            return int(resolved.value)
    if isinstance(resolved, mlir_ir.Value):
        owner = getattr(resolved, "owner", None)
        if owner is not None and getattr(owner, "name", "") == "arith.constant":
            attrs = owner.attributes
            if "value" in attrs:
                attr = attrs["value"]
                if isinstance(attr, mlir_ir.IntegerAttr):
                    return int(attr.value)
    return None


def _align_i64_value(
    value: mlir_ir.Value,
    byte_alignment: int,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    alignment = _const_i64(byte_alignment, loc=loc)
    alignment_minus_one = _const_i64(byte_alignment - 1, loc=loc)
    aligned = mlir_ir.Operation.create(
        "arith.addi",
        operands=[value, alignment_minus_one],
        results=[mlir_ir.IntegerType.get_signless(64)],
        loc=loc,
    ).results[0]
    aligned = mlir_ir.Operation.create(
        "arith.divui",
        operands=[aligned, alignment],
        results=[mlir_ir.IntegerType.get_signless(64)],
        loc=loc,
    ).results[0]
    return mlir_ir.Operation.create(
        "arith.muli",
        operands=[aligned, alignment],
        results=[mlir_ir.IntegerType.get_signless(64)],
        loc=loc,
    ).results[0]


def _refine_pointer_alignment(current_alignment: int, offset: Any) -> int:
    if current_alignment <= 1:
        return 1
    const_offset = _const_int_value(offset)
    if const_offset is None:
        return 1
    return _builtins.max(1, math.gcd(current_alignment, abs(const_offset)))


def _register_tla_tensor_type(
    value: mlir_ir.Value,
    tensor_type: TlaTensorTypeDescriptor,
) -> None:
    st = _runtime._current_frontend_state()
    if st is None:
        return
    st.tensor_type_by_value[value] = tensor_type


def _tla_tensor_type_for_mlir_value(v: mlir_ir.Value) -> TlaTensorTypeDescriptor:
    """Resolve structured ``!tla.tensor`` metadata for an SSA value."""
    st = _runtime._current_frontend_state()
    if st is not None:
        cached = st.tensor_type_by_value.get(v)
        if cached is not None:
            return cached
        host = st.tensor_host_by_value.get(v)
        if host is not None:
            desc = host.tla_tensor_type_descriptor()
            st.tensor_type_by_value[v] = desc
            return desc
    raise TlaLoweringError(
        "missing structured Tla tensor metadata for SSA value; tensor values used by "
        "Python lowering must come from a host tla.Tensor argument or a Tla Python op "
        f"that registered TlaTensorTypeDescriptor, got {str(v.type)!r}"
    )


def _tla_tensor_descriptor_from_type_or_value(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
) -> TlaTensorTypeDescriptor:
    if isinstance(source, mlir_ir.Value):
        return _tla_tensor_type_for_mlir_value(source)
    if isinstance(source, TlaTensorTypeDescriptor):
        return source
    raise TlaLoweringError(
        "expected a vector SSA value with registered TlaTensorTypeDescriptor "
        f"or a TlaTensorTypeDescriptor, got {type(source).__name__}"
    )


def _register_tla_tensor_metadata(
    value: mlir_ir.Value, metadata: dict[str, Any]
) -> None:
    st = _runtime._current_frontend_state()
    if st is None:
        return
    st.tensor_metadata_by_value[value] = metadata


def _tensor_metadata_field(value: mlir_ir.Value, field: str) -> Any:
    st = _runtime._current_frontend_state()
    if st is not None:
        cached = st.tensor_metadata_by_value.get(value)
        if cached is not None and field in cached:
            return cached[field]
    metadata = _tla_tensor_type_for_mlir_value(value).metadata()
    if st is not None:
        st.tensor_metadata_by_value[value] = metadata
    if field not in metadata:
        raise TlaLoweringError(f"unknown tensor metadata field: {field}")
    return metadata[field]


# Layout constants aligned with ``catlass/catlass.hpp`` and ``tla/layout.hpp``.
_CATLASS_BYTE_PER_C0 = 32
_CATLASS_C0_NUM_PER_FRACTAL = 16


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceil_div divisor must be positive")
    return (a + b - 1) // b


def _round_up(a: int, m: int) -> int:
    return _ceil_div(a, m) * m


def _mul_int_optional(a: int | None, b: int) -> int | None:
    if a is None:
        return None
    return a * b


def _as_index_expr_or_int(value: Any) -> Any:
    const = _const_int_value(value)
    if const is not None:
        return int(const)
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, _runtime._IndexExpr):
        return resolved
    if isinstance(resolved, mlir_ir.Value):
        return _runtime._IndexExpr(resolved)
    return value


def _components_to_index_tree(components: Any) -> Any:
    if isinstance(components, tuple):
        return tuple(_components_to_index_tree(x) for x in components)
    return _as_index_expr_or_int(components)


def _components_to_type_tree(components: Any) -> Any:
    if isinstance(components, tuple):
        return tuple(_components_to_type_tree(x) for x in components)
    const = _const_int_value(components)
    return int(const) if const is not None else None


def _tree_add(a: Any, b: Any) -> Any:
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise TlaLoweringError(
                "tensor metadata tree rank mismatch in coord addition"
            )
        return tuple(_tree_add(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, str) or isinstance(b, str):
        return "?"
    return a + b


def _tree_crop_origin(parent_origin: Any, tile_shape: Any, tile_coord: Any) -> Any:
    if isinstance(parent_origin, tuple):
        if not (isinstance(tile_shape, tuple) and isinstance(tile_coord, tuple)):
            raise TlaLoweringError("tensor metadata tree rank mismatch in origin crop")
        if len(parent_origin) != len(tile_shape) or len(parent_origin) != len(
            tile_coord
        ):
            raise TlaLoweringError("tensor metadata tree rank mismatch in origin crop")
        return tuple(
            _tree_crop_origin(po, ts, tc)
            for po, ts, tc in zip(parent_origin, tile_shape, tile_coord, strict=True)
        )
    if (
        isinstance(parent_origin, str)
        or isinstance(tile_shape, str)
        or isinstance(tile_coord, str)
    ):
        return "?"
    if (
        isinstance(parent_origin, int)
        and isinstance(tile_shape, int)
        and isinstance(tile_coord, int)
    ):
        rest = parent_origin - tile_coord
        return rest if rest < tile_shape else tile_shape
    rest = parent_origin - tile_coord
    rest_v = _runtime._coerce_index_value(rest)
    tile_v = _runtime._coerce_index_value(tile_shape)
    op = mlir_ir.Operation.create(
        "arith.minsi",
        operands=[rest_v, tile_v],
        results=[mlir_ir.IndexType.get()],
    )
    return _runtime._IndexExpr(op.results[0])


def _ceil_div_expr(a: Any, b: int) -> Any:
    if isinstance(a, str):
        return "?"
    return (a + (b - 1)) // b


def _round_up_expr(a: Any, m: int) -> Any:
    if isinstance(a, str):
        return "?"
    return _ceil_div_expr(a, m) * m


def _materialize_layout_trees_from_origin(
    origin_shape: Any, dtype: str, layout: str
) -> tuple[Any, Any, Any, Any] | None:
    if not isinstance(origin_shape, tuple) or len(origin_shape) != 2:
        return None
    rows, cols = origin_shape
    element_bytes = dtype_size_bytes(dtype)
    if element_bytes <= 0:
        return None
    ele_num_per_c0 = _builtins.max(1, _CATLASS_BYTE_PER_C0 // element_bytes)
    ele_num_per_fractal = _builtins.max(
        1, (_CATLASS_BYTE_PER_C0 * _CATLASS_C0_NUM_PER_FRACTAL) // element_bytes
    )
    c0_num_per_fractal = _CATLASS_C0_NUM_PER_FRACTAL
    coord = (0, 0)
    if layout == "row_major":
        return ((rows, cols), (cols, 1), coord, origin_shape)
    if layout == "column_major":
        return ((rows, cols), (1, rows), coord, origin_shape)
    if layout == "zN":
        rows_ru = _round_up_expr(rows, c0_num_per_fractal)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (ele_num_per_c0, _ceil_div_expr(cols, ele_num_per_c0)),
            ),
            ((ele_num_per_c0, ele_num_per_fractal), (1, rows_ru * ele_num_per_c0)),
            coord,
            origin_shape,
        )
    if layout == "nZ":
        cols_ru = _round_up_expr(cols, c0_num_per_fractal)
        return (
            (
                (ele_num_per_c0, _ceil_div_expr(rows, ele_num_per_c0)),
                (c0_num_per_fractal, _ceil_div_expr(cols, c0_num_per_fractal)),
            ),
            ((1, cols_ru * ele_num_per_c0), (ele_num_per_c0, ele_num_per_fractal)),
            coord,
            origin_shape,
        )
    if layout == "zZ":
        cols_ru = _round_up_expr(cols, ele_num_per_c0)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (ele_num_per_c0, _ceil_div_expr(cols, ele_num_per_c0)),
            ),
            ((ele_num_per_c0, cols_ru * c0_num_per_fractal), (1, ele_num_per_fractal)),
            coord,
            origin_shape,
        )
    if layout == "L0Clayout":
        rows_ru = _round_up_expr(rows, c0_num_per_fractal)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (c0_num_per_fractal, _ceil_div_expr(cols, c0_num_per_fractal)),
            ),
            ((c0_num_per_fractal, 256), (1, rows_ru * c0_num_per_fractal)),
            coord,
            origin_shape,
        )
    return None


def _remap_tensor_like_prefix_fields_for_layout_trees(
    origin_shape: Any,
    dtype: str,
    layout: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]] | None:
    """Derive shape/stride/coord/origin as nested tuple trees for ``layout`` (TLA-style when fractal).

    ``origin_shape`` must be a flat ``(N,)`` or ``(M, N)`` Tla index tree with ``int`` or
    ``None`` leaves. ``None`` represents an unknown dimension, spelled ``?`` in MLIR.
    Rank-1 ``row_major`` uses ``(N):(1)``, ``coord=(0,)``. Rank-2 **coord** is always
    ``(0, 0)``. Naming follows ``tla::GetTileLayout`` / fractal ``MakeLayout``
    (``rows`` / ``cols`` / ``ELE_NUM_PER_C0`` / ``C0_NUM_PER_FRACTAL``).
    """
    if isinstance(origin_shape, tuple) and len(origin_shape) == 1:
        length = origin_shape[0]
        if isinstance(length, tuple):
            return None
        layout_tag = layout.strip()
        if layout_tag == "row_major":
            return ((length,), (1,), (0,), (length,))
        return None

    origin_pair = _flat_dim_pair_from_tree(origin_shape)
    if origin_pair == (None, None) and origin_shape != (None, None):
        return None
    rows, cols = origin_pair
    element_bytes = dtype_size_bytes(dtype)
    if element_bytes <= 0:
        return None
    ele_num_per_c0 = _builtins.max(1, _CATLASS_BYTE_PER_C0 // element_bytes)
    ele_num_per_fractal = _builtins.max(
        1, (_CATLASS_BYTE_PER_C0 * _CATLASS_C0_NUM_PER_FRACTAL) // element_bytes
    )
    c0_num_per_fractal = _CATLASS_C0_NUM_PER_FRACTAL
    layout_tag = layout.strip()
    origin_shape_tree: tuple[Any, ...] = (rows, cols)
    coord_tree: tuple[Any, ...] = (0, 0)

    if layout_tag == "row_major":
        return ((rows, cols), (cols, 1), coord_tree, origin_shape_tree)
    if layout_tag == "column_major":
        return ((rows, cols), (1, rows), coord_tree, origin_shape_tree)
    if layout_tag == "zN":
        rows_round_up = None if rows is None else _round_up(rows, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, ele_num_per_c0)
        layout_shape = (
            (c0_num_per_fractal, ceil_div_rows),
            (ele_num_per_c0, ceil_div_cols),
        )
        stride_scale = _mul_int_optional(rows_round_up, ele_num_per_c0)
        layout_stride = (
            (ele_num_per_c0, ele_num_per_fractal),
            (1, stride_scale),
        )
        return layout_shape, layout_stride, coord_tree, origin_shape_tree
    if layout_tag == "nZ":
        cols_round_up = None if cols is None else _round_up(cols, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, ele_num_per_c0)
        ceil_div_cols = None if cols is None else _ceil_div(cols, c0_num_per_fractal)
        layout_shape = (
            (ele_num_per_c0, ceil_div_rows),
            (c0_num_per_fractal, ceil_div_cols),
        )
        stride_scale = _mul_int_optional(cols_round_up, ele_num_per_c0)
        layout_stride = (
            (1, stride_scale),
            (ele_num_per_c0, ele_num_per_fractal),
        )
        return layout_shape, layout_stride, coord_tree, origin_shape_tree
    if layout_tag == "zZ":
        cols_round_up = None if cols is None else _round_up(cols, ele_num_per_c0)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, ele_num_per_c0)
        layout_shape = (
            (c0_num_per_fractal, ceil_div_rows),
            (ele_num_per_c0, ceil_div_cols),
        )
        stride_scale = _mul_int_optional(cols_round_up, c0_num_per_fractal)
        layout_stride = (
            (ele_num_per_c0, stride_scale),
            (1, ele_num_per_fractal),
        )
        return layout_shape, layout_stride, coord_tree, origin_shape_tree
    if layout_tag == "L0Clayout":
        # Keep L0C consistent with tla::MakeLayout<..., L0C>, which uses
        # a fixed fractal element count (256) regardless of dtype.
        l0c_ele_num_per_fractal = 256
        rows_round_up = None if rows is None else _round_up(rows, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, c0_num_per_fractal)
        layout_shape = (
            (c0_num_per_fractal, ceil_div_rows),
            (c0_num_per_fractal, ceil_div_cols),
        )
        stride_scale = _mul_int_optional(rows_round_up, c0_num_per_fractal)
        layout_stride = (
            (c0_num_per_fractal, l0c_ele_num_per_fractal),
            (1, stride_scale),
        )
        return layout_shape, layout_stride, coord_tree, origin_shape_tree
    return None


def _flat_dim_pair_from_tree(tree: Any) -> tuple[int | None, int | None]:
    if not isinstance(tree, tuple) or len(tree) != 2:
        return (None, None)
    out: list[int | None] = []
    for item in tree:
        if item is None:
            out.append(None)
        elif isinstance(item, int):
            out.append(item)
        else:
            return (None, None)
    return (out[0], out[1])


def _remap_tensor_like_trees_for_layout(
    origin_shape: Any,
    dtype: str,
    layout: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]] | None:
    return _remap_tensor_like_prefix_fields_for_layout_trees(
        origin_shape, dtype, layout
    )


def _tla_value_element_type(value_type: mlir_ir.Type) -> mlir_ir.Type:
    element_type = _tla_type_bridge.value_element_type_get(
        value_type.context, value_type
    )
    if element_type is None:
        raise TlaLoweringError(f"expected typed !tla.value<element>, got {value_type}")
    return element_type


def _logical_tensor_shape_from_metadata(value: mlir_ir.Value) -> tuple[int | None, ...]:
    """Recover the logical tensor shape from registered/frontend tensor metadata."""
    for field in ("origin_shape", "shape"):
        shape = _tensor_metadata_field(value, field)
        if isinstance(shape, int):
            return (shape,)
        if isinstance(shape, tuple) and all(
            dim is None or isinstance(dim, int) for dim in shape
        ):
            return shape
    raise TlaLoweringError(
        "expected flat tensor metadata shape/origin_shape for register fragment"
    )


def _elem_from_type(type_str: str) -> str | None:
    """Element type from spelled ``!tla.value``."""
    if type_str.startswith("!tla.value<") and type_str.endswith(">"):
        return type_str[len("!tla.value<") : -1].strip() or None
    return None


def _layout_attr_from_value(value: mlir_ir.Value) -> str | None:
    owner = getattr(value, "owner", None)
    attrs = getattr(owner, "attributes", None)
    if attrs is None:
        return None
    attr = None
    for name in ("layouttag", "layout"):
        try:
            attr = attrs.get(name)
        except AttributeError:
            try:
                attr = attrs[name]
            except Exception:
                attr = None
        except Exception:
            attr = None
        if attr is not None:
            break
    if attr is None:
        return None
    text = str(attr).strip('"')
    return text or None


def _validate_mmad_contract(
    acc: mlir_ir.Value, lhs: mlir_ir.Value, rhs: mlir_ir.Value
) -> None:
    acc_desc = _tla_tensor_type_for_mlir_value(acc)
    lhs_desc = _tla_tensor_type_for_mlir_value(lhs)
    rhs_desc = _tla_tensor_type_for_mlir_value(rhs)

    addrspaces = (
        acc_desc.addrspace,
        lhs_desc.addrspace,
        rhs_desc.addrspace,
    )
    if addrspaces != ("l0c", "l0a", "l0b"):
        raise TlaLoweringError(
            "unsupported tla.mmad tile addrspaces; expected acc/lhs/rhs in "
            "l0c/l0a/l0b"
        )

    element_types = (
        lhs_desc.element_type,
        rhs_desc.element_type,
        acc_desc.element_type,
    )
    if element_types not in {
        ("f16", "f16", "f32"),
        ("bf16", "bf16", "f32"),
        ("f32", "f32", "f32"),
    }:
        raise TlaLoweringError(
            "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 -> f32, "
            "or f32,f32 -> f32 (L0C accumulator is fp32)"
        )

    lhs_m, lhs_k = _flat_dim_pair_from_tree(lhs_desc.origin_shape)
    rhs_k, rhs_n = _flat_dim_pair_from_tree(rhs_desc.origin_shape)
    acc_m, acc_n = _flat_dim_pair_from_tree(acc_desc.origin_shape)
    if None not in (lhs_m, lhs_k, rhs_k, rhs_n, acc_m, acc_n) and (
        lhs_k != rhs_k or lhs_m != acc_m or rhs_n != acc_n
    ):
        raise TlaLoweringError(
            "unsupported tla.mmad tile shape contract; expected lhs(MxK), rhs(KxN), acc(MxN)"
        )

    expected_layouts = ((acc, "L0Clayout"), (lhs, "zN"), (rhs, "nZ"))
    for operand, expected in expected_layouts:
        layout = _tla_tensor_type_for_mlir_value(operand).layout_tag
        layout = layout or _layout_attr_from_value(operand)
        if layout is not None and layout != expected:
            raise TlaLoweringError(
                "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ"
            )


def _flat_pair_sum_type_tree(a: Any, b: Any) -> tuple[int | None, int | None] | None:
    a_pair = _flat_dim_pair_from_tree(a)
    b_pair = _flat_dim_pair_from_tree(b)
    if a_pair == (None, None) and a != (None, None):
        return None
    if b_pair == (None, None) and b != (None, None):
        return None
    out: list[int | None] = []
    for lhs, rhs in zip(a_pair, b_pair, strict=True):
        out.append(None if lhs is None or rhs is None else lhs + rhs)
    return (out[0], out[1])


def _tree_contains_dynamic(tree: Any) -> bool:
    if isinstance(tree, tuple):
        return any(_tree_contains_dynamic(x) for x in tree)
    return tree is None


def _crop_origin_shape_type_tree(
    parent_origin: Any, tile_shape: Any, tile_coord: Any
) -> Any | None:
    origin_pair = _flat_dim_pair_from_tree(parent_origin)
    shape_pair = _flat_dim_pair_from_tree(tile_shape)
    coord_pair = _flat_dim_pair_from_tree(tile_coord)
    if (
        (origin_pair == (None, None) and parent_origin != (None, None))
        or (shape_pair == (None, None) and tile_shape != (None, None))
        or (coord_pair == (None, None) and tile_coord != (None, None))
    ):
        return None
    out: list[int | None] = []
    for dim, (origin, shape, coord) in enumerate(
        zip(origin_pair, shape_pair, coord_pair, strict=True), start=1
    ):
        if origin is None or shape is None or coord is None:
            out.append(None)
        else:
            if coord < 0:
                raise TlaLoweringError(
                    f"tile_view: element offset along dimension {dim} is negative ({coord}); "
                    "parent origin_shape requires non-negative offsets"
                )
            if coord >= origin:
                raise TlaLoweringError(
                    f"tile_view: element offset along dimension {dim} ({coord}) is out of range "
                    "for parent origin_shape; each offset must be strictly less than the "
                    "corresponding logical extent"
                )
            out.append(shape if origin - coord >= shape else origin - coord)
    return (out[0], out[1])


def _metadata_from_type_tree(tree: Any, dynamic_values: Iterable[Any]) -> Any:
    dyn_iter = iter(dynamic_values)

    def walk(node: Any) -> Any:
        if isinstance(node, tuple):
            return tuple(walk(x) for x in node)
        if node is None:
            return next(dyn_iter)
        return node

    return walk(tree)


def _format_tensor_type_descriptor(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
    shape_tree: Any,
    coord_tree: Any,
) -> TlaTensorTypeDescriptor:
    """Build the ``tile_view`` result tensor descriptor."""
    parent = _tla_tensor_descriptor_from_type_or_value(source)
    tile_shape = _components_to_type_tree(shape_tree)
    tile_coord = _components_to_type_tree(coord_tree)
    layout_remap = _remap_tensor_like_trees_for_layout(
        tile_shape, parent.element_type, parent.layout_tag
    )
    shape = layout_remap[0] if layout_remap is not None else tile_shape
    # ``tile_view`` follows the parent storage stride (TLA ``GetTileLayout``): only the
    # logical view shape/coord/origin update here; fractal stride is not re-derived from the tile.
    stride = parent.stride
    coord = _flat_pair_sum_type_tree(parent.coord, tile_coord)
    origin_shape = _crop_origin_shape_type_tree(
        parent.origin_shape, tile_shape, tile_coord
    )
    if origin_shape is None:
        if coord is not None and _tree_contains_dynamic(coord):
            origin_shape = (None, None)
        else:
            origin_shape = shape
    if coord is None:
        coord = tile_coord
    return TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape),
            stride=TlaIndexTreeType("stride", stride),
            origin_shape=TlaIndexTreeType("shape", origin_shape),
            layout_tag=parent.layout_tag,
        ),
        coord=coord,
        element_type=parent.element_type,
        addrspace=parent.addrspace,
        ptr_alignment=parent.ptr_alignment,
    )


def _format_tensor_type(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
    shape_value: mlir_ir.Value,
    coord_value: mlir_ir.Value,
) -> str:
    """Build ``tile_view`` result ``!tla.tensor<…>`` from source tensor + shape/coord SSA.

    **Stride**, **dtype**, **addr**, **layout** follow the parent tensor (including packed
    layouts: stride is never replaced by a ``tile_view``-local fractal remap).

    **Shape** (memory-layout field): for flat ``M,N`` tile sizes, when
    :func:`_remap_tensor_like_prefix_fields_for_layout_trees` applies to the parent's layout tag,
    the nested fractal spellings are taken from that remap—using the tile's flat logical
    ``M,N`` as the logical pair (aligned with ``tla::GetTileLayout`` for non-MxScale paths).
    Otherwise the ``!tla.shape<…>`` spelling is used unchanged.

    **Coord** and **origin** follow ``tla::TileViewImpl`` / ``GetTileLayout`` / ``CropOriginShape``:
    element offset is the Hadamard product already encoded in ``coord_value``'s ``!tla.coord<…>``;
    ``coordNew = Add(parent.coord, offset)`` and ``origin`` is ``CropOriginShape(parent.origin,
    tileShape, offset)`` for flat trees: :func:`_crop_origin_shape_type_tree` applies the same ``min``
    semantics per dimension, emitting ``?`` where a leaf is dynamic. When **coord**
    carries ``?`` but origin cannot be flattened (nested fractal), **origin** uses ``?,?`` instead
    of mirroring the memory **shape** tree.

    Out-of-range tile starts (``offset_i >= parent_origin_i`` or negative offset) are rejected
    inside :func:`_crop_origin_shape_type_tree` with :class:`TlaLoweringError` where the offset is
    statically known.

    ``source`` must be an SSA value with registered structured tensor metadata or an explicit
    :class:`~catlass.types.TlaTensorTypeDescriptor`.
    """
    return _format_tensor_type_descriptor(source, shape_value, coord_value).to_asm()


_format_tile_type = _format_tensor_type


def _is_integer(value: Any) -> bool:
    """Return whether ``value`` is a static ``int``, index/integer SSA, or a bound index."""
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return False
    if isinstance(resolved, int):
        return True
    if isinstance(resolved, mlir_ir.Value) and isinstance(
        resolved.type, (mlir_ir.IndexType, mlir_ir.IntegerType)
    ):
        return True
    if _category(resolved) == "index":
        return True
    return False


def _flatten_tla_tuple(a: IndexTree) -> tuple[Any, ...]:
    """Flatten a nested tuple tree to leaves (depth-first leaf order)."""
    if not isinstance(a, tuple):
        return (a,)
    return tuple(chain.from_iterable(_flatten_tla_tuple(x) for x in a))


def _check_index_tree_group_depth(
    op_name: str, tree: Any, *, _tuple_depth: int = 0
) -> None:
    if not isinstance(tree, tuple):
        return
    if len(tree) == 0:
        _op_error(op_name, "expected non-empty tuple in index tree")
    if _tuple_depth >= 2:
        _op_error(
            op_name,
            "Tla index trees support only top-level leaves or one-level leaf groups",
        )
    for child in tree:
        _check_index_tree_group_depth(op_name, child, _tuple_depth=_tuple_depth + 1)


def _check_shape(shape: IndexTree) -> None:
    """Validate a shape tree (positive static sizes, nested tuple of shapes, or dynamic index)."""
    _check_index_tree_group_depth("make_shape", shape)
    if _is_integer(shape):
        resolved = _resolve_bound_value(shape)
        if isinstance(resolved, int) and resolved <= 0:
            _op_error(
                "make_shape",
                f"Expected size in shape to be strictly positive, but got {resolved}",
            )
        return
    if isinstance(shape, tuple):
        if len(shape) == 0:
            _op_error("make_shape", "expected non-empty tuple in shape tree")
        for s in shape:
            _check_shape(s)
        return
    _op_error(
        "make_shape",
        f"Expected Shape, which is a positive integer or tuple of Shapes, but got {_type_name(shape)}",
    )


def _check_coord(coord: IndexTree) -> None:
    """Validate a Coord tree: leaves are index-like; static leaves must be >= 0.

    Tla does not support ``None`` coord leaves in packing yet; leaves must satisfy :func:`_is_integer`.
    Compile-time-known negative leaves are rejected; dynamic SSA leaves are not checked here.
    """
    _check_index_tree_group_depth("make_coord", coord)
    if isinstance(coord, tuple):
        if len(coord) == 0:
            _op_error("make_coord", "expected non-empty tuple in coord tree")
        flat = _flatten_tla_tuple(coord)
        if len(flat) == 0:
            _op_error(
                "make_coord",
                f"Expected Coord with at least one leaf, but got {coord!r}",
            )
        if not all(_is_integer(c) for c in flat):
            _op_error(
                "make_coord",
                f"Expected Coord, whose leaves are integers, but got {coord!r}",
            )
    elif _is_integer(coord):
        flat = (coord,)
    else:
        _op_error(
            "make_coord",
            f"Expected Coord, which is an integer or tuple of Coords, but got {_type_name(coord)}",
        )
        return

    for c in flat:
        static = _const_int_value(c)
        if static is not None and static < 0:
            _op_error(
                "make_coord",
                f"Expected coord leaf >= 0, but got {static} (in {coord!r})",
            )


def _check_stride(stride: IndexTree) -> None:
    """Validate a Stride tree: leaves are index-like; static leaves must be > 0.

    Tla strides are index trees only (no scaled-basis leaves in this frontend).
    Compile-time-known non-positive leaves are rejected; dynamic SSA leaves are not checked here.
    """
    _check_index_tree_group_depth("make_stride", stride)
    if isinstance(stride, tuple):
        if len(stride) == 0:
            _op_error("make_stride", "expected non-empty tuple in stride tree")
        flat = _flatten_tla_tuple(stride)
        if len(flat) == 0:
            _op_error(
                "make_stride",
                f"Expected Stride with at least one leaf, but got {stride!r}",
            )
        if not all(_is_integer(s) for s in flat):
            _op_error(
                "make_stride",
                f"Expected Stride, whose leaves are integers, but got {stride!r}",
            )
    elif _is_integer(stride):
        flat = (stride,)
    else:
        _op_error(
            "make_stride",
            f"Expected Stride, which is an integer or tuple of Strides, but got {_type_name(stride)}",
        )
        return

    for s in flat:
        static = _const_int_value(s)
        if static is not None and static <= 0:
            _op_error(
                "make_stride",
                f"Expected stride leaf strictly positive, but got {static} (in {stride!r})",
            )


def _transform_leaf(f: Callable[..., Any], *args: Any) -> Any:
    if all(isinstance(t, tuple) for t in args):
        return tuple(_transform_leaf(f, *_a) for _a in zip(*args))
    if all(not isinstance(t, tuple) for t in args):
        return f(*args)
    raise TypeError(f"profile of input tuples doesn't match: {args}")


def _is_static(x: Any) -> bool:
    if isinstance(x, mlir_ir.Value):
        owner = getattr(x, "owner", None)
        if (
            owner is not None
            and str(getattr(owner, "name", "") or "") == "arith.constant"
        ):
            return True
        return False
    if isinstance(x, tuple):
        return all(_is_static(a) for a in x)
    if isinstance(x, (bool, int)):
        return True
    if isinstance(x, Numeric) and isinstance(getattr(x, "value", None), (bool, int)):
        return _const_int_value(x) is not None
    return False


def _pack_x(
    x: tuple[Any, ...],
    packer: Callable[[tuple[Any, ...]], tuple[str, list[mlir_ir.Value]]],
    op_name: str,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    x = _transform_leaf(_resolve_bound_value, x)
    res_ty, dyn_elems = packer(x)
    dyn_elems = [t for t in dyn_elems if not _is_static(t)]
    return mlir_ir.Operation.create(
        f"tla.{op_name}",
        results=[_coerce_type(res_ty)],
        operands=dyn_elems,
        loc=loc,
    ).results[0]


def _pack_tree(
    op_name: str,
    kind: str,
    components: tuple[Any, ...],
) -> tuple[TlaIndexTreeType, list[mlir_ir.Value]]:
    type_tree: list[Any] = []

    def pack_one(c: Any) -> tuple[str, list[mlir_ir.Value], Any]:
        if isinstance(c, list):
            _op_error(
                op_name,
                "expected nested tuple tree for make_* components, got list (use parentheses, not brackets)",
            )
        if isinstance(c, tuple):
            if len(c) == 0:
                _op_error(op_name, "expected non-empty nested tuple in tree")
            child_packs = [pack_one(x) for x in c]
            dyns: list[mlir_ir.Value] = []
            for _, d, _ in child_packs:
                dyns.extend(d)
            return (
                f"({','.join(ty for ty, _, _ in child_packs)})",
                dyns,
                tuple(tree for _, _, tree in child_packs),
            )
        _require_index(op_name, "leaf", c, 0)
        const = _const_int_value(c)
        if const is not None:
            return (str(const), [], int(const))
        return ("?", [_as_index_value(c)], None)

    parts: list[str] = []
    dyn: list[mlir_ir.Value] = []
    for c in components:
        frag, d, tree = pack_one(c)
        parts.append(frag)
        dyn.extend(d)
        type_tree.append(tree)
    return TlaIndexTreeType(kind, tuple(type_tree)), dyn


def _pack_shape(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_shape(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_shape", "shape", t),
        "make_shape",
        loc=loc,
    )


def _pack_coord(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_coord(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_coord", "coord", t),
        "make_coord",
        loc=loc,
    )


def _pack_stride(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_stride(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_stride", "stride", t),
        "make_stride",
        loc=loc,
    )


def _dtype_to_str(value: Any) -> str:
    if isinstance(value, mlir_ir.Type):
        try:
            return Numeric.from_mlir_type(value).dtype
        except TypeError:
            return str(value)
    if isinstance(value, type) and issubclass(value, Numeric):
        if not value.dtype:
            raise TypeError(
                f"expected concrete Numeric element type, got abstract {value!r}"
            )
        return value.dtype
    return str(value)


def _looks_dtype_literal(value: Any) -> bool:
    if isinstance(value, mlir_ir.Type):
        return True
    return (
        isinstance(value, type)
        and issubclass(value, Numeric)
        and bool(getattr(value, "dtype", ""))
    )


def _op_error(op_name: str, message: str) -> None:
    raise TlaCoreAPIError(f"tla.{op_name}: {message}")


def _type_name(value: Any) -> str:
    resolved = _resolve_bound_value(value)
    if resolved is not value:
        return _type_name(resolved)
    return type(value).__name__


def _category(value: Any) -> str | None:
    if isinstance(value, _Shape):
        return "shape"
    if isinstance(value, _Coord):
        return "coord"
    if isinstance(value, _Stride):
        return "stride"
    if isinstance(value, _Layout):
        return "layout"
    if isinstance(value, _Pointer):
        return "pointer"
    if isinstance(value, _Tensor):
        return "tensor"
    category = getattr(value, "__tla_category__", None)
    if isinstance(category, str):
        return category
    category = _runtime._resolve_frontend_bound_category(value)
    if isinstance(category, str):
        return category
    resolved = _resolve_bound_value(value)
    if resolved is not value:
        category = getattr(resolved, "__tla_category__", None)
        if isinstance(category, str):
            return category
        category = _runtime._resolve_frontend_bound_category(resolved)
        if isinstance(category, str):
            return category
        value = resolved
    if isinstance(value, _MutexValue):
        return "mutex"
    if isinstance(value, mlir_ir.Value):
        if isinstance(value.type, mlir_ir.IndexType):
            return "index"
        category = _tla_type_bridge.tla_type_category(value.type)
        if category is not None:
            return category
    value_type = getattr(value, "type", None)
    if value_type is not None:
        category = _tla_type_bridge.tla_type_category(value_type)
        if category is not None:
            return category
    return None


def _token(value: Any) -> str | None:
    if isinstance(value, str):
        token = value.strip().lower()
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            return None
        token = name.strip().lower()
    return token


def _name_token(value: Any) -> str | None:
    if isinstance(value, str):
        token = value.strip()
        return token or None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        token = name.strip()
        return token or None
    return None


def _require_arg_count(op_name: str, args: tuple[Any, ...], expected: int) -> None:
    if len(args) != expected:
        _op_error(op_name, f"expected {expected} argument(s), got {len(args)}")


def _require_no_kwargs(op_name: str, kwargs: dict[str, Any]) -> None:
    if kwargs:
        _op_error(
            op_name, f"does not accept keyword arguments: {', '.join(sorted(kwargs))}"
        )


def _require_index(op_name: str, name: str, value: Any, position: int) -> None:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected index, got bool",
        )
    if isinstance(resolved, int):
        return
    if _category(resolved) == "index":
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): expected index, got {_type_name(value)}",
    )


def _require_category(
    op_name: str, name: str, value: Any, expected: str, position: int
) -> None:
    if _category(value) != expected:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected {expected}, got {_type_name(value)}",
        )


def _require_categories(
    op_name: str,
    name: str,
    value: Any,
    expected: tuple[str, ...],
    position: int,
) -> None:
    if _category(value) not in expected:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected one of {expected}, got {_type_name(value)}",
        )


def _require_shape(op_name: str, value: Any, position: int) -> None:
    if _category(value) == "shape":
        return
    if isinstance(value, tuple):
        _check_shape(value, op_name=op_name, label="shape", position=position)
        return
    _op_error(
        op_name,
        f"invalid argument 'shape' (position {position}): expected shape, got {_type_name(value)}",
    )


def _require_coord(op_name: str, value: Any, position: int) -> None:
    if _category(value) == "coord":
        return
    if isinstance(value, tuple):
        _check_shape(
            value,
            op_name=op_name,
            label="coord",
            position=position,
            require_positive_static_int=False,
        )
        return
    _op_error(
        op_name,
        f"invalid argument 'coord' (position {position}): expected coord, got {_type_name(value)}",
    )


def _require_literal(op_name: str, name: str, value: Any, position: int) -> None:
    if not isinstance(value, mlir_ir.Value):
        bound = _runtime._resolve_frontend_bound_value(value)
        if bound is not None:
            raise TlaLoweringError(f"tla.{op_name} requires a literal")
    if isinstance(value, (int, float, bool, str, mlir_ir.Type)):
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): expected literal, got {_type_name(value)}",
    )


def _require_pipe(op_name: str, name: str, value: Any, position: int) -> None:
    token = _token(value)
    if token is None or token not in _PIPE_VALUES:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected pipe, got {_type_name(value)}",
        )


def _pipe_attr_from_token(
    pipe: PipeLike | str, *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Attribute:
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    pipe_value = str(_token(pipe)).lower()
    return mlir_ir.Attribute.parse(f"#tla.pipe<{pipe_value}>", context=ctx)


def _ensure_no_explicit_mutex_access_in_guard() -> None:
    state = _runtime._current_frontend_state()
    if state is None or state.mutex_guard_depth <= 0:
        return
    raise TlaCoreAPIError(
        "tla.mutex_guard body cannot contain explicit mutex lock/unlock calls"
    )


def _raw_operation(op_or_view: Any) -> mlir_ir.Operation:
    operation = getattr(op_or_view, "operation", op_or_view)
    if not isinstance(operation, mlir_ir.Operation):
        raise TlaLoweringError(
            "expected MLIR operation while scanning tla.mutex_guard body, got "
            f"{type(op_or_view).__name__}"
        )
    return operation


def _walk_mutex_guard_ops(ops: Sequence[Any]) -> list[mlir_ir.Operation]:
    walked: list[mlir_ir.Operation] = []

    def visit(op_or_view: Any) -> None:
        op = _raw_operation(op_or_view)
        walked.append(op)
        if op.name == "tla.vec.func":
            return
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    for op in ops:
        visit(op)
    return walked


def _infer_copy_mutex_pipe(copy_op: mlir_ir.Operation) -> str:
    operands = list(copy_op.operands)
    if len(operands) != 2 and len(operands) != 3:
        raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    src_addrspace = _tla_tensor_type_for_mlir_value(operands[1]).addrspace.lower()
    if src_addrspace == "l0c":
        if len(operands) != 3:
            raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    elif len(operands) != 2:
        raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    if src_addrspace == "gm":
        return "mte2"
    if src_addrspace == "l1":
        return "mte1"
    if src_addrspace == "ub":
        return "mte3"
    if src_addrspace == "l0c":
        return "fix"
    raise TlaLoweringError(
        "tla.mutex_guard cannot infer pipe for tla.copy with source addrspace "
        f"{src_addrspace!r}"
    )

def _infer_mutex_guard_pipe(body_ops: Sequence[Any]) -> str:
    inferred: list[str] = []
    for op in _walk_mutex_guard_ops(body_ops):
        name = op.name
        if name in {"tla.mutex_lock", "tla.mutex_unlock"}:
            raise TlaCoreAPIError(
                "tla.mutex_guard body cannot contain explicit mutex lock/unlock calls"
            )
        if name == "tla.copy":
            inferred.append(_infer_copy_mutex_pipe(op))
        elif name == "tla.mmad":
            inferred.append("cube")
        elif name == "tla.vec.func":
            inferred.append("vector")
    if not inferred:
        raise TlaLoweringError(
            "tla.mutex_guard body must emit at least one tla.copy, tla.mmad, "
            "or tla.vec.func"
        )
    unique = set(inferred)
    if len(unique) != 1:
        pipes = ", ".join(sorted(unique))
        raise TlaLoweringError(f"tla.mutex_guard body inferred multiple pipes: {pipes}")
    return inferred[0]


def _require_cross_mode(op_name: str, value: Any, position: int) -> None:
    token = _token(value)
    if token is None or token not in _CROSS_MODE_VALUES:
        _op_error(
            op_name,
            f"invalid argument 'mode' (position {position}): expected cross_mode, got {_type_name(value)}",
        )


def _require_pointer_addrspace(op_name: str, value: Any, position: int) -> str:
    """Require :class:`AddressSpace` for ``make_ptr`` / :class:`~catlass.utils.localmem_allocator.LocalmemAllocator`.

    Returns the MLIR addrspace keyword (``str(enum)`` == ``enum.name``). Callers that only validate may ignore it.
    """
    if not isinstance(value, AddressSpace):
        _op_error(
            op_name,
            f"invalid argument 'mem_space' (position {position}): expected AddressSpace, got {_type_name(value)}",
        )
    return str(value)


def _require_bool_or_value(op_name: str, name: str, value: Any, position: int) -> None:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return
    if _category(resolved) == "bool":
        return
    if _category(resolved) == "value":
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): expected bool|value, got {_type_name(value)}",
    )


_DEBUG_PRINT_I32_MIN = -(2**31)
_DEBUG_PRINT_I32_MAX = 2**31 - 1


def _debug_print_operand(value: Any, *, loc: mlir_ir.Location | None) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _op_error(
            "debug_print",
            "unsupported value type bool; expected a signless i32 or f32 scalar",
        )
    if isinstance(resolved, int):
        if not _DEBUG_PRINT_I32_MIN <= resolved <= _DEBUG_PRINT_I32_MAX:
            _op_error(
                "debug_print", f"Python int {resolved} is outside signless i32 range"
            )
        return _const_i32(resolved, loc=loc)
    if isinstance(resolved, float):
        return _const_f32(resolved, loc=loc)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, (int, float, bool)):
            dtype = type(resolved).dtype.lower()
            if dtype == "i32":
                return _const_i32(int(resolved.value), loc=loc)
            if dtype == "f32":
                return _const_f32(float(resolved.value), loc=loc)
            _op_error(
                "debug_print",
                f"unsupported value type {dtype}; expected a signless i32 or f32 scalar",
            )
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, VectorSSA):
        resolved = _resolve_bound_value(resolved.value)
    if not isinstance(resolved, mlir_ir.Value):
        _op_error(
            "debug_print",
            f"unsupported value type {_type_name(value)}; expected a signless i32 or f32 scalar",
        )

    value_type = resolved.type
    if isinstance(value_type, mlir_ir.F32Type):
        return resolved
    if mlir_ir.IntegerType.isinstance(value_type):
        int_type = mlir_ir.IntegerType(value_type)
        if int_type.width == 32 and int_type.is_signless:
            return resolved
    _op_error(
        "debug_print",
        f"unsupported value type {value_type}; expected a signless i32 or f32 scalar",
    )


def debug_print(*args: Any, **kwargs: Any) -> None:
    """Emit one typed i32 or f32 debug value inside a cube or vector region."""
    if kwargs:
        _op_error("debug_print", "does not accept keyword arguments")
    if len(args) != 1:
        _op_error(
            "debug_print", f"expects exactly one positional argument; got {len(args)}"
        )
    value = _resolve_bound_value(args[0])
    if isinstance(value, bool):
        _op_error(
            "debug_print",
            "unsupported value type bool; expected a signless i32 or f32 scalar",
        )
    if (
        isinstance(value, int)
        and not _DEBUG_PRINT_I32_MIN <= value <= _DEBUG_PRINT_I32_MAX
    ):
        _op_error("debug_print", f"Python int {value} is outside signless i32 range")
    _require_frontend_state("debug_print")
    _runtime._require_enclosing_cube_or_vector("debug_print")
    loc = _capture_user_loc()
    _tla_ops_gen.debug_print(_debug_print_operand(value, loc=loc), loc=loc)


debug_print.__signature__ = inspect.Signature(
    [inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY)]
)


def _require_dtype(op_name: str, name: str, value: Any, position: int) -> None:
    if isinstance(value, mlir_ir.Type):
        return
    if (
        isinstance(value, type)
        and issubclass(value, Numeric)
        and getattr(value, "dtype", "")
    ):
        return
    if isinstance(value, str):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            "use a concrete Numeric type (e.g. tla.Float32) or mlir_ir.Type, not a str.",
        )
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): "
        f"expected mlir_ir.Type or concrete Numeric (e.g. tla.Float32), got {_type_name(value)}",
    )


def _require_byte_alignment(op_name: str, value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _op_error(
            op_name,
            f"invalid argument 'byte_alignment' (position {position}): expected positive_int, "
            f"got {_type_name(value)}",
        )
    return int(value)


def _require_allocation_dtype(op_name: str, dtype: Any) -> tuple[type[Numeric], int]:
    if (
        not isinstance(dtype, type)
        or not issubclass(dtype, Numeric)
        or not getattr(dtype, "dtype", "")
    ):
        _op_error(
            op_name,
            f"invalid argument 'dtype' (position 1): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    width = int(getattr(dtype, "width", 0) or 0)
    if width <= 0 or width % 8 != 0:
        _op_error(
            op_name,
            f"unsupported allocation dtype {dtype.dtype}; expected byte-addressable "
            "fixed-width scalar Numeric",
        )
    element_bytes = dtype_size_bytes(str(dtype.dtype))
    if element_bytes <= 0:
        _op_error(
            op_name,
            f"unsupported allocation dtype {dtype.dtype}; expected byte-addressable "
            "fixed-width scalar Numeric",
        )
    return dtype, element_bytes


def _static_allocation_size_bytes(
    op_name: str,
    shape: ShapeLike,
    dtype: type[Numeric],
    element_bytes: int,
) -> int:
    _check_shape(shape)
    num_elements = 1
    for dim in _flatten_tla_tuple(shape):
        dim_const = _const_int_value(dim)
        if dim_const is None:
            raise TlaLoweringError(
                f"tla.{op_name} requires a static shape (compile-time constants); "
                "dynamic shapes are not supported."
            )
        if dim_const <= 0:
            _op_error(
                op_name,
                f"Expected size in shape to be strictly positive, but got {dim_const}",
            )
        num_elements *= int(dim_const)

    size_bytes = num_elements * element_bytes
    if size_bytes <= 0 or size_bytes > 9_223_372_036_854_775_807:
        raise TlaLoweringError(
            f"tla.{op_name} allocation size_bytes must be in [1, 2**63-1] "
            f"for tla.alloc_ptr {{size_bytes : i64}}; got {size_bytes} "
            f"for dtype {dtype.dtype}"
        )
    return size_bytes


@dsl_user_op
def make_shape(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaShape:
    """Build a packed Tla shape from nested tuple components."""
    if len(components) == 0:
        _op_error("make_shape", "expected at least 1 component")
    _require_frontend_state("make_shape")
    v = _pack_shape(components, loc=loc)
    return _Shape(shape_value=v, components=components)


@dsl_user_op
def make_coord(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaCoord:
    """Build a packed Tla coordinate from nested tuple components."""
    if len(components) == 0:
        _op_error("make_coord", "expected at least 1 component")
    _require_frontend_state("make_coord")
    v = _pack_coord(components, loc=loc)
    return _Coord(coord_value=v, components=components)


@dsl_user_op
def make_stride(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaStride:
    """Build a packed Tla stride from nested tuple components."""
    if len(components) == 0:
        _op_error("make_stride", "expected at least 1 component")
    _require_frontend_state("make_stride")
    v = _pack_stride(components, loc=loc)
    return _Stride(stride_value=v, components=tuple(components))


@dsl_user_op
def make_layout(
    shape: _Shape,
    stride: _Stride,
    *,
    origin_shape: _Shape | None = None,
    layoutTag: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> TlaLayout:
    """Combine packed :func:`make_shape` and :func:`make_stride` into ``!tla.layout`` (``tla.make_layout``).

    ``origin_shape`` defaults to ``shape``; a third operand is emitted only when its SSA differs from ``shape``'s.
    """
    if not isinstance(shape, _Shape) or not isinstance(stride, _Stride):
        _op_error(
            "make_layout",
            "expected shape from tla.make_shape (TlaShape) and stride from tla.make_stride (TlaStride); "
            f"got shape={_type_name(shape)}, stride={_type_name(stride)}",
        )
    if origin_shape is not None and not isinstance(origin_shape, _Shape):
        _op_error(
            "make_layout",
            "expected origin_shape from tla.make_shape (TlaShape) or None; "
            f"got {_type_name(origin_shape)}",
        )
    _require_frontend_state("make_layout")
    layout_token = _resolve_arch_layout_tag(layoutTag, for_op="make_layout")
    # Rank/layout consistency is enforced by C++ LayoutType::verify via LayoutType.get.
    shape_val = shape._shape_value
    stride_val = stride._stride_value
    origin_for_type: mlir_ir.Value | None = (
        origin_shape._shape_value if origin_shape is not None else None
    )
    layout_ty = LayoutType.get(
        shape_val, stride_val, origin_for_type, layout_tag=layout_token
    )
    origin_ssa: mlir_ir.Value | None = None
    if origin_shape is not None and origin_shape._shape_value is not shape_val:
        origin_ssa = origin_shape._shape_value
    attrs: dict[str, mlir_ir.Attribute] = {}
    if layout_token != "row_major":
        attrs["layoutTag"] = mlir_ir.StringAttr.get(layout_token)
    operands: list[mlir_ir.Value] = [shape_val, stride_val]
    if origin_ssa is not None:
        operands.append(origin_ssa)
    op = mlir_ir.Operation.create(
        "tla.make_layout",
        operands=operands,
        results=[layout_ty],
        attributes=attrs,
        loc=loc,
    )
    return _Layout(
        layout_value=op.results[0],
        shape=shape,
        stride=stride,
        origin_shape=origin_shape,
        layout_tag=layout_token,
    )


def _emit_tile_view(
    source: Any,
    shape: _Shape,
    coord: _Coord,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    _require_category("tile_view", "source", source, "tensor", 0)
    if not isinstance(shape, _Shape) or not isinstance(coord, _Coord):
        _op_error(
            "tile_view",
            f"expected shape from tla.make_shape (TlaShape) and coord from tla.make_coord (TlaCoord); "
            f"got shape={_type_name(shape)}, coord={_type_name(coord)}",
        )
    _require_frontend_state("tile_view")
    source_value = _as_value(source)
    view_ty = _format_tensor_type_descriptor(
        source_value, shape._components, coord._components
    )
    result = _tla_ops_gen.tile_view(
        _coerce_type(view_ty),
        source_value,
        shape._shape_value,
        coord._coord_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, view_ty)
    try:
        source_meta = {
            "stride": _tensor_metadata_field(source_value, "stride"),
            "coord": _tensor_metadata_field(source_value, "coord"),
            "origin_shape": _tensor_metadata_field(source_value, "origin_shape"),
        }
        tile_shape_tree = _components_to_index_tree(shape._components)
        tile_coord_tree = _components_to_index_tree(coord._components)
        shape_meta = _metadata_from_type_tree(
            view_ty.shape, list(_flatten_tla_tuple(tile_shape_tree))
        )
        metadata = {
            "shape": shape_meta,
            "stride": source_meta["stride"],
            "coord": _tree_add(source_meta["coord"], tile_coord_tree),
            "origin_shape": _tree_crop_origin(
                source_meta["origin_shape"], tile_shape_tree, tile_coord_tree
            ),
            "dtype": view_ty.element_type,
            "addrspace": view_ty.addrspace,
            "layout_tag": view_ty.layout_tag,
        }
        _register_tla_tensor_metadata(result, metadata)
    except Exception:
        # Keep lowering permissive: metadata property access falls back to type parsing.
        pass
    return _Tensor(result)


@dsl_user_op
def tile_view(
    source: Any,
    shape: _Shape,
    coord: _Coord,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Create a tile view using tile-coordinate granularity on a ``!tla.tensor`` source."""
    if not isinstance(shape, _Shape) or not isinstance(coord, _Coord):
        _op_error(
            "tile_view",
            f"expected shape from tla.make_shape (TlaShape) and coord from tla.make_coord (TlaCoord); "
            f"got shape={_type_name(shape)}, coord={_type_name(coord)}",
        )
    normalized_coord = normalize_tile_view_coord(
        shape_components=shape._components,
        coord_components=coord._components,
    )
    return _emit_tile_view(
        source,
        shape,
        make_coord(*normalized_coord, loc=loc),
        loc=loc,
    )


@dsl_user_op
def make_tensor(
    ptr: Any,
    layout: TlaLayout,
    coord: CoordLike | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Construct a ``!tla.tensor`` from an explicit pointer, layout, and coord.

    Unlike :func:`make_tensor_like` (which clones layout/coord from a reference tensor),
    this takes the layout and coord directly, mirroring the
    ``!tla.tensor<layout, coord, ptr>`` type structure:

        tla.make_tensor(ptr, tla.make_layout(shape, stride), coord=tla.make_coord(...))

    ``coord`` defaults to a zero coord matching the layout's rank (rank-2 layout ->
    ``make_coord(0, 0)``, rank-1 -> ``make_coord(0)``). Element type and address space
    come from ``ptr``'s ``!tla.ptr`` type; the layout tag, shape, stride, and origin come
    from the ``!tla.layout`` operand (origin defaults to ``shape``).

    Lowering supports linear layouts (RowMajor/ColumnMajor). Like :func:`make_tensor_like`,
    a full compile requires ``ptr`` to carry backing storage; an allocator-backed pointer
    (from :class:`~catlass.utils.LocalmemAllocator` + :func:`recast_ptr`) is the supported
    form for runnable kernels.
    """
    _require_category("make_tensor", "ptr", ptr, "pointer", 0)
    if not isinstance(layout, _Layout):
        _op_error(
            "make_tensor",
            "invalid argument 'layout': expected tla.make_layout (TlaLayout); "
            f"got {_type_name(layout)}",
        )
    _require_frontend_state("make_tensor")
    ptr_value = _as_value(ptr)
    if not PtrType.isinstance(ptr_value.type):
        _op_error(
            "make_tensor",
            f"invalid argument 'ptr': expected !tla.ptr, got {ptr_value.type}",
        )
    ptr_ty = PtrType(ptr_value.type)
    addr = ptr_ty.addrspace
    try:
        dtype = _dtype_to_str(ptr_ty.pointee).lower()
    except TypeError as exc:
        raise TlaLoweringError(
            f"tla.make_tensor cannot derive element type from ptr pointee {ptr_ty.pointee}"
        ) from exc
    if dtype not in {"f16", "bf16", "f32", "i32", "i16", "i1", "i8"}:
        raise TlaLoweringError(
            f"tla.make_tensor expects a supported element type, got [{dtype}]"
        )

    shape_tree = _components_to_index_tree(layout._shape._components)
    stride_tree = _components_to_index_tree(layout._stride._components)
    origin_tree = (
        _components_to_index_tree(layout._origin_shape._components)
        if layout._origin_shape is not None
        else shape_tree
    )

    shape_rank = len(_flatten_tla_tuple(shape_tree))
    stride_rank = len(_flatten_tla_tuple(stride_tree))
    if shape_rank not in (1, 2) or stride_rank not in (1, 2):
        raise TlaLoweringError(
            f"tla.make_tensor supports at most 2-D layouts (got shape rank "
            f"{shape_rank}, stride rank {stride_rank})"
        )

    if coord is None:
        rank = shape_rank
        coord = make_coord(*([0] * rank), loc=loc)
    elif not isinstance(coord, _Coord):
        _op_error(
            "make_tensor",
            "invalid argument 'coord': expected tla.make_coord (TlaCoord) or None; "
            f"got {_type_name(coord)}",
        )
    coord_tree = _components_to_index_tree(coord._components)
    coord_rank = len(_flatten_tla_tuple(coord_tree))
    if coord_rank != shape_rank:
        raise TlaLoweringError(
            f"tla.make_tensor coord rank must match layout rank (got coord rank "
            f"{coord_rank}, expected {shape_rank})"
        )

    # Type trees spell dynamic leaves as ``None`` (``?`` in the ``!tla.tensor`` type);
    # the dynamic SSA values themselves travel in the make_shape / make_stride /
    # make_coord operands bundled into ``layout._layout_value`` / ``coord._coord_value``,
    # so the type only needs to mark which leaves are dynamic - same approach as
    # ``_format_tensor_type_descriptor`` (``tile_view``). The index trees above (which
    # carry the concrete ``int`` / ``_IndexExpr`` leaf values) back the metadata below.
    shape_type_tree = _components_to_type_tree(layout._shape._components)
    stride_type_tree = _components_to_type_tree(layout._stride._components)
    origin_type_tree = (
        _components_to_type_tree(layout._origin_shape._components)
        if layout._origin_shape is not None
        else shape_type_tree
    )
    coord_type_tree = _components_to_type_tree(coord._components)

    result_desc = TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape_type_tree),
            stride=TlaIndexTreeType("stride", stride_type_tree),
            origin_shape=TlaIndexTreeType("shape", origin_type_tree),
            layout_tag=layout._layout_tag,
        ),
        coord=coord_type_tree,
        element_type=dtype,
        addrspace=addr,
        ptr_alignment=ptr_ty.alignment,
    )
    op = mlir_ir.Operation.create(
        "tla.make_tensor",
        operands=[ptr_value, layout._layout_value, coord._coord_value],
        results=[_coerce_type(result_desc)],
        loc=loc,
    )
    out = op.results[0]
    _register_tla_tensor_type(out, result_desc)
    try:
        # Metadata carries the concrete leaf values (``int`` / ``_IndexExpr``) so that
        # downstream ops can do coord arithmetic on dynamic leaves; equivalent to
        # ``result_desc.metadata()`` for the static fields but preserving ``_IndexExpr``
        # for dynamic shape/stride/coord/origin (like ``tile_view`` does).
        _register_tla_tensor_metadata(
            out,
            {
                "shape": shape_tree,
                "stride": stride_tree,
                "coord": coord_tree,
                "origin_shape": origin_tree,
                "dtype": dtype,
                "addrspace": addr,
                "layout_tag": layout._layout_tag,
            },
        )
    except Exception:
        # Metadata property access falls back to type parsing when unavailable.
        pass
    return _Tensor(out)


@dsl_user_op
def make_tensor_like(
    ptr: Any,
    like: TileLike,
    layoutTag: Any | None = None,
    dst_dtype: DTypeLike | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Create a tensor over ``ptr`` from ``like`` using structured Tla tensor metadata.

    The ``layoutTag`` must be a ``tla.arch`` layout sentinel (e.g. ``tla.arch.RowMajor``,
    ``tla.arch.zN``); raw strings are not accepted. It selects a layout policy aligned with
    ``tla/layout.hpp``. Remapping uses logical ``M,N`` from ``like``'s flat
    **origin_shape** tree; nested origin trees skip remap. **shape** and **stride** are recomputed
    from that pair (e.g. ``zN`` nested 2×2 fractal spelling). **coord** is always ``0,0`` for
    every layout tag that participates in remap; **origin_shape** in the result matches the same
    flat logical pair. The tensor element type defaults to ``ptr``'s ``!tla.ptr``
    pointee, while its address space follows the pointer's memspace; L0 pointer names
    are remapped to tensor ABI names. ``like`` supplies only the tensor
    shape/layout/coord structure. ``dst_dtype`` is deprecated but continues to override
    the pointer element type when provided; it will be removed in a future release. It
    accepts a concrete
    :class:`~catlass.base_dsl.typing.Numeric` (e.g. ``tla.Float32``) or an
    ``mlir_ir.Type``; string dtype tokens are not accepted.
    The ``ptr`` operand is required by
    ``tla.make_tensor_like`` for lowering to attach backing storage.
    """
    _require_category("make_tensor_like", "like", like, "tensor", 1)
    if dst_dtype is not None:
        warnings.warn(
            "tla.make_tensor_like argument `dst_dtype` is deprecated and will be "
            "removed in a future release; it currently overrides the element type, "
            "so use a typed `ptr` instead.",
            category=FutureWarning,
            stacklevel=3,
        )
        _require_dtype("make_tensor_like", "dst_dtype", dst_dtype, 3)
    _require_frontend_state("make_tensor_like")
    ptr_value = _as_value(ptr)
    like_value = _as_value(like)
    try:
        like_type = _tla_tensor_type_for_mlir_value(like_value)
    except TlaLoweringError as exc:
        raise TlaLoweringError(
            "tla.make_tensor_like expects ``like`` to carry structured Tla tensor metadata; "
            f"got {str(like_value.type)!r}"
        ) from exc
    if not PtrType.isinstance(ptr_value.type):
        _op_error(
            "make_tensor_like",
            f"invalid argument 'ptr': expected !tla.ptr, got {ptr_value.type}",
        )
    # Keep frontend MLIR pointer spelling aligned with the pointer operand.
    ptr_ty = PtrType(ptr_value.type)
    if dst_dtype is not None:
        dtype = _dtype_to_str(dst_dtype).lower()
    else:
        try:
            dtype = _dtype_to_str(ptr_ty.pointee).lower()
        except TypeError as exc:
            raise TlaLoweringError(
                "tla.make_tensor_like cannot derive element type from ptr pointee "
                f"{ptr_ty.pointee}"
            ) from exc
    if dtype not in {"f16", "bf16", "f32", "i32", "i16", "i1", "i8"}:
        raise TlaLoweringError(
            f"tla.make_tensor_like expects a supported element type, got [{dtype}]"
        )
    addr = ptr_ty.addrspace

    # Infer layout if not provided
    if layoutTag is None:
        if addr == "l0a":
            layout = "zN"
        elif addr == "l0b":
            layout = "nZ"
        elif addr == "l0c":
            layout = "L0Clayout"
        elif addr == "l1":
            if like_type.layout_tag in ("row_major", "zN"):
                layout = "zN"
            elif like_type.layout_tag in ("column_major", "nZ"):
                layout = "nZ"
            else:
                raise TlaLoweringError(
                    f"tla.make_tensor_like cannot infer layout for addrspace l1 "
                    f"with likeTensor layoutTag '{like_type.layout_tag}'; please specify layoutTag explicitly"
                )
        else:
            raise TlaLoweringError(
                f"tla.make_tensor_like cannot infer layout for addrspace '{addr}'; "
                f"please specify layoutTag explicitly"
            )
    else:
        # Validate user-provided layoutTag
        if not isinstance(layoutTag, _LayoutTagSentinel):
            _op_error(
                "make_tensor_like",
                "invalid argument 'layoutTag': "
                "expected a tla.arch layout sentinel (e.g. tla.arch.zN) or None; "
                f"got {_type_name(layoutTag)}",
            )
        layout = _name_token(layoutTag)
        if layout is None:
            _op_error(
                "make_tensor_like",
                "invalid argument 'layoutTag': "
                f"expected a tla.arch layout sentinel with a token name; got {_type_name(layoutTag)}",
            )

    ptr_alignment = ptr_ty.alignment
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        like_type.origin_shape, dtype, layout
    )
    if remapped is not None:
        shape, stride, coord, origin = remapped
    else:
        shape = like_type.shape
        stride = like_type.stride
        coord = like_type.coord
        origin = like_type.origin_shape
    result_desc = TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape),
            stride=TlaIndexTreeType("stride", stride),
            origin_shape=TlaIndexTreeType("shape", origin),
            layout_tag=layout,
        ),
        coord=coord,
        element_type=dtype,
        addrspace=addr,
        ptr_alignment=ptr_alignment,
    )
    if remapped is not None:
        shape, stride, coord, origin = remapped
        result_desc = result_desc.with_updates(
            shape=shape,
            stride=stride,
            coord=coord,
            origin_shape=origin,
        )
    op = mlir_ir.Operation.create(
        "tla.make_tensor_like",
        operands=[ptr_value, like_value],
        results=[_coerce_type(result_desc)],
        attributes={"layoutTag": mlir_ir.StringAttr.get(layout)},
        loc=loc,
    )
    out = op.results[0]
    _register_tla_tensor_type(out, result_desc)
    try:
        like_meta = {
            "shape": _tensor_metadata_field(like_value, "shape"),
            "stride": _tensor_metadata_field(like_value, "stride"),
            "coord": _tensor_metadata_field(like_value, "coord"),
            "origin_shape": _tensor_metadata_field(like_value, "origin_shape"),
        }
        metadata = {
            "shape": like_meta["shape"],
            "stride": like_meta["stride"],
            "coord": like_meta["coord"],
            "origin_shape": like_meta["origin_shape"],
            "dtype": dtype,
            "addrspace": addr,
            "layout_tag": layout,
        }
        if remapped is not None:
            remapped_trees = _materialize_layout_trees_from_origin(
                like_meta["origin_shape"], dtype, layout
            )
            if remapped_trees is not None:
                metadata["shape"] = remapped_trees[0]
                metadata["stride"] = remapped_trees[1]
                metadata["coord"] = remapped_trees[2]
                metadata["origin_shape"] = remapped_trees[3]
            else:
                metadata = result_desc.metadata()
        _register_tla_tensor_metadata(out, metadata)
    except Exception:
        pass
    return _Tensor(out)


# (src addrspace, dst addrspace) routes and the region each must be nested in.
_COPY_CUBE_ROUTES = {
    ("gm", "l1"), ("l1", "l0a"), ("l1", "l0b"), ("l0c", "gm"),
    ("l0c", "ub"), ("l1", "ub"),
}
_COPY_VECTOR_ROUTES = {("gm", "ub"), ("ub", "gm"), ("ub", "l1")}


@dsl_user_op
def copy(dst: TileLike, src: TileLike, params: CopyParams | None = None, *, loc: mlir_ir.Location | None = None) -> None:
    """Copy between Tla tensor/view values.

    Frontend policy is intentionally minimal: ``tla.copy`` accepts only tensor
    operands and leaves layout/route selection to lowering.
    """
    _require_category("copy", "dst", dst, "tensor", 0)
    _require_category("copy", "src", src, "tensor", 1)
    _require_frontend_state("copy")
    dst_value = _as_value(dst)
    src_value = _as_value(src)

    # Cube data-path copies (GM->L1, L1->L0A/L0B, L0C->GM, L0C->UB, L1->UB) must
    # live in a tla.cube region; vector staging copies (GM<->UB, UB->L1) must
    # live in a tla.vector region. Mirrors tla.copy's MLIR verifier. Reading .addrspace
    # needs registered tensor metadata, which is unavailable for values carried
    # through scf.if/scf.for; when it can't be resolved, skip the frontend check
    # and let the MLIR verifier enforce placement.
    try:
        _route = (
            _tla_tensor_type_for_mlir_value(src_value).addrspace.lower(),
            _tla_tensor_type_for_mlir_value(dst_value).addrspace.lower(),
        )
    except TlaLoweringError:
        _route = None
    if _route in _COPY_CUBE_ROUTES:
        _runtime._require_enclosing_region("copy", "cube")
    elif _route in _COPY_VECTOR_ROUTES:
        _runtime._require_enclosing_region("copy", "vector")

    if _route is not None and _route[0] == "l0c":
        if params is None:
            params = CopyL0C2DstParams() # use default
        if isinstance(params, CopyL0C2DstParams):
            params._validate()
            if params.quant_mode != QuantMode.NO_QUANT:
                raise NotImplementedError(f"currently unsupported quant mode {params.quant_mode}")
            if params.relu_enable != False:
                raise NotImplementedError(f"currently unsupported relu_enable {params.relu_enable}")
            if (dst.addrspace == "ub") and (src.dtype != dst.dtype) and (
                params.l0c2ub_mode == L0C2UBMode.SPLIT_M or params.l0c2ub_mode == L0C2UBMode.SPLIT_N):
                raise TlaLoweringError(f"When copy l0c to ub with split mode, src and dst dtype must be same , got {src.dtype} {dst.dtype}")

            ctx = loc.context if loc is not None else mlir_ir.Context.current
            quant_mode_attr = mlir_ir.Attribute.parse(f"#tla.quant_mode<{params.quant_mode}>", context=ctx)
            l0c2ub_mode_attr = mlir_ir.Attribute.parse(f"#tla.l0c2ub_mode<{params.l0c2ub_mode}>", context=ctx)
            quant_scale_or_tensor = None
            if params.quant_mode == QuantMode.PER_TENSOR:
                quant_scale_or_tensor = _const_f32(params.quant_scale)
            elif params.quant_mode == QuantMode.PER_CHANNEL:
                quant_scale_or_tensor = _as_value(params.quant_tensor)
            params_value = _tla_ops_gen.CopyL0C2DstParams(
                _tla_type_bridge.copy_l0c2dst_params_type_get(ctx),
                params.unit_flag,
                params.relu_enable,
                quant_mode_attr,
                l0c2ub_mode_attr,
                quant_scale_or_tensor=quant_scale_or_tensor
            )
        else:
            raise TlaLoweringError(
                f"tla.copy operand `params` expects to be a CopyL0C2DstParams when {src.addrspace} -> {dst.addrspace}"
            )
    else:
        params_value = None

    return _tla_ops_gen.copy(dst_value, src_value, params=params_value, loc=loc)


@dsl_user_op
def flag(
    name: str,
    src_pipe: PipeLike | None = None,
    dst_pipe: PipeLike | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaFlag:
    """Materialize a synchronization flag. Legacy one-arg form is supported."""
    if not isinstance(name, str):
        _op_error(
            "flag",
            f"invalid argument 'name' (position 0): expected str, got {_type_name(name)}",
        )
    if src_pipe is None and dst_pipe is None:
        _require_frontend_state("flag")
        ctx = loc.context if loc is not None else mlir_ir.Context.current
        all_attr = mlir_ir.Attribute.parse("#tla.pipe<all>", context=ctx)
        return _tla_ops_gen.flag(
            _tla_type_bridge.flag_type_get(ctx),
            name,
            all_attr,
            all_attr,
            loc=loc,
        )
    if src_pipe is None or dst_pipe is None:
        _op_error(
            "flag",
            "expected either 1 argument (name) or 3 arguments (name, src_pipe, dst_pipe)",
        )
    _require_pipe("flag", "src_pipe", src_pipe, 1)
    _require_pipe("flag", "dst_pipe", dst_pipe, 2)
    _require_frontend_state("flag")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    src_value = str(_token(src_pipe)).lower()
    dst_value = str(_token(dst_pipe)).lower()
    src_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{src_value}>", context=ctx)
    dst_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{dst_value}>", context=ctx)
    return _tla_ops_gen.flag(
        _tla_type_bridge.flag_type_get(ctx),
        name,
        src_attr,
        dst_attr,
        loc=loc,
    )


@dsl_user_op
def cross_flag(
    name: str,
    *,
    mode: int = 2,
    loc: mlir_ir.Location | None = None,
) -> TlaCrossFlag:
    """Materialize a named cross-core synchronization flag.
    Source and destination pipes are specified by the corresponding set and wait operations.
    Mode 4 selects 1:1 AIC-to-AIV synchronization, addressing AIV0 and AIV1 independently.
    """
    if not isinstance(name, str):
        _op_error(
            "cross_flag",
            f"invalid argument 'name' (position 0): expected str, got {_type_name(name)}",
        )
    if not isinstance(mode, int) or isinstance(mode, bool) or mode not in (0, 1, 2, 4):
        _op_error(
            "cross_flag",
            f"invalid argument 'mode': expected one of 0, 1, 2, or 4, got {mode!r}",
        )
    _require_frontend_state("cross_flag")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    return _tla_ops_gen.cross_flag(
        _tla_type_bridge.cross_flag_type_get(ctx, mode),
        name,
        loc=loc,
    )


def _cross_flag_aiv_id_attr(
    op_name: str,
    cross_flag_value: CrossFlagLike,
    aiv_id: int | None,
    *,
    loc: mlir_ir.Location | None,
) -> mlir_ir.IntegerAttr | None:
    mode = _tla_type_bridge.cross_flag_mode(_as_value(cross_flag_value).type)
    if mode == 4:
        if not isinstance(aiv_id, int) or isinstance(aiv_id, bool) or aiv_id not in (0, 1):
            _op_error(
                op_name,
                "invalid argument 'aiv_id': mode 4 requires compile-time 0 or 1, "
                f"got {aiv_id!r}",
            )
    elif aiv_id is not None:
        _op_error(
            op_name,
            f"invalid argument 'aiv_id': mode {mode} requires None, got {aiv_id!r}",
        )
    if aiv_id is None:
        return None
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=ctx), aiv_id
    )


@dsl_user_op
def cross_core_set_flag(
    cross_flag_value: CrossFlagLike,
    pipe: PipeLike,
    aiv_id: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Set a cross-core synchronization flag from ``pipe``."""
    _require_category(
        "cross_core_set_flag", "flag", cross_flag_value, "cross_flag", 0
    )
    _require_pipe("cross_core_set_flag", "pipe", pipe, 1)
    _require_frontend_state("cross_core_set_flag")
    _runtime._require_enclosing_cube_or_vector("cross_core_set_flag")
    aiv_id_attr = _cross_flag_aiv_id_attr(
        "cross_core_set_flag", cross_flag_value, aiv_id, loc=loc
    )
    return _tla_ops_gen.cross_core_set_flag(
        _as_value(cross_flag_value),
        _pipe_attr_from_token(pipe, loc=loc),
        aiv_id=aiv_id_attr,
        loc=loc,
    )


@dsl_user_op
def cross_core_wait_flag(
    cross_flag_value: CrossFlagLike,
    pipe: PipeLike,
    aiv_id: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Wait on a cross-core synchronization flag on ``pipe``."""
    _require_category(
        "cross_core_wait_flag", "flag", cross_flag_value, "cross_flag", 0
    )
    _require_pipe("cross_core_wait_flag", "pipe", pipe, 1)
    _require_frontend_state("cross_core_wait_flag")
    _runtime._require_enclosing_cube_or_vector("cross_core_wait_flag")
    aiv_id_attr = _cross_flag_aiv_id_attr(
        "cross_core_wait_flag", cross_flag_value, aiv_id, loc=loc
    )
    return _tla_ops_gen.cross_core_wait_flag(
        _as_value(cross_flag_value),
        _pipe_attr_from_token(pipe, loc=loc),
        aiv_id=aiv_id_attr,
        loc=loc,
    )


@dsl_user_op
def set_flag(flag_value: FlagLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Set a synchronization flag."""
    _require_category("set_flag", "flag", flag_value, "flag", 0)
    _require_frontend_state("set_flag")
    _runtime._require_enclosing_cube_or_vector("set_flag")
    return _tla_ops_gen.set_flag(_as_value(flag_value), loc=loc)


@dsl_user_op
def wait_flag(flag_value: FlagLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Wait on a synchronization flag."""
    _require_category("wait_flag", "flag", flag_value, "flag", 0)
    _require_frontend_state("wait_flag")
    _runtime._require_enclosing_cube_or_vector("wait_flag")
    return _tla_ops_gen.wait_flag(_as_value(flag_value), loc=loc)


@dsl_user_op
def pipe_barrier(pipe: PipeLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Insert a pipe barrier for a specific pipe."""
    _require_pipe("pipe_barrier", "pipe", pipe, 0)
    _require_frontend_state("pipe_barrier")
    _runtime._require_enclosing_cube_or_vector("pipe_barrier")
    ctx = loc.context if loc is not None else mlir_ir.Context()
    pipe_value = str(_token(pipe)).lower()
    cube_pipes = [ "cube", "mte1", "mte2", "fix", "all" ]
    vector_pipes = [ "mte2", "mte3", "all" ] # NOTE: arch 3510 do not support pipe_barrier<pipe_v> specific
    if _runtime._has_enclosing_region("cube") and pipe_value not in cube_pipes:
        raise TlaLoweringError(f"in cube pipe_barrier only support {cube_pipes}, got {pipe_value}")
    elif _runtime._has_enclosing_region("vector") and pipe_value not in vector_pipes:
        raise TlaLoweringError(f"in vector pipe_barrier only support {vector_pipes}, got {pipe_value}")
    pipe_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{pipe_value}>", context=ctx)
    return _tla_ops_gen.pipe_barrier(pipe_attr, loc=loc)


@dsl_user_op
def mutex(
    resource: str,
    id: int = -1,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaMutex:
    """Materialize a mutex associated with a semantic resource."""
    if not isinstance(resource, str):
        _op_error(
            "mutex",
            f"invalid argument 'resource' (position 0): expected non-empty str, got {_type_name(resource)}",
        )
    resource_value = resource.strip()
    if not resource_value:
        _op_error("mutex", "resource must be non-empty")
    if isinstance(id, bool) or not isinstance(id, int):
        _op_error(
            "mutex",
            f"invalid argument 'id' (position 1): expected int in {{-1, 0..31}}, got {_type_name(id)}",
        )
    if id != -1 and not 0 <= id <= 31:
        _op_error("mutex", "id must be -1 or in range 0..31")
    _require_frontend_state("mutex")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    value = _tla_ops_gen.mutex(
        _tla_type_bridge.mutex_type_get(ctx), resource_value, int(id), loc=loc
    )
    return _MutexValue(value, resource_value, int(id))


@dsl_user_op
def mutex_guard(
    *mutexes: MutexLike, loc: mlir_ir.Location | None = None
) -> _MutexGuard:
    """Create a context manager that wraps a block with inferred mutex access."""
    if not mutexes:
        _op_error("mutex_guard", "expected at least one mutex")
    return _MutexGuard(tuple(mutexes), loc=loc)


def _emit_mutex_lock_op(
    mutex_value: MutexLike, *, pipe: PipeLike | str, loc: mlir_ir.Location | None = None
) -> None:
    pipe_attr = _pipe_attr_from_token(pipe, loc=loc)
    return _tla_ops_gen.mutex_lock(_as_value(mutex_value), pipe_attr, loc=loc)


def _emit_mutex_unlock_op(
    mutex_value: MutexLike, *, pipe: PipeLike | str, loc: mlir_ir.Location | None = None
) -> None:
    pipe_attr = _pipe_attr_from_token(pipe, loc=loc)
    return _tla_ops_gen.mutex_unlock(_as_value(mutex_value), pipe_attr, loc=loc)


@dsl_user_op
def mutex_lock(
    mutex_value: MutexLike, *, pipe: PipeLike, loc: mlir_ir.Location | None = None
) -> None:
    """Acquire a mutex from the specified pipe."""
    _ensure_no_explicit_mutex_access_in_guard()
    _require_category("mutex_lock", "mutex", mutex_value, "mutex", 0)
    _require_pipe("mutex_lock", "pipe", pipe, 1)
    _require_frontend_state("mutex_lock")
    _runtime._require_enclosing_cube_or_vector("mutex_lock")
    return _emit_mutex_lock_op(mutex_value, pipe=pipe, loc=loc)


@dsl_user_op
def mutex_unlock(
    mutex_value: MutexLike, *, pipe: PipeLike, loc: mlir_ir.Location | None = None
) -> None:
    """Release a mutex from the specified pipe."""
    _ensure_no_explicit_mutex_access_in_guard()
    _require_category("mutex_unlock", "mutex", mutex_value, "mutex", 0)
    _require_pipe("mutex_unlock", "pipe", pipe, 1)
    _require_frontend_state("mutex_unlock")
    _runtime._require_enclosing_cube_or_vector("mutex_unlock")
    return _emit_mutex_unlock_op(mutex_value, pipe=pipe, loc=loc)

@dsl_user_op
def local_mem_bar(
    src:MemType,
    dst:MemType,
    *,
    loc: mlir_ir.Location | None = None,
):
    # MemType pair → encoded I32 imm (matching hivmave.membar encoding)
    _local_mem_bar_barrier_kind = {
        (MemType.VEC_STORE, MemType.VEC_LOAD): 1,
        (MemType.VEC_LOAD, MemType.VEC_STORE): 2,
        (MemType.VEC_STORE, MemType.VEC_STORE): 3,
        (MemType.VEC_STORE, MemType.SCALAR_LOAD): 5,
        (MemType.VEC_STORE, MemType.SCALAR_STORE): 7,
        (MemType.VEC_LOAD, MemType.SCALAR_STORE): 6,
        (MemType.SCALAR_STORE, MemType.VEC_LOAD): 9,
        (MemType.SCALAR_STORE, MemType.VEC_STORE): 11,
        (MemType.SCALAR_LOAD, MemType.VEC_STORE): 10,
        (MemType.VEC_ALL, MemType.VEC_ALL): 0,
        (MemType.VEC_ALL, MemType.SCALAR_ALL): 4,
        (MemType.SCALAR_ALL, MemType.VEC_ALL): 8,
    }
    # check support
    if (src, dst) not in _local_mem_bar_barrier_kind:
        _op_error(
            "local_mem_bar",
            f"unsupported src and dst: {src.name} and {dst.name}",
        )
    _require_frontend_state("local_mem_bar")
    _runtime._require_enclosing_region("local_mem_bar", "vec.func")
    _tla_ops_gen.local_mem_bar(_local_mem_bar_barrier_kind[(src, dst)], loc=loc)
@dsl_user_op
def range(
    start: IndexLike,
    end: IndexLike | None = None,
    step: IndexLike | None = None,
    *,
    unroll: int = -1,
    unroll_full: bool = False,
    prefetch_stages: int | None = None,
    pipelining: int | None = None,
    loc: mlir_ir.Location | None = None,
) -> _ast_helpers.FrontendRange:
    """Create a frontend Tla dynamic range. Supports Python range arities."""
    del loc
    if pipelining is not None:
        if prefetch_stages is not None:
            _op_error(
                "range",
                "cannot specify both prefetch_stages and pipelining",
            )
        prefetch_stages = pipelining
    _validate_range_loop_attrs(
        unroll=unroll,
        unroll_full=unroll_full,
        prefetch_stages=prefetch_stages,
    )
    if end is None and step is None:
        _require_index("range", "end", start, 0)
        return _ast_helpers.range(
            start,
            unroll=unroll,
            unroll_full=unroll_full,
            prefetch_stages=prefetch_stages,
        )
    if step is None:
        _require_index("range", "start", start, 0)
        _require_index("range", "end", end, 1)
        return _ast_helpers.range(
            start,
            end,
            unroll=unroll,
            unroll_full=unroll_full,
            prefetch_stages=prefetch_stages,
        )
    if end is None:
        _op_error("range", "expected 1, 2, or 3 arguments")
    _require_index("range", "start", start, 0)
    _require_index("range", "end", end, 1)
    _require_index("range", "step", step, 2)
    return _ast_helpers.range(
        start,
        end,
        step,
        unroll=unroll,
        unroll_full=unroll_full,
        prefetch_stages=prefetch_stages,
    )


def _validate_range_loop_attrs(
    *,
    unroll: int,
    unroll_full: bool,
    prefetch_stages: int | None,
) -> None:
    if isinstance(unroll, bool) or not isinstance(unroll, int):
        _op_error("range", "unroll must be a compile-time Python int")
    if not isinstance(unroll_full, bool):
        _op_error("range", "unroll_full must be a compile-time Python bool")
    if prefetch_stages is not None:
        if isinstance(prefetch_stages, bool) or not isinstance(prefetch_stages, int):
            _op_error(
                "range", "prefetch_stages must be a compile-time Python int or None"
            )
        if prefetch_stages < 0:
            _op_error("range", "prefetch_stages must be non-negative")


def _require_constexpr_range_bound(op_name: str, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _op_error(op_name, f"{name} must be a compile-time Python int")
    return value


@dsl_user_op
def range_constexpr(
    start: int,
    end: int | None = None,
    step: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> Any:
    """Create a frontend-time static range for unrolled Python loops."""
    del loc
    if end is None and step is None:
        return _ast_helpers.range_constexpr(
            _require_constexpr_range_bound("range_constexpr", "end", start),
        )
    if step is None:
        return _ast_helpers.range_constexpr(
            _require_constexpr_range_bound("range_constexpr", "start", start),
            _require_constexpr_range_bound("range_constexpr", "end", end),
        )
    if end is None:
        _op_error("range_constexpr", "expected 1, 2, or 3 arguments")
    return _ast_helpers.range_constexpr(
        _require_constexpr_range_bound("range_constexpr", "start", start),
        _require_constexpr_range_bound("range_constexpr", "end", end),
        _require_constexpr_range_bound("range_constexpr", "step", step),
    )


@dsl_user_op
def cube(*, loc: mlir_ir.Location | None = None) -> TlaRegion:
    """Create a cube region stub for lowering-only usage."""
    return _region_stub("cube")


@dsl_user_op
def vector(*, loc: mlir_ir.Location | None = None) -> TlaRegion:
    """Create a vector region stub for lowering-only usage."""
    return _region_stub("vector")


_VEC_FUNC_MODES = {"simd", "SIMD", "simt", "SIMT"}


def _validate_vec_func_mode(mode: str) -> None:
    if not isinstance(mode, str):
        _op_error("vec.func", f"mode must be a string; got {_type_name(mode)}")
    if mode not in _VEC_FUNC_MODES:
        accepted = ", ".join(sorted(repr(value) for value in _VEC_FUNC_MODES))
        _op_error("vec.func", f"mode must be one of {accepted}; got {mode!r}")


@dsl_user_op
def _vec_func(*, mode: str = "simd", loc: mlir_ir.Location | None = None) -> TlaRegion:
    """Create a vector function region stub for lowering-only usage."""
    del loc
    _validate_vec_func_mode(mode)
    return _region_stub("vec.func")


@dsl_user_op
def mmad(
    acc: TileLike,
    lhs: TileLike,
    rhs: TileLike,
    init_c: bool | ValueLike | None = None,
    unit_flag: int | ValueLike | None = None,
    acc_type: DTypeLike | None = None,
    loc: mlir_ir.Location | None = None,
    **extra_kwargs: Any,
) -> TlaValue:
    """Emit a matrix-multiply-accumulate operation over Tla tiles.

    When ``acc_type`` is provided, it must be a concrete
    :class:`~catlass.base_dsl.typing.Numeric` (e.g. ``tla.Float32``) or an ``mlir_ir.Type``;
    string dtype tokens are not accepted.
    """
    if extra_kwargs:
        _op_error(
            "mmad",
            f"unknown keyword argument(s): {', '.join(sorted(extra_kwargs))}",
        )
    _require_category("mmad", "acc", acc, "tensor", 0)
    _require_category("mmad", "lhs", lhs, "tensor", 1)
    _require_category("mmad", "rhs", rhs, "tensor", 2)
    _require_frontend_state("mmad")
    _runtime._require_enclosing_region("mmad", "cube")

    if init_c is None:
        init_c = False
    _require_bool_or_value("mmad", "init_c", init_c, 3)
    init_c_value = _as_i1_value(init_c)

    if unit_flag is None:
        unit_flag = 0
    _require_index("mmad", "unit_flag", unit_flag, 4)
    if isinstance(unit_flag, int):
        if unit_flag not in [0b00, 0b10, 0b11]:
            raise TlaLoweringError(
                "tla.mmad operand 'unit_flag' expects values [0b00, 0b10, 0b11], "
                f"got [{unit_flag}]"
            )
        unit_flag_value = _const_i64(unit_flag)
    elif _category(unit_flag) == "index": # _IndexExpr
        unit_flag_value = _as_i64_value(unit_flag)
    else:
        raise TlaLoweringError("tla.mmad unit_flag must be a int")

    if acc_type is not None:
        _require_dtype("mmad", "acc_type", acc_type, 5)
        acc_type_value = _dtype_to_str(acc_type).lower()
        if acc_type_value not in {"f16", "bf16", "f32"}:
            raise TlaLoweringError(
                "tla.mmad attribute 'acc_type' expects dtype(s) [bf16, f16, f32], "
                f"got [{acc_type_value}]"
            )
    acc_value = _as_value(acc)
    lhs_value = _as_value(lhs)
    rhs_value = _as_value(rhs)
    _validate_mmad_contract(acc_value, lhs_value, rhs_value)
    return _tla_ops_gen.mmad(
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        loc=loc,
    )


@dsl_user_op
def full(
    value: Any,
    dtype: Any,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Create a 1-D vector SSA filled with a Python scalar literal."""
    state = _runtime._current_frontend_state()
    if state is None or not state.active_regions:
        raise TlaCoreAPIError("tla.full is only allowed inside tla.vec.func")
    _runtime._require_enclosing_region("full", "vec.func")
    _require_dtype("full", "dtype", dtype, 1)
    if not (
        isinstance(dtype, type)
        and issubclass(dtype, Numeric)
        and getattr(dtype, "dtype", "")
    ):
        _op_error(
            "full",
            f"invalid argument 'dtype' (position 1): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, mlir_ir.Value):
            _op_error("full", "value must be a Python scalar literal or host Numeric")
        resolved = resolved.value
    if not isinstance(resolved, (bool, int, float)):
        _op_error("full", "value must be a Python scalar literal or host Numeric")
    dtype_token = str(dtype.dtype).strip().lower()
    if dtype_token not in _FULL_SUPPORTED_DTYPES:
        _op_error(
            "full",
            f"unsupported vector element dtype {dtype.dtype}; supported dtypes are "
            f"{', '.join(sorted(_FULL_SUPPORTED_DTYPES))}",
        )
    desc = _vector_tile_descriptor(dtype, dtype_token=dtype_token)
    scalar_value = int(resolved) if isinstance(resolved, bool) else resolved
    context = loc.context if loc is not None else mlir_ir.Context.current
    scalar = _scalar_constant_for_element_type(
        "full",
        scalar_value,
        desc.element_mlir_type(context),
        loc=loc,
    )
    result = _tla_ops_gen.full(_coerce_type(desc), scalar, loc=loc)
    _register_tla_tensor_type(result, desc)
    _register_tla_tensor_metadata(result, desc.metadata())
    return VectorSSA(result)


def _vector_tile_descriptor(dtype: type[Numeric], *, dtype_token: str) -> TlaTensorTypeDescriptor:
    element_bytes = dtype_size_bytes(dtype_token)
    lanes = _vector_lane_count(element_bytes)
    return TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", lanes),
            stride=TlaIndexTreeType("stride", 1),
            origin_shape=TlaIndexTreeType("shape", lanes),
            layout_tag="row_major",
        ),
        coord=0,
        element_type=dtype_token,
        addrspace="ub",
        ptr_alignment=_builtins.max(1, element_bytes),
    )


@dsl_user_op
def arange(
    base: Any = 0,
    *,
    order: str = "increase",
    dtype: Any,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Create a 1-D vector SSA with monotonically increasing or decreasing values.

    - ``order="increase"`` (default): Create a 1-D vector SSA filled with ``base + lane`` (monotonic increase)

    - ``order="decrease"``: Create a 1-D vector SSA filled with ``base + VL - 1 - lane`` (monotonic decrease)

    Maps directly to AVE ``vci`` with ``INCREASE`` or ``DECREASE``;
    adjacent lanes are always spaced by 1 (no ``step`` parameter).

    Example:
    ```python
    # Store an ascending sequence [0, 1, 2, ..., 63] into ``dst`` (a ``tla.Tensor`` on ``tla.AddressSpace.ub``)
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    dst_tile.store(tla.arange(0, dtype=tla.Int32))

    # Or with descending order, store [63, 62, ..., 0]
    dst_tile.store(tla.arange(0, order="decrease", dtype=tla.Int32))
    ```
    """
    op_name = "arange"
    order = str(order).lower()
    if order not in _ARANGE_ORDERS:
        _op_error(
            op_name,
            f"order must be one of {sorted(_ARANGE_ORDERS)}; got {order!r}",
        )
    _require_dtype(op_name, "dtype", dtype, 2)
    if not (
        isinstance(dtype, type)
        and issubclass(dtype, Numeric)
        and getattr(dtype, "dtype", "")
    ):
        _op_error(
            op_name,
            f"invalid argument 'dtype' (position 2): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    dtype_token = str(dtype.dtype).strip().lower()
    if dtype_token not in _ARANGE_SUPPORTED_DTYPES:
        _op_error(
            op_name,
            f"unsupported vector element dtype {dtype.dtype}; supported dtypes are "
            f"{', '.join(sorted(_ARANGE_SUPPORTED_DTYPES))}",
        )
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    desc = _vector_tile_descriptor(dtype, dtype_token=dtype_token)
    context = loc.context if loc is not None else mlir_ir.Context.current
    element_type = desc.element_mlir_type(context)
    const = _const_int_value(base)
    if const is not None:
        start_value = _scalar_constant_for_element_type(
            op_name, const, element_type, loc=loc
        )
    else:
        resolved = _resolve_bound_value(base)
        if isinstance(resolved, mlir_ir.Value):
            if resolved.type == element_type:
                start_value = resolved
            elif isinstance(resolved.type, mlir_ir.IndexType):
                start_value = mlir_ir.Operation.create(
                    "arith.index_cast",
                    operands=[resolved],
                    results=[element_type],
                    loc=loc,
                ).results[0]
            else:
                _op_error(
                    op_name,
                    "base must be an integer literal or index SSA value",
                )
        elif isinstance(resolved, _runtime._IndexExpr):
            index_value = _runtime._coerce_index_value(resolved)
            start_value = mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[index_value],
                results=[element_type],
                loc=loc,
            ).results[0]
        else:
            _op_error(
                op_name,
                "base must be an integer literal or index SSA value",
            )
    result = _tla_ops_gen.arange(
        _coerce_type(desc), start_value, order=order, loc=loc
    )
    _register_tla_tensor_type(result, desc)
    _register_tla_tensor_metadata(result, desc.metadata())
    return VectorSSA(result)


def _emit_vector_binary(
    op_name: str,
    emitter: Any,
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Shared lowering for element-wise vector-vector binary ops.

    Optional ``mask`` (a ``MaskSSA`` from ``tla.create_mask`` or
    ``tla.update_mask``) controls which lanes are computed; masked-out lanes
    are undefined/zeroed.
    """
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _require_category(op_name, "rhs", rhs, "vector_ssa", 1)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    lhs_value = _as_value(lhs)
    rhs_value = _as_value(rhs)
    lhs_desc = _tla_tensor_type_for_mlir_value(lhs_value)
    mask_value = _as_value(mask) if mask is not None else None
    result = emitter(
        lhs_value.type,
        lhs_value,
        rhs_value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, lhs_desc)
    return VectorSSA(result)


def _as_vector_scalar_numeric(value: Any) -> Numeric | None:
    """Canonicalize vector–scalar rhs to ``Numeric`` via ``as_numeric``.

    Returns ``None`` if ``value`` is not a scalar operand.
    """
    if isinstance(value, VectorSSA) or _category(value) == "vector_ssa":
        return None
    if isinstance(value, Numeric):
        return value
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        return resolved
    if isinstance(resolved, (bool, int, float)):
        return as_numeric(resolved)
    if isinstance(resolved, mlir_ir.Value):
        # Also accept bare ``ir.Value`` via ``as_numeric``.
        try:
            return as_numeric(resolved)
        except (TypeError, ValueError, KeyError):
            return None
    return None


def _numeric_ir_value_for_element_type(
    op_name: str,
    scalar: Numeric,
    element_type: mlir_ir.Type,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Materialize scalar for vector–scalar ops via ``to`` + ``ir_value``.

    Host Python numbers keep literal ``arith.constant`` emission (range / fraction
    checks). Dynamic SSA uses ``Numeric.to`` + ``ir_value``.
    """
    if isinstance(scalar.value, (bool, int, float)):
        return _scalar_constant_for_element_type(
            op_name, scalar.value, element_type, loc=loc
        )
    dest_cls = Numeric.from_mlir_type(element_type)
    return scalar.to(dest_cls, loc=loc).ir_value(loc=loc)


def _scalar_constant_for_element_type(
    op_name: str,
    scalar: Any,
    element_type: mlir_ir.Type,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Emit ``arith.constant`` from a host scalar literal only."""
    resolved = _resolve_bound_value(scalar)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, mlir_ir.Value):
            _op_error(
                op_name,
                f"invalid argument 'rhs' (position 1): expected scalar literal, got Numeric SSA",
            )
        resolved = resolved.value
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        _op_error(
            op_name,
            f"invalid argument 'rhs' (position 1): expected scalar, got {_type_name(scalar)}",
        )
    if isinstance(element_type, mlir_ir.IndexType) or mlir_ir.IntegerType.isinstance(
        element_type
    ):
        if isinstance(resolved, float) and not resolved.is_integer():
            _op_error(
                op_name,
                f"invalid argument 'rhs' (position 1): expected integer scalar for "
                f"{element_type}, got {resolved!r}",
            )
        int_value = int(resolved)
        # Range check before IntegerAttr.get (signed bounds for signless integer types).
        if isinstance(element_type, mlir_ir.IndexType):
            lo, hi = -(2**63), 2**63 - 1
        else:
            int_ty = mlir_ir.IntegerType(element_type)
            width = int(int_ty.width)
            if int_ty.is_unsigned:
                lo, hi = 0, 2**width - 1
            else:
                lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
        if not (lo <= int_value <= hi):
            _op_error(
                op_name,
                f"integer scalar {int_value} out of range for {element_type} "
                f"(valid range [{lo}, {hi}])",
            )
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={"value": mlir_ir.IntegerAttr.get(element_type, int_value)},
            loc=loc,
        ).results[0]
    if (
        mlir_ir.F16Type.isinstance(element_type)
        or mlir_ir.F32Type.isinstance(element_type)
        or mlir_ir.F64Type.isinstance(element_type)
        or mlir_ir.BF16Type.isinstance(element_type)
    ):
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={"value": mlir_ir.FloatAttr.get(element_type, float(resolved))},
            loc=loc,
        ).results[0]
    _op_error(
        op_name, f"unsupported vector element type for scalar literal: {element_type}"
    )


def _emit_vector_scalar_binary(
    op_name: str,
    emitter: Any,
    lhs: VectorSSA,
    rhs: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if rhs_num is None:
        _op_error(
            op_name,
            f"invalid argument 'rhs' (position 1): expected scalar, got {_type_name(rhs)}",
        )
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    lhs_value = _as_value(lhs)
    lhs_desc = _tla_tensor_type_for_mlir_value(lhs_value)
    rhs_value = _numeric_ir_value_for_element_type(
        op_name, rhs_num, lhs_desc.element_mlir_type(lhs_value.type.context), loc=loc
    )
    mask_value = _as_value(mask) if mask is not None else None
    result = emitter(
        lhs_value.type,
        lhs_value,
        rhs_value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, lhs_desc)
    return VectorSSA(result)


def _emit_commutative_vector_scalar_binary(
    op_name: str,
    lhs: Any,
    rhs: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
    scalar_op_name: str | None = None,
) -> VectorSSA:
    scalar_op_name = scalar_op_name or op_name
    lhs_category = _category(lhs)
    rhs_category = _category(rhs)
    lhs_num = _as_vector_scalar_numeric(lhs)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if lhs_category == "vector_ssa" and rhs_num is not None:
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), lhs, rhs, mask=mask, loc=loc
        )
    if lhs_num is not None and rhs_category == "vector_ssa":
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), rhs, lhs, mask=mask, loc=loc
        )
    _op_error(op_name, "expected vector-scalar operands")


def _emit_vector_binary_or_scalar(
    op_name: str,
    vector_emitter: Any,
    scalar_op_name: str,
    lhs: Any,
    rhs: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
    commutative: bool = False,
) -> VectorSSA:
    lhs_category = _category(lhs)
    rhs_category = _category(rhs)
    lhs_num = _as_vector_scalar_numeric(lhs)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if lhs_category == "vector_ssa" and rhs_category == "vector_ssa":
        return _emit_vector_binary(op_name, vector_emitter, lhs, rhs, mask=mask, loc=loc)
    if lhs_category == "vector_ssa" and rhs_num is not None:
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), lhs, rhs, mask=mask, loc=loc
        )
    if commutative and lhs_num is not None and rhs_category == "vector_ssa":
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), rhs, lhs, mask=mask, loc=loc
        )
    if lhs_category != "vector_ssa":
        _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _op_error(op_name, "expected vector-vector or vector-scalar operands")


_FLOAT_UNARY_ELEMENT_TYPES = frozenset({"f16", "f32"})
_INTEGER_ABS_ELEMENT_TYPES = frozenset({"i8", "i16", "i32"})
_ABS_ELEMENT_TYPES = _FLOAT_UNARY_ELEMENT_TYPES | _INTEGER_ABS_ELEMENT_TYPES
_BITWISE_UNARY_TYPES = _ABS_ELEMENT_TYPES


def _emit_vector_unary(
    op_name: str,
    emitter: Any,
    operand: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 1)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    element_type = str(_tla_tensor_type_for_mlir_value(operand_value).element_type)
    if op_name in {"exp", "log", "sqrt"}:
        if element_type not in _FLOAT_UNARY_ELEMENT_TYPES:
            _op_error(
                op_name,
                f"tla.{op_name} requires f16 or f32 element type, "
                f"got {element_type}",
            )
    elif op_name in {"abs", "neg"} and element_type not in _ABS_ELEMENT_TYPES:
        _op_error(
            op_name,
            f"tla.{op_name} requires f16/f32 or i8/i16/i32 element type, "
            f"got {element_type}",
        )
    mask_value = _as_value(mask) if mask is not None else None
    result_desc = _tla_tensor_type_for_mlir_value(operand_value)
    result = emitter(
        operand_value.type,
        operand_value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, result_desc)
    return VectorSSA(result)


def _make_unary_op(mnemonic: str) -> Callable[..., VectorSSA]:
    @dsl_user_op
    def _unary(
        operand: VectorSSA,
        *,
        mask: MaskSSA | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> VectorSSA:
        return _emit_vector_unary(
            mnemonic,
            getattr(_tla_ops_gen, mnemonic),
            operand,
            mask=mask,
            loc=loc,
        )

    _unary.__name__ = mnemonic
    return _unary


exp = _make_unary_op("exp")
log = _make_unary_op("log")
sqrt = _make_unary_op("sqrt")
abs = _make_unary_op("abs")
neg = _make_unary_op("neg")


_INTERLEAVE_ELEMENT_TYPES = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "bf16",
        "f32",
    }
)

@dsl_user_op
def interleave(
    src0: VectorSSA,
    src1: VectorSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA]:
    _require_category("interleave", "src0", src0, "vector_ssa", 0)
    _require_category("interleave", "src1", src1, "vector_ssa", 1)
    _require_frontend_state("interleave")
    _runtime._require_enclosing_region("interleave", "vec.func")

    src0_value = _as_value(src0)
    src1_value = _as_value(src1)

    src_desc = _tla_tensor_type_for_mlir_value(src0_value)
    element_type = str(src_desc.element_type).lower()
    if element_type not in _INTERLEAVE_ELEMENT_TYPES:
        _op_error(
            "interleave",
            f"unsupported element type {src_desc.element_type}; supported types are "
            f"{', '.join(sorted(_INTERLEAVE_ELEMENT_TYPES))}",
        )

    dst0_value, dst1_value = _tla_ops_gen.interleave(
        src0_value.type,
        src0_value.type,
        src0_value,
        src1_value,
        loc=loc,
    )

    _register_tla_tensor_type(dst0_value, src_desc)
    _register_tla_tensor_type(dst1_value, src_desc)

    return VectorSSA(dst0_value), VectorSSA(dst1_value)


@dsl_user_op
def deinterleave(
    src0: VectorSSA,
    src1: VectorSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA]:
    _require_category("deinterleave", "src0", src0, "vector_ssa", 0)
    _require_category("deinterleave", "src1", src1, "vector_ssa", 1)
    _require_frontend_state("deinterleave")
    _runtime._require_enclosing_region("deinterleave", "vec.func")

    src0_value = _as_value(src0)
    src1_value = _as_value(src1)

    src_desc = _tla_tensor_type_for_mlir_value(src0_value)
    element_type = str(src_desc.element_type).lower()
    if element_type not in _INTERLEAVE_ELEMENT_TYPES:
        _op_error(
            "deinterleave",
            f"unsupported element type {src_desc.element_type}; supported types are "
            f"{', '.join(sorted(_INTERLEAVE_ELEMENT_TYPES))}",
        )

    dst0_value, dst1_value = _tla_ops_gen.deinterleave(
        src0_value.type,
        src0_value.type,
        src0_value,
        src1_value,
        loc=loc,
    )

    _register_tla_tensor_type(dst0_value, src_desc)
    _register_tla_tensor_type(dst1_value, src_desc)

    return VectorSSA(dst0_value), VectorSSA(dst1_value)


@dsl_user_op
def bitwise_not(
    operand: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    return _emit_bitwise_unary(
        "bitwise_not",
        _tla_ops_gen.bitwise_not,
        operand,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def add(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise add for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "add",
        _tla_ops_gen.add,
        "adds",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def sub(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise subtract for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "sub", _tla_ops_gen.sub, "subs", lhs, rhs, mask=mask, loc=loc
    )


@dsl_user_op
def mul(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise multiply for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "mul",
        _tla_ops_gen.mul,
        "muls",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def max(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise maximum for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "max",
        _tla_ops_gen.max,
        "maxs",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def min(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise minimum for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "min",
        _tla_ops_gen.min,
        "mins",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def div(
    lhs: Any,
    rhs: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise divide for loaded vector SSA values."""
    return _emit_vector_binary_or_scalar(
        "div", _tla_ops_gen.div, "divs", lhs, rhs, mask=mask, loc=loc
    )


def _reduction_result_descriptor(
    operand_value: mlir_ir.Value,
) -> TlaTensorTypeDescriptor:
    operand_desc = _tla_tensor_type_for_mlir_value(operand_value)
    return TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", 1),
            stride=TlaIndexTreeType("stride", 1),
            origin_shape=TlaIndexTreeType("shape", 1),
            layout_tag=operand_desc.layout_tag,
        ),
        coord=0,
        element_type=operand_desc.element_type,
        addrspace=operand_desc.addrspace,
        ptr_alignment=operand_desc.ptr_alignment,
    )


def _emit_vector_reduce(
    operand: VectorSSA,
    kind: ReductionOp,
    *,
    mask: MaskSSA,
    init_value: Any | None = None,
    reduction_profile: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    op_name = "VectorSSA.reduce"
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    _require_category(op_name, "mask", mask, "mask_ssa", 1)
    if init_value is not None:
        raise NotImplementedError(f"{op_name} only supports init_value=None")
    if reduction_profile is not None:
        raise NotImplementedError(f"{op_name} only supports reduction_profile=None")
    if not isinstance(kind, ReductionOp):
        _op_error(
            op_name,
            "invalid argument 'kind' (position 1): "
            f"expected ReductionOp, got {_type_name(kind)}",
        )
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    operand_desc = _tla_tensor_type_for_mlir_value(operand_value)
    _check_reduction_element_type_supported(op_name, operand_desc.element_type)
    mask_value = _as_value(mask)
    result_desc = _reduction_result_descriptor(operand_value)
    result = _tla_ops_gen.reduce(
        result_desc.to_mlir_type(operand_value.type.context),
        operand_value,
        kind.value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, result_desc)
    return VectorSSA(result)


@dsl_user_op
def where(
    mask: MaskSSA,
    x: VectorSSA,
    y: VectorSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit element-wise select for loaded vector SSA values.

    Lanes where ``mask`` (a ``MaskSSA`` from ``tla.create_mask`` or
    ``tla.update_mask``) is active take the corresponding lane of ``x``; the
    remaining lanes take ``y``. ``x`` and ``y`` must have identical
    ``!tla.tensor`` types. Lowers to ``ave.hir.vsel``.
    """
    _require_category("where", "mask", mask, "mask_ssa", 0)
    _require_category("where", "x", x, "vector_ssa", 1)
    _require_category("where", "y", y, "vector_ssa", 2)
    _require_frontend_state("where")
    _runtime._require_enclosing_region("where", "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    x_desc = _tla_tensor_type_for_mlir_value(x_value)
    mask_value = _as_value(mask)
    result = _tla_ops_gen.where(
        x_value.type,
        mask_value,
        x_value,
        y_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, x_desc)
    return VectorSSA(result)


@dsl_user_op
def squeeze(
    src: VectorSSA,
    mask: MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Pack lanes of ``src`` selected by ``mask`` into low indices of the result.

    Lanes where ``mask`` is inactive are dropped; the remaining selected lanes
    are written contiguously from lane 0. Trailing lanes in the result vector
    are zeroed. This matches AscendC ``Squeeze`` semantics.
    """
    _require_category("squeeze", "src", src, "vector_ssa", 0)
    _require_category("squeeze", "mask", mask, "mask_ssa", 1)
    _require_frontend_state("squeeze")
    _runtime._require_enclosing_region("squeeze", "vec.func")
    src_value = _as_value(src)
    mask_value = _as_value(mask)
    src_desc = _tla_tensor_type_for_mlir_value(src_value)
    result = _tla_ops_gen.squeeze(
        src_value.type,
        src_value,
        mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, src_desc)
    return VectorSSA(result)


def _emit_bitwise_unary(
    op_name: str,
    emitter: Any,
    operand: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    operand_category = _category(operand)
    expected = ("mask_ssa", "vector_ssa")
    if operand_category not in expected:
        _require_categories(op_name, "operand", operand, expected, 0)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 1)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    mask_value = _as_value(mask) if mask is not None else None
    if operand_category == "mask_ssa":
        return MaskSSA(
            emitter(operand_value.type, operand_value, mask=mask_value, loc=loc)
        )

    result_desc = _tla_tensor_type_for_mlir_value(operand_value)
    element_type = str(result_desc.element_type)
    if element_type not in _BITWISE_UNARY_TYPES:
        _op_error(
            op_name,
            f"tla.{op_name} requires f16/f32 or i8/i16/i32 element type, "
            f"got {element_type}",
        )
    result = emitter(
        operand_value.type,
        operand_value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, result_desc)
    return VectorSSA(result)


def _emit_bitwise_binary(
    op_name: str,
    emitter: Any,
    src0_reg: Any,
    src1_reg: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    src0_category = _category(src0_reg)
    src1_category = _category(src1_reg)
    expected = ("mask_ssa", "vector_ssa")
    if src0_category not in expected:
        _require_categories(op_name, "src0_reg", src0_reg, expected, 0)
    if src1_category not in expected:
        _require_categories(op_name, "src1_reg", src1_reg, expected, 1)
    if src0_category != src1_category:
        _op_error(
            op_name,
            "src0_reg and src1_reg must both be MaskReg values or both be RegTensor values",
        )
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    src0_value = _as_value(src0_reg)
    mask_value = _as_value(mask) if mask is not None else None
    result = emitter(
        src0_value.type,
        src0_value,
        _as_value(src1_reg),
        mask=mask_value,
        loc=loc,
    )
    if src0_category == "mask_ssa":
        return MaskSSA(result)

    result_desc = _tla_tensor_type_for_mlir_value(src0_value)
    _register_tla_tensor_type(result, result_desc)
    return VectorSSA(result)


def _tla_mask_type(context: mlir_ir.Context) -> mlir_ir.Type:
    """Return ``!tla.mask`` parsed in the same MLIR context as the surrounding SSA values."""
    _tla_type_bridge.load_tla_dialect(context)
    return mlir_ir.Type.parse("!tla.mask", context=context)


@dsl_user_op
def cmp(
    lhs: VectorSSA,
    rhs: Any,
    mode: str,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA:
    """Return a mask for element-wise vector compare."""
    mode = str(mode).lower()
    if mode not in _MASK_CMP_MODES:
        _op_error(
            "cmp",
            f"mode must be one of {_MASK_CMP_MODES}; got {mode!r}",
        )
    _require_category("cmp", "lhs", lhs, "vector_ssa", 0)
    rhs_category = _category(rhs)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if rhs_category != "vector_ssa" and rhs_num is None:
        _op_error(
            "cmp",
            f"invalid argument 'rhs' (position 1): expected vector or scalar, got {_type_name(rhs)}",
        )
    if mask is not None:
        _require_category("cmp", "mask", mask, "mask_ssa", 2)
    _require_frontend_state("cmp")
    _runtime._require_enclosing_region("cmp", "vec.func")
    lhs_value = _as_value(lhs)
    if rhs_category == "vector_ssa":
        rhs_value = _as_value(rhs)
    else:
        assert rhs_num is not None
        lhs_desc = _tla_tensor_type_for_mlir_value(lhs_value)
        rhs_value = _numeric_ir_value_for_element_type(
            "cmp", rhs_num, lhs_desc.element_mlir_type(lhs_value.type.context), loc=loc
        )
        _check_compare_element_type_supported("cmp", lhs_desc.element_type)
    mask_ty = _tla_mask_type(lhs_value.type.context)
    mask_value = _as_value(mask) if mask is not None else None
    return MaskSSA(
        _tla_ops_gen.cmp(
            mask_ty, lhs_value, rhs_value, mode, mask=mask_value, loc=loc
        )
    )


@dsl_user_op
def bitwise_and(
    src0_reg: Any,
    src1_reg: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Emit element-wise bitwise AND for MaskReg or RegTensor SSA values."""
    return _emit_bitwise_binary(
        "bitwise_and",
        _tla_ops_gen.bitwise_and,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def bitwise_or(
    src0_reg: Any,
    src1_reg: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Emit element-wise bitwise OR for MaskReg or RegTensor SSA values."""
    return _emit_bitwise_binary(
        "bitwise_or",
        _tla_ops_gen.bitwise_or,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def bitwise_xor(
    src0_reg: Any,
    src1_reg: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Emit element-wise bitwise XOR for MaskReg or RegTensor SSA values."""
    return _emit_bitwise_binary(
        "bitwise_xor",
        _tla_ops_gen.bitwise_xor,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def gather(
    x: TileLike,
    y: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Gather elements from a UB tensor according to vector indices.

    ``x`` is a tensor view pointing to UB memory (from ``tla.tile_view``).
    ``y`` is a vector of per-lane indices (from ``.load()``).
    Returns a vector register with gathered elements.
    """
    _require_category("gather", "x", x, "tensor", 0)
    _require_category("gather", "y", y, "vector_ssa", 1)
    _require_frontend_state("gather")
    _runtime._require_enclosing_region("gather", "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    x_desc = _tla_tensor_descriptor_from_type_or_value(x_value)
    y_desc = _tla_tensor_descriptor_from_type_or_value(y_value)

    # validate x addrspace is ub
    if x_desc.addrspace.lower() != "ub":
        _op_error(
            "gather",
            f"invalid argument 'x' (position 0): expected addrspace ub, got {x_desc.addrspace}",
        )

    # validate x element type
    _GATHER_SUPPORTED_X_ELEM_TYPES = frozenset(
        {
            "i1",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
            "f16",
            "bf16",
            "f32",
        }
    )
    if x_desc.element_type.lower() not in _GATHER_SUPPORTED_X_ELEM_TYPES:
        _op_error(
            "gather",
            f"invalid argument 'x' (position 0): unsupported element type "
            f"{x_desc.element_type}; supported types are "
            f"{', '.join(sorted(_GATHER_SUPPORTED_X_ELEM_TYPES))}",
        )

    # validate y element type
    _GATHER_SUPPORTED_Y_ELEM_TYPES = frozenset(
        {
            "i1",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
        }
    )
    if y_desc.element_type.lower() not in _GATHER_SUPPORTED_Y_ELEM_TYPES:
        _op_error(
            "gather",
            f"invalid argument 'y' (position 1): unsupported element type "
            f"{y_desc.element_type}; supported types are "
            f"{', '.join(sorted(_GATHER_SUPPORTED_Y_ELEM_TYPES))}",
        )
    if mask is not None:
        _require_category("gather", "mask", mask, "mask_ssa", 2)
    mask_value = _as_value(mask) if mask is not None else None
    result = _tla_ops_gen.gather(
        x_value.type,
        x_value,
        y_value,
        mask=mask_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, x_desc)
    return VectorSSA(result)


@dsl_user_op
def arch_block_idx(*, loc: mlir_ir.Location | None = None) -> TlaIndex:
    """Return block index in Tla execution model."""
    _require_frontend_state("arch.block_idx")
    value = _tla_ops_gen.arch_block_idx(mlir_ir.IndexType.get(), loc=loc)
    return _wrap_frontend_value(value)


@dsl_user_op
def arch_sub_block_idx(*, loc: mlir_ir.Location | None = None) -> TlaIndex:
    """Return sub-block index in Tla execution model."""
    _require_frontend_state("arch.sub_block_idx")
    value = _tla_ops_gen.arch_sub_block_idx(mlir_ir.IndexType.get(), loc=loc)
    return _wrap_frontend_value(value)


@dsl_user_op
def arch_block_dim(*, loc: mlir_ir.Location | None = None) -> TlaIndex:
    """Return block dimension in Tla execution model."""
    _require_frontend_state("arch.block_dim")
    value = _tla_ops_gen.arch_block_dim(mlir_ir.IndexType.get(), loc=loc)
    return _wrap_frontend_value(value)


@dsl_user_op
def allocate(
    shape: ShapeLike,
    dtype: type[Numeric],
    mem_scope: AddressSpace,
    byte_alignment: int,
    *,
    loc: mlir_ir.Location | None = None,
) -> PointerTypeHint:
    """Allocate static on-chip scratch memory and return a typed ``!tla.ptr``.

    ``shape`` is an element shape (or element count), not bytes. It must be fully
    static; ``size_bytes`` is computed as ``prod(shape) * sizeof(dtype)`` and
    stored on the resulting ``tla.alloc_ptr`` op.
    """
    _require_frontend_state("allocate")
    dtype, element_bytes = _require_allocation_dtype("allocate", dtype)
    align = _require_byte_alignment("allocate", byte_alignment, 3)
    addr_token = _require_pointer_addrspace("allocate", mem_scope, 2)
    if mem_scope in (AddressSpace.generic, AddressSpace.gm):
        _op_error(
            "allocate",
            "invalid argument 'mem_scope' (position 2): expected on-chip AddressSpace "
            "(l1, l0a, l0b, l0c, ub)",
        )
    size_bytes = _static_allocation_size_bytes(
        "allocate", shape, dtype, element_bytes
    )

    ctx = loc.context if loc is not None else mlir_ir.Context.current
    ptr_ty = PtrType.get(dtype.mlir_type(ctx), addr_token, align, context=ctx)
    i64_ty = mlir_ir.IntegerType.get_signless(64, context=ctx)
    op = mlir_ir.Operation.create(
        "tla.alloc_ptr",
        operands=[],
        results=[ptr_ty],
        attributes={
            "size_bytes": mlir_ir.IntegerAttr.get(i64_ty, size_bytes),
        },
        loc=loc,
    )
    return _Pointer(op.results[0], alloc_size_bytes=size_bytes)


@dsl_user_op
def make_ptr(
    dtype: type[Numeric] | None,
    value: int | mlir_ir.Value | Numeric,
    mem_space: AddressSpace = AddressSpace.gm,
    *,
    assumed_align: int | None = None,
    loc: mlir_ir.Location | None = None,
) -> PointerTypeHint:
    """Build a :class:`Pointer` from an integer bit pattern via ``tla.inttoptr``."""
    _require_frontend_state("make_ptr")

    if dtype is not None and (
        not isinstance(dtype, type) or not issubclass(dtype, Numeric) or not dtype.dtype
    ):
        _op_error("make_ptr", f"expects dtype to be a type of Numeric, but got {dtype}")
    dt = Int8 if dtype is None else dtype
    ctx = loc.context if loc is not None else mlir_ir.Context()
    pointee = dt.mlir_type(ctx)
    addr_token = _require_pointer_addrspace("make_ptr", mem_space, 2)
    bytes_per_elt = _builtins.max(1, int(dt.width) // 8)
    align = assumed_align if assumed_align is not None else bytes_per_elt
    if bytes_per_elt % align != 0 and align % bytes_per_elt != 0:
        _op_error(
            "make_ptr",
            f"element size {bytes_per_elt} is incompatible with assumed_align={align}",
        )
    if align <= 0 or (align & (align - 1)) != 0:
        _op_error("make_ptr", "assumed_align must be a positive power of 2")
    out_ptr_ty = PtrType.get(pointee, addr_token, align, context=ctx)
    addr_ssa = _coerce_inttoptr_address(addr_token, value, loc)
    return _Pointer(_tla_ops_gen.inttoptr(out_ptr_ty, addr_ssa, loc=loc, ip=None))


@dsl_user_op
def recast_ptr(
    ptr: PointerTypeHint,
    *,
    dtype: type[Numeric],
    loc: mlir_ir.Location | None = None,
) -> Any:
    """Change the logical pointee type of a ``!tla.ptr`` (dtype-only; no swizzle).

    The operand must already be a ``!tla.ptr`` (typically from :func:`make_ptr`).
    Allocator or bare integer addresses are not accepted; use :func:`make_ptr`
    first, then ``recast_ptr``.

    Preserves address space and alignment; element size vs alignment is not
    re-validated here (that check exists only on :func:`make_ptr`).
    """
    _require_frontend_state("recast_ptr")
    _require_category("recast_ptr", "ptr", ptr, "pointer", 0)
    if not isinstance(dtype, type) or not issubclass(dtype, Numeric) or not dtype.dtype:
        _op_error(
            "recast_ptr", f"expects dtype to be a type of Numeric, but got {dtype}"
        )
    ctx = loc.context if loc is not None else mlir_ir.Context()
    alloc_size_bytes = getattr(ptr, "_alloc_size_bytes", None)
    p = _coerce_pointer_arg(ptr)
    src_ty = p._ptr_ty
    new_pointee = dtype.mlir_type(ctx)
    out_ptr_ty = PtrType.get(
        new_pointee, src_ty.addrspace, src_ty.alignment, context=ctx
    )
    return _Pointer(
        _tla_ops_gen.recast_ptr(out_ptr_ty, p.value, loc=loc, ip=None),
        alloc_size_bytes=alloc_size_bytes,
    )


def _emit_tensor_ptr(
    source: mlir_ir.Value, loc: mlir_ir.Location | None = None
) -> _Pointer:
    """Emit ``tla.tensor_ptr`` extracting the backing ``!tla.ptr`` of a tensor value.

    Shared by :meth:`catlass.tla.tensor._Tensor.ptr` and the execution-mode
    ``_ArgProxy.ptr`` so kernel-argument proxies support ``arg.ptr`` the same way as
    frontend ``_Tensor`` values.
    """
    _require_frontend_state("tensor_ptr")
    ptr_ty = _tla_type_bridge.tensor_ptr_type_get(source.type)
    op = mlir_ir.Operation.create(
        "tla.tensor_ptr",
        operands=[source],
        results=[ptr_ty],
        loc=loc,
    )
    return _Pointer(op.results[0])


_require_generated("tile_view")
_require_generated("copy")
_require_generated("load")
_require_generated("flag")
_require_generated("cross_flag")
_require_generated("cross_core_set_flag")
_require_generated("cross_core_wait_flag")
_require_generated("set_flag")
_require_generated("store")
_require_generated("wait_flag")
_require_generated("pipe_barrier")
_require_generated("mutex")
_require_generated("mutex_lock")
_require_generated("mutex_unlock")
_require_generated("cube")
_require_generated("vector")
_require_generated("mmad")
_require_generated("add")
_require_generated("adds")
_require_generated("sub")
_require_generated("subs")
_require_generated("mul")
_require_generated("muls")
_require_generated("maxs")
_require_generated("mins")
_require_generated("div")
_require_generated("where")
_require_generated("squeeze")
_require_generated("bitwise_not")
_require_generated("bitwise_and")
_require_generated("bitwise_or")
_require_generated("bitwise_xor")
_require_generated("divs")
_require_generated("reduce")
_require_generated("interleave")
_require_generated("deinterleave")
for _unary_op_name in ("exp", "log", "sqrt", "abs", "neg"):
    _require_generated(_unary_op_name)
_require_generated("cmp")
_require_generated("arch_block_idx")
_require_generated("arch_sub_block_idx")
_require_generated("arch_block_dim")
_require_generated("inttoptr")
_require_generated("recast_ptr")

arch = _Namespace()
arch._set("block_idx", arch_block_idx)
arch._set("sub_block_idx", arch_sub_block_idx)
arch._set("block_dim", arch_block_dim)
arch._set("L1", _runtime.utils.L1)
arch._set("L0A", _runtime.utils.L0A)
arch._set("L0B", _runtime.utils.L0B)
arch._set("L0C", _runtime.utils.L0C)
arch._set("UB", _runtime.utils.UB)
arch._set("SCALAR", _runtime.pipes.SCALAR)
arch._set("VECTOR", _runtime.pipes.VECTOR)
arch._set("CUBE", _runtime.pipes.CUBE)
arch._set("MTE1", _runtime.pipes.MTE1)
arch._set("MTE2", _runtime.pipes.MTE2)
arch._set("MTE3", _runtime.pipes.MTE3)
arch._set("FIX", _runtime.pipes.FIX)
arch._set("zN", _LayoutTagSentinel("zN"))
arch._set("nZ", _LayoutTagSentinel("nZ"))
arch._set("zZ", _LayoutTagSentinel("zZ"))
arch._set("nN", _LayoutTagSentinel("nN"))
arch._set("RowMajor", _LayoutTagSentinel("row_major"))
arch._set("ColumnMajor", _LayoutTagSentinel("column_major"))
arch._set("L0Clayout", _LayoutTagSentinel("L0Clayout"))

vec = _Namespace()
vec._set("func", _vec_func)


class _MaskPatternSentinel:
    """A fixed ``tla.mask`` pattern (e.g. ``tla.mask.ALL``, ``tla.mask.VL8``).

    Pass it to ``tla.create_mask(pattern=...)`` to materialize a ``MaskSSA``;
    ops that take ``mask=`` accept only a ``MaskSSA`` (from ``tla.create_mask``
    or ``tla.update_mask``), not a pattern sentinel directly.
    """

    __slots__ = ("_token",)

    def __init__(self, token: str) -> None:
        self._token = token

    def __repr__(self) -> str:
        return f"tla.mask.{self._token}"


# AVE pge patterns exposed under the tla.mask namespace.
_MASK_PATTERN_TOKENS = (
    "ALL",
    "ALLF",
    "VL1",
    "VL2",
    "VL3",
    "VL4",
    "VL8",
    "VL16",
    "VL32",
    "VL64",
    "VL128",
    "M3",
    "M4",
    "H",
    "Q",
)


def _mask_elem_type(
    op_name: str, dtype: Any, loc: mlir_ir.Location | None
) -> mlir_ir.Type:
    """Resolve a mask ``dtype`` (Numeric type or mlir.ir.Type) to an element type.

    The element width fixes the mask lane count (256 bytes / dtype size).
    """
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    if isinstance(dtype, mlir_ir.Type):
        return dtype
    if isinstance(dtype, type) and issubclass(dtype, Numeric) and dtype.dtype:
        return dtype.mlir_type(ctx)
    _op_error(
        op_name,
        f"expects dtype to be a Numeric type (e.g. tla.Float32) or mlir.ir.Type, got {dtype}",
    )


@dsl_user_op
def create_mask(
    *,
    pattern: Any | None = None,
    dtype: Any = Float32,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA:
    """Create a vector mask inside a vector region from a fixed pattern.

    ``pattern=tla.mask.ALL`` (or ``VL8``, ``H``, ``Q``, ``M4``, ...) emits a
    fixed AVE pge pattern. ``dtype`` (default ``f32``) fixes the mask lane count
    (256 bytes / dtype size) and must match the enclosing vector region's element
    width. For tail handling over a loop, use :func:`update_mask`.
    """
    if pattern is None:
        _op_error("create_mask", "pattern is required")
    _require_frontend_state("create_mask")
    _runtime._require_enclosing_region("create_mask", "vec.func")
    elem_type = _mask_elem_type("create_mask", dtype, loc)
    mask_ty = mlir_ir.Type.parse("!tla.mask")
    token = pattern._token if isinstance(pattern, _MaskPatternSentinel) else str(pattern)
    return MaskSSA(
        _tla_ops_gen.create_mask(
            mask_ty, pattern=token, dtype=mlir_ir.TypeAttr.get(elem_type), loc=loc
        )
    )


@dsl_user_op
def update_mask(
    true_shape: Any,
    dtype: Any = Float32,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[MaskSSA, Any]:
    """Create a tail mask and the remaining element count.

    Maps to AVE ``plt``. Given ``true_shape`` (the number of elements still to
    process), returns ``(mask, new_true_shape)`` where ``mask`` activates lane
    ``i`` iff ``i < true_shape`` (saturating to all-true once ``true_shape``
    reaches the lane count) and ``new_true_shape = true_shape - lanes`` with
    ``lanes = 256 bytes / dtype size`` (= 64 for ``f32``).

    Seed a loop-carried counter with the total element count and thread
    ``new_true_shape`` back each iteration to mask successive chunks, including
    the partial tail. ``dtype`` (default ``f32``) fixes the lane count and must
    match the enclosing vector region's element width.
    """
    _require_frontend_state("update_mask")
    _runtime._require_enclosing_region("update_mask", "vec.func")
    elem_type = _mask_elem_type("update_mask", dtype, loc)
    true_shape_value = _as_index_value(true_shape)
    mask_ty = mlir_ir.Type.parse("!tla.mask")
    index_ty = mlir_ir.IndexType.get()
    mask_value, new_true_shape = _tla_ops_gen.update_mask(
        mask_ty, index_ty, true_shape_value, mlir_ir.TypeAttr.get(elem_type), loc=loc
    )
    return MaskSSA(mask_value), _runtime._IndexExpr(new_true_shape)


_mask_namespace = _Namespace()
for _mask_pattern_token in _MASK_PATTERN_TOKENS:
    _mask_namespace._set(_mask_pattern_token, _MaskPatternSentinel(_mask_pattern_token))


def __getattr__(name: str) -> Any:
    if name == "mask":
        return _mask_namespace
    raise AttributeError(name)


def _resolve_arch_layout_tag(value: Any | None, *, for_op: str) -> str:
    """Normalize ``Tensor(..., layout_tag=...)`` to the MLIR layout token string."""
    if value is None:
        token = _name_token(arch.RowMajor)
        assert token is not None
        return token
    if not isinstance(value, _LayoutTagSentinel):
        raise TypeError(
            f"{for_op}: layout_tag must be a tla.arch layout sentinel "
            f"(e.g. tla.arch.RowMajor); got {_type_name(value)}"
        )
    token = _name_token(value)
    if token is None:
        raise TypeError(f"{for_op}: layout sentinel produced no token: {value!r}")
    return token


setattr(_runtime.utils, "LocalmemAllocator", LocalmemAllocator)

__all__ = [
    "TlaCoreAPIError",
    "dsl_user_op",
    "arch",
    "mask",
    "create_mask",
    "update_mask",
    "tile_view",
    "make_tensor",
    "make_tensor_like",
    "copy",
    "debug_print",
    "flag",
    "cross_flag",
    "cross_core_set_flag",
    "cross_core_wait_flag",
    "set_flag",
    "wait_flag",
    "pipe_barrier",
    "mutex",
    "mutex_guard",
    "mutex_lock",
    "mutex_unlock",
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
    "squeeze",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "exp",
    "log",
    "sqrt",
    "abs",
    "neg",
    "interleave",
    "deinterleave",
    "gather",
    "ReductionOp",
    "cmp",
    "make_ptr",
    "allocate",
    "recast_ptr",
    "make_shape",
    "make_coord",
    "make_stride",
    "make_layout",
    "IndexTree",
    "range_constexpr",
    "_Pointer",
    "VectorSSA",
    "LocalmemAllocator",
]
