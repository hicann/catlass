"""Typing protocols, ``Numeric`` element types, and ``Pointer`` ABC for Tla DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    get_origin,
    runtime_checkable,
)

from mlir import ir as mlir_ir  # type: ignore[assignment]

_T = TypeVar("_T")

# -----------------------------------------------------------------------------
# Numeric element types (``Type[Numeric]`` parity — minimal subset)
# -----------------------------------------------------------------------------

_TOKEN_TO_NUMERIC: dict[str, type["Numeric"]] = {}


class Numeric(ABC):
    """Compile-time element type for Tla kernels (minimal ``Numeric`` subset).

    ``Pointer.dtype`` / ``Pointer.value_type`` return these type objects. Scalar constructors
    ``Int8(x)``, ``Float32(x)``, … return :class:`catlass.types.Scalar` instances.

    ``width`` is the element size in **bits** (element ``width`` in bits); storage size in
    bytes is ``max(1, width // 8)``.
    """

    dtype: ClassVar[str] = ""
    width: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.dtype:
            if cls.width <= 0:
                raise TypeError(
                    f"{cls.__qualname__}: concrete Numeric must set positive class-level width (bits)"
                )
            if cls.dtype not in _TOKEN_TO_NUMERIC:
                _TOKEN_TO_NUMERIC[cls.dtype] = cls

    def __new__(cls, value: Any) -> Any:
        from ..types import Scalar

        if not cls.dtype:
            raise TypeError("Numeric is abstract")
        return Scalar(value, cls.dtype)

    @classmethod
    def mlir_type(cls, ctx: mlir_ir.Context) -> mlir_ir.Type:
        with ctx:
            return _elem_token_to_mlir_type(cls.dtype)

    @classmethod
    def from_mlir_type(cls, ty: mlir_ir.Type) -> type[Numeric]:
        tok = _mlir_type_to_elem_token(ty)
        try:
            return _TOKEN_TO_NUMERIC[tok]
        except KeyError as exc:
            raise TypeError(f"unsupported element type for Numeric: {ty!r}") from exc


def _elem_token_to_mlir_type(token: str) -> mlir_ir.Type:
    token = token.strip().lower()
    if token == "index":
        return mlir_ir.IndexType.get()
    if token == "f16":
        return mlir_ir.F16Type.get()
    if token == "bf16":
        return mlir_ir.BF16Type.get()
    if token == "f32":
        return mlir_ir.F32Type.get()
    if token == "f64":
        return mlir_ir.F64Type.get()
    if token.startswith("i") and token[1:].isdigit():
        return mlir_ir.IntegerType.get_signless(int(token[1:]))
    if token.startswith("si") and token[2:].isdigit():
        return mlir_ir.IntegerType.get_signed(int(token[2:]))
    if token.startswith("u") and token[1:].isdigit():
        return mlir_ir.IntegerType.get_unsigned(int(token[1:]))
    if token.startswith("ui") and token[2:].isdigit():
        return mlir_ir.IntegerType.get_unsigned(int(token[2:]))
    raise TypeError(f"unsupported Tla element type token: {token!r}")


def _mlir_type_to_elem_token(ty: mlir_ir.Type) -> str:
    if isinstance(ty, mlir_ir.IndexType):
        return "index"
    if isinstance(ty, mlir_ir.F16Type):
        return "f16"
    if isinstance(ty, mlir_ir.BF16Type):
        return "bf16"
    if isinstance(ty, mlir_ir.F32Type):
        return "f32"
    if isinstance(ty, mlir_ir.F64Type):
        return "f64"
    if isinstance(ty, mlir_ir.IntegerType):
        int_type = mlir_ir.IntegerType(ty)
        prefix = "u" if int_type.is_unsigned else "i"
        return f"{prefix}{int_type.width}"
    from catlass import _tla_type_bridge

    element_type = _tla_type_bridge.value_element_type_get(ty.context, ty)
    if element_type is not None:
        return _mlir_type_to_elem_token(element_type)
    raise TypeError(f"unsupported element type for Numeric: {ty!r}")


class Bool(Numeric):
    dtype: ClassVar[str] = "i1"
    width: ClassVar[int] = 1


class Int8(Numeric):
    dtype: ClassVar[str] = "i8"
    width: ClassVar[int] = 8


class Int16(Numeric):
    dtype: ClassVar[str] = "i16"
    width: ClassVar[int] = 16


class Int32(Numeric):
    dtype: ClassVar[str] = "i32"
    width: ClassVar[int] = 32


class Int64(Numeric):
    dtype: ClassVar[str] = "i64"
    width: ClassVar[int] = 64


class UInt8(Numeric):
    dtype: ClassVar[str] = "u8"
    width: ClassVar[int] = 8


class UInt16(Numeric):
    dtype: ClassVar[str] = "u16"
    width: ClassVar[int] = 16


class UInt32(Numeric):
    dtype: ClassVar[str] = "u32"
    width: ClassVar[int] = 32


class UInt64(Numeric):
    dtype: ClassVar[str] = "u64"
    width: ClassVar[int] = 64


class Float16(Numeric):
    dtype: ClassVar[str] = "f16"
    width: ClassVar[int] = 16


class BFloat16(Numeric):
    dtype: ClassVar[str] = "bf16"
    width: ClassVar[int] = 16


class Float32(Numeric):
    dtype: ClassVar[str] = "f32"
    width: ClassVar[int] = 32


class Float64(Numeric):
    dtype: ClassVar[str] = "f64"
    width: ClassVar[int] = 64


class Index(Numeric):
    dtype: ClassVar[str] = "index"
    width: ClassVar[int] = 64


class Constexpr(Generic[_T]):
    """Type marker for compile-time-only frontend parameters.

    Parameters annotated as ``Constexpr[...]`` are consumed during lowering and are
    excluded from the runtime kernel ABI.
    """

    @classmethod
    def is_constexpr_annotation(cls, annotation: Any) -> bool:
        if annotation is cls:
            return True
        origin = get_origin(annotation)
        return origin is cls


@runtime_checkable
class JitArgument(Protocol):
    """Protocol for objects that can be passed to JIT-compiled Tla functions."""

    def __c_pointers__(self) -> list[int]:
        """Return list of integer pointers for runtime execution."""
        raise NotImplementedError

    def __get_mlir_types__(self, context: Any | None = None) -> list[Any]:
        """Return list of MLIR types for function signatures."""
        raise NotImplementedError

    def __extract_mlir_values__(self) -> list[Any]:
        raise NotImplementedError

    def __new_from_mlir_values__(self, values: list[Any]) -> Any:
        """Reconstruct an object from MLIR values in the JIT trace."""
        raise NotImplementedError


class Pointer(ABC):
    """Abstract JIT pointer (typed ``Pointer`` protocol).

    Concrete instances include ``catlass.core_api._Pointer`` (``dtype`` is ``Type[Numeric]``)
    and allocator helpers (``dtype`` may be a string token).
    """

    @property
    @abstractmethod
    def dtype(self) -> Any: ...

    @property
    def value_type(self) -> Any:
        return self.dtype

    @abstractmethod
    def __get_mlir_types__(self, context: Any | None = None) -> list[Any]: ...

    @abstractmethod
    def __extract_mlir_values__(self) -> list[Any]: ...

    @abstractmethod
    def __new_from_mlir_values__(self, values: list[Any]) -> Pointer: ...


__all__ = [
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
    "Float16",
    "BFloat16",
    "Float32",
    "Float64",
    "Index",
    "Pointer",
    "Constexpr",
    "JitArgument",
]
