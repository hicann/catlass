"""Typing protocols, ``Numeric`` value model, and ``Pointer`` ABC."""

from __future__ import annotations

import ctypes
import operator
import struct
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast as typing_cast,
    get_origin,
    runtime_checkable,
)

import numpy as np
from mlir import ir as mlir_ir  # type: ignore[assignment]

from .op import dsl_user_op

_T = TypeVar("_T")

_TOKEN_TO_NUMERIC: dict[str, type["Numeric"]] = {}


def _binary_op(
    op: Callable[..., Any],
    *,
    flip: bool = False,
) -> Callable[..., Any]:
    """Binary op wrapper on Numeric (emits ``arith.*``)."""

    _COMPARE_OPS = (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )

    def wrapper(
        lhs: "Numeric",
        rhs: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        from mlir.dialects import arith

        if not isinstance(rhs, Numeric):
            if not isinstance(rhs, (int, float, bool, mlir_ir.Value)):
                return NotImplemented
            rhs = as_numeric(rhs)

        from .. import runtime as _runtime

        in_frontend = _runtime._current_frontend_state() is not None
        host_fold = (
            not in_frontend
            and isinstance(lhs.value, (int, float, bool))
            and isinstance(rhs.value, (int, float, bool))
        )

        # No silent promotion: both operands must already share a concrete type.
        if type(lhs) is not type(rhs):
            raise TypeError(
                f"Numeric operands must have the same type, got "
                f"{type(lhs).__name__} and {type(rhs).__name__}; "
                f"cast explicitly with .to(...) before the operation"
            )

        # ``/`` is float-only; integers must use ``//`` (no truncate / promote).
        if op is operator.truediv and type(lhs).is_integer:
            raise TypeError(
                "Numeric '/' is only supported for float types; "
                "use '//' for integer division, or cast to Float32 first"
            )

        # ``**`` is float-only; Ascend/TLA backend does not lower ``math.ipowi``.
        if op is operator.pow and type(lhs).is_integer:
            raise TypeError(
                "Numeric '**' is only supported for float types; "
                "cast to Float32 first (integer pow is not supported by the backend)"
            )

        res_type: type[Numeric] = Bool if op in _COMPARE_OPS else type(lhs)

        if host_fold:
            a, b = lhs.value, rhs.value
            if flip:
                a, b = b, a
            return res_type(op(a, b))

        if not in_frontend:
            raise RuntimeError(
                "Numeric arithmetic on SSA values requires TLA frontend lowering context"
            )

        lhs_ir = lhs.ir_value(loc=loc)
        rhs_ir = rhs.ir_value(loc=loc)
        if flip:
            lhs_ir, rhs_ir = rhs_ir, lhs_ir

        if op in _COMPARE_OPS:
            fpred, ipred = {
                operator.eq: ("OEQ", "eq"),
                operator.ne: ("ONE", "ne"),
                operator.lt: ("OLT", "slt"),
                operator.le: ("OLE", "sle"),
                operator.gt: ("OGT", "sgt"),
                operator.ge: ("OGE", "sge"),
            }[op]
            if type(lhs).is_float:
                result = arith.CmpFOp(
                    getattr(arith.CmpFPredicate, fpred), lhs_ir, rhs_ir, loc=loc
                ).result
            else:
                if op in (operator.lt, operator.le, operator.gt, operator.ge) and not type(
                    lhs
                ).signed:
                    ipred = {"slt": "ult", "sle": "ule", "sgt": "ugt", "sge": "uge"}[
                        ipred
                    ]
                result = arith.CmpIOp(
                    getattr(arith.CmpIPredicate, ipred), lhs_ir, rhs_ir, loc=loc
                ).result
            return Bool(result)

        unsigned = not res_type.signed
        if op is operator.add:
            name = "arith.addf" if res_type.is_float else "arith.addi"
        elif op is operator.sub:
            name = "arith.subf" if res_type.is_float else "arith.subi"
        elif op is operator.mul:
            name = "arith.mulf" if res_type.is_float else "arith.muli"
        elif op is operator.truediv:
            # Integer ``/`` already rejected above; only float ``divf`` remains.
            name = "arith.divf"
        elif op is operator.floordiv:
            if res_type.is_float:
                raise TypeError(
                    "Numeric '//' is only supported for integer types; use '/' for floats"
                )
            name = "arith.divui" if unsigned else "arith.divsi"
        elif op is operator.mod:
            if res_type.is_float:
                raise TypeError("Numeric '%' is only supported for integer types")
            name = "arith.remui" if unsigned else "arith.remsi"
        elif op is operator.pow:
            # Integer ``**`` already rejected above; only float ``powf`` remains.
            name = "math.powf"
        elif op is operator.and_:
            if res_type.is_float:
                raise TypeError("Numeric '&' is only supported for integer types")
            name = "arith.andi"
        elif op is operator.or_:
            if res_type.is_float:
                raise TypeError("Numeric '|' is only supported for integer types")
            name = "arith.ori"
        elif op is operator.xor:
            if res_type.is_float:
                raise TypeError("Numeric '^' is only supported for integer types")
            name = "arith.xori"
        elif op is operator.lshift:
            if res_type.is_float:
                raise TypeError("Numeric '<<' is only supported for integer types")
            name = "arith.shli"
        elif op is operator.rshift:
            if res_type.is_float:
                raise TypeError("Numeric '>>' is only supported for integer types")
            name = "arith.shrui" if unsigned else "arith.shrsi"
        else:
            raise TypeError(f"unsupported Numeric operator: {op!r}")

        result = mlir_ir.Operation.create(
            name,
            operands=[lhs_ir, rhs_ir],
            results=[res_type.mlir_type()],
            loc=loc,
        ).results[0]
        return res_type(result)

    return wrapper


class DslType(type):
    """Metaclass for DSL types: ``is_abstract`` + type identity."""

    _is_abstract: bool

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        new_cls = super().__new__(cls, name, bases, attrs)
        new_cls._is_abstract = is_abstract
        return new_cls

    @property
    def is_abstract(cls) -> bool:
        return cls._is_abstract


class NumericMeta(DslType):
    """Metaclass for numeric types.

    Class-creation kwargs: ``width``, ``dtype`` (TLA token), ``mlir_type`` factory,
    ``is_abstract``.
    """

    width: int
    dtype: str
    signed: bool
    _mlir_type_factory: Callable[[], mlir_ir.Type] | None

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 0,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        signed: bool = True,
        **kwargs: Any,
    ) -> Any:
        del kwargs

        def _extract_mlir_values(self: "Numeric") -> tuple[mlir_ir.Value, ...]:
            return (self.ir_value(),)

        def _new_from_mlir_values(
            self: "Numeric", values: list[Any] | tuple[Any, ...]
        ) -> "Numeric":
            if len(values) != 1:
                raise TypeError(
                    f"Numeric expects 1 MLIR value after dynamic if, got {len(values)}"
                )
            return type(self)(values[0])

        new_attrs = {
            "__extract_mlir_values__": _extract_mlir_values,
            "__new_from_mlir_values__": _new_from_mlir_values,
        }
        # Prefer explicit methods defined on the class body.
        new_attrs.update(attrs)
        new_cls = super().__new__(
            cls, name, bases, new_attrs, is_abstract=is_abstract
        )
        new_cls.width = width
        new_cls.dtype = dtype
        new_cls.signed = signed
        new_cls._mlir_type_factory = staticmethod(mlir_type) if mlir_type else None
        if dtype and not is_abstract:
            token = dtype.strip().lower()
            if token not in _TOKEN_TO_NUMERIC:
                _TOKEN_TO_NUMERIC[token] = typing_cast(type[Numeric], new_cls)
        return new_cls

    def mlir_type(cls, ctx: mlir_ir.Context | None = None) -> mlir_ir.Type:
        """Return MLIR type for this Numeric class (optional ``ctx``)."""
        factory = cls._mlir_type_factory
        if factory is None:
            raise TypeError(f"{cls.__name__} has no mlir_type factory")
        if ctx is None:
            return factory()
        with ctx:
            return factory()

    def is_same_kind(cls, other: type) -> bool:
        return (cls.is_integer and getattr(other, "is_integer", False)) or (
            cls.is_float and getattr(other, "is_float", False)
        )

    @property
    def is_integer(cls) -> bool:
        return False

    @property
    def is_float(cls) -> bool:
        return False


class IntegerMeta(NumericMeta):
    """Metaclass for integer numeric types."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        signed: bool = True,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Inject ``__c_pointers__`` on Integer family (host bits as ``list[int]``).
        def _c_pointers(self: "Integer") -> list[int]:
            if isinstance(self.value, mlir_ir.Value):
                raise TypeError(
                    f"{type(self).__name__} SSA value cannot provide host __c_pointers__"
                )
            if width == 1:
                return [1 if bool(self.value) else 0]
            if signed:
                c_value = getattr(ctypes, f"c_int{width}")(int(self.value))
                return [c_value.value & ((1 << width) - 1)]
            return [int(self.value) & ((1 << width) - 1)]

        # Class body wins if it defines ``__c_pointers__``; else metaclass injects.
        injected = {"__c_pointers__": _c_pointers}
        merged = {**injected, **attrs}
        return super().__new__(
            cls,
            name,
            bases,
            merged,
            width=width,
            dtype=dtype,
            mlir_type=mlir_type,
            is_abstract=is_abstract,
            signed=signed,
            **kwargs,
        )

    @property
    def is_integer(cls) -> bool:
        return True

    @property
    def is_float(cls) -> bool:
        return False


class FloatMeta(NumericMeta):
    """Metaclass for floating-point numeric types."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super().__new__(
            cls,
            name,
            bases,
            attrs,
            width=width,
            dtype=dtype,
            mlir_type=mlir_type,
            is_abstract=is_abstract,
            signed=True,
            **kwargs,
        )

    @property
    def is_integer(cls) -> bool:
        return False

    @property
    def is_float(cls) -> bool:
        return True


@dsl_user_op
def cast(
    obj: Any,
    type_: type["Numeric"],
    *,
    loc: mlir_ir.Location | None = None,
) -> "Numeric":
    """Cast ``obj`` to a concrete Numeric type.

    ``cast`` constructs the target type (``type_(obj)``). Cross-type SSA
    conversion is done in ``Numeric.__init__`` / ``.to`` (via arith ops), not by
    rebinding the raw SSA value.
    """
    if not isinstance(type_, NumericMeta):
        raise TypeError(f"cast target must be a Numeric type, got {type_!r}")
    if type_.is_abstract:
        if not isinstance(obj, type_):
            raise TypeError(
                f"can't cast {obj!r} to abstract {type_.__name__}. "
                "Pass a concrete type instead, e.g. Int32, Float32."
            )
        return obj
    ctor = typing_cast(Callable[..., Numeric], type_)
    return ctor(obj, loc=loc)


def _mlir_i(width: int, *, unsigned: bool = False) -> Callable[[], mlir_ir.Type]:
    def factory() -> mlir_ir.Type:
        if unsigned:
            return mlir_ir.IntegerType.get_unsigned(width)
        return mlir_ir.IntegerType.get_signless(width)

    return factory


def _mlir_f16() -> mlir_ir.Type:
    return mlir_ir.F16Type.get()


def _mlir_bf16() -> mlir_ir.Type:
    return mlir_ir.BF16Type.get()


def _mlir_f32() -> mlir_ir.Type:
    return mlir_ir.F32Type.get()


def _mlir_index() -> mlir_ir.Type:
    return mlir_ir.IndexType.get()


class Numeric(metaclass=NumericMeta, is_abstract=True):
    """Numeric type and value (``DslType`` / ``NumericMeta`` first-class).

    - Type tag: ``dtype=tla.Float16``; class attr ``dtype`` is the TLA token string.
    - Value: ``tla.Int32(5)`` / ``tensor[i]`` → concrete Numeric; arithmetic via operators.
    """

    dtype: ClassVar[str] = ""
    width: ClassVar[int] = 0
    signed: ClassVar[bool] = True

    def __new__(cls, value: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if cls.is_abstract:
            raise TypeError(
                f"{cls.__name__} is abstract; use a concrete type (e.g. Int32, Float32)"
            )
        if value is None:
            raise TypeError(f"{cls.__name__}() missing required argument: 'value'")
        return object.__new__(cls)

    def __init__(
        self,
        value: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        # Cross-type conversion: host via Python; SSA via ``.to`` (emits arith ops).
        # Entry is construction / ``cast``, not a bare SSA unwrap.
        cls = type(self)
        if isinstance(value, Numeric):
            if type(value) is cls:
                value = value.value
            elif isinstance(value.value, (bool, int, float)):
                value = float(value.value) if cls.is_float else int(value.value)
            else:
                converted = value.to(cls, loc=loc)
                self.value = converted.value
                self.__tla_category__ = "numeric"
                if isinstance(self.value, mlir_ir.Value):
                    from .. import runtime as _runtime

                    _runtime._bind_frontend_value(self, self.value)
                    _runtime._bind_frontend_category(self, "numeric")
                    _runtime._bind_frontend_category(self.value, "numeric")
                return
        if isinstance(value, mlir_ir.Value):
            expected = cls.mlir_type()
            if str(value.type) != str(expected):
                src_cls = Numeric.from_mlir_type(value.type)
                value = src_cls(value).to(cls, loc=loc).value
            self.value: Any = value
            self.__tla_category__ = "numeric"
            from .. import runtime as _runtime

            _runtime._bind_frontend_value(self, value)
            _runtime._bind_frontend_category(self, "numeric")
            _runtime._bind_frontend_category(value, "numeric")
        elif isinstance(value, (bool, int, float)):
            self.value = value
            self.__tla_category__ = "numeric"
        else:
            raise TypeError(
                f"{type(self).__name__} expects bool|int|float|mlir.ir.Value|Numeric, "
                f"got {type(value).__name__}"
            )

    @classmethod
    def from_dtype_token(cls, token: str) -> type[Numeric]:
        """Resolve a TLA element token (e.g. ``\"i32\"``) to a concrete Numeric class."""
        key = token.strip().lower()
        try:
            return _TOKEN_TO_NUMERIC[key]
        except KeyError as exc:
            raise TypeError(f"unsupported Tla element type token: {token!r}") from exc

    @classmethod
    def from_mlir_type(cls, ty: mlir_ir.Type) -> type[Numeric]:
        if isinstance(ty, mlir_ir.IndexType):
            token = "index"
        elif isinstance(ty, mlir_ir.F16Type):
            token = "f16"
        elif isinstance(ty, mlir_ir.BF16Type):
            token = "bf16"
        elif isinstance(ty, mlir_ir.F32Type):
            token = "f32"
        elif isinstance(ty, mlir_ir.F64Type):
            raise TypeError(f"unsupported element type for Numeric: {ty!r}")
        elif isinstance(ty, mlir_ir.IntegerType):
            int_type = mlir_ir.IntegerType(ty)
            prefix = "u" if int_type.is_unsigned else "i"
            token = f"{prefix}{int_type.width}"
        else:
            raise TypeError(f"unsupported element type for Numeric: {ty!r}")
        try:
            return _TOKEN_TO_NUMERIC[token]
        except KeyError as exc:
            raise TypeError(f"unsupported element type for Numeric: {ty!r}") from exc

    def ir_value(self, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
        if isinstance(self.value, mlir_ir.Value):
            return self.value
        cls = type(self)
        element_type = cls.mlir_type()
        if cls.is_float:
            return mlir_ir.Operation.create(
                "arith.constant",
                results=[element_type],
                attributes={
                    "value": mlir_ir.FloatAttr.get(element_type, float(self.value))
                },
                loc=loc,
            ).results[0]
        if isinstance(self.value, float) and not float(self.value).is_integer():
            raise TypeError(
                f"expected integer literal for {cls.__name__}, got {self.value!r}"
            )
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={
                "value": mlir_ir.IntegerAttr.get(element_type, int(self.value))
            },
            loc=loc,
        ).results[0]

    @dsl_user_op
    def to(
        self,
        dtype: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        if dtype is int:
            if isinstance(self.value, (int, float, bool)):
                return int(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python int")
        if dtype is float:
            if isinstance(self.value, (int, float, bool)):
                return float(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python float")
        if dtype is bool:
            if isinstance(self.value, (int, float, bool)):
                return bool(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python bool")
        if dtype is mlir_ir.Value:
            return self.ir_value(loc=loc)
        if not isinstance(dtype, NumericMeta):
            raise TypeError(f"unsupported to() target: {dtype!r}")
        if dtype.is_abstract:
            raise TypeError(
                f"can't convert to abstract {dtype.__name__}; "
                "use a concrete type (e.g. Int32, Float32)"
            )
        if dtype is type(self):
            return self
        # Host: construct target (Python cast in __init__). SSA: emit arith ops.
        if isinstance(self.value, (bool, int, float)):
            return dtype(self, loc=loc)
        src = self.ir_value(loc=loc)
        dst_ty = dtype.mlir_type()
        src_ty = type(self)
        if str(src.type) == str(dst_ty):
            return dtype(src)
        if src_ty.is_integer and dtype.is_integer:
            if dtype.width == src_ty.width:
                return self.bitcast(dtype, loc=loc)
            cast_name = (
                ("arith.extui" if not src_ty.signed else "arith.extsi")
                if dtype.width > src_ty.width
                else "arith.trunci"
            )
            result = mlir_ir.Operation.create(
                cast_name, operands=[src], results=[dst_ty], loc=loc
            ).results[0]
            return dtype(result)
        if src_ty.is_float and dtype.is_float:
            cast_name = (
                "arith.extf" if dtype.width > src_ty.width else "arith.truncf"
            )
            result = mlir_ir.Operation.create(
                cast_name, operands=[src], results=[dst_ty], loc=loc
            ).results[0]
            return dtype(result)
        if src_ty.is_integer and dtype.is_float:
            cast_name = "arith.uitofp" if not src_ty.signed else "arith.sitofp"
            result = mlir_ir.Operation.create(
                cast_name, operands=[src], results=[dst_ty], loc=loc
            ).results[0]
            return dtype(result)
        if src_ty.is_float and dtype.is_integer:
            cast_name = "arith.fptoui" if not dtype.signed else "arith.fptosi"
            result = mlir_ir.Operation.create(
                cast_name, operands=[src], results=[dst_ty], loc=loc
            ).results[0]
            return dtype(result)
        raise TypeError(
            f"unsupported Numeric.to({dtype.__name__}) from {type(self).__name__}"
        )

    @dsl_user_op
    def bitcast(
        self,
        dtype: type["Numeric"],
        *,
        loc: mlir_ir.Location | None = None,
    ) -> "Numeric":
        if not isinstance(dtype, NumericMeta) or dtype.is_abstract:
            raise TypeError(
                f"bitcast dtype must be a concrete Numeric type, got {dtype!r}"
            )
        if dtype is type(self):
            return self
        if dtype.width != type(self).width:
            raise ValueError(
                f"bitcast requires same bit width: {type(self).__name__}({type(self).width}) "
                f"vs {dtype.__name__}({dtype.width})"
            )
        src = self.ir_value(loc=loc)
        result = mlir_ir.Operation.create(
            "arith.bitcast",
            operands=[src],
            results=[dtype.mlir_type()],
            loc=loc,
        ).results[0]
        return dtype(result)

    def __repr__(self) -> str:
        if isinstance(self.value, mlir_ir.Value):
            return f"{type(self).__name__}(<{self.value.type}>)"
        return f"{type(self).__name__}({self.value!r})"

    def __int__(self) -> int:
        if isinstance(self.value, (int, float, bool)):
            return int(self.value)
        raise TypeError(f"{type(self).__name__} SSA cannot be converted to Python int")

    def __index__(self) -> int:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return int(self.value)
        raise TypeError(f"{type(self).__name__} SSA cannot be used as a Python index")

    def __bool__(self) -> bool:
        # ``__bool__``: static host values only.
        if isinstance(self.value, (int, float, bool)):
            return bool(self.value)
        raise TypeError(
            f"{type(self).__name__} SSA cannot be converted to Python bool; "
            "use explicit comparison in the DSL"
        )

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        ctx = context
        if ctx is None:
            ctx = mlir_ir.Context.current
        with ctx:
            return [type(self).mlir_type()]

    def __c_pointers__(self) -> list[int]:
        # Base stub; concrete packing lives on IntegerMeta / Float* subclasses.
        raise ValueError(
            "only support built-in types: bool, (u)int{8, 16, 32, 64}, "
            f"float{{16, 32, 64}}, bf16, index; got {type(self).__name__}"
        )

    @dsl_user_op
    def __add__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.add)(self, other, loc=loc)

    @dsl_user_op
    def __radd__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.add)(self, other, loc=loc)

    @dsl_user_op
    def __sub__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.sub)(self, other, loc=loc)

    @dsl_user_op
    def __rsub__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.sub, flip=True)(self, other, loc=loc)

    @dsl_user_op
    def __mul__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.mul)(self, other, loc=loc)

    @dsl_user_op
    def __rmul__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.mul)(self, other, loc=loc)

    @dsl_user_op
    def __truediv__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.truediv)(self, other, loc=loc)

    @dsl_user_op
    def __rtruediv__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.truediv, flip=True)(self, other, loc=loc)

    @dsl_user_op
    def __floordiv__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.floordiv)(self, other, loc=loc)

    @dsl_user_op
    def __rfloordiv__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.floordiv, flip=True)(self, other, loc=loc)

    @dsl_user_op
    def __mod__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.mod)(self, other, loc=loc)

    @dsl_user_op
    def __rmod__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.mod, flip=True)(self, other, loc=loc)

    @dsl_user_op
    def __neg__(self, *, loc: mlir_ir.Location | None = None) -> "Numeric":
        if isinstance(self.value, (int, float, bool)):
            return type(self)(-self.value)
        from .. import runtime as _runtime

        if _runtime._current_frontend_state() is None:
            raise RuntimeError("Numeric.__neg__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        if type(self).is_float:
            result = mlir_ir.Operation.create(
                "arith.negf", operands=[v], results=[v.type], loc=loc
            ).results[0]
        else:
            zero = type(self)(0).ir_value(loc=loc)
            result = mlir_ir.Operation.create(
                "arith.subi", operands=[zero, v], results=[v.type], loc=loc
            ).results[0]
        return type(self)(result)

    @dsl_user_op
    def __abs__(self, *, loc: mlir_ir.Location | None = None) -> "Numeric":
        if isinstance(self.value, (int, float, bool)):
            return type(self)(abs(self.value))
        from .. import runtime as _runtime

        if _runtime._current_frontend_state() is None:
            raise RuntimeError("Numeric.__abs__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        if type(self).is_float:
            result = mlir_ir.Operation.create(
                "math.absf", operands=[v], results=[v.type], loc=loc
            ).results[0]
        else:
            result = mlir_ir.Operation.create(
                "math.absi", operands=[v], results=[v.type], loc=loc
            ).results[0]
        return type(self)(result)

    @dsl_user_op
    def __eq__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        # Same-type mismatches must raise (do not swallow into NotImplemented,
        # or Python falls back to identity ``==`` / ``!=``).
        return _binary_op(operator.eq)(self, other, loc=loc)

    @dsl_user_op
    def __ne__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        return _binary_op(operator.ne)(self, other, loc=loc)
    @dsl_user_op
    def __lt__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        return _binary_op(operator.lt)(self, other, loc=loc)

    @dsl_user_op
    def __le__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        return _binary_op(operator.le)(self, other, loc=loc)

    @dsl_user_op
    def __gt__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        return _binary_op(operator.gt)(self, other, loc=loc)

    @dsl_user_op
    def __ge__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> Any:
        return _binary_op(operator.ge)(self, other, loc=loc)

    @dsl_user_op
    def __pow__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.pow)(self, other, loc=loc)

    @dsl_user_op
    def __rpow__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.pow, flip=True)(self, other, loc=loc)

    @staticmethod
    def _from_python_value(value: Any) -> "Numeric":
        """Map Python / MLIR scalar to a concrete Numeric."""
        if isinstance(value, Numeric):
            return value
        if isinstance(value, bool):
            return Bool(value)
        if isinstance(value, int):
            if (value <= 2147483647) and (value >= -2147483648):
                return Int32(value)
            return Int64(value)
        if isinstance(value, float):
            return Float32(value)
        if isinstance(value, mlir_ir.Value):
            return Numeric.from_mlir_type(value.type)(value)
        raise ValueError(
            f"unable to convert {value} in type {type(value)} to Numeric"
        )


def as_numeric(obj: Any) -> Numeric:
    """Convert a Python primitive or MLIR value to a Numeric."""
    if isinstance(obj, Numeric):
        return obj
    return Numeric._from_python_value(obj)


class Integer(Numeric, metaclass=IntegerMeta, width=32, mlir_type=_mlir_i(32), is_abstract=True):
    """Abstract integer numeric family."""

    @dsl_user_op
    def __invert__(self, *, loc: mlir_ir.Location | None = None) -> "Integer":
        cls = typing_cast(type[Integer], type(self))
        if isinstance(self.value, (int, bool)):
            return cls(~int(self.value))
        from .. import runtime as _runtime

        if _runtime._current_frontend_state() is None:
            raise RuntimeError("Integer.__invert__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        ones = cls(-1).ir_value(loc=loc)
        result = mlir_ir.Operation.create(
            "arith.xori", operands=[v, ones], results=[v.type], loc=loc
        ).results[0]
        return cls(result)

    @dsl_user_op
    def __lshift__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.lshift)(self, other, loc=loc)

    @dsl_user_op
    def __rlshift__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"Cannot left shift {other_} with {self}")
        return other_.__lshift__(self, loc=loc)

    @dsl_user_op
    def __rshift__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.rshift)(self, other, loc=loc)

    @dsl_user_op
    def __rrshift__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"Cannot right shift {other_} with {self}")
        return other_.__rshift__(self, loc=loc)

    @dsl_user_op
    def __and__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.and_)(self, other, loc=loc)

    @dsl_user_op
    def __rand__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return self.__and__(other, loc=loc)

    @dsl_user_op
    def __or__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.or_)(self, other, loc=loc)

    @dsl_user_op
    def __ror__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return self.__or__(other, loc=loc)

    @dsl_user_op
    def __xor__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return _binary_op(operator.xor)(self, other, loc=loc)

    @dsl_user_op
    def __rxor__(
        self, other: Any, *, loc: mlir_ir.Location | None = None
    ) -> "Numeric":
        return self.__xor__(other, loc=loc)


class Float(Numeric, metaclass=FloatMeta, width=32, mlir_type=_mlir_f32, is_abstract=True):
    """Abstract floating-point numeric family."""


class Bool(Integer, metaclass=IntegerMeta, width=1, dtype="i1", signed=True, mlir_type=_mlir_i(1)):
    pass


class Int8(Integer, metaclass=IntegerMeta, width=8, dtype="i8", signed=True, mlir_type=_mlir_i(8)):
    pass


class Int16(Integer, metaclass=IntegerMeta, width=16, dtype="i16", signed=True, mlir_type=_mlir_i(16)):
    pass


class Int32(Integer, metaclass=IntegerMeta, width=32, dtype="i32", signed=True, mlir_type=_mlir_i(32)):
    pass


class Int64(Integer, metaclass=IntegerMeta, width=64, dtype="i64", signed=True, mlir_type=_mlir_i(64)):
    pass


class UInt8(
    Integer, metaclass=IntegerMeta, width=8, dtype="u8", signed=False, mlir_type=_mlir_i(8, unsigned=True)
):
    pass


class UInt16(
    Integer,
    metaclass=IntegerMeta,
    width=16,
    dtype="u16",
    signed=False,
    mlir_type=_mlir_i(16, unsigned=True),
):
    pass


class UInt32(
    Integer,
    metaclass=IntegerMeta,
    width=32,
    dtype="u32",
    signed=False,
    mlir_type=_mlir_i(32, unsigned=True),
):
    pass


class UInt64(
    Integer,
    metaclass=IntegerMeta,
    width=64,
    dtype="u64",
    signed=False,
    mlir_type=_mlir_i(64, unsigned=True),
):
    pass


class Float16(Float, metaclass=FloatMeta, width=16, dtype="f16", mlir_type=_mlir_f16):
    @staticmethod
    def _host_bits(value: float) -> int:
        f16_val = np.float16(value)
        return int(f16_val.view(np.uint16))

    def __c_pointers__(self) -> list[int]:
        if isinstance(self.value, mlir_ir.Value):
            raise TypeError("Float16 SSA value cannot provide host __c_pointers__")
        return [Float16._host_bits(float(self.value))]


class BFloat16(Float, metaclass=FloatMeta, width=16, dtype="bf16", mlir_type=_mlir_bf16):
    def __c_pointers__(self) -> list[int]:
        if isinstance(self.value, mlir_ir.Value):
            raise TypeError("BFloat16 SSA value cannot provide host __c_pointers__")
        bits = int(np.float32(self.value).view(np.uint32))
        # Round-to-nearest-even when truncating f32 → bf16 (not plain >> 16).
        if (bits & 0x7F800000) != 0x7F800000:
            bits = (bits + 0x00007FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return [int(np.uint16(bits >> 16))]


class Float32(Float, metaclass=FloatMeta, width=32, dtype="f32", mlir_type=_mlir_f32):
    @staticmethod
    def _host_bits(value: float) -> int:
        return struct.unpack("I", struct.pack("f", float(value)))[0]

    def __c_pointers__(self) -> list[int]:
        if isinstance(self.value, mlir_ir.Value):
            raise TypeError("Float32 SSA value cannot provide host __c_pointers__")
        return [Float32._host_bits(float(self.value))]


class Index(
    Integer, metaclass=IntegerMeta, width=64, dtype="index", signed=True, mlir_type=_mlir_index
):
    pass


class Constexpr(Generic[_T]):
    """Type marker for compile-time-only frontend parameters."""

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
        raise NotImplementedError

    def __get_mlir_types__(self, context: Any | None = None) -> list[Any]:
        raise NotImplementedError

    def __extract_mlir_values__(self) -> list[Any]:
        raise NotImplementedError

    def __new_from_mlir_values__(self, values: list[Any]) -> Any:
        raise NotImplementedError


class Pointer(ABC):
    """Abstract JIT pointer (typed ``Pointer`` protocol)."""

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
    "DslType",
    "NumericMeta",
    "IntegerMeta",
    "FloatMeta",
    "Numeric",
    "Integer",
    "Float",
    "cast",
    "as_numeric",
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
    "Index",
    "Pointer",
    "Constexpr",
    "JitArgument",
]
