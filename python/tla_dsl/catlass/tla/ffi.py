"""Declaration API for user-provided TLA device functions."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Callable, TypeAlias, get_type_hints

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..base_dsl.op import dsl_user_op
from ..base_dsl.typing import Numeric, TypedPointer


_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


ExternArgType: TypeAlias = TypedPointer | type[Numeric]


@dataclass(frozen=True)
class ExternFunction:
    """A C ABI entry point supplied by user source code."""

    source: str
    symbol: str
    arg_types: tuple[ExternArgType, ...]

    @dsl_user_op
    def __call__(
        self,
        *args: object,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Emit a call to this external function during frontend lowering."""

        # Importing here keeps declaration independent from the frontend's
        # comparatively heavy core_api module.
        from ..core_api import _emit_extern_call

        _emit_extern_call(self, args, loc=loc)


def _validate_symbol(symbol: object) -> str:
    if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError(f"tla.extern name must be a C identifier, got {symbol!r}")
    return symbol


def _validate_arg_type(annotation: object, *, parameter: str) -> ExternArgType:
    if isinstance(annotation, TypedPointer):
        return annotation
    if (
        isinstance(annotation, type)
        and issubclass(annotation, Numeric)
        and annotation.dtype
    ):
        return annotation
    raise TypeError(
        "tla.extern parameter annotations must be Pointer[dtype, memory_space] "
        f"or a concrete Numeric type; parameter {parameter!r} has {annotation!r}"
    )


def extern(
    *,
    source: str,
    name: str | None = None,
) -> Callable[[Callable[..., None]], ExternFunction]:
    """Declare one C ABI entry point supplied by inline Ascend C source."""

    # Validity check
    if not isinstance(source, str):
        raise TypeError(f"tla.extern source must be str, got {type(source).__name__}")
    if not source.strip():
        raise ValueError("tla.extern source must not be empty")
    explicit_symbol = None if name is None else _validate_symbol(name)

    def decorator(fn: Callable[..., None]) -> ExternFunction:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn)

        # Validity check
        return_annotation = hints.get("return", signature.return_annotation)
        if return_annotation not in (None, type(None)):
            if return_annotation is inspect.Signature.empty:
                detail = "is missing"
            else:
                detail = f"must be None, got {return_annotation!r}"
            raise TypeError(f"tla.extern return annotation for {fn.__name__} {detail}")

        arg_types: list[ExternArgType] = []
        for parameter in signature.parameters.values():
            # Validity check
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError(
                    "tla.extern parameters must be positional and fixed; "
                    f"parameter {parameter.name!r} has kind {parameter.kind.description}"
                )
            if parameter.default is not inspect.Parameter.empty:
                raise TypeError(
                    "tla.extern parameters must not have default values; "
                    f"parameter {parameter.name!r} has default {parameter.default!r}"
                )
            annotation = hints.get(parameter.name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"tla.extern parameter {parameter.name!r} is missing an annotation"
                )
            # Add to arg_types after validation
            arg_types.append(_validate_arg_type(annotation, parameter=parameter.name))

        symbol = (
            explicit_symbol
            if explicit_symbol is not None
            else _validate_symbol(fn.__name__)
        )
        return ExternFunction(
            source=source,
            symbol=symbol,
            arg_types=tuple(arg_types),
        )

    return decorator


__all__ = [
    "extern",
]
