"""Declaration API for user-provided TLA device functions."""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeAlias, get_type_hints

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..base_dsl.op import dsl_user_op
from ..base_dsl.typing import Numeric, TypedPointer


_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


ExternArgType: TypeAlias = TypedPointer | type[Numeric]
_IncludeDir: TypeAlias = str | os.PathLike[str]


@dataclass(frozen=True)
class ExternFunction:
    """A C ABI entry point supplied by user source code."""

    source: str
    symbol: str
    arg_types: tuple[ExternArgType, ...]
    include_dirs: tuple[Path, ...] = ()

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
    include_dirs: _IncludeDir | Sequence[_IncludeDir] = (),
) -> Callable[[Callable[..., None]], ExternFunction]:
    """Directory: Decorators
    Description:
        Declare a Ascend C ABI entry point implemented by inline Ascend C source. The
        decorated Python function body is not executed; its name and annotations
        describe the symbol and argument ABI emitted by calls inside a TLA kernel.

    Parameters:
        - *`source`* (`str`): Non-empty Ascend C translation-unit source that
          defines the declared C ABI symbol.
        - *`name`* (`str | None`): Optional C symbol name. Defaults to the
          decorated Python function name.
        - *`include_dirs`* (`str | os.PathLike[str] | Sequence[str | os.PathLike[str]]`):
          Optional include search directory or ordered directories. Relative paths
          are resolved against the file containing the extern declaration.

    Constraints:
        - `name` must be a valid C identifier, and `source` must be non-empty.
        - Parameters must be positional, have no defaults, and be annotated with
          `tla.Pointer[dtype, address_space]` or a concrete TLA numeric type.
        - The return annotation must be `None`.
        - Calls must be inside exactly one `tla.cube()` or `tla.vector()` region
          and outside `tla.vec.func()`.
        - Within one kernel, a symbol may use only one extern declaration object;
          declarations sharing the same source must use identical ordered
          `include_dirs`.

    Example:
    ```python
    SOURCE = r'''
    #include "kernel_operator.h"
    extern "C" {
    [aicore] __attribute__((always_inline)) void store_value(
        uint64_t dst_addr, int32_t value) {
      auto dst = reinterpret_cast<__gm__ int32_t *>(dst_addr);
      dst[0] = value;
    }
    }
    '''

    @tla.extern(source=SOURCE, include_dirs="include")
    def store_value(
        dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
        value: tla.Int32,
    ) -> None: ...

    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.cube():
            store_value(dst.ptr, 42)
    ```

    """

    # Validity check
    if not isinstance(source, str):
        raise TypeError(f"tla.extern source must be str, got {type(source).__name__}")
    if not source.strip():
        raise ValueError("tla.extern source must not be empty")
    explicit_symbol = None if name is None else _validate_symbol(name)

    def decorator(fn: Callable[..., None]) -> ExternFunction:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn)
        symbol = (
            explicit_symbol
            if explicit_symbol is not None
            else _validate_symbol(fn.__name__)
        )
        source_file = inspect.getsourcefile(fn)
        declaration_dir = (
            Path(source_file).expanduser().resolve().parent
            if source_file
            else Path.cwd()
        )
        include_dir_values = (
            (include_dirs,)
            if isinstance(include_dirs, (str, os.PathLike))
            else include_dirs
        )
        resolved_include_dir_list: list[Path] = []
        for include_dir in include_dir_values:
            path = Path(include_dir).expanduser()
            if not path.is_absolute():
                path = declaration_dir / path
            resolved_include_dir_list.append(path.resolve())
        resolved_include_dirs = tuple(resolved_include_dir_list)

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

        return ExternFunction(
            source=source,
            symbol=symbol,
            arg_types=tuple(arg_types),
            include_dirs=resolved_include_dirs,
        )

    return decorator


__all__ = [
    "extern",
]
