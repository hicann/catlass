"""Shared helpers for user-facing DSL operation wrappers."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints

from mlir import ir as mlir_ir  # type: ignore[assignment]

from .. import runtime as _runtime


def _capture_user_loc() -> mlir_ir.Location | None:
    frame = inspect.currentframe()
    caller = (
        frame.f_back.f_back
        if frame is not None
        and frame.f_back is not None
        and frame.f_back.f_back is not None
        else None
    )
    if caller is None:
        return None
    frame_info = inspect.getframeinfo(caller)
    positions = getattr(frame_info, "positions", None)
    col_offset = int(getattr(positions, "col_offset", 0) or 0)
    lineno = int(getattr(positions, "lineno", frame_info.lineno) or frame_info.lineno)
    if lineno <= 0:
        return mlir_ir.Location.unknown()
    file_loc = mlir_ir.Location.file(frame_info.filename, lineno, col_offset)
    return mlir_ir.Location.name(frame_info.function, childLoc=file_loc)


def _record_category(value: Any, category: str) -> None:
    _runtime._bind_frontend_category(value, category)
    try:
        setattr(value, "__tla_category__", category)
    except (AttributeError, TypeError):
        pass


def dsl_user_op(op_func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach caller source location to user-facing DSL op calls."""
    # Lazy import: ``types`` imports ``typing``, so a top-level import here
    # would cycle when ``typing`` decorates Numeric with ``dsl_user_op``.
    return_category: str | None = None
    try:
        from ..types import annotation_to_category

        return_annotation = get_type_hints(
            op_func, globalns=op_func.__globals__
        ).get("return")
        return_category = annotation_to_category(return_annotation)
    except Exception:
        return_category = None

    @wraps(op_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        loc = kwargs.pop("loc", None)
        if loc is None and _runtime._current_frontend_state() is not None:
            loc = _capture_user_loc()
        elif loc is not None and not isinstance(loc, mlir_ir.Location):
            raise TypeError(
                f"loc must be mlir.ir.Location or None, got {type(loc).__name__}"
            )
        result = op_func(*args, loc=loc, **kwargs)
        if return_category is not None and result is not None:
            _record_category(result, return_category)
        return result

    return wrapper
