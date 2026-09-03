"""Runtime JIT argument adapters for host launch packing.

Constexpr annotation checks and the adapter registry used when packing
launch arguments.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, get_origin

from ..typing import is_constexpr_annotation


class _PointerLaunchArg:
    """Launch argument that exposes a single device pointer via ``__c_pointers__``."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

    def __c_pointers__(self) -> list[int]:
        return [self._ptr]


class JitArgAdapterRegistry:
    """Map Python types to launch-time DSL objects with ``__c_pointers__``."""

    jit_arg_adapter_registry: dict[type[Any], Callable[[Any], Any]] = {}

    @classmethod
    def register_jit_arg_adapter(
        cls, python_type: type[Any]
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Register a JIT argument adapter for ``python_type``."""

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            if python_type in cls.jit_arg_adapter_registry:
                raise RuntimeError(
                    f"JIT argument adapter for {python_type} is already registered!"
                )
            cls.jit_arg_adapter_registry[python_type] = fn
            return fn

        return decorator

    @classmethod
    def clear(cls) -> None:
        cls.jit_arg_adapter_registry.clear()

    @classmethod
    def get_registered_adapter(cls, arg: object) -> Callable[[Any], Any] | None:
        """Return the registered adapter for ``type(arg)``, or ``None``."""
        return cls.jit_arg_adapter_registry.get(type(arg), None)


register_jit_arg_adapter = JitArgAdapterRegistry.register_jit_arg_adapter


def _adapt_from_data_ptr(obj: Any) -> Any:
    """If ``obj`` exposes ``data_ptr``, wrap it as a pointer launch arg."""
    data_ptr = getattr(obj, "data_ptr", None)
    if callable(data_ptr):
        return _PointerLaunchArg(int(data_ptr()))
    if data_ptr is not None and not callable(data_ptr):
        return _PointerLaunchArg(int(data_ptr))
    return obj


def _owner_marks_classmethod(owner: Callable[..., Any] | None) -> bool:
    """Whether ``owner`` is a ``classmethod`` (raw or ``__func__``-wrapped)."""
    if owner is None:
        return False
    if isinstance(owner, classmethod):
        return True
    return isinstance(getattr(owner, "__func__", None), classmethod)


def is_arg_annotation_constexpr(
    arg_annotation: Any,
    arg_name: str,
    arg_index: int,
    owning_func: Optional[Callable[..., Any]],
) -> bool:
    """True when the parameter is compile-time only for Host launch packing.

    Treats method receivers (``self`` / classmethod ``cls``) as compile-time, and
    accepts bare ``Constexpr`` / ``Constexpr[...]`` including postponed string
    forms (same rules as ``is_constexpr_annotation``).
    """
    # First positional receiver is never packed into the launch ABI.
    if arg_index == 0 and arg_name == "self":
        return True
    if arg_index == 0 and arg_name == "cls" and _owner_marks_classmethod(owning_func):
        return True

    return is_constexpr_annotation(arg_annotation)


def is_argument_constexpr(
    arg: Any,
    arg_annotation: Any,
    arg_name: str,
    arg_index: int,
    owning_func: Callable[..., Any],
) -> bool:
    """True when a bound value must stay Host-side (not a device launch arg)."""
    if arg is None:
        return True
    if is_arg_annotation_constexpr(arg_annotation, arg_name, arg_index, owning_func):
        return True
    # ``Type[X]`` / bare type tokens participate in specialization, not launch.
    if isinstance(arg, type) and (
        arg_annotation is inspect.Parameter.empty or get_origin(arg_annotation) is type
    ):
        return True
    return False


def _is_compile_time_callable(value: Any) -> bool:
    """True for plain callables / ``functools.partial`` / ``@tla.jit`` wrappers.

    Used when deciding whether a host value is a Constexpr-callable candidate
    (must still be annotated ``tla.Constexpr[...]`` to enter staging).
    """

    return (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or isinstance(value, functools.partial)
        or getattr(value, "_tla_jit", False) is True
    )


__all__ = [
    "JitArgAdapterRegistry",
    "register_jit_arg_adapter",
    "_PointerLaunchArg",
    "is_arg_annotation_constexpr",
    "is_argument_constexpr",
    "_is_compile_time_callable",
]
