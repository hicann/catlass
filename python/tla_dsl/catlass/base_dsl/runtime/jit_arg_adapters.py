"""Dataclass tree reconstruction and existing host launch adapters."""

from __future__ import annotations

import functools
import inspect
from dataclasses import fields
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    get_origin,
    get_type_hints,
)

from ..typing import Numeric, is_constexpr_annotation
from ..utils.tree_utils import (
    ArgumentTreeError,
    TreeMetadata,
)


def _is_constexpr_annotation(annotation: Any) -> bool:
    forward_arg = getattr(annotation, "__forward_arg__", None)
    if forward_arg is not None:
        annotation = forward_arg
    return is_constexpr_annotation(annotation)


@lru_cache(maxsize=None)
def _resolved_field_annotations(cls: type) -> dict[str, Any]:
    """Resolve deferred field annotations once per class."""
    try:
        return get_type_hints(cls)
    except (NameError, TypeError):
        return getattr(cls, "__annotations__", {})


def _dataclass_members(
    value: Any,
) -> tuple[tuple[str, ...], list[Any], tuple[str, ...]]:
    names: list[str] = []
    values: list[Any] = []
    constexpr_names: list[str] = []
    declared_names = {item.name for item in fields(value)}
    extra_names = (
        vars(value).keys() - declared_names if hasattr(value, "__dict__") else set()
    )
    if extra_names:
        raise ArgumentTreeError(
            f"dataclass {type(value).__name__} has undeclared instance fields: "
            f"{', '.join(sorted(extra_names))}"
        )
    annotations = _resolved_field_annotations(type(value))
    for item in fields(value):
        try:
            field_value = getattr(value, item.name)
        except AttributeError as exc:
            raise ArgumentTreeError(
                f"dataclass {type(value).__name__} is missing field {item.name!r}"
            ) from exc
        names.append(item.name)
        values.append(field_value)
        if _is_constexpr_annotation(annotations.get(item.name, item.type)):
            constexpr_names.append(item.name)
    return tuple(names), values, tuple(constexpr_names)


def _flatten_dataclass(
    value: Any, expected_metadata: TreeMetadata | None
) -> tuple[TreeMetadata, list[Any]]:
    if expected_metadata is not None:
        metadata = expected_metadata
        names = metadata.fields
        extra_names = vars(value).keys() - names
        if extra_names:
            raise ArgumentTreeError(
                f"dataclass {type(value).__name__} has undeclared instance fields: "
                f"{', '.join(sorted(extra_names))}"
            )
        try:
            return metadata, [getattr(value, name) for name in names]
        except AttributeError as exc:
            raise ArgumentTreeError(
                f"dataclass {type(value).__name__} is missing a compiled field"
            ) from exc
    names, values, constexpr_names = _dataclass_members(value)
    _validate_dataclass_kernel_arg(value)
    return (
        TreeMetadata(
            python_type=type(value),
            fields=names,
            constexpr_indices=tuple(
                i for i, name in enumerate(names) if name in constexpr_names
            ),
            child_keys=names,
        ),
        values,
    )


_DATACLASS_DEFAULT_ONLY_PARAMS = (
    ("init", True),
    ("repr", True),
    ("eq", True),
    ("order", False),
    ("unsafe_hash", False),
    ("match_args", True),
    ("slots", False),
    ("weakref_slot", False),
)


def _validate_dataclass_options(value: Any) -> None:
    cls = type(value)
    params = getattr(cls, "__dataclass_params__", None)
    if params is not None:
        for name, default in _DATACLASS_DEFAULT_ONLY_PARAMS:
            if hasattr(params, name) and getattr(params, name) != default:
                raise ArgumentTreeError(
                    f"dataclass {cls.__name__} is used as a kernel argument but was "
                    f"declared with {name}={getattr(params, name)!r} (default "
                    f"{default!r}); only frozen= and kw_only= may be customized"
                )
    if getattr(cls, "__slots__", None):
        raise ArgumentTreeError(
            f"dataclass {cls.__name__} is used as a kernel argument but was declared "
            "with slots=True; only frozen= and kw_only= may be customized"
        )


def _validate_dataclass_kernel_arg(value: Any) -> None:
    """Directory: Decorators
    Description:
        Pack a Host-side kernel argument dataclass through the recursive
        struct-like argument tree.  Runtime fields become ordered ABI leaves;
        ``tla.Constexpr`` fields remain compile-time values.

    Parameters:
        - *``cls``*: The class decorated with ``@dataclass``.
        - *``frozen``*: Whether Python field assignment is disabled (default false).
        - *``kw_only``*: Whether construction requires keyword arguments (default false).

    Constraints:
        - Fields must be supported leaves or supported ordered aggregates.
        - Reconstruction calls ``cls(**fields)``: the constructor must accept
          every declared field as a keyword argument. A generated constructor
          does not accept ``field(init=False)`` fields.
        - Construction, including ``__post_init__``, runs during compilation.
        - ``slots=True`` and non-default dataclass options are rejected by the
          argument-tree validator.
        - ``Constexpr`` fields are not compared at launch. Recompile after
          changing static configuration; an existing kernel keeps its specialization.

    Example:
        ```python
        @dataclass(frozen=True)
        class Aux:
            tile: tla.Constexpr[int]
            bias: tla.Tensor
            limit: float

        compiled = tla.compile(kernel, Aux(128, bias, 0.5), ...)
        compiled(Aux(128, bias, 0.5), ...)
        ```
    """
    _validate_dataclass_options(value)


_CONSTEXPR_RO_CACHE: dict[tuple[type, tuple[str, ...]], type] = {}


def _constexpr_readonly_dataclass_cls(
    cls: type, constexpr_names: frozenset[str]
) -> type:
    key = (cls, tuple(sorted(constexpr_names)))
    cached = _CONSTEXPR_RO_CACHE.get(key)
    if cached is not None:
        return cached

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if name in constexpr_names and name in self.__dict__:
            raise AttributeError(
                f"{cls.__name__}.{name} is a tla.Constexpr field and is read-only "
                "(compile-time constant)"
            )
        cls.__setattr__(self, name, value)

    readonly = type(f"{cls.__name__}ConstexprRO", (cls,), {"__setattr__": __setattr__})
    _CONSTEXPR_RO_CACHE[key] = readonly
    return readonly


def _unflatten_dataclass(metadata: TreeMetadata, children: Iterable[Any]) -> Any:
    cls = metadata.python_type
    if metadata.constexpr_fields:
        cls = _constexpr_readonly_dataclass_cls(
            cls, frozenset(metadata.constexpr_fields)
        )
    return cls(**dict(zip(metadata.fields, children, strict=True)))


def _value_mlir_types(value: Any, context: Any) -> tuple[Any, ...]:
    """Resolve a value's declared MLIR types for tree construction."""

    return tuple(value.__get_mlir_types__(context))


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


def _adapt_registered_launch_args(args: Sequence[Any]) -> Sequence[Any]:
    """Apply existing launch adapters without changing logical leaf positions."""

    registry = JitArgAdapterRegistry.jit_arg_adapter_registry
    if not registry:
        return args
    adapted = None
    for index, arg in enumerate(args):
        adapter = registry.get(type(arg))
        if (
            adapter is None
            or isinstance(arg, Numeric)
            or hasattr(arg, "__c_pointers__")
        ):
            continue
        if adapted is None:
            adapted = list(args)
        adapted[index] = adapter(arg)
    return args if adapted is None else tuple(adapted)


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
