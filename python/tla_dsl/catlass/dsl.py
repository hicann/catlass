"""Tla DSL decorators and lowering entry points."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, is_dataclass
import inspect
from typing import Any, Callable, Mapping, Sequence

from . import runtime as _runtime
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.typing import is_constexpr_annotation
from .base_dsl.compiler import CompileCallable, compile
from .execution import TlaKernelArtifact


def _kernel_signature(fn: Callable[..., Any] | None) -> inspect.Signature | None:
    if fn is None:
        return None
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


#: Compile/launch kwargs consumed by the runtime rather than by the kernel
#: signature. A kernel parameter sharing one of these names is ambiguous, so
#: ``_bind_kernel_call_args`` reports the clash instead of guessing.
_RESERVED_CALL_KWARGS = frozenset(
    {"options", "block_num", "device", "stream", "cache", "cache_dir", "type_args"}
)


def _bind_kernel_call_args(
    fn: Callable[..., Any] | None,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve a call into one positional host value per kernel parameter.

    Lowering takes host values positionally, but callers write ordinary Python
    calls — by name, out of order, relying on defaults — and kernel arguments
    share one ``**kwargs`` with the runtime options. Binding is delegated to
    ``Signature.bind_partial`` so keyword, keyword-only and defaulted parameters
    follow the same rules they would in a plain Python call.

    Returns the bound values and the kwargs left over for the runtime.
    """
    sig = _kernel_signature(fn)
    if sig is None:
        return tuple(args), dict(kwargs or {})
    params = list(sig.parameters.values())
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if any(p.kind in variadic for p in params):
        # No fixed parameter list to bind against; leave the call untouched.
        return tuple(args), dict(kwargs or {})

    remaining = dict(kwargs or {})
    by_name = {}
    for param in params[len(args) :]:
        if param.name not in remaining:
            continue
        if param.name in _RESERVED_CALL_KWARGS:
            raise TypeError(
                f"kernel parameter {param.name!r} collides with the runtime "
                f"option of the same name; pass it positionally or rename the "
                f"parameter"
            )
        by_name[param.name] = remaining.pop(param.name)

    try:
        bound = sig.bind_partial(*args, **by_name)
    except TypeError:
        # Not a call this signature accepts; let the arity check report it.
        return tuple(args), dict(kwargs or {})
    bound.apply_defaults()

    # Values are positional from here on, so stop at the first parameter the
    # call left unbound rather than shifting every later value left.
    values: list[Any] = []
    for param in params:
        if param.name not in bound.arguments:
            break
        values.append(bound.arguments[param.name])
    # Hand back anything the truncation dropped, so it is not silently lost.
    for name, value in by_name.items():
        if name not in {p.name for p in params[: len(values)]}:
            remaining[name] = value
    return tuple(values), remaining


def _constexpr_param_mask(fn: Callable[..., Any] | None) -> tuple[bool, ...]:
    """One flag per kernel parameter: is it annotated ``tla.Constexpr[...]``?"""
    sig = _kernel_signature(fn)
    if sig is None:
        return ()
    return tuple(is_constexpr_annotation(p.annotation) for p in sig.parameters.values())


def _strip_constexpr_launch_args(
    args: Sequence[Any], fn: Callable[..., Any] | None
) -> tuple[Any, ...]:
    """Drop ``Constexpr`` params from a launch arg list.

    Constexpr params are baked into the compiled kernel and have no ABI slot, so
    they must not reach the launch payload — mirrors how ``get_rectified_args``
    skips ``Constexpr`` dataclass fields.
    """
    mask = _constexpr_param_mask(fn)
    if not any(mask):
        return tuple(args)
    return tuple(a for i, a in enumerate(args) if not (i < len(mask) and mask[i]))


def _get_typed_call_args(
    args: Sequence[Any], fn: Callable[..., Any] | None = None
) -> Sequence[Any] | None:
    # ``Constexpr`` params are compile-time host values with no MLIR type and no
    # kernel block arg, so they are passed through verbatim whatever their type
    # (``str``, tuple, enum, …). Without this they would fall into the ``None``
    # branch below and the kernel body would silently see ``None`` instead of the
    # value — and every variant would collapse onto one cached kernel.
    mask = _constexpr_param_mask(fn)
    args, _ = _bind_kernel_call_args(fn, args)
    inferred: list[Any] = []
    has_constexpr_value = False
    for pos, arg in enumerate(args):
        if pos < len(mask) and mask[pos]:
            inferred.append(arg)
            if arg is not None:
                has_constexpr_value = True
            continue
        resolver = getattr(arg, "__get_mlir_types__", None)
        if callable(resolver):
            inferred.append(arg)
        elif is_dataclass(arg) and not isinstance(arg, type):
            # Plain stdlib ``@dataclass`` instances are unpacked into per-field
            # scalar kernel args; keep the instance so lowering can read fields.
            inferred.append(arg)
        elif isinstance(arg, (bool, int, float)):
            # Preserve host literals for ``tla.Constexpr[...]`` / numeric params.
            # Plain ints must not be erased to None or Constexpr lowering sees NoneType.
            inferred.append(arg)
        else:
            inferred.append(None)
    if not has_constexpr_value and all(item is None for item in inferred):
        return None
    return tuple(inferred)


from .catlass_dsl.tla import KernelLauncher


_ACTIVE_JIT_HELPER_TRANSFORMER: contextvars.ContextVar[
    Callable[["TlaJitFunction"], Callable[..., Any]] | None
] = contextvars.ContextVar("tla_active_jit_helper_transformer", default=None)


@contextlib.contextmanager
def _jit_helper_transformer(
    transform: Callable[["TlaJitFunction"], Callable[..., Any]],
):
    """Use transformed ``@tla.jit`` helpers while lowering one root function."""

    token = _ACTIVE_JIT_HELPER_TRANSFORMER.set(transform)
    try:
        yield
    finally:
        _ACTIVE_JIT_HELPER_TRANSFORMER.reset(token)


@dataclass
class TlaJitFunction:
    """Wrapper for Tla DSL JIT/kernels that can emit and execute Tla IR."""

    fn: Callable[..., Any]
    kind: str
    options: Mapping[str, Any]
    decorator_location: DSLLocation | None = None
    _mlir: str | None = None
    _base_dsl: BaseDSL | None = None
    _lowered: Any | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.kind == "kernel":
            return KernelLauncher(
                self, launch_kwargs=dict(kwargs), launch_args=tuple(args)
            )
        transform = _ACTIVE_JIT_HELPER_TRANSFORMER.get()
        if transform is not None:
            return transform(self)(*args, **kwargs)
        return self.fn(*args, **kwargs)

    def compile(
        self, *, type_args: Sequence[Any] | None = None, **kwargs: Any
    ) -> TlaKernelArtifact:
        """Directory: Compile and Launch / Compile
        Description:
            Compile this `@tla.kernel` function and return a `TlaKernelArtifact`.
            Use `tla.compile(fn, *args, options=...)` as the usual Host entry; call
            `.compile()` only when you already hold a `TlaJitFunction` and need the
            raw `TlaKernelArtifact`.

            Parameters:
            - *`type_args`* (`Sequence[Any] | None`): Host tensors / scalars used as
              compile type samples. Optional; default `None` (no tensor
              specialization).
            - *`kwargs`*: Host compile kwargs. Pass `options="--npu-arch 3510"` to
              select the public chip name. Cache / IR-dump knobs use `CATLASS_DSL_*`
              environment variables.

            Constraints:
            - `type_args` are compile-time type samples; they need not be bound NPU
              buffers (`make_fake_tensor` is valid).
            - Pass the public chip name with `options="--npu-arch 3510"`;
              unsupported tokens raise at compile time.

            Example:
            ```python
            artifact = my_kernel.compile(
                type_args=[tx, ty],
                options="--npu-arch 3510",
            )
            ```

        """
        runtime = _runtime.runtime_options_from_kwargs(kwargs)
        return _runtime.compile_kernel(
            self.fn,
            kind=self.kind,
            options=self.options,
            runtime=runtime,
            type_args=type_args,
            decorator_location=self.decorator_location,
        )

    @property
    def mlir(self) -> str:
        if self._mlir is None:
            base_dsl = self._base_dsl or BaseDSL()
            lowered = base_dsl._lower(
                self.fn,
                kind=self.kind,
                options=dict(self.options),
                location=self.decorator_location,
            )
            self._lowered = lowered
            self._mlir = lowered.asm()
            self._base_dsl = base_dsl
        return self._mlir

    def dump_mlir(self, *, type_args: Sequence[Any] | None = None) -> str:
        """Directory: Compile and Launch / Inspect
        Description:
            Return the TLA IR (`tlair`) MLIR text for this kernel. Does not compile
            to a device binary and does not launch.

            Parameters:
            - *`type_args`* (`Sequence[Any] | None`): Host tensors / scalars used as
              type samples, same as `.compile()`. Optional; default `None`.

            Constraints:
            - `type_args` follow the same rules as `.compile()`.
            - The returned string is frontend TLA IR (`tlair`), not the HIVM/LLVM
              form stored on `TlaKernelArtifact.lowered_llvm`.

            Example:
            ```python
            text = my_kernel.dump_mlir(type_args=[fa, fb])
            print(text[:500])
            ```

        """
        base_dsl = self._base_dsl or BaseDSL()
        if type_args is None and self._mlir is not None:
            return self._mlir
        lowered = base_dsl._lower(
            self.fn,
            kind=self.kind,
            options=dict(self.options),
            type_args=type_args,
            location=self.decorator_location,
        )
        mlir = lowered.asm()
        if type_args is None:
            self._mlir = mlir
            self._lowered = lowered
        self._base_dsl = base_dsl
        return mlir


def jit(fn: Callable[..., Any]) -> TlaJitFunction:
    """Decorate a helper for Tla DSL lowering."""

    _reject_async_dsl_function(fn, kind="jit")
    return TlaJitFunction(
        fn,
        kind="jit",
        options={},
        decorator_location=_capture_decorator_location(),
    )


def kernel(
    fn: Callable[..., Any] | None = None,
    *,
    auto_sync: str | None = None,
) -> TlaJitFunction | Callable[[Callable[..., Any]], TlaJitFunction]:
    """Directory: Decorators
    Description:
        Mark a Python function as a TLA kernel entry. The function body is not
        executed on the Host. Returns a `TlaJitFunction`. Calling that object
        returns a `KernelLauncher` without launching; call the launcher or
        `.launch(...)` to compile and run.

        Parameters:
        - *`fn`* (`Callable[..., Any] | None`): The function being decorated.
          Use `@tla.kernel` or `@tla.kernel(auto_sync=...)`; call `tla.kernel(fn)`
          only when decorator syntax is unavailable.
        - *`auto_sync`* (`str | None`): Optional. `"v0"` inserts automatic local
          mutexes. Default `None` (synchronization stays explicit).

        Constraints:
        - The decorated function must not be defined with Python `async def`.
        - `auto_sync` must be `"v0"` or `None`.
        - Kernel parameter types:

          | Kind | Types |
          | --- | --- |
          | Tensor | `tla.Tensor` |
          | Python scalars | `bool` / `int` / `float` |
          | `tla` scalars | `Bool`, `Int8/16/32/64`, `UInt8/16/32/64`, `Float16/32`, `BFloat16` |
          | Compile-time | `tla.Constexpr[...]` |
          | Struct | `@dataclass` whose fields are among the above |

        Example:
        ```python
        @tla.kernel
        def vadd(src: tla.Tensor, dst: tla.Tensor) -> None:
            with tla.vector():
                tla.copy(src, dst)

        @tla.kernel(auto_sync="v0")
        def vadd_auto(src: tla.Tensor, dst: tla.Tensor) -> None:
            with tla.vector():
                tla.copy(src, dst)

        vadd(tx, ty, options="--npu-arch 3510")(block_num=1)
        ```

    """

    if auto_sync not in (None, "v0"):
        raise ValueError(
            f"tla.kernel auto_sync must be 'v0' or None, got {auto_sync!r}"
        )

    def decorate(target: Callable[..., Any]) -> TlaJitFunction:
        if not callable(target):
            raise TypeError("tla.kernel expects a callable")
        _reject_async_dsl_function(target, kind="kernel")
        options = {} if auto_sync is None else {"auto_sync": auto_sync}
        return TlaJitFunction(
            target,
            kind="kernel",
            options=options,
            decorator_location=_capture_decorator_location(),
        )

    if fn is None:
        return decorate
    return decorate(fn)


def _reject_async_dsl_function(fn: Callable[..., Any], *, kind: str) -> None:
    unwrapped = inspect.unwrap(fn)
    if not (
        inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)
    ):
        return
    filename = inspect.getsourcefile(unwrapped) or "<unknown>"
    try:
        source_lines, lineno = inspect.getsourcelines(unwrapped)
    except (OSError, IOError, TypeError):
        source_lines, lineno = [], 1
    text = source_lines[0] if source_lines else None
    offset = len(text) - len(text.lstrip()) + 1 if text is not None else None
    error = SyntaxError(f"async Tla {kind} functions are not supported")
    error.filename = filename
    error.lineno = lineno
    error.offset = offset
    error.text = text
    raise error


def _capture_decorator_location() -> DSLLocation | None:
    frame = inspect.currentframe()
    if frame is None:
        return None
    caller = frame.f_back
    while caller is not None and caller.f_code.co_filename == __file__:
        caller = caller.f_back
    if caller is None:
        return None
    filename = caller.f_code.co_filename
    return DSLLocation(
        filename=filename,
        lineno=int(caller.f_lineno),
        col_offset=0,
        function_name=caller.f_code.co_name,
    )


__all__ = [
    "DSLLocation",
    "BaseDSL",
    "TlaJitFunction",
    "KernelLauncher",
    "CompileCallable",
    "compile",
    "jit",
    "kernel",
]
