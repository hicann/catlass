"""Tla DSL decorators and lowering entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Any, Callable, Mapping, Sequence
import contextvars

from . import runtime as _runtime
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.compiler import CompileCallable, compile
from .execution import TlaKernelArtifact, TlaExecutionResult, TlaUnsupportedAbiError


_JIT_TYPE_ARGS: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "tla_jit_type_args", default=None
)


class KernelLauncher:
    """Launch wrapper for zero-arg Tla kernels."""

    def __init__(
        self,
        fn: "TlaJitFunction",
        *,
        launch_kwargs: dict[str, Any] | None = None,
        launch_args: Sequence[Any] | None = None,
    ) -> None:
        self._fn = fn
        self._launch_kwargs = dict(launch_kwargs or {})
        self._launch_args = tuple(launch_args or ())
        self._runtime = None
        self._artifact = None
        type_args = _resolve_jit_type_args(self._fn.fn.__name__)
        if type_args is None and self._launch_args:
            type_args = _infer_type_args_from_runtime(self._launch_args)
        should_eager_compile = (
            type_args is not None or not inspect.signature(self._fn.fn).parameters
        )
        if should_eager_compile:
            runtime = _runtime.runtime_options_for_launch(
                _runtime.runtime_options_from_kwargs(self._launch_kwargs)
            )
            self._runtime = runtime
            self._artifact = _compile_kernel_compat(
                fn=self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )

    def launch(
        self,
        *,
        block: int | None = None,
        type_args: Sequence[Any] | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        if "grid" in kwargs or "grid" in self._launch_kwargs:
            raise TlaUnsupportedAbiError(
                "`grid` is not supported. Use `block` with an integer value."
            )
        launch_kwargs = {**self._launch_kwargs, **kwargs}
        if "block" in launch_kwargs:
            if block is not None:
                raise TlaUnsupportedAbiError("`block` specified multiple times.")
            block = launch_kwargs.pop("block")
        if block is None:
            block = 1
        if not isinstance(block, int):
            raise TlaUnsupportedAbiError("`block` must be an int.")
        launch_args = self._launch_args
        if args is not None:
            if launch_args:
                raise TlaUnsupportedAbiError("`args` specified multiple times.")
            launch_args = tuple(args)
        if type_args is None:
            type_args = _resolve_jit_type_args(self._fn.fn.__name__)
        if type_args is None and launch_args:
            type_args = _infer_type_args_from_runtime(launch_args)
        launch_kwargs["grid"] = (int(block), 1, 1)
        if (
            self._runtime is not None
            and "cache_dir" not in launch_kwargs
            and self._runtime.cache_dir is not None
            and not self._runtime.cache_enabled
        ):
            launch_kwargs["cache_dir"] = self._runtime.cache_dir
            launch_kwargs["cache"] = False
        runtime = _runtime.runtime_options_for_launch(
            _runtime.runtime_options_from_kwargs(launch_kwargs)
        )
        artifact = self._artifact
        if artifact is None or self._runtime != runtime:
            artifact = _compile_kernel_compat(
                fn=self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )
            self._artifact = artifact
            self._runtime = runtime
        print(f"kernel.o: {artifact.kernel_binary_path}")
        return _runtime.execute_kernel(
            artifact,
            runtime=runtime,
            launch_args=launch_args,
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> TlaExecutionResult:
        return self.launch(args=args, **kwargs)


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
        type_args = kwargs.pop("type_args", None)
        if type_args is None and args:
            inferred = _infer_type_args_from_runtime(args)
            if inferred is not None:
                type_args = {"__default__": inferred}
        if type_args is None:
            return self.fn(*args, **kwargs)
        token = _JIT_TYPE_ARGS.set(type_args)
        try:
            return self.fn(*args, **kwargs)
        finally:
            _JIT_TYPE_ARGS.reset(token)

    def compile(
        self, *, type_args: Sequence[Any] | None = None, **kwargs: Any
    ) -> TlaKernelArtifact:
        runtime = _runtime.runtime_options_from_kwargs(kwargs)
        return _compile_kernel_compat(
            fn=self.fn,
            kind=self.kind,
            options=self.options,
            runtime=runtime,
            type_args=type_args,
            decorator_location=self.decorator_location,
        )

    def run(
        self,
        *args: Any,
        type_args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        if args:
            raise TlaUnsupportedAbiError(
                "Phase-1 execution supports zero-argument kernels only."
            )
        runtime = _runtime.runtime_options_from_kwargs(kwargs)
        artifact = _compile_kernel_compat(
            fn=self.fn,
            kind=self.kind,
            options=self.options,
            runtime=runtime,
            type_args=type_args,
            decorator_location=self.decorator_location,
        )
        return _runtime.execute_kernel(
            artifact, runtime=runtime, launch_args=args, launch_kwargs=kwargs
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


@dataclass
class AscendNpuIrKernelFunction:
    """Wrapper for file-backed AscendNPU-IR kernels."""

    fn: Callable[..., Any]
    mlir_file: Path
    decorator_location: DSLLocation | None = None

    def compile(self, **kwargs: Any) -> TlaKernelArtifact:
        runtime = _runtime.runtime_options_from_kwargs(kwargs)
        return _compile_kernel_compat(
            fn=self.fn,
            kind="ascendnpuir_kernel",
            options={
                "mlir_file": str(self.mlir_file),
                "entrypoint": self.fn.__name__,
            },
            runtime=runtime,
            type_args=None,
            decorator_location=self.decorator_location,
        )

    def launch(
        self,
        *launch_args: Any,
        block: int | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        if "grid" in kwargs:
            raise TlaUnsupportedAbiError(
                "`grid` is not supported. Use `block` with an integer value."
            )
        if block is None:
            block = 1
        if not isinstance(block, int):
            raise TlaUnsupportedAbiError("`block` must be an int.")
        if launch_args and args is not None:
            raise TlaUnsupportedAbiError("Launch arguments specified multiple times.")
        if args is None:
            args = launch_args
        self._validate_launch_args(args)
        launch_kwargs = dict(kwargs)
        launch_kwargs["grid"] = (int(block), 1, 1)
        _apply_ascendnpuir_launch_target_defaults(launch_kwargs, self.mlir_file)
        runtime = _runtime.runtime_options_for_launch(
            _runtime.runtime_options_from_kwargs(launch_kwargs)
        )
        artifact = _compile_kernel_compat(
            fn=self.fn,
            kind="ascendnpuir_kernel",
            options={
                "mlir_file": str(self.mlir_file),
                "entrypoint": self.fn.__name__,
            },
            runtime=runtime,
            type_args=None,
            decorator_location=self.decorator_location,
        )
        print(f"kernel.o: {artifact.kernel_binary_path}")
        return _runtime.execute_kernel(
            artifact,
            runtime=runtime,
            launch_args=tuple(args),
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TlaUnsupportedAbiError(
                "ascendnpuir kernels accept launch arguments positionally. "
                "Use `.launch(..., device=..., block=...)` for launch options."
            )
        self._validate_launch_args(args)
        return AscendNpuIrKernelInvocation(self, launch_args=tuple(args))

    def _validate_launch_args(self, args: Sequence[Any]) -> None:
        parameters = tuple(inspect.signature(self.fn).parameters.values())
        required = [
            param
            for param in parameters
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and param.default is inspect._empty
        ]
        if len(args) < len(required):
            raise TlaUnsupportedAbiError(
                f"{self.fn.__name__} expects at least {len(required)} launch "
                f"arguments, got {len(args)}."
            )
        if not any(
            param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters
        ) and len(args) > len(parameters):
            raise TlaUnsupportedAbiError(
                f"{self.fn.__name__} expects at most {len(parameters)} launch "
                f"arguments, got {len(args)}."
            )


@dataclass
class AscendNpuIrKernelInvocation:
    """Launch wrapper for a file-backed AscendNPU-IR kernel invocation."""

    kernel: AscendNpuIrKernelFunction
    launch_args: Sequence[Any]

    def launch(self, **kwargs: Any) -> TlaExecutionResult:
        return self.kernel.launch(*self.launch_args, **kwargs)


def _apply_ascendnpuir_launch_target_defaults(
    launch_kwargs: dict[str, Any], mlir_file: Path
) -> None:
    if any(
        key in launch_kwargs
        for key in ("arch_scope", "target_arch", "core_type", "kernel_mode")
    ):
        return
    try:
        mlir_text = Path(mlir_file).read_text()
    except OSError:
        return

    core_type: str | None = None
    target_arch: str | None = None
    if (
        "hivm.module_core_type<AIC>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIC>" in mlir_text
    ):
        core_type = "aic"
    elif (
        "hivm.module_core_type<AIV>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir_text
    ):
        core_type = "aiv"

    if "dav-c310" in mlir_text or 'hacc.target<"Ascend950PR_9589">' in mlir_text:
        target_arch = "c310"

    if core_type is not None:
        launch_kwargs.setdefault("core_type", core_type)
        launch_kwargs.setdefault("kernel_mode", core_type)
    if target_arch is not None:
        launch_kwargs.setdefault("target_arch", target_arch)


def jit(fn: Callable[..., Any]) -> TlaJitFunction:
    """Decorate a helper for Tla DSL lowering."""

    return TlaJitFunction(
        fn,
        kind="jit",
        options={},
        decorator_location=_capture_decorator_location(),
    )


def kernel(fn: Callable[..., Any]) -> TlaJitFunction:
    """Decorate a Tla kernel entry point."""

    return TlaJitFunction(
        fn,
        kind="kernel",
        options={},
        decorator_location=_capture_decorator_location(),
    )


def ascendnpuir_kernel(filename: str | Path) -> Callable[[Callable[..., Any]], Any]:
    """Decorate a kernel backed by a pre-authored AscendNPU-IR MLIR file."""

    def decorator(fn: Callable[..., Any]) -> AscendNpuIrKernelFunction:
        location = _capture_decorator_location()
        return AscendNpuIrKernelFunction(
            fn=fn,
            mlir_file=_resolve_source_path(filename, location),
            decorator_location=location,
        )

    return decorator


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


def _resolve_source_path(filename: str | Path, location: DSLLocation | None) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path.resolve()
    if location is not None:
        return (Path(location.filename).resolve().parent / path).resolve()
    return path.resolve()


def _compile_kernel_compat(
    *,
    fn: Callable[..., Any],
    kind: str,
    options: Mapping[str, Any],
    runtime: Any,
    type_args: Sequence[Any] | None,
    decorator_location: DSLLocation | None,
) -> TlaKernelArtifact:
    try:
        return _runtime.compile_kernel(
            fn,
            kind=kind,
            options=options,
            runtime=runtime,
            type_args=type_args,
            decorator_location=decorator_location,
        )
    except TypeError as exc:
        if "decorator_location" not in str(exc):
            raise
        return _runtime.compile_kernel(
            fn,
            kind=kind,
            options=options,
            runtime=runtime,
            type_args=type_args,
        )


def _resolve_jit_type_args(kernel_name: str) -> Sequence[Any] | None:
    type_args = _JIT_TYPE_ARGS.get()
    if type_args is None:
        return None
    if isinstance(type_args, Mapping):
        if kernel_name in type_args:
            return type_args.get(kernel_name)
        return type_args.get("__default__")
    return type_args


def _infer_type_args_from_runtime(args: Sequence[Any]) -> Sequence[Any] | None:
    inferred: list[Any] = []
    for arg in args:
        resolver = getattr(arg, "__get_mlir_types__", None)
        if callable(resolver):
            inferred.append(arg)
        else:
            inferred.append(None)
    if all(item is None for item in inferred):
        return None
    return tuple(inferred)


__all__ = [
    "DSLLocation",
    "BaseDSL",
    "TlaJitFunction",
    "AscendNpuIrKernelFunction",
    "AscendNpuIrKernelInvocation",
    "KernelLauncher",
    "CompileCallable",
    "compile",
    "jit",
    "kernel",
    "ascendnpuir_kernel",
]
