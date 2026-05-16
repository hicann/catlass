from __future__ import annotations

from typing import Any

from ..execution import TlaUnsupportedAbiError
from .jit_executor import TlaJitExecutor


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> TlaJitExecutor:
        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        from ..dsl import (
            AscendNpuIrKernelFunction,
            AscendNpuIrKernelInvocation,
            TlaJitFunction,
            KernelLauncher,
            _infer_type_args_from_runtime,
        )

        if isinstance(func, KernelLauncher):
            func = func._fn
        if isinstance(func, AscendNpuIrKernelInvocation):
            func = func.kernel
        if isinstance(func, AscendNpuIrKernelFunction):
            if args:
                func._validate_launch_args(args)
            return TlaJitExecutor(func.compile(**kwargs))
        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError(
                "tla.compile expects a @tla.jit, @tla.kernel, or "
                "@tla.ascendnpuir_kernel function."
            )
        type_args = kwargs.pop("type_args", None)
        if type_args is None and args:
            inferred = _infer_type_args_from_runtime(args)
            if inferred is not None:
                type_args = inferred
        return TlaJitExecutor(func.compile(type_args=type_args, **kwargs))


compile = CompileCallable()


__all__ = ["TlaJitExecutor", "CompileCallable", "compile"]
