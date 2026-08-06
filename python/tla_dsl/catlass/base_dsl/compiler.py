from __future__ import annotations

from typing import Any

from ..execution import TlaUnsupportedAbiError
from .jit_executor import TlaJitExecutor


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> TlaJitExecutor:
        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        from ..dsl import TlaJitFunction, _get_typed_call_args
        from ..catlass_dsl.tla import KernelLauncher

        if isinstance(func, KernelLauncher):
            func = func._fn
        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError(
                "tla.compile expects a @tla.jit or @tla.kernel function."
            )
        type_args = kwargs.pop("type_args", None)
        if type_args is None and args:
            inferred = _get_typed_call_args(args)
            if inferred is not None:
                type_args = inferred
        return TlaJitExecutor(func.compile(type_args=type_args, **kwargs))


compile = CompileCallable()


__all__ = ["TlaJitExecutor", "CompileCallable", "compile"]
