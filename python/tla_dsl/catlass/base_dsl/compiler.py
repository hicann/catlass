from __future__ import annotations

from typing import Any

from .jit_executor import JitCompiledFunction, JitModule


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> JitCompiledFunction:
        """Directory: Compile and Launch / Compile
        Description:
            Compile a `@tla.jit` or `@tla.kernel` function and return a callable
            `JitCompiledFunction`. This is the public `tla.compile` entry. Call the
            returned object to launch (`compiled(*tensors, block_num=...)`). Use this
            path when you compile once and launch the same binary repeatedly.

            Parameters:
            - *`func`* (`TlaJitFunction`): Decorated `@tla.kernel` function. Required.
            - *`args`* (`Any`): Host tensors / scalars / `@dataclass` instances used
              as compile type samples (e.g. `from_dlpack` or `make_fake_tensor`
              results).
            - *`kwargs`*: Host compile kwargs. Pass `options="--npu-arch 3510"` to
              select the public chip name. Cache / IR dump / force-recompile use
              `CATLASS_DSL_*` environment variables.

            Constraints:
            - `func` must be a `@tla.jit` or `@tla.kernel` `TlaJitFunction`.
            - `args` are compile-time type samples; they need not be bound NPU
              buffers (`make_fake_tensor` is valid).
            - Pass the public chip name with `options="--npu-arch 3510"`;
              unsupported tokens raise at compile time.
            - Launch kwargs such as `block_num` / `stream` belong on the returned
              compiled function, not on `tla.compile`.

            Example:
            ```python
            compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
            compiled(tx, ty, block_num=1)
            compiled(tx, ty, block_num=1)  # launch again with the same binary
            ```

        """
        from ..execution import TlaUnsupportedAbiError
        from ..dsl import (
            TlaJitFunction,
            _bind_kernel_call_args,
            _get_typed_call_args,
        )

        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError(
                "tla.compile expects a @tla.jit or @tla.kernel function."
            )
        if "type_args" in kwargs:
            raise TypeError(
                "tla.compile() does not accept `type_args`; "
                "pass compile-time sample arguments positionally."
            )
        # Kernel args passed by name arrive mixed in with the compile options.
        args, kwargs = _bind_kernel_call_args(func.fn, args, kwargs)
        type_args = _get_typed_call_args(args, fn=func.fn) if args else None
        return func.compile(type_args=type_args, **kwargs)


compile = CompileCallable()


__all__ = [
    "JitCompiledFunction",
    "JitModule",
    "CompileCallable",
    "compile",
]
