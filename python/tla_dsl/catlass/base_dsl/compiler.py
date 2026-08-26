from __future__ import annotations

import shlex
from typing import Any

from .jit_executor import TlaJitExecutor


def _parse_compile_options_from_str(options: Any) -> dict[str, str]:
    """Parse ``options="--npu-arch 3510"`` token string."""
    if options is None:
        return {}
    if not isinstance(options, str):
        raise TypeError(
            "compile options must be a string "
            '(e.g. options="--npu-arch 3510"), '
            f"got {type(options).__name__}"
        )
    text = options.strip()
    if not text:
        return {}
    tokens = shlex.split(text)
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            parsed[key.replace("-", "_")] = value
            index += 1
            continue
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise ValueError(f"Missing value for compile option {token!r}")
            parsed[key] = tokens[index + 1]
            index += 2
            continue
        raise ValueError(f"Unexpected token in compile options: {token!r}")
    return parsed


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> TlaJitExecutor:
        """Directory: Compile and Launch / Compile
        Description:
            Compile a `@tla.kernel` function and return a callable executor wrapping
            the `TlaKernelArtifact`. This is the public `tla.compile` entry. Call the
            returned executor to launch (`artifact(*tensors, block_num=...)`). Use
            this path when you compile once and launch the same binary repeatedly.

            Parameters:
            - *`func`* (`TlaJitFunction`): Decorated `@tla.kernel` function. Required.
            - *`args`* (`Any`): Host tensors / scalars / `@dataclass` instances used
              as compile type samples (e.g. `from_dlpack` or `make_fake_tensor`
              results).
            - *`kwargs`*: Host compile kwargs. Pass `options="--npu-arch 3510"` to
              select the public chip name. Cache / IR dump / force-recompile use
              `CATLASS_DSL_*` environment variables.

            Constraints:
            - `func` must be a `@tla.kernel` `TlaJitFunction`.
            - `args` are compile-time type samples; they need not be bound NPU
              buffers (`make_fake_tensor` is valid).
            - Pass the public chip name with `options="--npu-arch 3510"`;
              unsupported tokens raise at compile time.
            - Launch kwargs such as `block_num` / `stream` belong on the returned
              executor (`artifact(...)` / `TlaJitExecutor.launch`), not on
              `tla.compile`.

            Example:
            ```python
            artifact = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
            artifact(tx, ty, block_num=1)
            artifact(tx, ty, block_num=1)  # launch again with the same binary
            ```

        """
        from ..execution import TlaUnsupportedAbiError
        from ..dsl import (
            TlaJitFunction,
            _bind_kernel_call_args,
            _get_typed_call_args,
        )
        from ..catlass_dsl.tla import KernelLauncher

        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        if isinstance(func, KernelLauncher):
            func = func._fn
        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError(
                "tla.compile expects a @tla.jit or @tla.kernel function."
            )
        type_args = kwargs.pop("type_args", None)
        # Kernel args passed by name arrive mixed in with the compile options.
        args, kwargs = _bind_kernel_call_args(func.fn, args, kwargs)
        if type_args is None and args:
            inferred = _get_typed_call_args(args, func.fn)
            if inferred is not None:
                type_args = inferred
        return TlaJitExecutor(func.compile(type_args=type_args, **kwargs))


compile = CompileCallable()


__all__ = ["TlaJitExecutor", "CompileCallable", "compile"]
