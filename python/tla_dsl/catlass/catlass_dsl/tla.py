"""Tla kernel launch API."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Sequence

from .. import runtime as _runtime
from ..dsl import (
    _bind_kernel_call_args,
    _get_typed_call_args,
    _strip_constexpr_launch_args,
)
from ..execution import TlaExecutionResult, TlaUnsupportedAbiError

if TYPE_CHECKING:
    from ..dsl import TlaJitFunction


class KernelLauncher:
    """Runtime launch wrapper returned when a ``@tla.kernel`` function is called.

    Typical usage::

        launcher = my_kernel(tx, ty, options="--npu-arch 3510")
        launcher(block_num=1)          # or launcher.launch(block_num=1)

    The first call returns this object without launching. Compilation happens
    when the launcher is constructed with launch args (or the kernel has no
    parameters), or on ``.launch`` when there is no cached artifact or runtime
    options changed; otherwise ``.launch`` reuses the cached artifact. To
    compile once and launch repeatedly with compile and launch separated, use
    ``tla.compile`` and the returned executor.
    See Host API reference (Compile and Launch).
    """

    def __init__(
        self,
        fn: "TlaJitFunction",
        *,
        launch_kwargs: dict[str, Any] | None = None,
        launch_args: Sequence[Any] | None = None,
    ) -> None:
        self._fn = fn
        # Kernel args passed by name arrive mixed in with the runtime options.
        bound_args, bound_kwargs = _bind_kernel_call_args(
            fn.fn, tuple(launch_args or ()), launch_kwargs or {}
        )
        self._launch_kwargs = bound_kwargs
        self._launch_args = bound_args
        self._runtime = None
        self._artifact = None
        type_args = (
            _get_typed_call_args(self._launch_args, self._fn.fn)
            if self._launch_args
            else None
        )
        should_eager_compile = (
            type_args is not None or not inspect.signature(self._fn.fn).parameters
        )
        if should_eager_compile:
            runtime = _runtime.runtime_options_for_launch(
                _runtime.runtime_options_from_kwargs(self._launch_kwargs)
            )
            self._runtime = runtime
            self._artifact = _runtime.compile_kernel(
                self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )

    def launch(
        self,
        *,
        block_num: int | None = None,
        type_args: Sequence[Any] | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        """Directory: Compile and Launch / Launch
        Description:
            Launch a `@tla.kernel` on the NPU. Obtain this object by calling the
            decorated kernel (`launcher = my_kernel(*tensors, options=...)`); that
            call does not launch. Then call `launcher.launch(...)` or invoke the
            launcher (`launcher(block_num=...)`, equivalent to
            `.launch(args=..., **kwargs)`). Compiles when there is no cached artifact
            or runtime options changed; reuses the cached artifact when runtime
            options are unchanged. To compile once and launch repeatedly with compile
            and launch separated, use `tla.compile` and the returned
            `TlaJitExecutor`.

        Parameters:
            - *`block_num`* (`int | None`): Number of blocks to launch. Optional;
              default `1` (also taken from kwargs stored when the launcher was
              constructed). Must be an `int` when provided.
            - *`type_args`* (`Sequence[Any] | None`): Compile-time type samples.
              Optional; inferred from `args` / constructor launch args when omitted.
            - *`args`* (`Sequence[Any] | None`): Explicit runtime argument sequence.
              Optional. Cannot be combined with launch args already stored on the
              launcher from the first call (`my_kernel(*tensors)`).
            - *`stream`* (`Any`, via `**kwargs`): Optional ACL stream handle (often
              an `int`). When omitted, uses `torch.npu.current_stream` if available;
              otherwise set `stream=` explicitly or `CATLASS_DSL_NPU_DEVICE`.
            - *`options`* (`str`, via `**kwargs`): Public chip name, e.g.
              `options="--npu-arch 3510"`. May be set on the first call or here;
              kwargs from this call override values stored when the launcher was
              constructed.

        Constraints:
            - Calling `@tla.kernel` returns `KernelLauncher` and does not launch.
              This method (or calling the launcher) compiles when there is no cached
              artifact or runtime options changed, then launches; it does not
              recompile when runtime options are unchanged and a cached artifact
              exists.
            - When the first call already passed tensors (`my_kernel(tx, ty)`), pass
              only launch kwargs such as `block_num` on the second call / `.launch`.
            - Repeated `.launch` on the same launcher reuses the cached artifact
              when runtime options are unchanged. Use `tla.compile` when compile and
              launch must be separated explicitly.
            - `args=` must not be set if the launcher already holds launch args
              (`TlaUnsupportedAbiError`).
            - Launch args must be bound NPU buffers for tensors (`from_dlpack`);
              `make_fake_tensor` samples are for compile / type samples only.
            - `block_num` must be an `int` (default `1`).

        Example:
        ```python
        @tla.kernel
        def vadd(src: tla.Tensor, dst: tla.Tensor) -> None:
            with tla.vector():
                tla.copy(src, dst)

        vadd(tx, ty, options="--npu-arch 3510")(block_num=1)
        # or:
        launcher = vadd(tx, ty, options="--npu-arch 3510")
        launcher.launch(block_num=1)
        # or pass args on .launch when the first call had none:
        vadd(options="--npu-arch 3510").launch(args=(tx, ty), block_num=1)
        ```

        """
        launch_kwargs = {**self._launch_kwargs, **kwargs}
        if block_num is not None:
            launch_kwargs["block_num"] = block_num
        block_num = launch_kwargs.pop("block_num", 1)
        if not isinstance(block_num, int):
            raise TlaUnsupportedAbiError("`block_num` must be an int.")
        launch_args = self._launch_args
        if args is not None:
            if launch_args:
                raise TlaUnsupportedAbiError("`args` specified multiple times.")
            launch_args = tuple(args)
        if type_args is None and launch_args:
            type_args = _get_typed_call_args(launch_args, self._fn.fn)
        launch_kwargs["block_num"] = int(block_num)
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
            artifact = _runtime.compile_kernel(
                self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )
            self._artifact = artifact
            self._runtime = runtime
        return _runtime.execute_kernel(
            artifact,
            runtime=runtime,
            launch_args=_strip_constexpr_launch_args(launch_args, self._fn.fn),
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> TlaExecutionResult:
        return self.launch(args=args, **kwargs)


__all__ = ["KernelLauncher"]
