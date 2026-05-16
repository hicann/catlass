from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..execution import (
    TlaKernelArtifact,
    TlaExecutionResult,
    TlaRuntimeUnavailableError,
    TlaUnsupportedAbiError,
    execute_kernel,
)


@dataclass(frozen=True)
class TlaJitExecutor:
    """Callable wrapper around a compiled Tla kernel artifact."""

    artifact: TlaKernelArtifact

    def launch(
        self,
        *launch_args: Any,
        block: int | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        if "grid" in kwargs and block is not None:
            raise TlaUnsupportedAbiError("Use either `block` or `grid`, not both.")
        if launch_args and args is not None:
            raise TlaUnsupportedAbiError("Launch arguments specified multiple times.")
        if args is None:
            args = launch_args
        launch_kwargs = dict(kwargs)
        if block is not None:
            if not isinstance(block, int):
                raise TlaUnsupportedAbiError("`block` must be an int.")
            launch_kwargs["grid"] = (int(block), 1, 1)
        runtime = self.artifact.runtime
        if runtime is None:
            raise TlaRuntimeUnavailableError(
                "Compiled artifact is missing runtime options and cannot be launched."
            )
        return execute_kernel(
            self.artifact,
            runtime=runtime,
            launch_args=tuple(args),
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> TlaExecutionResult:
        return self.launch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.artifact, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TlaJitExecutor):
            return self.artifact == other.artifact
        return self.artifact == other
