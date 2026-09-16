"""Source-aware formatting for execution-mode frontend failures."""

from __future__ import annotations

import dis
import linecache
import os
import sys
import traceback
from dataclasses import dataclass
from types import CodeType, TracebackType


@dataclass(frozen=True)
class SourceLocation:
    """A source span suitable for rendering a compact code frame."""

    filename: str
    lineno: int
    col_offset: int = 0
    end_col_offset: int | None = None


def _env_truthy(name: str, *, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def verbose_errors_enabled() -> bool:
    """Whether user-requested compiler detail should be included in errors."""

    return _env_truthy("CATLASS_DSL_VERBOSE_ERRORS", default="0")


def traceback_location_for_code(
    exc: BaseException, code: CodeType
) -> SourceLocation | None:
    """Return the last traceback position belonging to *code*."""

    matched = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is code:
            matched = tb
        tb = tb.tb_next
    if matched is None:
        return None
    positions = {
        instruction.offset: instruction.positions
        for instruction in dis.get_instructions(code)
    }
    position = positions.get(matched.tb_lasti)
    lineno = int(
        (position.lineno if position is not None else None) or matched.tb_lineno
    )
    col_offset = int((position.col_offset if position is not None else None) or 0)
    end_col_offset = (
        None
        if position is None or position.end_col_offset is None
        else int(position.end_col_offset)
    )
    return SourceLocation(
        filename=code.co_filename,
        lineno=lineno,
        col_offset=col_offset,
        end_col_offset=end_col_offset,
    )


def syntax_error_location(exc: SyntaxError) -> SourceLocation | None:
    """Return explicit parser or frontend-validation coordinates when present."""

    filename = exc.filename
    lineno = exc.lineno
    if not isinstance(filename, str) or not filename or not isinstance(lineno, int):
        return None
    if lineno <= 0:
        return None
    offset = exc.offset if isinstance(exc.offset, int) else 1
    col_offset = max(0, offset - 1)
    end_offset = exc.end_offset if isinstance(exc.end_offset, int) else None
    end_col_offset = (
        max(col_offset + 1, end_offset - 1) if end_offset is not None else None
    )
    return SourceLocation(
        filename=filename,
        lineno=lineno,
        col_offset=col_offset,
        end_col_offset=end_col_offset,
    )


def traceback_location_for_user_code(exc: BaseException) -> SourceLocation | None:
    """Return the deepest traceback position outside Catlass implementation code."""

    matched = None
    tb = exc.__traceback__
    while tb is not None:
        if not _is_internal_frame(tb.tb_frame.f_code.co_filename):
            matched = tb
        tb = tb.tb_next
    if matched is None:
        return None
    return traceback_location_for_code(exc, matched.tb_frame.f_code)


def render_code_frame(
    location: SourceLocation,
    *,
    context_lines: int = 2,
    source_text: str | None = None,
) -> str:
    """Render a stable plain-text source frame from a file or supplied text."""

    lines = (
        source_text.splitlines()
        if source_text is not None
        else linecache.getlines(location.filename)
    )
    if not lines or location.lineno <= 0 or location.lineno > len(lines):
        return ""
    first = max(1, location.lineno - context_lines)
    last = min(len(lines), location.lineno + context_lines)
    width = len(str(last))
    rendered: list[str] = []
    for lineno in range(first, last + 1):
        source = lines[lineno - 1].rstrip("\n")
        marker = ">" if lineno == location.lineno else " "
        rendered.append(f"{marker}{lineno:>{width}} | {source}")
        if lineno != location.lineno:
            continue
        expanded_prefix = source[: location.col_offset].expandtabs()
        span = max(
            1,
            (location.end_col_offset or location.col_offset + 1) - location.col_offset,
        )
        rendered.append(f" {' ' * width} | {' ' * len(expanded_prefix)}{'^' * span}")
    return "\n".join(rendered)


class FrontendDiagnosticError(RuntimeError):
    """A frontend failure with source context and an optional causal trace."""

    def __init__(
        self,
        summary: str,
        *,
        location: SourceLocation | None = None,
        reason: str = "",
        cause_trace: str = "",
    ) -> None:
        super().__init__(summary)
        self.location = location
        self.reason = reason
        self.cause_trace = cause_trace

    def format(self, *, verbose: bool = False) -> str:
        rendered = self.args[0]
        if self.location is not None:
            rendered += (
                f"\n --> {self.location.filename}:{self.location.lineno}:"
                f"{self.location.col_offset + 1}"
            )
            frame = render_code_frame(self.location)
            if frame:
                rendered += f"\n{frame}"
        if self.reason:
            rendered += f"\nreason: {self.reason}"
        if verbose and self.cause_trace:
            rendered += f"\n\nCaptured cause traceback:\n{self.cause_trace.rstrip()}"
        return rendered

    def __str__(self) -> str:
        return self.format(verbose=verbose_errors_enabled())


def capture_cause_trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


_CATLASS_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_delegate_excepthook = sys.excepthook


def _is_catlass_diagnostic(exc: BaseException) -> bool:
    return exc.__class__.__module__.startswith("catlass.") and callable(
        getattr(exc, "format", None)
    )


def _is_internal_frame(filename: str) -> bool:
    try:
        return os.path.commonpath(
            (_CATLASS_PACKAGE_DIR, os.path.abspath(filename))
        ) == (_CATLASS_PACKAGE_DIR)
    except ValueError:
        return False


def _without_internal_frames(trace: TracebackType | None) -> TracebackType | None:
    """Copy a traceback while excluding Catlass implementation frames."""

    kept: list[TracebackType] = []
    while trace is not None:
        if not _is_internal_frame(trace.tb_frame.f_code.co_filename):
            kept.append(trace)
        trace = trace.tb_next
    filtered: TracebackType | None = None
    for frame in reversed(kept):
        filtered = TracebackType(
            filtered, frame.tb_frame, frame.tb_lasti, frame.tb_lineno
        )
    return filtered


def _catlass_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """Hide implementation frames for uncaught, formatted Catlass errors."""

    if _is_catlass_diagnostic(exc_value) and not verbose_errors_enabled():
        _delegate_excepthook(
            exc_type, exc_value, _without_internal_frames(exc_traceback)
        )
        return
    _delegate_excepthook(exc_type, exc_value, exc_traceback)


def install_excepthook() -> None:
    """Install filtering while retaining the host hook active at installation."""

    global _delegate_excepthook
    if sys.excepthook is _catlass_excepthook:
        return
    _delegate_excepthook = sys.excepthook
    sys.excepthook = _catlass_excepthook


install_excepthook()


__all__ = [
    "FrontendDiagnosticError",
    "SourceLocation",
    "capture_cause_trace",
    "install_excepthook",
    "render_code_frame",
    "syntax_error_location",
    "traceback_location_for_code",
    "traceback_location_for_user_code",
    "verbose_errors_enabled",
]
