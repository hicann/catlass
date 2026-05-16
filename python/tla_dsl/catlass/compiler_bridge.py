"""Typed helpers for running the Tla pass pipeline from Python MLIR modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Sequence

from . import _tla_type_bridge

if TYPE_CHECKING:
    from mlir import ir as mlir_ir


class BridgeUnavailableError(RuntimeError):
    """Raised when the in-process compile bridge is unavailable."""


@dataclass(frozen=True)
class TlaLoweringResult:
    """Result of lowering a live ``tla``-dialect MLIR module through the typed bridge."""

    lowered_mlir: str
    pass_ir_dump: str = ""


def lower_tlair_module_to_mlir(
    module: mlir_ir.Module,
    *,
    mlir_print_ir_before: Sequence[str] = (),
    mlir_print_ir_after: Sequence[str] = (),
    mlir_print_ir_before_all: bool = False,
    mlir_print_ir_after_all: bool = False,
) -> TlaLoweringResult:
    """Lower an existing MLIR module through the Tla pipeline.

    Pass-print arguments intentionally mirror the TlaCompile flag names while
    staying structured Python args instead of CLI strings.
    """

    ext = _load_bridge_extension()
    result = ext.lower_to_mlir(
        module,
        list(mlir_print_ir_before),
        list(mlir_print_ir_after),
        bool(mlir_print_ir_before_all),
        bool(mlir_print_ir_after_all),
    )
    return TlaLoweringResult(
        lowered_mlir=str(result["lowered_mlir"]),
        pass_ir_dump=str(result.get("pass_ir_dump", "")),
    )


def _load_bridge_extension() -> ModuleType:
    try:
        return _tla_type_bridge._load_bridge_extension()
    except _tla_type_bridge.TlaTypeBridgeUnavailableError as exc:
        raise BridgeUnavailableError(str(exc)) from exc


def resolve_bridge_extension_path() -> Path | None:
    return _tla_type_bridge._resolve_bridge_extension_path()


__all__ = [
    "BridgeUnavailableError",
    "TlaLoweringResult",
    "lower_tlair_module_to_mlir",
    "resolve_bridge_extension_path",
]
