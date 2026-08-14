"""Tests for PR #114: location handling refactor — direct mlir_ir.Location creation
in dsl_user_op, removal of _CapturedLocation intermediate class."""

from __future__ import annotations

import pytest
from mlir import ir as mlir_ir

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.core_api import _Shape

# ---------------------------------------------------------------------------
# _CapturedLocation / _to_mlir_location removal
# ---------------------------------------------------------------------------


def test_captured_location_class_is_removed() -> None:
    """_CapturedLocation dataclass no longer exists in runtime module."""
    assert not hasattr(runtime_mod, "_CapturedLocation")


def test_to_mlir_location_function_is_removed() -> None:
    """_to_mlir_location conversion function no longer exists in runtime module."""
    assert not hasattr(runtime_mod, "_to_mlir_location")


# ---------------------------------------------------------------------------
# _capture_caller_location now returns mlir_ir.Location directly
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# dsl_user_op loc type validation
# ---------------------------------------------------------------------------


def test_dsl_op_rejects_non_mlir_location_loc() -> None:
    """Passing a non-mlir_ir.Location, non-None loc raises TypeError."""
    with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
        tla.make_shape(1, 2, loc="bad_location")


def test_dsl_op_rejects_int_as_loc() -> None:
    with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
        tla.make_shape(1, 2, loc=42)





# ---------------------------------------------------------------------------
# Auto-capture location
# ---------------------------------------------------------------------------




def test_loc_none_without_frontend_state_rejects_make_shape() -> None:
    """Host public ``make_shape`` requires frontend."""
    with pytest.raises(tla.TlaIRNotExecutableError, match="tla.make_shape"):
        tla.make_shape(1, 2, loc=None)


# ---------------------------------------------------------------------------
# Backward compatibility: representative ops work without explicit loc
# ---------------------------------------------------------------------------


