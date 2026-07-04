"""Tests for PR #114: location handling refactor — direct mlir_ir.Location creation
in dsl_user_op, removal of _CapturedLocation intermediate class."""

from __future__ import annotations

import pytest
from mlir import ir as mlir_ir

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.core_api import _Shape
from catlass.runtime import TlaIRNotExecutableError


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


def test_capture_caller_location_returns_mlir_location() -> None:
    """_capture_caller_location returns mlir_ir.Location, not None or a plain dataclass."""
    with runtime_mod._eager_capture():
        loc = runtime_mod._capture_caller_location()
        assert isinstance(loc, mlir_ir.Location)
        assert loc is not None


# ---------------------------------------------------------------------------
# dsl_user_op loc type validation
# ---------------------------------------------------------------------------


def test_dsl_op_rejects_non_mlir_location_loc() -> None:
    """Passing a non-mlir_ir.Location, non-None loc raises TypeError."""
    with runtime_mod._eager_capture():
        with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
            tla.make_shape(1, 2, loc="bad_location")


def test_dsl_op_rejects_int_as_loc() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(TypeError, match="loc must be mlir.ir.Location or None"):
            tla.make_shape(1, 2, loc=42)


def test_dsl_op_accepts_explicit_mlir_location() -> None:
    """Passing a valid mlir_ir.Location as loc is accepted and op succeeds."""
    with runtime_mod._eager_capture() as state:
        explicit_loc = mlir_ir.Location.file("test_file.py", 10, 5)
        sh = tla.make_shape(8, 16, loc=explicit_loc)
        assert sh is not None
    mlir = state.module.operation.get_asm(
        print_generic_op_form=True, assume_verified=False
    )
    assert "tla.make_shape" in mlir


def test_dsl_op_accepts_unknown_location() -> None:
    with runtime_mod._eager_capture() as state:
        sh = tla.make_shape(4, loc=mlir_ir.Location.unknown())
        assert sh is not None
    assert "tla.make_shape" in state.module.operation.get_asm(
        print_generic_op_form=True, assume_verified=False
    )


def test_dsl_op_accepts_named_location() -> None:
    """Passing a fused NameLoc (file + function name) is accepted."""
    with runtime_mod._eager_capture() as state:
        file_loc = mlir_ir.Location.file("my_kernel.py", 42, 7)
        named_loc = mlir_ir.Location.name("my_func", childLoc=file_loc)
        sh = tla.make_shape(4, 8, loc=named_loc)
        st = tla.make_stride(8, 1)
        tla.make_layout(sh, st)
    assert "tla.make_layout" in state.module.operation.get_asm(
        print_generic_op_form=True, assume_verified=False
    )


# ---------------------------------------------------------------------------
# Auto-capture location
# ---------------------------------------------------------------------------


def test_auto_capture_with_frontend_state() -> None:
    """When loc is omitted and frontend state exists, location is auto-captured."""
    with runtime_mod._eager_capture():
        sh = tla.make_shape(1, 2)
        assert isinstance(sh, _Shape)
        loc = sh._shape_value.owner.location
        assert isinstance(loc, mlir_ir.Location)


def test_auto_captured_location_contains_calling_file() -> None:
    """Auto-captured location should carry the calling source file and line info."""
    with runtime_mod._eager_capture():
        sh = tla.make_shape(1, 2)
        loc_str = str(sh._shape_value.owner.location)
    assert ".py" in loc_str


def test_loc_none_without_frontend_state_raises() -> None:
    """Explicit loc=None without frontend state raises TlaIRNotExecutableError (non-executable Tla IR path), not TypeError."""
    with pytest.raises(TlaIRNotExecutableError):
        tla.make_shape(1, 2, loc=None)


# ---------------------------------------------------------------------------
# Backward compatibility: representative ops work without explicit loc
# ---------------------------------------------------------------------------


def test_representative_ops_work_without_explicit_loc() -> None:
    """Representative DSL ops across categories work with auto-captured location."""
    # Region-requiring verbs (set_flag/wait_flag/pipe_barrier/cross_core_*/
    # mutex_lock/unlock) can't be emitted here: they must be nested in a
    # tla.cube/tla.vector region, and region ops are not enterable under
    # _eager_capture. The auto-location path is identical for every dsl_user_op,
    # so representative region-free ops across categories still exercise it.
    with runtime_mod._eager_capture() as state:
        sh = tla.make_shape(1, 2)
        tla.make_coord(0, 0)
        st = tla.make_stride(1, 100)
        tla.make_layout(sh, st)
        tla.flag("ready")
        tla.cross_flag("x", tla.pipes.MTE3, tla.pipes.SCALAR)
        tla.mutex(resource="l0a_ping", id=-1)

    mlir = state.module.operation.get_asm(
        print_generic_op_form=True, assume_verified=False
    )
    for op_name in (
        "tla.make_shape",
        "tla.make_layout",
        "tla.flag",
        "tla.cross_flag",
        "tla.mutex",
    ):
        assert op_name in mlir
