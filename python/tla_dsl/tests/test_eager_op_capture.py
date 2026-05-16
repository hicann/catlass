from __future__ import annotations

import pytest

import catlass as tla
from catlass.core_api import _category
import catlass.runtime as runtime_mod


def test_eager_capture_emits_ops_into_module() -> None:
    with runtime_mod._eager_capture() as state:
        _ = tla.make_shape(100, 200)
        _ = tla.make_coord(0, 0)
        _ = tla.make_stride(1, 100)
        sh114 = tla.make_shape(1, 1, 4)
        st114 = tla.make_stride(1, 1, 1)
        _ = tla.make_layout(sh114, st114)

    mlir = state.module.operation.get_asm(
        print_generic_op_form=True,
        assume_verified=False,
    )
    assert "tla.make_shape" in mlir
    assert "tla.make_coord" in mlir
    assert "tla.make_stride" in mlir
    assert "tla.make_layout" in mlir


def test_helper_only_ops_still_work_without_capture() -> None:
    with pytest.raises(tla.TlaIRNotExecutableError, match="tla.make_coord"):
        _ = tla.make_coord(0, 0)
    with pytest.raises(tla.TlaIRNotExecutableError, match="tla.make_stride"):
        _ = tla.make_stride(1, 1)
    with pytest.raises(tla.TlaIRNotExecutableError, match="tla.make_shape"):
        _ = tla.make_shape(1, 2)
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_layout"):
        _ = tla.make_layout((1, 1, 4), (1, 1, 1))


def test_non_capture_tile_behavior_is_unchanged() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(1, 2), tla.Float16, origin_shape=tla.make_shape(1, 2)
        )
    with pytest.raises(tla.TlaIRNotExecutableError):
        _ = tla.tile_view(
            mem,
            tla.make_shape(1, 2),
            tla.make_coord(0, 0),
        )


def test_region_helpers_keep_region_stub_behavior() -> None:
    with runtime_mod._eager_capture() as state:
        stub = tla.cube()
    assert stub.__class__.__name__ == "_RegionStub"
    assert _category(stub) == "region"
    mlir = state.module.operation.get_asm(
        print_generic_op_form=True,
        assume_verified=False,
    )
    assert "tla.cube" not in mlir
