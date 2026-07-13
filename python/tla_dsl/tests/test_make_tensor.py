from __future__ import annotations

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError


# --- Frontend IR emission -----------------------------------------------------


@tla.kernel
def make_tensor_makeptr_kernel() -> None:
    ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l1)
    local = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1)),
        coord=tla.make_coord(0, 0),
    )
    _ = local


@tla.kernel
def make_tensor_default_coord_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256)
    # coord omitted: must default to a zero coord matching the rank-2 layout.
    local = tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
    )
    with tla.vector():
        tla.copy(tile, local)


def test_make_tensor_emits_op_with_explicit_coord() -> None:
    mlir = make_tensor_makeptr_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    assert "tla.make_tensor_like" not in mlir
    # Result tensor type carries the explicit layout/coord/ptr.
    assert "!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>" in mlir
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.ptr<f32, l1" in mlir


def test_make_tensor_coord_defaults_to_zero_matching_rank() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float32,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = make_tensor_default_coord_kernel.dump_mlir(type_args=(mem,))
    assert "tla.make_tensor" in mlir
    # rank-2 layout -> default coord is (0, 0).
    assert "!tla.coord<0,0>" in mlir


def test_make_tensor_rank1_default_coord_is_single_zero() -> None:
    @tla.kernel
    def _rank1_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        local = tla.make_tensor(
            ptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        )
        _ = local

    mlir = _rank1_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    # rank-1 layout keeps a rank-1 coord in the frontend IR (rank-2 promotion happens
    # in the C++ lowering); the default coord leaf is 0.
    assert "!tla.coord<0>" in mlir
    assert "!tla.coord<0,0>" not in mlir


# --- Preconditions ------------------------------------------------------------


def test_make_tensor_rejects_non_layout() -> None:
    with runtime_mod._eager_capture():
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        with pytest.raises(tla.TlaCoreAPIError, match="tla.make_tensor"):
            tla.make_tensor(
                ptr,
                tla.make_shape(16, 16),  # not a tla.make_layout result
                coord=tla.make_coord(0, 0),
            )


def test_make_tensor_rejects_non_pointer() -> None:
    with runtime_mod._eager_capture():
        layout = tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
        with pytest.raises(tla.TlaCoreAPIError, match="tla.make_tensor"):
            tla.make_tensor(
                tla.make_shape(16, 16),  # not a !tla.ptr
                layout,
                coord=tla.make_coord(0, 0),
            )


def test_make_tensor_rejects_bad_coord_type() -> None:
    with runtime_mod._eager_capture():
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        layout = tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
        with pytest.raises(tla.TlaCoreAPIError, match="tla.make_tensor"):
            # A _Shape is the wrong type for coord (expected tla.make_coord / None).
            tla.make_tensor(ptr, layout, coord=tla.make_shape(16, 16))


def test_make_tensor_rejects_higher_rank_layout() -> None:
    with runtime_mod._eager_capture():
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        # 3-D layout: exceeds the max 2-D supported by make_tensor.
        layout = tla.make_layout(
            tla.make_shape(2, 3, 4), tla.make_stride(12, 4, 1)
        )
        with pytest.raises(TlaLoweringError, match="at most 2-D"):
            tla.make_tensor(ptr, layout, coord=tla.make_coord(0, 0, 0))


def test_make_tensor_rejects_coord_rank_mismatch() -> None:
    with runtime_mod._eager_capture():
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        # rank-2 layout but rank-1 coord.
        layout = tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
        with pytest.raises(TlaLoweringError, match="coord rank must match"):
            tla.make_tensor(ptr, layout, coord=tla.make_coord(0))

