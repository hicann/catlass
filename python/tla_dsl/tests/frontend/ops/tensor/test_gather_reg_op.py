"""Tests for tla.gather Reg→Reg path (vselr)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import catlass.tla as tla


def _f16_pair() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Int16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
    )


@tla.kernel
def gather_reg_vector_kernel(
    src: tla.Tensor, idx: tla.Tensor, dst: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(128), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            s = src_tile.load()
            i = idx_tile.load()
            dst_tile.store(tla.gather(s, i))


def test_gather_reg_op_emits_mlir() -> None:
    src, idx, dst = _f16_pair()
    mlir = gather_reg_vector_kernel.dump_mlir(type_args=(src, idx, dst))
    assert "tla.gather" in mlir
    assert "tla.gather_reg" not in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.gather")
    lines = [line for line in mlir.splitlines() if "tla.gather" in line]
    assert len(lines) == 1
    # Reg→Reg form: both operands are !tla.vector
    assert "!tla.vector" in lines[0]
