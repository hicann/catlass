from __future__ import annotations

import catlass as tla
import catlass.runtime as runtime_mod


def _vector_tensor(dtype: type[tla.Numeric] = tla.Float32) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(64),
            dtype,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def vector_interleave_kernel(
    src0: tla.Tensor,
    src1: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            out0, out1 = tla.interleave(src0_tile.load(), src1_tile.load())
            dst0_tile.store(out0)
            dst1_tile.store(out1)


@tla.kernel
def vector_deinterleave_kernel(
    src0: tla.Tensor,
    src1: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            out0, out1 = tla.deinterleave(src0_tile.load(), src1_tile.load())
            dst0_tile.store(out0)
            dst1_tile.store(out1)


def test_vector_interleave_public_export_exists() -> None:
    assert callable(tla.interleave)
    assert callable(tla.deinterleave)


def test_vector_interleave_emits_tla_op() -> None:
    tensor = _vector_tensor()
    mlir = vector_interleave_kernel.dump_mlir(type_args=(tensor, tensor, tensor, tensor))

    assert "tla.interleave" in mlir
    assert "tla.vector" in mlir
    assert "tla.vec.func" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.interleave")

    interleave_lines = [line for line in mlir.splitlines() if "tla.interleave" in line]
    assert len(interleave_lines) == 1
    assert "->" in interleave_lines[0]


def test_vector_deinterleave_emits_tla_op() -> None:
    tensor = _vector_tensor()
    mlir = vector_deinterleave_kernel.dump_mlir(
        type_args=(tensor, tensor, tensor, tensor)
    )

    assert "tla.deinterleave" in mlir
    assert "tla.vector" in mlir
    assert "tla.vec.func" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.deinterleave")

    deinterleave_lines = [
        line
        for line in mlir.splitlines()
        if "tla.deinterleave" in line
    ]
    assert len(deinterleave_lines) == 1
    assert "->" in deinterleave_lines[0]
