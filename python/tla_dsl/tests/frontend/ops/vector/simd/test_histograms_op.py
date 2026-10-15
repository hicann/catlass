"""Tests for tla.histograms Reg FREQUENCY/ACCUMULATE histogram op."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import catlass.tla as tla


def _u8_u16_tensor_pair() -> tuple[tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.UInt8,
            (256,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(256,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.UInt16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
    )


@tla.kernel
def histograms_vector_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            keys = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt8)
            hist_lo = tla.full(0, dtype=tla.Int16)
            hist_hi = tla.full(0, dtype=tla.Int16)
            hist_lo = tla.histograms(
                keys,
                hist_lo,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN0,
                    type=tla.params.HistogramsType.ACCUMULATE,
                ),
                mask=mask,
            )
            hist_hi = tla.histograms(
                keys,
                hist_hi,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN1,
                    type=tla.params.HistogramsType.ACCUMULATE,
                ),
                mask=mask,
            )
            # Store BIN0 half; BIN1 is also exercised in IR emission.
            dst_tile.store(hist_lo)
            _ = hist_hi


@tla.kernel
def histograms_with_dst_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            keys = src_tile.load()
            hist = dst_tile.load()
            hist = tla.histograms(
                keys,
                hist,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN0,
                    type=tla.params.HistogramsType.FREQUENCY,
                ),
            )
            dst_tile.store(hist)


def test_histograms_op_emits_mlir() -> None:
    src, dst = _u8_u16_tensor_pair()
    mlir = histograms_vector_kernel.dump_mlir(type_args=(src, dst))
    assert "tla.histograms" in mlir
    assert "mode <bin0>" in mlir
    assert "mode <bin1>" in mlir
    assert "type <accumulate>" in mlir
    assert "tla.full" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.histograms")
    hist_lines = [line for line in mlir.splitlines() if "tla.histograms" in line]
    assert len(hist_lines) == 2


def test_histograms_with_dst_emits_mlir() -> None:
    src, dst = _u8_u16_tensor_pair()
    mlir = histograms_with_dst_kernel.dump_mlir(type_args=(src, dst))
    assert "tla.histograms" in mlir
    assert "mode <bin0>" in mlir
    assert "type <frequency>" in mlir
