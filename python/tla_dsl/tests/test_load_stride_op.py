from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.params import BlockLoadParams


def _ub_tensor(
    dtype: type[tla.Numeric],
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _make_stride_kernel(params: BlockLoadParams, view: int) -> Any:
    @tla.kernel
    def _load_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(view), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(view), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(src_tile.load(params))

    return _load_stride_kernel


# (id, block_stride, post_update_stride, expect_repeat): BlockLoadParams always
# emits block_stride; repeat_stride is only emitted for a non-zero
# post_update_stride (the vsldb compile-time address pre-offset).
_POSITIVE_CASES = [
    ("default", 0, 0, False),
    ("block-only", 4, 0, False),
    ("block-repeat", 4, 2, True),
    ("boundary-65535", 0xFFFF, 0xFFFF, True),
]

# vsldb stubs cover 2/4-byte lanes; VL is 256 bytes worth of elements.
_DTYPES = [
    (tla.Int16, "i16", 128),
    (tla.Int32, "i32", 64),
    (tla.Float16, "f16", 128),
    (tla.Float32, "f32", 64),
]


@pytest.mark.parametrize(
    ("case_id", "block_stride", "post_update_stride", "expect_repeat"),
    _POSITIVE_CASES,
)
@pytest.mark.parametrize(("dtype", "ir_dtype", "vl_ele"), _DTYPES)
def test_load_stride_emits_tlair(
    compiler_tlair: Any,
    case_id: str,
    block_stride: int,
    post_update_stride: int,
    expect_repeat: bool,
    dtype: type[tla.Numeric],
    ir_dtype: str,
    vl_ele: int,
) -> None:
    mlir = compiler_tlair(
        _make_stride_kernel(
            BlockLoadParams(
                block_stride=block_stride, post_update_stride=post_update_stride
            ),
            vl_ele,
        ),
        type_args=(
            _ub_tensor(dtype=dtype, extent=vl_ele),
            _ub_tensor(dtype=dtype, extent=vl_ele),
        ),
    )

    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    assert f"block_stride = {block_stride} : i32" in load_lines[0]
    if expect_repeat:
        assert f"repeat_stride = {post_update_stride} : i32" in load_lines[0]
    else:
        assert "repeat_stride" not in load_lines[0]
    assert f"!tla.vector<{vl_ele}x{ir_dtype}>" in load_lines[0]


# vsldb packs each stride into a 16-bit immediate field and has no b8 shim:
# out-of-range strides and 1-byte element types are rejected at the frontend
# before any IR is emitted.
@pytest.mark.parametrize(
    ("block_stride", "post_update_stride"),
    [
        (-1, 0),
        (0x10000, 0),
        (4, -1),
        (4, 0x10000),
    ],
)
def test_load_stride_rejects_out_of_range_strides(
    compiler_tlair: Any,
    block_stride: int,
    post_update_stride: int,
) -> None:
    with pytest.raises(TlaLoweringError, match=r"must be in \[0, 65535\]"):
        compiler_tlair(
            _make_stride_kernel(
                BlockLoadParams(
                    block_stride=block_stride, post_update_stride=post_update_stride
                ),
                128,
            ),
            type_args=(
                _ub_tensor(dtype=tla.Int16, extent=128),
                _ub_tensor(dtype=tla.Int16, extent=128),
            ),
        )


@pytest.mark.parametrize("dtype", [tla.Int8, tla.UInt8])
def test_load_stride_rejects_1byte_elements(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
) -> None:
    with pytest.raises(TlaLoweringError, match="2/4-byte element type"):
        compiler_tlair(
            _make_stride_kernel(BlockLoadParams(block_stride=4), 256),
            type_args=(
                _ub_tensor(dtype=dtype, extent=256),
                _ub_tensor(dtype=dtype, extent=256),
            ),
        )
