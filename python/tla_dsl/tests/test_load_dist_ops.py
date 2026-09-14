from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.params import LoadDist, NormalLoadParams


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
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


def _make_load_kernel(load_dist: LoadDist, src_view: int, dst_view: int) -> Any:
    @tla.kernel
    def _load_dist_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(src_view), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(dst_view), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(src_tile.load(NormalLoadParams(load_dist=load_dist)))

    return _load_dist_kernel


# (mode, load_dist, dtype, src_view, dst_view, ir_vector): the src tile view
# matches what each dist actually reads (brc: 1 element; us/unpack: VL/2;
# e2b: one element per DataBlock) and the dst view is a full VL register.
_SUFFIXED_MODES = [
    ("brc_b16", LoadDist.DIST_BRC_B16, tla.Int16, 1, 128, "128xi16"),
    ("us_b16", LoadDist.DIST_US_B16, tla.Int16, 64, 128, "128xi16"),
    ("unpack_b16", LoadDist.DIST_UNPACK_B16, tla.Int16, 64, 128, "128xi16"),
    ("e2b_b16", LoadDist.DIST_E2B_B16, tla.Int16, 8, 128, "128xi16"),
    ("e2b_b32", LoadDist.DIST_E2B_B32, tla.Int32, 8, 64, "64xi32"),
]


@pytest.mark.parametrize(
    ("mode", "load_dist", "dtype", "src_view", "dst_view", "ir_vector"),
    _SUFFIXED_MODES,
)
def test_load_dist_emits_tlair(
    compiler_tlair: Any,
    mode: str,
    load_dist: LoadDist,
    dtype: type[tla.Numeric],
    src_view: int,
    dst_view: int,
    ir_vector: str,
) -> None:
    mlir = compiler_tlair(
        _make_load_kernel(load_dist, src_view, dst_view),
        type_args=(
            _ub_tensor(dtype=dtype, extent=src_view),
            _ub_tensor(dtype=dtype, extent=dst_view),
        ),
    )

    assert f"#tla.load_dist<{mode}>" in mlir
    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    # Every distribution mode fills a full VL register of its dtype.
    assert f"!tla.vector<{ir_vector}>" in load_lines[0]


def _reject_cases() -> list[Any]:
    b16_modes = [
        ("brc_b16", LoadDist.DIST_BRC_B16, 1, 64, "b16|B16|2-byte"),
        ("us_b16", LoadDist.DIST_US_B16, 64, 64, "b16|B16|2-byte"),
        ("unpack_b16", LoadDist.DIST_UNPACK_B16, 64, 64, "b16|B16|2-byte"),
        ("e2b_b16", LoadDist.DIST_E2B_B16, 8, 8, "E2B_B16|2-byte"),
    ]
    cases = []
    for mode, load_dist, src_view, dst_view, match in b16_modes:
        for dtype in (tla.Float32, tla.Int8, tla.Int32):
            cases.append(
                pytest.param(
                    mode, load_dist, src_view, dst_view, match, dtype,
                    id=f"{mode}-{dtype.__name__}",
                )
            )
    for dtype in (tla.Int16, tla.Int8, tla.Float16):
        cases.append(
            pytest.param(
                "e2b_b32", LoadDist.DIST_E2B_B32, 8, 8, "E2B_B32|4-byte", dtype,
                id=f"e2b_b32-{dtype.__name__}",
            )
        )
    return cases


@pytest.mark.parametrize(
    ("mode", "load_dist", "src_view", "dst_view", "match", "dtype"),
    _reject_cases(),
)
def test_load_dist_rejects_wrong_dtype(
    compiler_tlair: Any,
    mode: str,
    load_dist: LoadDist,
    src_view: int,
    dst_view: int,
    match: str,
    dtype: type[tla.Numeric],
) -> None:
    # Suffixed modes are locked to their element width; other element types
    # must be rejected at the frontend before any IR is emitted.
    with pytest.raises(TlaLoweringError, match=match):
        compiler_tlair(
            _make_load_kernel(load_dist, src_view, dst_view),
            type_args=(
                _ub_tensor(dtype=dtype, extent=src_view),
                _ub_tensor(dtype=dtype, extent=dst_view),
            ),
        )


@pytest.mark.parametrize(
    ("dtype", "ir_dtype", "db_ele", "vl_ele"),
    [
        (tla.Int8, "i8", 32, 256),
        (tla.Int16, "i16", 16, 128),
        (tla.Int32, "i32", 8, 64),
    ],
)
def test_load_blk_emits_tlair(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    ir_dtype: str,
    db_ele: int,
    vl_ele: int,
) -> None:
    # Unlike the suffixed modes, ``blk`` has no element-size suffix and works
    # for any dtype: the transfer always reads one 32-byte DataBlock worth of
    # elements (b8: 32, b16: 16, b32: 8) and replicates it across all VL/DB
    # (= 8) DataBlocks of a VL register.
    mlir = compiler_tlair(
        _make_load_kernel(LoadDist.DIST_BLK, db_ele, vl_ele),
        type_args=(
            _ub_tensor(dtype=dtype, extent=db_ele),
            _ub_tensor(dtype=dtype, extent=vl_ele),
        ),
    )

    assert "#tla.load_dist<blk>" in mlir
    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    assert f"!tla.vector<{vl_ele}x{ir_dtype}>" in load_lines[0]
