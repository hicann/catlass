from __future__ import annotations

from pathlib import Path

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass._mlir_bindings import tla_ops_gen


@tla.kernel
def _mask_logic_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            zero = tla.sub(reg, reg)
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            h_mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            q_mask = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            m4_mask = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)

            not_mask = tla.not_(q_mask, all_mask)
            and_mask = tla.and_(h_mask, m4_mask, all_mask)
            or_mask = tla.or_(q_mask, m4_mask, all_mask)
            xor_mask = tla.xor(h_mask, m4_mask, all_mask)

            tmp0 = tla.where(not_mask, reg, zero)
            tmp1 = tla.where(and_mask, tmp0, zero)
            tmp2 = tla.where(or_mask, tmp1, zero)
            dst_tile.store(tla.where(xor_mask, tmp2, zero), mask=all_mask)


def _tensor_args() -> tuple[tla.Tensor, tla.Tensor]:
    with runtime_mod._eager_capture():
        src = tla.Tensor(
            tla.make_shape(64), tla.Float32, origin_shape=tla.make_shape(64)
        )
        dst = tla.Tensor(
            tla.make_shape(64), tla.Float32, origin_shape=tla.make_shape(64)
        )
    return src, dst


@pytest.mark.parametrize(
    ("binding_name", "op_name"),
    (
        ("mask_not", "tla.mask_not"),
        ("mask_and", "tla.mask_and"),
        ("mask_or", "tla.mask_or"),
        ("mask_xor", "tla.mask_xor"),
    ),
)
def test_mask_logic_bindings_and_public_ops_emit_mlir(
    binding_name: str, op_name: str
) -> None:
    assert hasattr(tla_ops_gen, binding_name)
    assert op_name in _mask_logic_kernel.dump_mlir(type_args=_tensor_args())


def test_mask_logic_ops_are_in_mask_logic_info() -> None:
    pass_source = Path(
        __file__
    ).parents[1] / "csrc/mlir/lib/Passes/ConvertTlaToVectorPass.cpp"
    source = pass_source.read_text(encoding="utf-8")
    start = source.index("static std::optional<MaskLogicUnaryInfo> getMaskLogicUnaryInfo")
    end = source.index(
        "static std::optional<AnyVectorOperationInfo> getAnyVectorOperationInfo", start
    )
    body = source[start:end]

    for op_name in ("MaskNotOp", "MaskAndOp", "MaskOrOp", "MaskXorOp"):
        assert f"::tla::{op_name}" in body


def test_mask_logic_rejects_non_mask_predicate() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.not_.*mask"):
        _invalid_mask_logic_kernel.dump_mlir(type_args=_tensor_args())


@tla.kernel
def _invalid_mask_logic_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            dst_tile.store(reg, mask=tla.not_(all_mask, reg))
