from __future__ import annotations

from typing import Any

import pytest

import catlass as tla
from catlass.execution_lowering import UnsupportedExecutionLowering
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
def vector_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MAX, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MIN, mask=reduce_mask)


@tla.kernel
def vector_masked_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H)
            src_reg = src_tile.load()
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=mask)


@tla.kernel
def vector_unsigned_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask)


@tla.kernel
def reduce_non_reduction_op_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce("add", mask=reduce_mask)


@tla.kernel
def reduce_init_value_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, init_value=0.0)


@tla.kernel
def reduce_profile_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, reduction_profile=0)


@tla.kernel
def reduce_none_keywords_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(
                tla.ReductionOp.ADD,
                mask=reduce_mask,
                init_value=None,
                reduction_profile=None,
            )


@tla.kernel
def vector_reduce_no_mask_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load().reduce(tla.ReductionOp.ADD)


def test_vector_reduce_public_export_exists() -> None:
    assert [op.value for op in tla.ReductionOp] == ["add", "max", "min"]


def test_vector_reduce_emits_supported_kinds() -> None:
    mlir = vector_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert mlir.count("tla.reduce") == 3
    for kind in ("add", "max", "min"):
        assert f'kind = "{kind}"' in mlir


def test_vector_reduce_emits_mask() -> None:
    mlir = vector_masked_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    reduce_line = next(line for line in mlir.splitlines() if "tla.reduce" in line)
    assert "tla.reduce" in reduce_line
    assert "!tla.mask" in reduce_line
    assert 'kind = "add"' in reduce_line


def test_unsigned_reduction_has_no_init_profile_attrs() -> None:
    mlir = vector_unsigned_reduce_kernel.dump_mlir(type_args=(_vector_tensor(tla.UInt16),))

    assert "!tla.ptr<ui16" in mlir
    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_reduce_accepts_none_semantic_keywords() -> None:
    mlir = reduce_none_keywords_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_unsupported_reduction_element_type_is_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce.*unsupported reduction element type i64",
    ):
        vector_reduce_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int64),))


def test_reduce_rejects_non_reduction_op() -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, UnsupportedExecutionLowering), match="expected ReductionOp"
    ):
        reduce_non_reduction_op_kernel.dump_mlir(type_args=(_vector_tensor(),))


@pytest.mark.parametrize(
    "kernel, match",
    (
        (reduce_init_value_keyword_kernel, "only supports init_value=None"),
        (
            reduce_profile_keyword_kernel,
            "only supports reduction_profile=None",
        ),
    ),
)
def test_reduce_hides_unsupported_semantic_keywords(
    kernel: Any,
    match: str,
) -> None:
    with pytest.raises(UnsupportedExecutionLowering, match=match):
        kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_reduce_requires_explicit_mask() -> None:
    with pytest.raises(
        UnsupportedExecutionLowering,
        match=r"missing.*keyword-only.*argument.*'mask'",
    ):
        vector_reduce_no_mask_kernel.dump_mlir(type_args=(_vector_tensor(),))
