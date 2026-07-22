from __future__ import annotations

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _vector_tensor() -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(64),
            tla.Float32,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def readonly_vector_if_kernel(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            if limit > 0:
                dst_tile.store(value)


@tla.kernel
def vector_if_result_kernel(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            if limit > 0:
                value = tla.abs(value)
            dst_tile.store(value)


@tla.kernel
def vector_if_expr_kernel(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = src_tile.load()
            value = tla.abs(source) if limit > 0 else source
            dst_tile.store(value)


@tla.kernel
def vector_for_carried_kernel(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            for _ in tla.range(limit):
                value = tla.abs(value)
            dst_tile.store(value)


@tla.kernel
def vector_while_kernel(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            index = 0
            while index < limit:
                dst_tile.store(value)
                index += 1


def _type_args() -> tuple[object, object, int]:
    tensor = _vector_tensor()
    return tensor, tensor, 2


def test_resultless_if_can_capture_vector_ssa_read_only() -> None:
    mlir = readonly_vector_if_kernel.dump_mlir(type_args=_type_args())

    assert "scf.if" in mlir
    assert "!tla.vector<64xf32>" in mlir
    assert "tla.store" in mlir


@pytest.mark.parametrize(
    ("kernel", "message"),
    (
        (vector_if_result_kernel, "Dynamic if carried values"),
        (vector_if_expr_kernel, "Conditional expression results"),
        (vector_for_carried_kernel, "Dynamic for carried values"),
        (
            vector_while_kernel,
            "while loops are not currently supported inside tla.vec.func",
        ),
    ),
)
def test_unsupported_vector_ssa_control_flow_is_rejected(
    kernel: object, message: str
) -> None:
    with pytest.raises(Exception, match=message):
        kernel.dump_mlir(type_args=_type_args())  # type: ignore[attr-defined]
