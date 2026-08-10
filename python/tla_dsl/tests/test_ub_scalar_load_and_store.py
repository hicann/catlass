from __future__ import annotations

import re

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering


def _tensor(
    *shape: int,
    addrspace: tla.AddressSpace = tla.AddressSpace.ub,
    dtype: type = tla.Float32,
) -> tla.Tensor:
    if not shape:
        shape = (1,)
    with runtime_mod._eager_capture():
        tla_shape = tla.make_shape(*shape)
        coord = None
        stride = None
        if len(shape) > 2:
            contiguous_strides = []
            running_stride = 1
            for dim in reversed(shape):
                contiguous_strides.append(running_stride)
                running_stride *= dim
            coord = tla.make_coord(*(0 for _ in shape))
            stride = tla.make_stride(*reversed(contiguous_strides))
        return tla.Tensor(
            tla_shape,
            dtype,
            addrspace=addrspace,
            origin_shape=tla_shape,
            coord=coord,
            stride=stride,
            layout_tag=tla.arch.RowMajor,
        )


def _assert_dynamic_index_from_arg(mlir: str, op_line: str) -> None:
    match = re.search(r"\[(%[-\w.$]+)\]", op_line)
    assert match is not None
    index = match.group(1)
    if index != "%arg1":
        assert re.search(
            rf"^\s*{re.escape(index)} = arith\.index_cast %arg1\b",
            mlir,
            re.MULTILINE,
        )
    assert "arith.constant 7 : index" not in mlir


@tla.kernel
def scalar_load_kernel(slot: tla.Tensor) -> None:
    slot_tile = tla.tile_view(slot, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        _ = slot_tile[0]


@tla.kernel
def scalar_load_offset_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        _ = slot[7]


@tla.kernel
def scalar_load_dynamic_offset_kernel(slot: tla.Tensor, offset: tla.Index) -> None:
    with tla.vector():
        _ = slot[offset]


@tla.kernel
def scalar_load_rank2_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        _ = slot[1, 2]


@tla.kernel
def gm_scalar_load_offset_kernel(slot: tla.Tensor) -> None:
    _ = slot[7]


@tla.kernel
def scalar_load_bad_index_kernel(slot: tla.Tensor) -> None:
    slot_tile = tla.tile_view(slot, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        _ = slot_tile[-1]


@tla.kernel
def scalar_load_past_end_kernel(slot: tla.Tensor) -> None:
    slot_tile = tla.tile_view(slot, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        _ = slot_tile[1]


@tla.kernel
def scalar_load_outside_vector_kernel(slot: tla.Tensor) -> None:
    _ = slot[0]


@tla.kernel
def scalar_load_inside_vec_func_kernel(slot: tla.Tensor) -> None:
    slot_tile = tla.tile_view(slot, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = slot_tile[0]


@tla.kernel
def scalar_store_kernel(slot: tla.Tensor) -> None:
    slot_tile = tla.tile_view(slot, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        slot_tile[0] = 3.5


@tla.kernel
def scalar_store_offset_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        slot[7] = tla.Float32(4.5)


@tla.kernel
def scalar_store_dynamic_offset_kernel(slot: tla.Tensor, offset: tla.Index) -> None:
    with tla.vector():
        slot[offset] = 5.5


@tla.kernel
def scalar_load_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        dst[3] = src[2]


@tla.kernel
def gm_scalar_load_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst[3] = src[2]


@tla.kernel
def scalar_store_bad_index_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        slot[-1] = 1.0


@tla.kernel
def scalar_store_past_end_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        slot[1] = 1.0


@tla.kernel
def scalar_store_outside_vector_kernel(slot: tla.Tensor) -> None:
    slot[0] = 1.0


@tla.kernel
def scalar_store_inside_vec_func_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            slot[0] = 1.0


@tla.kernel
def scalar_store_rank2_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        slot[1, 2] = 1.0


@tla.kernel
def scalar_store_rank3_kernel(slot: tla.Tensor) -> None:
    with tla.vector():
        slot[0, 0, 0] = 1.0


def test_scalar_subscript_emits_tla_scalar_load() -> None:
    mlir = scalar_load_kernel.dump_mlir(type_args=(_tensor(),))

    assert "tla.scalar_load" in mlir
    assert "tla.load_scalar" not in mlir
    assert "!tla.ptr<f32, ub, 4>" in mlir
    assert "-> f32" in next(
        line for line in mlir.splitlines() if "tla.scalar_load" in line
    )


def test_bool_scalar_load_uses_standard_bool_wrapper() -> None:
    mlir = scalar_load_kernel.dump_mlir(type_args=(_tensor(dtype=tla.Bool),))

    assert "tla.scalar_load" in mlir
    assert "-> i1" in next(
        line for line in mlir.splitlines() if "tla.scalar_load" in line
    )


def test_scalar_subscript_accepts_nonzero_offset_in_longer_tensor() -> None:
    mlir = scalar_load_offset_kernel.dump_mlir(type_args=(_tensor(64),))

    load_line = next(line for line in mlir.splitlines() if "tla.scalar_load" in line)
    assert "tla.scalar_load" in load_line
    assert "arith.constant 7 : index" in mlir
    assert "[%c7]" in load_line


def test_scalar_subscript_accepts_dynamic_index_ssa() -> None:
    mlir = scalar_load_dynamic_offset_kernel.dump_mlir(
        type_args=(_tensor(64), 7)
    )

    load_line = next(line for line in mlir.splitlines() if "tla.scalar_load" in line)
    assert "tla.scalar_load" in load_line
    _assert_dynamic_index_from_arg(mlir, load_line)


def test_scalar_subscript_accepts_rank2_ub_tensor() -> None:
    mlir = scalar_load_rank2_kernel.dump_mlir(type_args=(_tensor(2, 4),))

    load_line = next(line for line in mlir.splitlines() if "tla.scalar_load" in line)
    assert "arith.constant 1 : index" in mlir
    assert "arith.constant 2 : index" in mlir
    assert "[%c1, %c2]" in load_line


def test_scalar_load_accepts_ub_tensor_inside_vec_func() -> None:
    mlir = scalar_load_inside_vec_func_kernel.dump_mlir(type_args=(_tensor(),))

    load_line = next(line for line in mlir.splitlines() if "tla.scalar_load" in line)
    assert "tla.vec.func" in mlir
    assert "[%c0]" in load_line


def test_scalar_subscript_dispatches_gm_to_tla_scalar_load() -> None:
    mlir = gm_scalar_load_offset_kernel.dump_mlir(
        type_args=(_tensor(64, addrspace=tla.AddressSpace.gm),)
    )

    assert "tla.scalar_load" in mlir
    assert "tla.load_scalar" not in mlir


def test_scalar_subscript_assignment_emits_tla_scalar_store() -> None:
    mlir = scalar_store_kernel.dump_mlir(type_args=(_tensor(),))

    assert "tla.scalar_store" in mlir
    assert "tla.store_scalar" not in mlir
    assert "!tla.ptr<f32, ub, 4>" in mlir
    assert "3.500000e+00 : f32" in mlir


def test_scalar_store_accepts_nonzero_offset_in_longer_tensor() -> None:
    mlir = scalar_store_offset_kernel.dump_mlir(type_args=(_tensor(64),))

    store_line = next(
        line for line in mlir.splitlines() if "tla.scalar_store" in line
    )
    assert "arith.constant 7 : index" in mlir
    assert "[%c7]" in store_line


def test_scalar_store_accepts_dynamic_index_ssa() -> None:
    mlir = scalar_store_dynamic_offset_kernel.dump_mlir(
        type_args=(_tensor(64), 7)
    )

    store_line = next(
        line for line in mlir.splitlines() if "tla.scalar_store" in line
    )
    _assert_dynamic_index_from_arg(mlir, store_line)


def test_scalar_store_accepts_rank2_ub_tensor() -> None:
    mlir = scalar_store_rank2_kernel.dump_mlir(type_args=(_tensor(2, 4),))

    store_line = next(
        line for line in mlir.splitlines() if "tla.scalar_store" in line
    )
    assert "arith.constant 1 : index" in mlir
    assert "arith.constant 2 : index" in mlir
    assert "[%c1, %c2]" in store_line


def test_scalar_store_accepts_ub_tensor_inside_vec_func() -> None:
    mlir = scalar_store_inside_vec_func_kernel.dump_mlir(type_args=(_tensor(),))

    store_line = next(
        line for line in mlir.splitlines() if "tla.scalar_store" in line
    )
    assert "tla.vec.func" in mlir
    assert "[%c0]" in store_line


def test_scalar_store_accepts_scalar_ssa_from_load() -> None:
    mlir = scalar_load_store_kernel.dump_mlir(type_args=(_tensor(64), _tensor(64)))

    assert "tla.scalar_load" in mlir
    assert "tla.scalar_store" in mlir
    assert "tla.load_scalar" not in mlir
    assert "tla.store_scalar" not in mlir


def test_scalar_load_store_dispatches_gm_to_scalar_access_ops() -> None:
    gm = _tensor(64, addrspace=tla.AddressSpace.gm)
    mlir = gm_scalar_load_store_kernel.dump_mlir(type_args=(gm, gm))

    assert "tla.scalar_load" in mlir
    assert "tla.scalar_store" in mlir
    assert "tla.load_scalar" not in mlir
    assert "tla.store_scalar" not in mlir


@pytest.mark.parametrize(
    "kernel,tensor,match",
    (
        (scalar_load_bad_index_kernel, _tensor(), "out of bounds for length 1"),
        (scalar_load_past_end_kernel, _tensor(), "out of bounds for length 1"),
        (
            scalar_load_outside_vector_kernel,
            _tensor(),
            r"must be nested inside tla\.vector",
        ),
    ),
)
def test_scalar_load_rejects_invalid_usage(kernel, tensor, match: str) -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, TlaLoweringError, UnsupportedExecutionLowering),
        match=match,
    ):
        kernel.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    "kernel,tensor,match",
    (
        (scalar_store_bad_index_kernel, _tensor(), "out of bounds for length 1"),
        (scalar_store_past_end_kernel, _tensor(), "out of bounds for length 1"),
        (
            scalar_store_outside_vector_kernel,
            _tensor(),
            r"must be nested inside tla\.vector",
        ),
        (
            scalar_store_rank3_kernel,
            _tensor(1, 1, 1),
            "index rank must match tensor logical rank",
        ),
    ),
)
def test_scalar_store_rejects_invalid_usage(kernel, tensor, match: str) -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, TlaLoweringError, UnsupportedExecutionLowering),
        match=match,
    ):
        kernel.dump_mlir(type_args=(tensor,))
