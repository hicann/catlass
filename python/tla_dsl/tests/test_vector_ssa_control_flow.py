from __future__ import annotations

import pytest

import catlass as tla


def _ub_tensor(dtype: object, elements: int = 64) -> object:
    ptr = tla.allocate(elements, dtype, tla.AddressSpace.ub, 256)
    return tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(elements), tla.make_stride(1)),
        coord=tla.make_coord(0),
    )


@tla.kernel
def readonly_vector_if_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            if limit > 0:
                dst_tile.store(value)


@tla.kernel
def vector_if_result_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            if limit > 0:
                value = tla.abs(value)
            dst_tile.store(value)


@tla.kernel
def vector_if_expr_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = src_tile.load()
            value = tla.abs(source) if limit > 0 else source
            dst_tile.store(value)


@tla.kernel
def vector_for_carried_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            for _ in tla.range(limit):
                value = tla.abs(value)
            dst_tile.store(value)


@tla.kernel
def vector_while_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            index = 0
            while index < limit:
                dst_tile.store(value)
                index += 1


@tla.kernel
def mask_if_result_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            if limit > 0:
                mask = tla.bitwise_not(mask)
            dst_tile.store(value, mask=mask)


@tla.kernel
def mask_if_expr_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            source = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask = tla.bitwise_not(source) if limit > 0 else source
            dst_tile.store(value, mask=mask)


@tla.kernel
def mask_for_carried_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            for _ in tla.range(limit):
                mask = tla.bitwise_not(mask)
            dst_tile.store(value, mask=mask)


@tla.kernel
def mixed_nested_carrier_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            for _ in tla.range(limit):
                if limit > 1:
                    value = tla.abs(value)
                    mask = tla.bitwise_not(mask)
            dst_tile.store(value, mask=mask)


@tla.kernel
def mask_while_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            index = 0
            while index < limit:
                dst_tile.store(value, mask=mask)
                mask = tla.bitwise_not(mask)
                index += 1


@tla.kernel
def mismatched_vector_element_expr_kernel(limit: int) -> None:
    f32_tile = _ub_tensor(tla.Float32)
    f16_tile = _ub_tensor(tla.Float16)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = f32_tile.load() if limit > 0 else f16_tile.load()
            dst_tile.store(value)


@tla.kernel
def mismatched_vector_lanes_expr_kernel(limit: int) -> None:
    full_tile = _ub_tensor(tla.Float32)
    short_tile = _ub_tensor(tla.Float32, 32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = full_tile.load() if limit > 0 else short_tile.load()
            dst_tile.store(value)


@tla.kernel
def mismatched_mask_expr_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = (
                tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
                if limit > 0
                else tla.create_mask(pattern=tla.mask.H, dtype=tla.Float16)
            )
            dst_tile.store(value, mask=mask)


@tla.kernel
def mismatched_pytree_expr_kernel(limit: int) -> None:
    src_tile = _ub_tensor(tla.Float32)
    dst_tile = _ub_tensor(tla.Float32)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            state = (value, mask) if limit > 0 else [value, mask]
            dst_tile.store(state[0], mask=state[1])


def _type_args() -> tuple[int]:
    return (2,)


def _assert_scalar_kernel_with_local_ub(mlir: str) -> None:
    signature = next(
        line.strip()
        for line in mlir.splitlines()
        if line.strip().startswith("tla.func @")
    )
    assert "(%arg0: i32)" in signature
    assert "!tla.tensor<" not in signature
    assert "tla.alloc_ptr" in mlir
    assert "tla.tile_view" not in mlir
    assert "!tla.ptr<f32, ub, 256>" in mlir


def test_resultless_if_can_capture_vector_ssa_read_only() -> None:
    mlir = readonly_vector_if_kernel.dump_mlir(type_args=_type_args())
    _assert_scalar_kernel_with_local_ub(mlir)

    assert "scf.if" in mlir
    assert "!tla.vector<64xf32>" in mlir
    assert "tla.store" in mlir


@pytest.mark.parametrize(
    "kernel",
    (
        vector_if_result_kernel,
        vector_if_expr_kernel,
        vector_for_carried_kernel,
        mask_if_result_kernel,
        mask_if_expr_kernel,
        mask_for_carried_kernel,
        mixed_nested_carrier_kernel,
    ),
)
def test_register_ssa_if_and_for_control_flow_is_supported(kernel: object) -> None:
    mlir = kernel.dump_mlir(type_args=_type_args())  # type: ignore[attr-defined]
    _assert_scalar_kernel_with_local_ub(mlir)

    assert "scf." in mlir
    assert "!tla.vector<64xf32>" in mlir
    if "mask" in kernel.fn.__name__:  # type: ignore[attr-defined]
        assert "!tla.mask<64>" in mlir
    assert "tla.store" in mlir


@pytest.mark.parametrize("kernel", (vector_while_kernel, mask_while_kernel))
def test_register_ssa_while_control_flow_is_rejected(kernel: object) -> None:
    with pytest.raises(
        Exception, match="while loops are not currently supported inside tla.vec.func"
    ):
        kernel.dump_mlir(type_args=_type_args())  # type: ignore[attr-defined]


def test_conditional_expression_rejects_mismatched_vector_element_type() -> None:
    with pytest.raises(
        Exception,
        match=r"else branch.*!tla\.vector<64xf16>.*expected !tla\.vector<64xf32>",
    ):
        mismatched_vector_element_expr_kernel.dump_mlir(type_args=(2,))


def test_conditional_expression_rejects_mismatched_vector_lanes() -> None:
    with pytest.raises(
        Exception,
        match=r"else branch.*!tla\.vector<32xf32>.*expected !tla\.vector<64xf32>",
    ):
        mismatched_vector_lanes_expr_kernel.dump_mlir(type_args=_type_args())


def test_conditional_expression_rejects_mismatched_mask_lanes() -> None:
    with pytest.raises(
        Exception,
        match=r"else branch.*!tla\.mask<128>.*expected !tla\.mask<64>",
    ):
        mismatched_mask_expr_kernel.dump_mlir(type_args=_type_args())


def test_conditional_expression_rejects_mismatched_pytree() -> None:
    with pytest.raises(Exception, match="else branch has incompatible structure"):
        mismatched_pytree_expr_kernel.dump_mlir(type_args=_type_args())
