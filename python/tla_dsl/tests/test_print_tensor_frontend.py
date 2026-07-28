from __future__ import annotations

import inspect

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _host_tensor(
    shape: tuple[int, ...] = (4, 4),
    *,
    dtype: type[tla.Numeric] = tla.Float32,
    addrspace: tla.AddressSpace = tla.AddressSpace.gm,
    layout: object = tla.arch.RowMajor,
    stride: tuple[int, ...] | None = None,
    alignment: int | None = None,
    coord: tuple[int, ...] | None = None,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape_value = tla.make_shape(*shape)
        stride_value = tla.make_stride(*(stride or ())) if stride is not None else None
        coord_value = tla.make_coord(*(coord or tuple(0 for _ in shape)))
        tensor = tla.Tensor(
            shape_value,
            dtype,
            addrspace=addrspace,
            origin_shape=shape_value,
            coord=coord_value,
            layout_tag=layout,
            stride=stride_value,
        )
        if alignment is not None:
            tensor._assumed_align = alignment
        return tensor


def _host_packed_tensor(layout: object) -> tla.Tensor:
    packed_shape = {
        "zN": ((16, 2), (8, 4)),
        "nZ": ((8, 4), (16, 2)),
        "zZ": ((16, 2), (8, 4)),
        "L0Clayout": ((16, 2), (16, 2)),
        "zNUnAlign": ((32, 1), (8, 4)),
    }[str(layout)]
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(*packed_shape),
            tla.Float32,
            addrspace=tla.AddressSpace.gm,
            origin_shape=tla.make_shape(32, 32),
            layout_tag=layout,
        )


@tla.kernel
def _aiv_print_tensor(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def _aiv_print_tensor_prefix(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 4)


@tla.kernel
def _aiv_print_tensor_runtime_length(
    value: tla.Tensor, length: tla.Int32
) -> None:
    with tla.vector():
        tla.print(value, length)


@tla.kernel
def _aic_print_tensor(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value)


@tla.kernel
def _aiv_print_ub_tensor(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 4)


@tla.kernel
def _aiv_print_dynamic_internal_ub_tensor(
    value: tla.Tensor, dim: "index"
) -> None:
    ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    shape = tla.make_shape(dim, 4)
    stride = tla.make_stride(4, 1)
    layout = tla.make_layout(shape, stride)
    local = tla.make_tensor(ptr, layout)
    with tla.vector():
        tla.print(local, 4)


@tla.kernel
def _regionless_print_tensor(value: tla.Tensor) -> None:
    tla.print(value)


@tla.kernel
def _mixed_print_tensor(value: tla.Tensor) -> None:
    with tla.cube():
        with tla.vector():
            tla.print(value)


@tla.kernel
def _print_tensor_length_false(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, False)


@tla.kernel
def _print_tensor_length_zero(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 0)


@tla.kernel
def _print_tensor_length_negative(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, -1)


@tla.kernel
def _print_tensor_length_too_large(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 262_113)


def test_print_has_positional_only_value_and_optional_length_surface() -> None:
    assert str(inspect.signature(tla.print)) == "(value, length=None, /)"


def test_print_tensor_emits_dedicated_tensor_marker() -> None:
    mlir = _aiv_print_tensor.dump_mlir(type_args=(_host_tensor(),))

    assert mlir.count("tla.print_tensor") == 1
    assert "tla.debug_print" not in mlir
    assert "!tla.ptr<f32, gm" in mlir


def test_print_tensor_emits_dedicated_tensor_marker_from_aic() -> None:
    mlir = _aic_print_tensor.dump_mlir(type_args=(_host_tensor(),))

    assert mlir.count("tla.print_tensor") == 1
    assert "tla.debug_print" not in mlir
    assert "tla.cube" in mlir


def test_print_tensor_accepts_single_element_rank_one_tensor() -> None:
    mlir = _aiv_print_tensor.dump_mlir(type_args=(_host_tensor((1,)),))

    assert mlir.count("tla.print_tensor") == 1


def test_print_tensor_accepts_large_source_with_prefix_length() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(_host_tensor((1024,)),))

    assert "tla.print_tensor" in mlir
    assert "shape = [1024]" in mlir or "shape = array<i64: 1024>" in mlir
    assert "length = %" in mlir
    assert "arith.constant 4 : i64" in mlir


def test_print_tensor_accepts_dynamic_rank_two_shape_with_explicit_length() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(tensor,))

    assert "tla.print_tensor" in mlir
    assert "shape = [-1, 4]" in mlir or "shape = array<i64: -1, 4>" in mlir
    assert "length = %" in mlir


def test_print_tensor_accepts_runtime_integer_length() -> None:
    mlir = _aiv_print_tensor_runtime_length.dump_mlir(
        type_args=(_host_tensor((32,)), tla.Int32(4))
    )

    assert "tla.print_tensor" in mlir
    assert "arith.extsi" in mlir
    assert "length = %" in mlir


def test_print_tensor_accepts_dynamic_rank_one_shape_with_explicit_length() -> None:
    tensor = _host_tensor((32,))
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(tensor,))

    assert "tla.print_tensor" in mlir
    assert "shape = [-1]" in mlir or "shape = array<i64: -1>" in mlir


def test_print_tensor_requires_explicit_length_for_dynamic_shape() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    with pytest.raises(tla.TlaCoreAPIError, match="explicit length"):
        _aiv_print_tensor.dump_mlir(type_args=(tensor,))


def test_print_tensor_accepts_aligned_aiv_ub_tensor() -> None:
    mlir = _aiv_print_ub_tensor.dump_mlir(
        type_args=(
            _host_tensor(
                addrspace=tla.AddressSpace.ub,
                alignment=32,
            ),
        )
    )

    assert "tla.print_tensor" in mlir
    assert "!tla.ptr<f32, ub" in mlir


def test_print_tensor_accepts_aligned_aiv_ub_offset() -> None:
    mlir = _aiv_print_ub_tensor.dump_mlir(
        type_args=(
            _host_tensor(
                (4,),
                addrspace=tla.AddressSpace.ub,
                alignment=256,
                coord=(8,),
            ),
        )
    )

    assert "tla.print_tensor" in mlir
    assert "!tla.coord<8>" in mlir


def test_print_tensor_defers_ub_base_alignment_to_runtime() -> None:
    mlir = _aiv_print_ub_tensor.dump_mlir(
        type_args=(
            _host_tensor(
                addrspace=tla.AddressSpace.ub,
                alignment=4,
            ),
        )
    )

    assert "tla.print_tensor" in mlir


def test_print_tensor_defers_ub_offset_alignment_to_runtime() -> None:
    mlir = _aiv_print_ub_tensor.dump_mlir(
        type_args=(
            _host_tensor(
                (4,),
                addrspace=tla.AddressSpace.ub,
                alignment=256,
                coord=(1,),
            ),
        )
    )

    assert "tla.print_tensor" in mlir
    assert "!tla.coord<1>" in mlir


@pytest.mark.parametrize(
    "layout",
    (
        tla.arch.zN,
        tla.arch.nZ,
        tla.arch.zZ,
        tla.arch.L0Clayout,
        tla.arch.zNUnAlign,
    ),
)
def test_print_tensor_accepts_generic_layouts(layout: object) -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(
        type_args=(_host_packed_tensor(layout),)
    )

    assert "tla.print_tensor" in mlir
    assert "shape = [32, 32]" in mlir or "shape = array<i64: 32, 32>" in mlir


def test_print_tensor_accepts_noncontiguous_linear_stride() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(
        type_args=(_host_tensor((2, 2), stride=(8, 1)),)
    )

    assert "tla.print_tensor" in mlir


def test_print_tensor_accepts_column_major_layout() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(
        type_args=(_host_tensor((32, 32), layout=tla.arch.ColumnMajor),)
    )

    assert "tla.print_tensor" in mlir


@pytest.mark.parametrize(
    ("tensor", "match"),
    (
        (_host_tensor(dtype=tla.Float16), "float32"),
        (_host_tensor((262_113,)), "explicit length"),
    ),
)
def test_print_tensor_rejects_unsupported_tensor_contract(
    tensor: tla.Tensor, match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        _aiv_print_tensor.dump_mlir(type_args=(tensor,))


def test_print_tensor_rejects_host_call() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="lowered Tla IR"):
        tla.print(_host_tensor())


def test_print_tensor_accepts_dynamic_aiv_ub_shape_at_aligned_base() -> None:
    tensor = _host_tensor(
        addrspace=tla.AddressSpace.ub,
        alignment=32,
    )
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_ub_tensor.dump_mlir(type_args=(tensor,))

    assert "tla.print_tensor" in mlir
    assert "shape = [-1, 4]" in mlir or "shape = array<i64: -1, 4>" in mlir


def test_print_tensor_accepts_dynamic_internal_ub_shape() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_dynamic_internal_ub_tensor.dump_mlir(
        type_args=(tensor, 4)
    )

    assert "tla.make_shape %" in mlir
    assert "!tla.shape<?,4>" in mlir
    assert "tla.print_tensor" in mlir


def test_print_tensor_rejects_regionless_and_mixed_placement() -> None:
    tensor = _host_tensor()
    with pytest.raises(tla.TlaCoreAPIError, match="tla.cube.*tla.vector"):
        _regionless_print_tensor.dump_mlir(type_args=(tensor,))
    with pytest.raises(tla.TlaCoreAPIError, match="mixed"):
        _mixed_print_tensor.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    "kernel",
    (
        _print_tensor_length_false,
        _print_tensor_length_zero,
        _print_tensor_length_negative,
        _print_tensor_length_too_large,
    ),
)
def test_print_tensor_rejects_invalid_length(kernel: object) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="length"):
        kernel.dump_mlir(type_args=(_host_tensor((32,)),))


def test_print_tensor_rejects_length_above_tensor_size() -> None:
    @tla.kernel
    def kernel(value: tla.Tensor) -> None:
        with tla.vector():
            tla.print(value, 5)

    with pytest.raises(tla.TlaCoreAPIError, match="element count"):
        kernel.dump_mlir(type_args=(_host_tensor((4,)),))


def test_print_tensor_accepts_exact_fifo_capacity() -> None:
    @tla.kernel
    def kernel(value: tla.Tensor) -> None:
        with tla.vector():
            tla.print(value, 262_112)

    mlir = kernel.dump_mlir(type_args=(_host_tensor((262_112,)),))
    assert "arith.constant 262112 : i64" in mlir
    assert "length = %" in mlir


def test_print_tensor_rejects_keyword_argument() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="keyword"):
        tla.print(value=_host_tensor())
