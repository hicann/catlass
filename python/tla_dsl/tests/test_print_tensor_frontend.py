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
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape_value = tla.make_shape(*shape)
        stride_value = tla.make_stride(*(stride or ())) if stride is not None else None
        coord_value = tla.make_coord(*(0 for _ in shape))
        return tla.Tensor(
            shape_value,
            dtype,
            addrspace=addrspace,
            origin_shape=shape_value,
            coord=coord_value,
            layout_tag=layout,
            stride=stride_value,
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
def _aic_print_tensor(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value)


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
        tla.print(value, 17)


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
    assert "length = 4" in mlir


@pytest.mark.parametrize(
    ("tensor", "match"),
    (
        (_host_tensor(dtype=tla.Float16), "float32"),
        (_host_tensor(addrspace=tla.AddressSpace.ub), "GM"),
        (_host_tensor(layout=tla.arch.ColumnMajor), "row-major"),
        (_host_tensor((2, 2), stride=(1, 2)), "contiguous row-major"),
        (_host_tensor((17,)), "explicit length"),
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


def test_print_tensor_rejects_dynamic_shape() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    with pytest.raises(tla.TlaCoreAPIError, match="static shape"):
        _aiv_print_tensor.dump_mlir(type_args=(tensor,))


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


def test_print_tensor_rejects_keyword_argument() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="keyword"):
        tla.print(value=_host_tensor())
