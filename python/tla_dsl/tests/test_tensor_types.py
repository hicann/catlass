from typing import Any
from unittest.mock import patch

import pytest

import catlass as tla
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.base_dsl.runtime.dlpack_types import DLDeviceType
from catlass.execution_lowering import TlaLoweringError
from mlir import ir as mlir_ir


class _DlpackSentinel:
    """Minimal ``__dlpack__`` stub; capsule contents are ignored when parse is mocked."""

    def __dlpack__(self) -> Any:
        return object()


def _from_dlpack_with_parsed(
    parsed: dict[str, Any],
    **from_dlpack_kwargs: Any,
) -> tla.Tensor:
    """Run ``from_dlpack`` with a mocked capsule parse returning ``parsed``."""

    def _fake_parse(_capsule: Any) -> dict[str, Any]:
        return parsed

    with patch.object(runtime_mod._Tensor, "_parse_capsule", _fake_parse):
        return runtime_mod.from_dlpack(_DlpackSentinel(), **from_dlpack_kwargs)


def _device_dlpack_fields(
    *,
    shape: tuple[int, ...] = (2, 3),
    strides: tuple[int, ...] = (3, 1),
    data_ptr: int = 0xCAFE,
    device_type: int = int(DLDeviceType.kDLNpuCandidate1),
    dtype_code: int = 2,
    dtype_bits: int = 32,
) -> dict[str, Any]:
    return {
        "data_ptr": data_ptr,
        "device_type": device_type,
        "device_id": 0,
        "shape": shape,
        "strides": strides,
        "dtype_code": dtype_code,
        "dtype_bits": dtype_bits,
        "dtype_lanes": 1,
    }


def test_tla_type_descriptors_construct_native_mlir_types() -> None:
    with mlir_ir.Context() as ctx:
        shape = tla.types.TlaIndexTreeType("shape", ((None, 16), 8))
        stride = tla.types.TlaIndexTreeType("stride", ((16, 1), 128))
        coord = tla.types.TlaIndexTreeType("coord", (0, 0))
        origin = tla.types.TlaIndexTreeType("shape", ((32, 16), 8))
        layout = tla.types.TlaLayoutDescriptor(shape, stride, origin)
        tensor = tla.types.TlaTensorTypeDescriptor(
            layout=tla.types.TlaLayoutDescriptor(
                shape, stride, origin, layout_tag="zN"
            ),
            coord=(0, 0, 0),
            element_type="f16",
            addrspace="gm",
            ptr_alignment=2,
        )

        assert str(shape.to_mlir_type(ctx)) == "!tla.shape<(?,16),8>"
        assert str(stride.to_mlir_type(ctx)) == "!tla.stride<(16,1),128>"
        assert str(coord.to_mlir_type(ctx)) == "!tla.coord<0,0>"
        assert str(layout.to_mlir_type(ctx)) == (
            "!tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, row_major>"
        )
        assert str(tensor.to_mlir_type(ctx)) == (
            "!tla.tensor<!tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, zN>, !tla.coord<0,0,0>, !tla.ptr<f16, gm, 2>>"
        )


def test_ptr_type_uses_bridge_accessors_for_nested_pointee() -> None:
    with mlir_ir.Context() as ctx:
        with mlir_ir.Location.unknown(ctx):
            pointee = mlir_ir.MemRefType.get((4,), tla.Float16.mlir_type(ctx))
        ptr_type = tla.types.PtrType.get(pointee, "l1", 32, context=ctx)

        assert tla.types.PtrType.isinstance(ptr_type)
        assert ptr_type.pointee == pointee
        assert ptr_type.addrspace == "l1"
        assert ptr_type.alignment == 32


def test_tla_type_descriptors_require_native_bridge(monkeypatch) -> None:
    monkeypatch.setattr(tla.types._tla_type_bridge, "_EXTENSION", None)
    monkeypatch.setattr(
        tla.types._tla_type_bridge, "_resolve_bridge_extension_path", lambda: None
    )

    with pytest.raises(
        tla.types._tla_type_bridge.TlaTypeBridgeUnavailableError,
        match="Tla type bridge extension not found",
    ):
        tla.types.TlaIndexTreeType("shape", (1, 2)).to_mlir_type()


def test_tensor_defaults_without_affecting_tla_type() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(1, 128), tla.Float32, origin_shape=tla.make_shape(1, 128)
        )

    assert tensor.__tla_type__() == (
        "!tla.tensor<!tla.layout<!tla.shape<1,128>, !tla.stride<128,1>, !tla.shape<1,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>"
    )


def test_nested_shape_emits_make_shape_style_type_fields() -> None:
    """Nested component trees use comma + parens per Tla index-tree field."""
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape((4, 8), 16),
            tla.Float16,
            origin_shape=tla.make_shape((4, 8), 16),
            coord=tla.make_coord((0, 0), 0),
            stride=tla.make_stride((128, 16), 1),
        )

    assert tensor.__tla_type__() == (
        "!tla.tensor<!tla.layout<!tla.shape<(4,8),16>, !tla.stride<(128,16),1>, !tla.shape<(4,8),16>, row_major>, !tla.coord<(0,0),0>, !tla.ptr<f16, gm, 2>>"
    )
    assert tensor._shape_tuple == (4, 8, 16)
    with pytest.raises(TypeError, match="not indexable"):
        len(tensor)


def test_deep_nested_shape_groups_are_rejected() -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(tla.TlaCoreAPIError, match="one-level leaf groups"),
    ):
        tla.make_shape(((1, 2), (3, 4)))


def test_deep_nested_tensor_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-level leaf groups"):
        tla.types.TlaIndexTreeType("shape", (((1, 2), (3, 4)),))


def test_tensor_string_metadata_is_rejected() -> None:
    tensor_type = (
        "!tla.tensor<!tla.shape<1,2>, !tla.stride<2,1>, "
        "!tla.coord<0,0>, !tla.shape<1,2>, f16, gm, row_major>"
    )
    with pytest.raises(TlaLoweringError, match="TlaTensorTypeDescriptor"):
        core_api_mod._tla_tensor_descriptor_from_type_or_value(tensor_type)  # type: ignore[arg-type]


def test_uncached_tensor_value_metadata_is_rejected() -> None:
    tensor_type = (
        "!tla.tensor<!tla.shape<1,2>, !tla.stride<2,1>, "
        "!tla.coord<0,0>, !tla.shape<1,2>, f16, gm, row_major>"
    )
    with mlir_ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(
            f"module {{ func.func @f(%arg0: {tensor_type}) {{ return }} }}"
        )
        arg = module.body.operations[0].regions[0].blocks[0].arguments[0]
        with pytest.raises(
            TlaLoweringError, match="missing structured Tla tensor metadata"
        ):
            core_api_mod._tla_tensor_type_for_mlir_value(arg)


def test_tensor_exposes_structured_type_descriptor() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l1,
            layout_tag=tla.arch.zN,
            origin_shape=tla.make_shape(16, 16),
        )

    desc = tensor.tla_tensor_type_descriptor()
    assert desc.shape == ((16, 1), (16, 1))
    assert desc.stride == ((16, 256), (1, 256))
    assert desc.coord == (0, 0)
    assert desc.origin_shape == (16, 16)
    assert desc.element_type == "f16"
    assert desc.addrspace == "l1"
    assert desc.layout_tag == "zN"
    assert desc.to_asm() == (
        "!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 2>>"
    )


def test_tensor_repr_includes_metadata() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(1, 128),
            tla.Float32,
            addrspace=tla.AddressSpace.ub,
            data_ptr=123,
            origin_shape=tla.make_shape(1, 128),
        )

    assert repr(tensor) == (
        "Tensor(shape=(1, 128), dtype='f32', addrspace='ub', data_ptr=123, "
        "origin_shape=(1, 128), coord=(0, 0), stride=(128, 1), "
        "layout_tag='row_major')"
    )


def test_tensor_dtype_rejects_raw_string() -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(TypeError, match="expected a Tla element type"),
    ):
        tla.Tensor(tla.make_shape(1, 2), "f16", origin_shape=tla.make_shape(1, 2))


def test_tensor_addrspace_rejects_raw_string() -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(TypeError, match="expected tla.AddressSpace"),
    ):
        tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            addrspace="gm",
            origin_shape=tla.make_shape(1, 2),
        )


def test_tensor_layout_tag_rejects_raw_string() -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(TypeError, match="layout_tag must be a tla.arch layout sentinel"),
    ):
        tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            layout_tag="row_major",
            origin_shape=tla.make_shape(1, 2),
        )


def test_tensor_c_pointers_emit_raw_device_pointer() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(4),
            tla.Float32,
            data_ptr=1234,
            origin_shape=tla.make_shape(4),
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
        )

    assert tensor.__c_pointers__() == [1234]


def test_runtime_tensor_is_not_indexable() -> None:
    tensor = _from_dlpack_with_parsed(
        _device_dlpack_fields(), layout_tag=tla.arch.RowMajor
    )
    with pytest.raises(TypeError, match="not indexable"):
        _ = tensor.data
    with pytest.raises(TypeError, match="not indexable"):
        tensor[0, 0] = 1.0


@pytest.mark.parametrize(
    "device_type",
    [int(DLDeviceType.kDLNpuCandidate1), int(DLDeviceType.kDLExtDev)],
)
def test_from_dlpack_row_major_metadata(device_type: int) -> None:
    parsed = _device_dlpack_fields(
        shape=(2, 3), strides=(3, 1), device_type=device_type
    )
    tensor = _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.RowMajor)

    assert tensor._shape_tuple == (2, 3)
    assert tensor.layout_tag == "row_major"
    assert tensor.stride == (3, 1)
    assert tensor.data_ptr == 0xCAFE
    assert tensor._external_binding is True
    tensor.prepare_for_launch()


def test_from_dlpack_column_major_row_major_physical() -> None:
    """``permute(1, 0).contiguous()``: row-major physical (k, m), logical (m, k)."""
    parsed = _device_dlpack_fields(shape=(3, 2), strides=(2, 1))
    tensor = _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.ColumnMajor)

    assert tensor._shape_tuple == (2, 3)
    assert tensor.layout_tag == "column_major"
    assert tensor.stride == (1, 2)
    assert tensor.origin_shape == (2, 3)


def test_from_dlpack_column_major_rejects_npu_column_major_physical() -> None:
    """Only ``permute(1, 0).contiguous()`` (row-major physical) is accepted for column-major."""
    parsed = _device_dlpack_fields(shape=(2, 1), strides=(1, 2))
    with pytest.raises(
        tla.types.RuntimeTensorError, match=r"permute\(1, 0\)\.contiguous\(\)"
    ):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.ColumnMajor)


def test_from_dlpack_explicit_origin_shape() -> None:
    parsed = _device_dlpack_fields(shape=(3, 2), strides=(2, 1))
    with runtime_mod._eager_capture():
        origin = tla.make_shape(5, 7)
    tensor = _from_dlpack_with_parsed(
        parsed,
        layout_tag=tla.arch.ColumnMajor,
        origin_shape=origin,
    )

    assert tensor.origin_shape == (5, 7)
    assert tensor._shape_tuple == (5, 7)
    assert tensor.stride == (1, 5)


def test_from_dlpack_explicit_origin_shape_skips_stride_validation() -> None:
    parsed = _device_dlpack_fields(shape=(3, 2), strides=(1, 3))
    with runtime_mod._eager_capture():
        origin = tla.make_shape(3, 2)
    tensor = _from_dlpack_with_parsed(
        parsed,
        layout_tag=tla.arch.RowMajor,
        origin_shape=origin,
    )

    assert tensor.origin_shape == (3, 2)


def test_from_dlpack_explicit_origin_shape_rejects_raw_tuple() -> None:
    parsed = _device_dlpack_fields()
    with pytest.raises(TypeError, match="tla.make_shape"):
        _from_dlpack_with_parsed(
            parsed,
            layout_tag=tla.arch.RowMajor,
            origin_shape=(2, 3),
        )


def test_from_dlpack_row_major_rejects_column_major_physical() -> None:
    parsed = _device_dlpack_fields(shape=(3, 2), strides=(1, 3))
    with pytest.raises(tla.types.RuntimeTensorError, match=r"tensor\.contiguous\(\)"):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.RowMajor)


def test_from_dlpack_row_major_non_2d_skips_stride_validation() -> None:
    """Non-2-D buffers skip row/col stride checks; layout remap may still reject them."""
    parsed = _device_dlpack_fields(shape=(2, 3, 4), strides=(1, 2, 6))
    with pytest.raises(
        tla.types.RuntimeTensorError, match="cannot derive layout metadata"
    ):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.RowMajor)


def test_from_dlpack_column_major_rejects_non_contiguous_physical() -> None:
    parsed = _device_dlpack_fields(shape=(3, 2), strides=(4, 2))
    with pytest.raises(
        tla.types.RuntimeTensorError, match=r"permute\(1, 0\)\.contiguous\(\)"
    ):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.ColumnMajor)


def test_from_dlpack_requires_layout_tag() -> None:
    with pytest.raises(TypeError, match=r"required keyword-only argument: 'layout_tag'"):
        runtime_mod.from_dlpack(_DlpackSentinel())


def test_from_dlpack_rejects_cpu_buffer() -> None:
    parsed = _device_dlpack_fields(device_type=int(DLDeviceType.kDLCPU))
    with pytest.raises(tla.types.RuntimeTensorError, match="Ascend/NPU device"):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.RowMajor)


def test_from_dlpack_rejects_scalar_tensor() -> None:
    parsed = _device_dlpack_fields(shape=(), strides=())
    with pytest.raises(
        tla.types.RuntimeTensorError, match="cannot derive layout metadata"
    ):
        _from_dlpack_with_parsed(parsed, layout_tag=tla.arch.RowMajor)


def test_mark_layout_dynamic_updates_stride_metadata() -> None:
    tensor = _from_dlpack_with_parsed(
        _device_dlpack_fields(shape=(4, 8), strides=(8, 1)),
        layout_tag=tla.arch.RowMajor,
    )
    updated = tensor.mark_layout_dynamic(leading_dim=1)
    assert updated._dynamic_stride_tree == (None, 1)


def test_manual_tensor_can_bind_device_view() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l1,
            layout_tag=tla.arch.zN,
            origin_shape=tla.make_shape(16, 16),
        )
    fields = _device_dlpack_fields(
        data_ptr=0xBEEF, shape=(16, 1, 16, 1), strides=(16, 256, 1, 256)
    )
    tensor.data_ptr = int(fields["data_ptr"])
    tensor._external_binding = True
    assert tensor.data_ptr == 0xBEEF


def test_prepare_for_launch_requires_binding() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 2),
            tla.Float32,
            origin_shape=tla.make_shape(2, 2),
        )
    with pytest.raises(RuntimeError, match="buffer is not bound"):
        tensor.prepare_for_launch()
