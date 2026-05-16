import pytest

import catlass as tla
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from mlir import ir as mlir_ir

np = pytest.importorskip("numpy")


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


def test_memref_type_uses_bridge_accessors_for_dynamic_shape() -> None:
    with mlir_ir.Context() as ctx:
        element_type = tla.Float32.mlir_type(ctx)
        memref_type = tla.types.MemrefType.get(
            (4, None), element_type, "gm", context=ctx
        )

        assert tla.types.MemrefType.isinstance(memref_type)
        assert memref_type.shape == (4, None)
        assert memref_type.element_type == element_type
        assert memref_type.addrspace == "gm"


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
    assert tensor.stale is False


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
    assert len(tensor) == 4
    assert len(tensor.data[0]) == 8
    assert len(tensor.data[0][0]) == 16
    assert tensor.data[0][0][0] is None


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


def test_tensor_repr_includes_stale_state() -> None:
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
        "stale=False, origin_shape=(1, 128), coord=(0, 0), stride=(128, 1), "
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


def test_tensor_tobytes_matches_dtype_layout() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 2), tla.Float32, origin_shape=tla.make_shape(2, 2)
        )
    tensor.data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).tobytes()
    assert tensor.tobytes() == expected
    assert tensor.size_bytes == len(expected)


def test_tensor_data_is_mutable_via_nested_indexing() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 3), tla.Float32, origin_shape=tla.make_shape(2, 3)
        )

    tensor[0][1] = 3.5
    tensor[1] = [1.0, 2.0, 3.0]

    assert tensor[0][1] == 3.5
    assert tensor[1][0] == 1.0
    assert tensor[1][2] == 3.0


def test_tensor_data_accepts_numpy_assignment() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 3), tla.Float32, origin_shape=tla.make_shape(2, 3)
        )

    tensor.data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert tensor[0][0] == 1.0
    assert tensor[0][2] == 3.0
    assert tensor[1][0] == 4.0
    assert tensor[1][2] == 6.0


def test_tensor_data_rejects_numpy_shape_mismatch() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 3), tla.Float32, origin_shape=tla.make_shape(2, 3)
        )

    with pytest.raises(ValueError, match="expected shape \\(2, 3\\)"):
        tensor.data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_stale_tensor_auto_downloads_on_first_access(monkeypatch) -> None:
    download_bytes = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    ).tobytes()

    class FakeRt:
        def malloc(self, size_bytes, policy):
            return (333, 0)

        def memcpy(self, dst, dst_size, src, src_size, kind):
            return 0

        def malloc_host(self, size_bytes):
            return (334, 0)

    class FakeUtil:
        def bytes_to_ptr(self, value):
            return 444

        def ptr_to_bytes(self, ptr, size_bytes):
            return download_bytes

    class FakeAcl:
        rt = FakeRt()
        util = FakeUtil()

    monkeypatch.setattr(tla.types, "_load_acl", lambda: FakeAcl())

    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 3), tla.Float32, origin_shape=tla.make_shape(2, 3)
        )
    tensor[0] = [1.0, 2.0, 3.0]
    tensor.upload_data()

    assert tensor.stale is True

    assert tensor[0][2] == 3.0
    assert tensor.stale is False

    tensor[0] = [1.0, 2.0, 3.0]
    assert tensor[0][2] == 3.0


def test_upload_data_allocates_once_and_reuses_device_pointer(monkeypatch) -> None:
    calls: list[object] = []

    class FakeRt:
        def malloc(self, size_bytes, policy):
            calls.append(("malloc", size_bytes, policy))
            return (987654, 0)

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls.append(("memcpy", dst, dst_size, src, src_size, kind))
            return 0

        def malloc_host(self, size_bytes):
            calls.append(("malloc_host", size_bytes))
            return (765432, 0)

    class FakeUtil:
        def bytes_to_ptr(self, value):
            calls.append(("bytes_to_ptr", value))
            return 123456

        def ptr_to_bytes(self, ptr, size_bytes):
            calls.append(("ptr_to_bytes", ptr, size_bytes))
            return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).tobytes()

    class FakeAcl:
        rt = FakeRt()
        util = FakeUtil()

    monkeypatch.setattr(
        tla.runtime,
        "_GLOBAL_RUNTIME_STATE",
        tla.runtime.TlaRuntimeState(device_id=0, stream=1),
    )
    monkeypatch.setattr(tla.types, "_load_acl", lambda: FakeAcl())

    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 2), tla.Float32, origin_shape=tla.make_shape(2, 2)
        )
    tensor.data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    tensor.upload_data()
    first_ptr = tensor.data_ptr
    tensor.download_data()
    tensor.upload_data()

    assert first_ptr == 987654
    assert tensor.data_ptr == 987654
    assert tensor.stale is True
    assert calls[0] == ("malloc", 16, 0)
    assert calls.count(("malloc", 16, 0)) == 1
    assert tla.runtime.runtime_state().device_ptrs == (987654,)
    memcpy_calls = [call for call in calls if call[0] == "memcpy"]
    assert len(memcpy_calls) == 3
    assert memcpy_calls[0][1] == 987654
    assert memcpy_calls[0][2] == 16
    assert memcpy_calls[0][4] == 16
    assert memcpy_calls[0][5] == 1
    assert memcpy_calls[1] == ("memcpy", 765432, 16, 987654, 16, 2)
    assert memcpy_calls[2][5] == 1


def test_download_data_allocates_host_once_and_reuses_host_pointer(monkeypatch) -> None:
    calls: list[object] = []
    download_bytes = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32).tobytes()

    class FakeRt:
        def malloc_host(self, size_bytes):
            calls.append(("malloc_host", size_bytes))
            return (222333, 0)

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls.append(("memcpy", dst, dst_size, src, src_size, kind))
            return 0

    class FakeUtil:
        def ptr_to_bytes(self, ptr, size_bytes):
            calls.append(("ptr_to_bytes", ptr, size_bytes))
            return download_bytes

    class FakeAcl:
        rt = FakeRt()
        util = FakeUtil()

    monkeypatch.setattr(
        tla.runtime,
        "_GLOBAL_RUNTIME_STATE",
        tla.runtime.TlaRuntimeState(device_id=0, stream=1),
    )
    monkeypatch.setattr(tla.types, "_load_acl", lambda: FakeAcl())

    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(2, 2),
            tla.Float32,
            data_ptr=999888,
            stale=True,
            origin_shape=tla.make_shape(2, 2),
        )

    tensor.download_data()
    assert tensor.host_ptr == 222333
    assert tensor.stale is False
    assert tensor[0][0] == 9.0
    assert tensor[1][1] == 6.0
    assert tla.runtime.runtime_state().host_ptrs == (222333,)

    tensor.stale = True
    tensor.download_data()
    assert calls.count(("malloc_host", 16)) == 1
    memcpy_calls = [call for call in calls if call[0] == "memcpy"]
    assert len(memcpy_calls) == 2
    assert memcpy_calls[0] == ("memcpy", 222333, 16, 999888, 16, 2)


def test_upload_data_marks_tensor_stale_again(monkeypatch) -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(1, 2), tla.Float32, origin_shape=tla.make_shape(1, 2)
        )

    tensor[0][0] = 7.0

    class FakeRt:
        def malloc(self, size_bytes, policy):
            return (111, 0)

        def memcpy(self, dst, dst_size, src, src_size, kind):
            return 0

        def malloc_host(self, size_bytes):
            return (112, 0)

    class FakeUtil:
        def bytes_to_ptr(self, value):
            return 222

        def ptr_to_bytes(self, ptr, size_bytes):
            return np.array([[7.0, 0.0]], dtype=np.float32).tobytes()

    class FakeAcl:
        rt = FakeRt()
        util = FakeUtil()

    monkeypatch.setattr(tla.types, "_load_acl", lambda: FakeAcl())
    tensor.upload_data()

    assert tensor.stale is True
    assert tensor[0][0] == 7.0
    assert tensor.stale is False


def test_download_data_requires_existing_device_pointer() -> None:
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float32,
            stale=True,
            origin_shape=tla.make_shape(1, 2),
        )

    with pytest.raises(RuntimeError, match="prior upload_data"):
        tensor.download_data()


def test_tensor_rejects_symbolic_string_shape() -> None:
    with pytest.raises(TypeError, match="tla.make_shape"):
        tla.Tensor("1x128", tla.Float32)
