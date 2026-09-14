from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import struct

from dataclasses import replace

import pytest


compiler_bridge = pytest.importorskip(
    "catlass.compiler_bridge", exc_type=ImportError
)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
tla = pytest.importorskip("catlass.tla", exc_type=ImportError)
jit_executor = pytest.importorskip(
    "catlass.base_dsl.jit_executor", exc_type=ImportError
)

_UNIFIED_FIELDS = (
    "allocated",
    "aligned",
    "offset",
    "size0",
    "size1",
    "size2",
    "size3",
    "stride0",
    "stride1",
    "stride2",
    "stride3",
    "originShape0",
    "originShape1",
)


def _bound_fake(
    shape_args: tuple,
    dtype: object,
    *,
    stride_args: tuple,
    coord_args: tuple,
    data_ptr: int,
    mark_dynamic: bool = False,
):
    """Fake host tensor with a synthetic ``data_ptr`` for launch-ABI packing tests.

    Built via :func:`make_fake_tensor` (unbound), then stamped with ``data_ptr`` so
    memref field packing can be exercised without a real DLPack buffer.
    """
    tensor = make_fake_tensor(
                 dtype,
                 (*shape_args,),
                 (*stride_args,),
                 origin_shape=(*shape_args,),
                 coord=(*coord_args,),
                 layout_tag=tla.arch.RowMajor,
             )
    if mark_dynamic:
        tensor = tensor.mark_layout_dynamic()
    tensor.data_ptr = int(data_ptr)
    tensor._external_binding = True
    return tensor


def _memref_field_layout(
    fields: tuple[str, ...] = _UNIFIED_FIELDS,
) -> compiler_bridge.KernelAbiLayout:
    arguments = []
    offset = 0
    for index, field in enumerate(fields):
        arguments.append(
            compiler_bridge.KernelAbiArgument(
                index=index,
                kind=compiler_bridge.KernelAbiArgumentKind.MEMREF_FIELD,
                scalar=None,
                mlir_type="memref<?x?x?x?xi32>",
                offset=offset,
                storage_size=8,
                alignment=4,
                logical_index=0,
                field=field,
            )
        )
        offset += 8
    return compiler_bridge.KernelAbiLayout(
        schema_version=4,
        entrypoint="kernel",
        total_size=((offset + 7) // 8) * 8,
        arguments=tuple(arguments),
    )


@pytest.mark.parametrize(
    ("shape", "dtype", "stride", "coord", "data_ptr", "expected"),
    (
        (
            (17,),
            tla.Int32,
            (1,),
            (0,),
            0xABCD00,
            (0xABCD00, 0xABCD00, 0, 17, 1, 1, 1, 1, 1, 1, 1, 17, 1),
        ),
        (
            (4, 8),
            tla.Float16,
            (8, 1),
            (0, 0),
            0x1000,
            (0x1000, 0x1000, 0, 4, 8, 1, 1, 8, 1, 1, 1, 4, 8),
        ),
    ),
)
def test_dynamic_memref_tuple_and_payload(
    shape, dtype, stride, coord, data_ptr, expected
) -> None:
    tensor = _bound_fake(
        shape,
        dtype,
        stride_args=stride,
        coord_args=coord,
        data_ptr=data_ptr,
    )

    assert tensor.build_memref_launch_fields() == expected
    assert execution._pack_launch_args(
        [tensor], _memref_field_layout()
    ) == struct.pack("<13Q", *expected)


def test_build_memref_launch_fields_tracks_pointer_change() -> None:
    tensor = _bound_fake(
        (4, 8), tla.Float16, stride_args=(8, 1), coord_args=(0, 0), data_ptr=0x1000
    )

    first = tensor.build_memref_launch_fields()
    tensor.data_ptr = 0x2000
    rebound = tensor.build_memref_launch_fields()
    assert rebound[:3] == (0x2000, 0x2000, 0)
    assert rebound[3:] == first[3:]


def test_pack_launch_args_rejects_noncanonical_memref_fields() -> None:
    fields = (_UNIFIED_FIELDS[1], _UNIFIED_FIELDS[0], *_UNIFIED_FIELDS[2:])

    with pytest.raises(
        execution.TlaUnsupportedAbiError,
        match="contiguous canonical 13-field run",
    ):
        execution._prepare_abi_packer(_memref_field_layout(fields))


def _static_layout(rank):
    fields = (
        ("allocated", "aligned", "offset")
        + tuple(f"size{i}" for i in range(rank))
        + tuple(f"stride{i}" for i in range(rank))
    )
    layout = _memref_field_layout(fields)
    mlir_type = "memref<" + "x".join(["2"] * rank + ["f32"]) + ">"
    return replace(
        layout,
        arguments=tuple(replace(a, mlir_type=mlir_type) for a in layout.arguments),
    )


class _DescriptorProvider:
    def __init__(self, fields):
        self.fields = fields

    def build_memref_launch_fields(self):
        return self.fields


@pytest.mark.parametrize("mlir_type", ["memref<2xf32>", "!static_buffer"])
def test_static_memref_payload_uses_fields_not_type_spelling(mlir_type):
    layout = _static_layout(1)
    layout = replace(
        layout,
        arguments=tuple(replace(a, mlir_type=mlir_type) for a in layout.arguments),
    )
    payload = execution._pack_launch_args(
        (_DescriptorProvider(tuple(range(13))),), layout
    )
    assert payload == struct.pack("<5Q", 0, 1, 2, 3, 7)


@pytest.mark.parametrize("rank", [None, 1], ids=["dynamic", "static"])
@pytest.mark.parametrize("count", [12, 14], ids=["missing", "extra"])
def test_descriptor_provider_field_count(rank, count):
    layout = _memref_field_layout() if rank is None else _static_layout(rank)
    packer = execution._prepare_abi_packer(layout)

    error = execution.TlaUnsupportedAbiError if rank is not None else struct.error
    with pytest.raises(error, match="expected 13"):
        execution._pack_launch_args_prepared(
            (_DescriptorProvider(tuple(range(count))),), packer
        )


def test_memref_packing_preserves_tensor_binding_error():
    tensor = make_fake_tensor(
        tla.Float32, (2,), (1,), layout_tag=tla.arch.RowMajor
    )
    packer = execution._prepare_abi_packer(_static_layout(1))
    with pytest.raises(RuntimeError, match="Tensor buffer is not bound") as error:
        execution._pack_launch_args_prepared((tensor,), packer)
    assert type(error.value) is RuntimeError


def test_memref_packing_preserves_encoding_error():
    packer = execution._prepare_abi_packer(_static_layout(1))
    provider = _DescriptorProvider((1 << 64, *range(1, 13)))
    with pytest.raises(struct.error):
        execution._pack_launch_args_prepared((provider,), packer)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_mixed_static_memref_payload(rank):
    dynamic = _bound_fake(
        (8,),
        tla.Float32,
        stride_args=(1,),
        coord_args=(0,),
        data_ptr=0x1000,
        mark_dynamic=True,
    )
    first = _bound_fake(
        (2,) * rank,
        tla.Float32,
        stride_args=tuple(2**i for i in reversed(range(rank))),
        coord_args=(0,) * rank,
        data_ptr=0x2000,
    )
    second = _bound_fake(
        (2,) * rank,
        tla.Float32,
        stride_args=tuple(2**i for i in reversed(range(rank))),
        coord_args=(0,) * rank,
        data_ptr=0x3000,
    )
    dyn_layout = _memref_field_layout()
    sta_layout = _static_layout(rank)
    # Include two static groups to check independent logical indices and offsets.
    arguments = list(dyn_layout.arguments)
    for logical_index in (1, 2):
        for arg in sta_layout.arguments:
            arguments.append(
                replace(
                    arg,
                    index=len(arguments),
                    logical_index=logical_index,
                    offset=8 * len(arguments),
                )
            )
    layout = replace(
        dyn_layout, arguments=tuple(arguments), total_size=8 * len(arguments)
    )
    packer = execution._prepare_abi_packer(layout)
    for first_ptr, second_ptr in (
        (0x2000, 0x3000), (0x4000, 0x3000), (0x4000, 0x5000)
    ):
        first.data_ptr, second.data_ptr = first_ptr, second_ptr
        shape_stride = (2,) * rank + tuple(2**i for i in reversed(range(rank)))
        expected = (
            dynamic.build_memref_launch_fields()
            + (first_ptr, first_ptr, 0)
            + shape_stride
            + (second_ptr, second_ptr, 0)
            + shape_stride
        )
        assert execution._pack_launch_args_prepared(
            (dynamic, first, second), packer
        ) == struct.pack(f"<{len(expected)}Q", *expected)


def test_prepared_strided_memref_payload_uses_current_host_strides():
    packer = execution._prepare_abi_packer(_static_layout(2))
    for pointer, row_stride in ((0x1000, 6), (0x2000, 9)):
        tensor = _bound_fake(
            (2, 3),
            tla.Float32,
            stride_args=(row_stride, 1),
            coord_args=(0, 0),
            data_ptr=pointer,
        )
        assert execution._pack_launch_args_prepared((tensor,), packer) == struct.pack(
            "<7Q", pointer, pointer, 0, 2, 3, row_stride, 1
        )


@pytest.mark.parametrize(
    "fault", ["missing", "reordered", "duplicate", "type", "split"]
)
def test_invalid_static_memref_groups_rejected(fault):
    layout = _static_layout(2)
    args = list(layout.arguments)
    if fault == "missing":
        args.pop()
    elif fault == "reordered":
        args[3] = replace(args[3], field="stride0")
        args[5] = replace(args[5], field="size0")
    elif fault == "duplicate":
        args[4] = replace(args[4], field="size0")
    elif fault == "type":
        args[-1] = replace(args[-1], mlir_type="memref<2x2xi32>")
    elif fault == "split":
        args = [replace(a, logical_index=i) for i in (0, 1, 0) for a in args]
    args = tuple(replace(a, index=i, offset=8 * i) for i, a in enumerate(args))
    layout = replace(layout, arguments=args, total_size=8 * len(args))
    execution._validate_kernel_abi_layout(layout)
    message = {
        "type": "same MLIR type",
        "split": "one contiguous group per logical argument",
    }.get(fault, "contiguous canonical")
    with pytest.raises(execution.TlaUnsupportedAbiError, match=message):
        execution._prepare_abi_packer(layout)


@pytest.mark.parametrize("rank", [0, 5])
def test_unsupported_static_memref_field_groups_rejected(rank):
    with pytest.raises(execution.TlaUnsupportedAbiError, match="contiguous canonical"):
        execution._prepare_abi_packer(_static_layout(rank))


def _two_tensor_memref_field_layout() -> compiler_bridge.KernelAbiLayout:
    arguments = []
    offset = 0
    index = 0
    for logical_index in (0, 1):
        for field in _UNIFIED_FIELDS:
            arguments.append(
                compiler_bridge.KernelAbiArgument(
                    index=index,
                    kind=compiler_bridge.KernelAbiArgumentKind.MEMREF_FIELD,
                    scalar=None,
                    mlir_type="memref<?x?x?x?xf32>",
                    offset=offset,
                    storage_size=8,
                    alignment=4,
                    logical_index=logical_index,
                    field=field,
                )
            )
            offset += 8
            index += 1
    return compiler_bridge.KernelAbiLayout(
        schema_version=4,
        entrypoint="basic_mixed",
        total_size=((offset + 7) // 8) * 8,
        arguments=tuple(arguments),
    )


def test_mixed_handoff_uses_logical_abi_count_for_dynamic_gm(
    monkeypatch, tmp_path
) -> None:
    """Device split funcs expand each dynamic GM to memref+origins; host still
    passes one Tensor per logical arg. Packing must follow ABI logical_index."""

    def _make_tensor(ptr: int, rows: int, cols: int):
        return _bound_fake(
            (rows, cols),
            tla.Float32,
            stride_args=(cols, 1),
            coord_args=(0, 0),
            data_ptr=ptr,
            mark_dynamic=True,
        )

    # 2 logical tensors → 2*(memref + origin0 + origin1) = 6 device params.
    lowered = (
        "module { "
        "func.func @basic_mixed_mix_aic("
        "%a: memref<?x?x?x?xf32>, %ao0: index, %ao1: index, "
        "%b: memref<?x?x?x?xf32>, %bo0: index, %bo1: index"
        ') attributes {mix_mode = "mix"} '
        "func.func @basic_mixed_mix_aiv("
        "%a: memref<?x?x?x?xf32>, %ao0: index, %ao1: index, "
        "%b: memref<?x?x?x?xf32>, %bo0: index, %bo1: index"
        ') attributes {mix_mode = "mix"} }'
    )
    kernel_abi = _two_tensor_memref_field_layout()
    metadata = execution._analyze_artifact_static_metadata("module {}", lowered)
    assert metadata.logical_mixed_handoff is not None
    abi_packer = execution._prepare_abi_packer(
        kernel_abi,
        expected_entrypoint=metadata.logical_mixed_handoff.entrypoint,
    )
    compile_option = execution.TlaCompileOption(kernel_mode="mix")
    compiled = execution._new_jit_compiled_function(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=lowered,
        entrypoint="basic_mixed",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=kernel_abi,
        abi_packer=abi_packer,
        uses_scalar_print=metadata.uses_scalar_print,
        uses_tensor_print=metadata.uses_tensor_print,
        logical_mixed_handoff=metadata.logical_mixed_handoff,
        compile_option=compile_option,
        pass_ir_dump="",
    )
    from catlass.base_dsl.runtime import ascend_stream_adapter as stream_mod

    monkeypatch.setattr(stream_mod, "current_device", lambda: 0)
    monkeypatch.setattr(execution, "load_acl", lambda: None)
    loads = []
    monkeypatch.setattr(
        execution,
        "load_binary",
        lambda **kwargs: loads.append(kwargs) or (11, 12),
    )
    module = compiled.jit_module
    assert isinstance(module, jit_executor.JitModule)
    launches = []
    monkeypatch.setattr(
        execution,
        "launch_kernel",
        lambda **kwargs: launches.append(kwargs),
    )
    compiled(
        args=[_make_tensor(0x1000, 32, 16), _make_tensor(0x2000, 16, 32)],
        block_num=1,
        stream=0,
    )

    assert module.entrypoint == "basic_mixed"
    assert module.kernel_mode == "mix"
    assert loads[0]["kernel_mode"] == "mix"
    assert len(launches) == 1
    payload = launches[0]["payload"]
    assert len(payload) == 208
    values = struct.unpack("<26Q", payload)
    assert values[0:13] == (0x1000, 0x1000, 0, 32, 16, 1, 1, 16, 1, 1, 1, 32, 16)
    assert values[13:26] == (0x2000, 0x2000, 0, 16, 32, 1, 1, 32, 1, 1, 1, 16, 32)
