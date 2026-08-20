from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import struct

import pytest


compiler_bridge = pytest.importorskip(
    "catlass.compiler_bridge", exc_type=ImportError
)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
tla = pytest.importorskip("catlass.tla", exc_type=ImportError)

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


def _memref_field_layout() -> compiler_bridge.KernelAbiLayout:
    arguments = []
    offset = 0
    for index, field in enumerate(_UNIFIED_FIELDS):
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


def test_build_memref_launch_fields_rank1() -> None:
    tensor = _bound_fake((17,), tla.Int32, stride_args=(1,), coord_args=(0,), data_ptr=0xABCD00)
    assert tensor.build_memref_launch_fields() == {
        "allocated": 0xABCD00,
        "aligned": 0xABCD00,
        "offset": 0,
        "size0": 17,
        "size1": 1,
        "size2": 1,
        "size3": 1,
        "stride0": 1,
        "stride1": 1,
        "stride2": 1,
        "stride3": 1,
        "originShape0": 17,
        "originShape1": 1,
    }


def test_build_memref_launch_fields_rank2() -> None:
    tensor = _bound_fake(
        (4, 8), tla.Float16, stride_args=(8, 1), coord_args=(0, 0), data_ptr=0x1000
    )
    assert tensor.build_memref_launch_fields() == {
        "allocated": 0x1000,
        "aligned": 0x1000,
        "offset": 0,
        "size0": 4,
        "size1": 8,
        "size2": 1,
        "size3": 1,
        "stride0": 8,
        "stride1": 1,
        "stride2": 1,
        "stride3": 1,
        "originShape0": 4,
        "originShape1": 8,
    }


def test_build_memref_launch_fields_reuses_metadata_and_tracks_pointer_change() -> None:
    tensor = _bound_fake(
        (4, 8), tla.Float16, stride_args=(8, 1), coord_args=(0, 0), data_ptr=0x1000
    )

    first = tensor.build_memref_launch_fields()
    assert tensor.build_memref_launch_fields() is first

    tensor.data_ptr = 0x2000
    rebound = tensor.build_memref_launch_fields()
    assert rebound is not first
    assert rebound["allocated"] == 0x2000
    assert rebound["aligned"] == 0x2000
    assert rebound["offset"] == 0
    assert {key: value for key, value in rebound.items() if key not in {"allocated", "aligned"}} == {
        key: value for key, value in first.items() if key not in {"allocated", "aligned"}
    }


def test_pack_launch_args_expands_unified_memref_fields() -> None:
    tensor = _bound_fake((17,), tla.Int32, stride_args=(1,), coord_args=(0,), data_ptr=0xABCD00)
    payload = execution._pack_launch_args([tensor], _memref_field_layout())
    assert len(payload) == 104
    values = struct.unpack("<13Q", payload)
    assert values == (0xABCD00, 0xABCD00, 0, 17, 1, 1, 1, 1, 1, 1, 1, 17, 1)


def test_pack_launch_args_expands_rank2_unified_memref_fields() -> None:
    tensor = _bound_fake((4, 8), tla.Float16, stride_args=(8, 1), coord_args=(0, 0), data_ptr=0x1000)
    payload = execution._pack_launch_args([tensor], _memref_field_layout())
    assert len(payload) == 104
    values = struct.unpack("<13Q", payload)
    assert values == (0x1000, 0x1000, 0, 4, 8, 1, 1, 8, 1, 1, 1, 4, 8)


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


def test_mixed_handoff_uses_logical_abi_count_for_dynamic_gm(tmp_path) -> None:
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
    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=lowered,
        entrypoint="ignored",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_two_tensor_memref_field_layout(),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[_make_tensor(0x1000, 32, 16), _make_tensor(0x2000, 16, 32)],
        block_num=1,
    )

    assert plan.entrypoint == "basic_mixed"
    assert plan.kernel_mode == "mix"
    assert len(plan.payload) == 208
    values = struct.unpack("<26Q", plan.payload)
    assert values[0:13] == (0x1000, 0x1000, 0, 32, 16, 1, 1, 16, 1, 1, 1, 32, 16)
    assert values[13:26] == (0x2000, 0x2000, 0, 16, 32, 1, 1, 32, 1, 1, 1, 16, 32)
