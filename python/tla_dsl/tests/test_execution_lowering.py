from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.mixed_kernel_attrs import (
    MixedKernelModuleAttrInputs,
    build_mixed_kernel_entry_attrs,
    build_mixed_kernel_module_attrs,
    target_system_spec_contains_arch,
)




def test_execution_lowering_validates_make_shape_components() -> None:
    def bad_shape() -> None:
        tla.make_shape(16.0, 16)

    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
        _ = BaseDSL()._func(bad_shape, kind="kernel", options={}, type_args=())


def test_execution_lowering_validates_make_coord_components() -> None:
    def bad_coord() -> None:
        tla.make_coord(0.0, 0)

    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_coord"):
        _ = BaseDSL()._func(bad_coord, kind="kernel", options={}, type_args=())


def test_execution_only_mode_lowers_tla_range_loop() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    mlir = BaseDSL()._func(
        lowered,
        kind="kernel",
        options={},
        type_args=(mem_a,),
    )
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.make_coord" in mlir


def test_execution_only_mode_lowers_python_range_loop() -> None:
    def supported(mem_a: tla.Tensor) -> None:
        for _i in range(4):
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    mlir = BaseDSL()._func(
        supported,
        kind="kernel",
        options={},
        type_args=(mem_a,),
    )
    assert "scf.for" in mlir
    assert mlir.count("tla.make_coord") == 1


def test_mixed_kernel_module_attrs_are_formatted_correctly() -> None:
    inputs = MixedKernelModuleAttrInputs(
        target_name="Ascend910_9589",
        module_core_type="MIX",
        target_system_spec=(
            '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<'
            '#dlti.dl_entry<"ARCH", "dav-c310">>>'
        ),
    )

    attrs = build_mixed_kernel_module_attrs(inputs)

    assert attrs == {
        "dlti.target_system_spec": inputs.target_system_spec,
        "hacc.target": '#hacc.target<"Ascend910_9589">',
        "hivm.module_core_type": "#hivm.module_core_type<MIX>",
    }


def test_mixed_kernel_entry_attrs_are_formatted_correctly() -> None:
    attrs = build_mixed_kernel_entry_attrs()

    assert attrs == {
        "hacc.entry": True,
        "hacc.function_kind": "#hacc.function_kind<DEVICE>",
    }


def test_target_system_spec_contains_arch_detects_arch_presence() -> None:
    with_arch = (
        '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<'
        '#dlti.dl_entry<"ARCH", "dav-c310">>>'
    )
    without_arch = (
        '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<'
        '#dlti.dl_entry<"UB_SIZE", 2031616 : i32>>>'
    )

    assert target_system_spec_contains_arch(with_arch) is True
    assert target_system_spec_contains_arch(without_arch) is False


def test_mixed_kernel_module_attrs_require_arch_in_target_spec() -> None:
    inputs = MixedKernelModuleAttrInputs(
        target_name="Ascend910_9589",
        module_core_type="MIX",
        target_system_spec=(
            '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<'
            '#dlti.dl_entry<"UB_SIZE", 2031616 : i32>>>'
        ),
    )

    with pytest.raises(ValueError, match="ARCH"):
        build_mixed_kernel_module_attrs(inputs)


@pytest.mark.parametrize(
    ("target_name", "module_core_type", "target_system_spec"),
    (
        (
            "",
            "MIX",
            '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"ARCH", "dav-c310">>>',
        ),
        (
            "Ascend910_9589",
            "",
            '#dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"ARCH", "dav-c310">>>',
        ),
        ("Ascend910_9589", "MIX", ""),
    ),
)
def test_mixed_kernel_module_attr_inputs_reject_empty_fields(
    target_name: str, module_core_type: str, target_system_spec: str
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MixedKernelModuleAttrInputs(
            target_name=target_name,
            module_core_type=module_core_type,
            target_system_spec=target_system_spec,
        )


def _structured_tensor(dtype=tla.Float32):
    return make_fake_tensor(dtype, (8,), (1,))

class _NamedAux(NamedTuple):
    bias: tla.Tensor
    limit: float

@pytest.mark.parametrize(
    "container", ["direct", "tuple", "list", "namedtuple", "dataclass"]
)
def test_static_tensor_frontend_interface_is_independent_of_container(container):
    @dataclass
    class Config:
        bias: tla.Tensor

    tensor = _structured_tensor()
    values = {
        "direct": tensor,
        "tuple": (tensor,),
        "list": [tensor],
        "namedtuple": _NamedAux(tensor, 0.5),
        "dataclass": Config(tensor),
    }

    def get_tensor(aux):
        if container == "direct":
            return aux
        if container in ("tuple", "list"):
            return aux[0]
        return aux.bias

    def kernel(aux, output):
        src = get_tensor(aux)
        _ = (src.dtype, src.addrspace, src.layout_tag)
        tla.make_shape(*src.shape)
        tla.make_shape(*src.origin_shape)
        tla.make_stride(*src.stride)
        tla.make_coord(*src.coord)
        tla.allocate((8,), src.element_type, tla.AddressSpace.ub, 32)
        _ = src.ptr
        _ = src.__extract_mlir_values__()
        output[0] = src[0]

    lowered = BaseDSL()._lower(
        kernel, kind="kernel", options={}, type_args=(values[container], tensor)
    )
    asm = lowered.asm(generic=True)
    assert '"tla.scalar_load"' in asm
    assert '"tla.scalar_store"' in asm
    assert '"tla.tensor_ptr"' in asm

@dataclass
class _StaticConfig:
    tile: tla.Constexpr[int] = 128
    width: tla.Constexpr[int] = 256

class _HostInt(int):
    pass

class _HostFloat(float):
    pass

class _HostEnum(IntEnum):
    THREE = 3

@pytest.mark.parametrize("value,dtype,op", [
    (_HostInt(3), tla.Int32, "arith.addi"),
    (_HostEnum.THREE, tla.Int32, "arith.addi"),
    (_HostFloat(3), tla.Float32, "arith.addf"),
])
def test_numeric_subclass_uses_tree_lowering(value, dtype, op) -> None:

    def kernel(value, out):
        out[0] = value + value

    lowered = BaseDSL()._lower(
        kernel,
        kind="kernel",
        options={},
        type_args=(value, _structured_tensor(dtype=dtype)),
    )
    assert lowered.argument_trees[0].runtime_leaf_count == 1
    assert lowered.argument_trees[1] is not None
    assert op in lowered.module.operation.get_asm()

def test_top_level_constexpr_config_is_available_during_lowering() -> None:
    @tla.kernel
    def kernel(config: tla.Constexpr[_StaticConfig]) -> None:
        tla.allocate(config.tile, tla.Float32, tla.AddressSpace.ub, 256)

    narrow = kernel.dump_mlir(type_args=(_StaticConfig(tile=64),))
    wide = kernel.dump_mlir(type_args=(_StaticConfig(tile=128),))
    assert narrow != wide
    assert "tla.func @kernel()" in narrow

def test_recursive_tree_enters_execution_lowering() -> None:
    def kernel(x: tla.Tensor, aux) -> None:
        _bias = aux[0]
        _limit = aux[1][0]
        tla.make_coord(0, 0)

    x = _structured_tensor()
    bias = _structured_tensor()
    lowered = BaseDSL()._lower(
        kernel,
        kind="kernel",
        options={},
        type_args=(x, (bias, [3.0, None])),
    )
    assert len(lowered.argument_trees) == 2
    assert lowered.argument_trees[1] is not None
    assert lowered.argument_trees[1].runtime_leaf_count == 2
    assert lowered.asm(generic=True).count("tla.func") == 1
