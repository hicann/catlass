from __future__ import annotations

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
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

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2), tla.Float16, origin_shape=tla.make_shape(1, 2)
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


def test_execution_only_mode_handles_python_range_loop() -> None:
    def supported(mem_a: tla.Tensor) -> None:
        for _i in range(4):
            tla.make_coord(0, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2), tla.Float16, origin_shape=tla.make_shape(1, 2)
        )
    mlir = BaseDSL()._func(
        supported,
        kind="kernel",
        options={},
        type_args=(mem_a,),
    )
    assert mlir.count("tla.make_coord") == 4


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
