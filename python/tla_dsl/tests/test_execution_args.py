"""Smoke tests for ``TlaExecutionArgs`` (layout packing details live in hivm tests)."""

from __future__ import annotations

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

import catlass as tla
from catlass import compiler_bridge, execution
from catlass.base_dsl.jit_executor import TlaExecutionArgs


class _Ptr:
    def __c_pointers__(self):
        return [0x123456789ABCDEF0]


def _i32_ptr_layout() -> compiler_bridge.KernelAbiLayout:
    return compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=16,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.INTEGER,
                    32,
                    compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
                    None,
                ),
                mlir_type="i32",
                offset=0,
                storage_size=4,
                alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=1,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=8,
                storage_size=8,
                alignment=4,
            ),
        ),
    )


def test_execution_args_requires_kernel_abi() -> None:
    with pytest.raises(execution.TlaUnsupportedAbiError, match="kernel ABI layout"):
        TlaExecutionArgs().generate_launch_payload([tla.Int32(1)])


def test_execution_args_delegates_pack_to_layout_path() -> None:
    args = [tla.Int32(5), _Ptr()]
    layout = _i32_ptr_layout()
    assert TlaExecutionArgs(kernel_abi=layout).generate_launch_payload(
        args
    ) == execution._pack_launch_args(args, layout)


def test_execution_args_enforces_expected_arg_count() -> None:
    layout = _i32_ptr_layout()
    binder = TlaExecutionArgs(kernel_abi=layout, expected_arg_count=2)
    with pytest.raises(execution.TlaUnsupportedAbiError, match="expected 2"):
        binder.generate_launch_payload([tla.Int32(1)])
