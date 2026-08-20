"""Tests for passing stdlib ``@dataclass`` instances as kernel arguments.

``tla.dataclass`` is a plain re-export of :func:`dataclasses.dataclass`; the
frontend detects any stdlib dataclass instance, unpacks its fields into scalar
kernel args at lowering, and unpacks them again into the launch ABI.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

import catlass.tla as tla


@dataclass
class TilingData:
    tiling_int: tla.Int32 = 0
    tiling_float: tla.Float32 = 1.0


@dataclass(frozen=True)
class FrozenTilingData:
    tile_m: tla.Int32 = 16
    tile_n: tla.Int32 = 16


@dataclass
class PlainTilingData:
    tiling_int: int
    tiling_float: float


def test_dataclass_is_stdlib_dataclass() -> None:
    assert dataclasses.is_dataclass(TilingData)
    assert dataclasses.is_dataclass(TilingData())
    names = tuple(field.name for field in dataclasses.fields(TilingData))
    assert names == ("tiling_int", "tiling_float")


def test_dataclass_init_keeps_field_values() -> None:
    # Field values are not coerced to the annotated Numeric type.
    td = TilingData()
    assert td.tiling_int == 0
    assert td.tiling_float == 1.0

    td2 = TilingData(tiling_int=tla.Int32(7), tiling_float=tla.Float32(2.5))
    assert isinstance(td2.tiling_int, tla.Int32)
    assert isinstance(td2.tiling_float, tla.Float32)
    assert td2.tiling_int.value == 7
    assert td2.tiling_float.value == 2.5


def test_frozen_dataclass_raises_on_setattr() -> None:
    frozen = FrozenTilingData()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.tile_m = tla.Int32(32)


@dataclass
class MixedTensorData:
    t: tla.Tensor
    k: tla.Int32


@tla.kernel
def _kernel_dataclass_tensor_field(td: MixedTensorData, out: tla.Tensor) -> None:
    out[0] = td.t[0] + td.k


def test_kernel_dataclass_tensor_field_lowers() -> None:
    """A dataclass may carry a ``tla.Tensor`` field plus scalar fields."""
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = MixedTensorData(_gm_tensor_1d(8, dtype=tla.Int32), tla.Int32(1))
    mlir = _kernel_dataclass_tensor_field.dump_mlir(type_args=(td, out))
    assert "tla.func @_kernel_dataclass_tensor_field(%arg0: !tla.tensor" in mlir
    assert "tla.scalar_load %arg0" in mlir
    assert "arith.addi" in mlir


def test_execution_args_unpacks_dataclass_tensor_field() -> None:
    from catlass.base_dsl.jit_executor import TlaExecutionArgs
    from catlass import compiler_bridge

    class _MockTensor:
        def __c_pointers__(self) -> list[int]:
            return [0x123456789ABCDEF0]

    td = MixedTensorData(_MockTensor(), tla.Int32(7))
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="k",
        total_size=16,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=0,
                storage_size=8,
                alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=1,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.INTEGER,
                    32,
                    compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
                    None,
                ),
                mlir_type="i32",
                offset=8,
                storage_size=4,
                alignment=4,
            ),
        ),
    )
    payload = TlaExecutionArgs(kernel_abi=layout).generate_launch_payload([td])
    expected = (0x123456789ABCDEF0).to_bytes(8, "little") + (7).to_bytes(
        4, "little", signed=True
    ) + b"\0" * 4  # payload padded out to total_size 16
    assert payload == expected


def _gm_tensor_1d(length: int, *, dtype: type = tla.Int32) -> tla.Tensor:
    from catlass.tla.runtime import make_fake_tensor

    return make_fake_tensor(
               dtype,
               (length,),
               (1,),
               origin_shape=(length,),
               layout_tag=tla.arch.RowMajor,
           )


@tla.kernel
def _kernel_dataclass_arg(tiling: TilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_int + 1


@tla.kernel
def _kernel_plain_dataclass_arg(tiling: PlainTilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_int


@tla.kernel
def _kernel_dataclass_float_arg(tiling: TilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_float * 2.0


def test_kernel_lowers_dataclass_to_scalar_block_args() -> None:
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_dataclass_arg.dump_mlir(type_args=(TilingData(), out))
    assert "tla.func @_kernel_dataclass_arg(%arg0: i32, %arg1: f32" in mlir
    assert "arith.addi %arg0" in mlir
    assert "tla.scalar_store" in mlir


def test_kernel_lowers_plain_value_fields_by_value_type() -> None:
    """Plain ``int``/``float`` dataclass fields resolve to i32/f32 dynamically."""
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_plain_dataclass_arg.dump_mlir(type_args=(PlainTilingData(3, 2.5), out))
    assert "tla.func @_kernel_plain_dataclass_arg(%arg0: i32, %arg1: f32" in mlir
    assert "tla.scalar_store" in mlir


def test_kernel_dataclass_float_field_lowers_to_f32() -> None:
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_dataclass_float_arg.dump_mlir(type_args=(TilingData(), out))
    assert "tla.func @_kernel_dataclass_float_arg(%arg0: i32, %arg1: f32" in mlir
    assert "arith.mulf %arg1" in mlir


@dataclass(frozen=True)
class _FrozenSizes:
    l0a: int


@tla.kernel
def _kernel_frozen(tiling: _FrozenSizes, out: tla.Tensor) -> None:
    out[0] = tiling.l0a


def test_frozen_dataclass_unpacks_to_runtime_scalar() -> None:
    """Frozen dataclasses still unpack to runtime scalar block args."""
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_frozen.dump_mlir(type_args=(_FrozenSizes(1024), out))
    assert "tla.func @_kernel_frozen(%arg0: i32" in mlir
    assert "tla.scalar_store" in mlir


def test_execution_args_expands_dataclass_into_fields() -> None:
    from catlass.base_dsl.jit_executor import TlaExecutionArgs
    from catlass import compiler_bridge, execution

    td = TilingData(tiling_int=tla.Int32(7), tiling_float=tla.Float32(2.5))
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="k",
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
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.FLOAT,
                    32,
                    None,
                    compiler_bridge.KernelAbiFloatFormat.F32,
                ),
                mlir_type="f32",
                offset=4,
                storage_size=4,
                alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=2,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=8,
                storage_size=8,
                alignment=4,
            ),
        ),
    )

    class _Ptr:
        def __c_pointers__(self) -> list[int]:
            return [0x123456789ABCDEF0]

    payload = TlaExecutionArgs(kernel_abi=layout).generate_launch_payload(
        [td, _Ptr()]
    )
    expected = (
        (7).to_bytes(4, byteorder="little", signed=True)
        + execution._pack_scalar_argument(
            tla.Float32(2.5),
            compiler_bridge.KernelAbiScalarDescriptor(
                compiler_bridge.KernelAbiScalarCategory.FLOAT,
                32,
                None,
                compiler_bridge.KernelAbiFloatFormat.F32,
            ),
            "f32",
            4,
        )
        + (0x123456789ABCDEF0).to_bytes(8, byteorder="little", signed=False)
    )
    assert payload == expected


def test_execution_args_unpacks_plain_value_fields() -> None:
    from catlass.base_dsl.jit_executor import TlaExecutionArgs
    from catlass import compiler_bridge

    plain = PlainTilingData(7, 2.5)
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="k",
        total_size=8,
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
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.FLOAT,
                    32,
                    None,
                    compiler_bridge.KernelAbiFloatFormat.F32,
                ),
                mlir_type="f32",
                offset=4,
                storage_size=4,
                alignment=4,
            ),
        ),
    )
    payload = TlaExecutionArgs(kernel_abi=layout).generate_launch_payload([plain])
    expected = (7).to_bytes(4, "little", signed=True) + (
        tla.Float32(2.5).__c_pointers__()[0].to_bytes(4, "little")
    )
    assert payload == expected


@dataclass
class ConstexprTilingData:
    TILE_M: tla.Constexpr[int]
    TILE_N: tla.Constexpr[int]
    dyn: tla.Int32


@dataclass
class AllConstexprTilingData:
    TILE_M: tla.Constexpr[int]
    TILE_N: tla.Constexpr[int]


@tla.kernel
def _kernel_constexpr_mixed(td: ConstexprTilingData, out: tla.Tensor) -> None:
    _ = tla.make_shape(td.TILE_M, td.TILE_N)
    out[0] = td.dyn


def test_kernel_dataclass_constexpr_fields_not_in_signature() -> None:
    """``tla.Constexpr[...]`` fields add no block args; dynamic fields still do."""
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = ConstexprTilingData(TILE_M=16, TILE_N=32, dyn=tla.Int32(3))
    mlir = _kernel_constexpr_mixed.dump_mlir(type_args=(td, out))
    assert "tla.func @_kernel_constexpr_mixed(%arg0: i32, %arg1: !tla.tensor" in mlir
    assert "!tla.shape<16,32>" in mlir
    assert "tla.scalar_store" in mlir


def test_all_constexpr_dataclass_has_no_block_args() -> None:
    @tla.kernel
    def kernel(td: AllConstexprTilingData) -> None:
        _ = tla.make_shape(td.TILE_M, td.TILE_N)

    mlir = kernel.dump_mlir(type_args=(AllConstexprTilingData(16, 32),))
    assert "tla.func @kernel()" in mlir
    assert "!tla.shape<16,32>" in mlir


def test_execution_args_skips_constexpr_fields() -> None:
    from catlass.base_dsl.jit_executor import TlaExecutionArgs
    from catlass import compiler_bridge

    td = ConstexprTilingData(TILE_M=16, TILE_N=32, dyn=tla.Int32(7))
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="k",
        total_size=8,
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
        ),
    )
    payload = TlaExecutionArgs(kernel_abi=layout).generate_launch_payload([td])
    expected = (7).to_bytes(4, "little", signed=True) + b"\0" * 4
    assert payload == expected


def test_kernel_assign_to_constexpr_field_raises() -> None:
    @tla.kernel
    def kernel(td: ConstexprTilingData, out: tla.Tensor) -> None:
        td.TILE_M = 512
        out[0] = td.dyn

    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = ConstexprTilingData(TILE_M=16, TILE_N=32, dyn=tla.Int32(3))
    with pytest.raises(Exception, match="TILE_M.*read-only"):
        kernel.dump_mlir(type_args=(td, out))


def test_kernel_can_mutate_dynamic_field() -> None:
    @tla.kernel
    def kernel(td: ConstexprTilingData, out: tla.Tensor) -> None:
        td.dyn = tla.Int32(9)
        out[0] = td.dyn

    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = ConstexprTilingData(TILE_M=16, TILE_N=32, dyn=tla.Int32(3))
    mlir = kernel.dump_mlir(type_args=(td, out))
    assert "arith.constant 9" in mlir


@dataclass(frozen=True)
class FrozenConstexprTilingData:
    TILE_M: tla.Constexpr[int]
    dyn: tla.Int32


def test_frozen_constexpr_field_write_raises_readonly() -> None:
    @tla.kernel
    def kernel(td: FrozenConstexprTilingData, out: tla.Tensor) -> None:
        td.TILE_M = 512
        out[0] = td.dyn

    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = FrozenConstexprTilingData(TILE_M=16, dyn=tla.Int32(3))
    with pytest.raises(Exception, match="TILE_M.*read-only"):
        kernel.dump_mlir(type_args=(td, out))


@dataclass(frozen=True, kw_only=True)
class FrozenKwOnlyTilingData:
    TILE_M: tla.Constexpr[int]
    dyn: tla.Int32


@tla.kernel
def _kernel_frozen_kw_only(td: FrozenKwOnlyTilingData, out: tla.Tensor) -> None:
    out[0] = td.dyn


def test_dataclass_frozen_kw_only_is_kernel_arg() -> None:
    """``frozen=True, kw_only=True`` is allowed for a kernel-argument dataclass."""
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    td = FrozenKwOnlyTilingData(TILE_M=16, dyn=tla.Int32(3))
    mlir = _kernel_frozen_kw_only.dump_mlir(type_args=(td, out))
    assert "tla.func @_kernel_frozen_kw_only(%arg0: i32" in mlir
    assert "tla.scalar_store" in mlir


@pytest.mark.parametrize(
    ("dataclass_kwargs",),
    (
        ({"slots": True},),
        ({"eq": False},),
        ({"order": True},),
    ),
)
def test_dataclass_custom_stdlib_option_rejected(dataclass_kwargs: dict) -> None:
    """Non-default stdlib dataclass options are rejected at kernel compile time."""

    class _OptionallySlots:
        if dataclass_kwargs.get("slots"):
            __slots__ = ()

    Bad = dataclasses.dataclass(
        type(
            "_Bad",
            (_OptionallySlots,),
            {"__annotations__": {"x": tla.Int32}},
        ),
        **dataclass_kwargs,
    )

    @tla.kernel
    def kernel(td: Bad, out: tla.Tensor) -> None:
        out[0] = td.x

    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match="only frozen= and kw_only= may be customized"):
        kernel.dump_mlir(type_args=(Bad(1), out))
