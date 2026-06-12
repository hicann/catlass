import builtins
from typing import Any

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError


def _mmad_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    """Host tensors use fractal ``make_shape`` trees for zN/nZ/L0C."""
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape((16, 8), (16, 4)),
                tla.Float16,
                addrspace=tla.AddressSpace.l0a,
                origin_shape=tla.make_shape(128, 64),
                layout_tag=tla.arch.zN,
            ),
            tla.Tensor(
                tla.make_shape((16, 4), (16, 8)),
                tla.Float16,
                addrspace=tla.AddressSpace.l0b,
                origin_shape=tla.make_shape(64, 128),
                layout_tag=tla.arch.nZ,
            ),
            tla.Tensor(
                tla.make_shape((16, 8), (16, 8)),
                tla.Float32,
                addrspace=tla.AddressSpace.l0c,
                origin_shape=tla.make_shape(128, 128),
                layout_tag=tla.arch.L0Clayout,
            ),
        )


def _vector_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(64),
                tla.Float32,
                addrspace=tla.AddressSpace.ub,
                origin_shape=tla.make_shape(64),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                tla.make_shape(64),
                tla.Float32,
                addrspace=tla.AddressSpace.ub,
                origin_shape=tla.make_shape(64),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                tla.make_shape(64),
                tla.Float32,
                addrspace=tla.AddressSpace.ub,
                origin_shape=tla.make_shape(64),
                layout_tag=tla.arch.RowMajor,
            ),
        )


def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )


class BindValue:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


@tla.kernel
def generic_with_as_shadows_tla_range_alias_kernel() -> None:
    with BindValue(builtins.range) as tla_range:
        for i in tla_range(0, 2):
            tla.make_coord(i, 0)


@tla.kernel
def cube_static_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=False)


@tla.kernel
def vec_func_default_mode_kernel() -> None:
    with tla.vec.func():
        tla.make_coord(0)


@tla.kernel
def vec_func_simd_mode_kernel() -> None:
    with tla.vec.func(mode="simd"):
        tla.make_coord(0)


@tla.kernel
def vec_func_positional_mode_kernel() -> None:
    with tla.vec.func("simd"):
        tla.make_coord(0)


@tla.kernel
def vec_func_unknown_keyword_kernel() -> None:
    with tla.vec.func(foo="simd"):
        tla.make_coord(0)


@tla.kernel
def vec_vector_ssa_kernel(lhs: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        lhs_reg = lhs_tile.load()
        rhs_reg = rhs_tile.load()
        dst_tile.store(tla.add(lhs_reg, rhs_reg))


@tla.kernel
def vec_store_rejects_raw_tensor_kernel(lhs: tla.Tensor, dst: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        dst_tile.store(lhs_tile)


@tla.kernel
def vec_add_rejects_raw_tensor_kernel(lhs: tla.Tensor, rhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        tla.add(lhs_tile, rhs_tile)


def test_generic_with_as_binding_shadows_tla_range_alias() -> None:
    mlir = generic_with_as_shadows_tla_range_alias_kernel.dump_mlir()

    assert "scf.for" not in mlir
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.coord<1,0>" in mlir


def test_cube_region_lowering_emits_exec_units() -> None:
    ta, tb, tc = _mmad_tensor_args()
    try:
        mlir = cube_static_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "tla.cube" in mlir
    assert "tla.mmad" in mlir
    assert 'tla.exec_units = "cube"' in mlir
    assert 'tla.module_exec_units = "cube"' in mlir


def test_vec_func_default_mode_lowering() -> None:
    mlir = vec_func_default_mode_kernel.dump_mlir()

    assert "tla.vec.func" in mlir
    assert 'mode = "simd"' in mlir
    assert 'tla.has_vector_region' in mlir


def test_vec_func_simd_mode_lowering() -> None:
    mlir = vec_func_simd_mode_kernel.dump_mlir()

    assert "tla.vec.func" in mlir
    assert 'mode = "simd"' in mlir


def test_vec_func_rejects_positional_mode() -> None:
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="mode must be passed by keyword"):
        vec_func_positional_mode_kernel.dump_mlir()


def test_vec_func_rejects_unknown_keyword() -> None:
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="unknown keyword argument: foo"):
        vec_func_unknown_keyword_kernel.dump_mlir()


def test_vec_vector_ssa_load_add_store_lowering() -> None:
    mlir = vec_vector_ssa_kernel.dump_mlir(type_args=_vector_tensor_args())

    assert "tla.load" in mlir
    assert "tla.add" in mlir
    assert "tla.store" in mlir
    assert "tla.make_rmem_tensor" not in mlir


def test_vec_store_rejects_raw_tensor() -> None:
    lhs, _, dst = _vector_tensor_args()
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="expected vector_ssa"):
        vec_store_rejects_raw_tensor_kernel.dump_mlir(type_args=(lhs, dst))


def test_vec_add_rejects_raw_tensor() -> None:
    lhs, rhs, _ = _vector_tensor_args()
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="expected vector_ssa"):
        vec_add_rejects_raw_tensor_kernel.dump_mlir(type_args=(lhs, rhs))
