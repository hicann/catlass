from __future__ import annotations

import os
from collections.abc import Callable

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.tla.runtime import from_dlpack, make_fake_tensor


def _vector_tensor(dtype: object = tla.Float32) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_scalar_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            mask = tla.create_mask(pattern=tla.mask.H)
            _ = tla.add(reg, 1.0)
            _ = tla.sub(reg, 2.0)
            _ = tla.mul(3.0, reg)
            _ = tla.max(reg, 5.0)
            _ = tla.min(6.0, reg)
            _ = tla.div(reg, 4.0, mask=mask)


@tla.kernel
def bf16_scalar_literal_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.add(tile.load(), 1.0)


@tla.kernel
def invalid_scalar_lhs_sub_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            _ = tla.sub(3.0, reg)


@tla.kernel
def invalid_integer_scalar_fraction_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            _ = tla.add(reg, 1.9)


@tla.kernel
def vector_scalar_numeric_ssa_kernel(
    lhs: tla.Tensor, scalar_buf: tla.Tensor
) -> None:
    """Dynamic Numeric SSA as vector–scalar rhs."""
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            s = scalar_buf[0]
            _ = tla.add(reg, s)
            _ = tla.mul(s, reg)
            _ = tla.cmp(reg, s, mode="gt")


def test_vector_scalar_accepts_numeric_ssa() -> None:
    scalar_buf = make_fake_tensor(
                     tla.Float32,
                     (1,),
                     (1,),
                     addrspace=tla.AddressSpace.ub,
                     origin_shape=(1,),
                     layout_tag=tla.arch.RowMajor,
                 )
    mlir = vector_scalar_numeric_ssa_kernel.dump_mlir(
        type_args=(_vector_tensor(), scalar_buf)
    )
    assert "tla.adds" in mlir
    assert "tla.muls" in mlir
    assert "tla.cmp" in mlir
    # Dynamic scalar must not be forced through a fresh host arith.constant alone;
    # scalar load feeds the vector–scalar ops.
    assert "tla.scalar_load" in mlir or "memref.load" in mlir or "tla.load" in mlir


def test_vector_scalar_symbols_are_exported() -> None:
    for symbol in ("add", "sub", "mul", "max", "min", "div"):
        assert callable(getattr(tla, symbol))
    for symbol in (
        "adds",
        "subs",
        "muls",
        "maxs",
        "mins",
        "divs",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
    ):
        assert not hasattr(tla, symbol)


def test_vector_scalar_dispatch_and_masking() -> None:
    mlir = vector_scalar_kernel.dump_mlir(type_args=(_vector_tensor(),))

    for op_name in ("adds", "subs", "muls", "maxs", "mins", "divs"):
        assert mlir.count(f"tla.{op_name}") == 1
    divs_line = next(line for line in mlir.splitlines() if "tla.divs" in line)
    assert "mask" in divs_line
    assert "pass_thru" not in mlir


def test_bf16_scalar_literal_uses_bf16_constant(
    compiler_tlair: Callable[..., str],
) -> None:
    mlir = compiler_tlair(bf16_scalar_literal_kernel, type_args=(_vector_tensor(tla.BFloat16),))

    assert any("arith.constant" in line and "bf16" in line for line in mlir.splitlines())
    assert "tla.adds" in mlir


def test_noncommutative_vector_scalar_rejects_scalar_lhs() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="invalid argument 'lhs'"):
        invalid_scalar_lhs_sub_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_integer_vector_scalar_rejects_fractional_literal() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected integer scalar"):
        invalid_integer_scalar_fraction_kernel.dump_mlir(
            type_args=(_vector_tensor(tla.Int32),)
        )


# --- unsigned max/min signed-twin guard (u16 65535 vs 1) -------------------

_U16_N = 128
_U16_HI = 65535
_U16_LO = 1


@tla.kernel
def u16_max_gm_kernel(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_U16_N), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(_U16_N), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(_U16_N), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    y_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        y_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            xv = tla.tile_view(x_ub, tla.make_shape(_U16_N), tla.make_coord(0)).load()
            yv = tla.tile_view(y_ub, tla.make_shape(_U16_N), tla.make_coord(0)).load()
            tla.tile_view(z_ub, tla.make_shape(_U16_N), tla.make_coord(0)).store(
                tla.max(xv, yv)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def u16_min_gm_kernel(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_U16_N), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(_U16_N), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(_U16_N), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    y_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        y_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            xv = tla.tile_view(x_ub, tla.make_shape(_U16_N), tla.make_coord(0)).load()
            yv = tla.tile_view(y_ub, tla.make_shape(_U16_N), tla.make_coord(0)).load()
            tla.tile_view(z_ub, tla.make_shape(_U16_N), tla.make_coord(0)).store(
                tla.min(xv, yv)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _as_tla_tensor(buf):
    return from_dlpack(
        buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
        origin_shape=tuple(buf.shape),
    )


@pytest.fixture
def npu_device_for_unsigned_maxmin(
    monkeypatch: pytest.MonkeyPatch, isolated_compile_cache
) -> int:
    del isolated_compile_cache
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("CANN environment required: source set_env.sh")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not torch.npu.is_available():
        pytest.skip("No accessible Ascend NPU")
    device = int(os.environ.get("CATLASS_TEST_NPU_DEVICE", "2"))
    torch.npu.set_device(device)
    return device


def test_u16_max_min_65535_vs_1_on_npu(npu_device_for_unsigned_maxmin: int) -> None:
    """Signed twin would yield max→1 / min→65535; unsigned must keep 65535 / 1."""
    import torch

    del npu_device_for_unsigned_maxmin
    a = torch.full((_U16_N,), _U16_HI, dtype=torch.uint16, device="cpu").npu()
    b = torch.full((_U16_N,), _U16_LO, dtype=torch.uint16, device="cpu").npu()
    out = torch.zeros((_U16_N,), dtype=torch.uint16, device="npu")
    tx, ty, tz = _as_tla_tensor(a), _as_tla_tensor(b), _as_tla_tensor(out)

    art = tla.compile(u16_max_gm_kernel, tx, ty, tz, options="--npu-arch 3510")
    art(tx, ty, tz, block_num=1)
    torch.npu.synchronize()
    assert bool((out.cpu() == _U16_HI).all())

    art = tla.compile(u16_min_gm_kernel, tx, ty, tz, options="--npu-arch 3510")
    art(tx, ty, tz, block_num=1)
    torch.npu.synchronize()
    assert bool((out.cpu() == _U16_LO).all())
