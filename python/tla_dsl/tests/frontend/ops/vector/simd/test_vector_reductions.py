from __future__ import annotations

import os
from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import UnsupportedExecutionLowering
import catlass.runtime as runtime_mod
from catlass.tla.runtime import from_dlpack, make_fake_tensor


def _vector_tensor(dtype: type[tla.Numeric] = tla.Float32, shape: int = 64) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (shape,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(shape,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MAX, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MIN, mask=reduce_mask)


@tla.kernel
def vector_reduce_i64_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int64)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)



@tla.kernel
def vector_masked_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H)
            src_reg = src_tile.load()
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=mask)


@tla.kernel
def vector_unsigned_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask)


@tla.kernel
def reduce_non_reduction_op_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce("add", mask=reduce_mask)


@tla.kernel
def reduce_init_value_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, init_value=0.0)


@tla.kernel
def reduce_profile_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, reduction_profile=0)


@tla.kernel
def reduce_none_keywords_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(
                tla.ReductionOp.ADD,
                mask=reduce_mask,
                init_value=None,
                reduction_profile=None,
            )


@tla.kernel
def vector_reduce_no_mask_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load().reduce(tla.ReductionOp.ADD)


def test_vector_reduce_public_export_exists() -> None:
    assert [op.value for op in tla.ReductionOp] == ["add", "max", "min"]


def test_vector_reduce_emits_supported_kinds() -> None:
    mlir = vector_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert mlir.count("tla.reduce") == 3
    for kind in ("add", "max", "min"):
        assert f'kind = "{kind}"' in mlir


def test_vector_reduce_emits_mask() -> None:
    mlir = vector_masked_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    reduce_line = next(line for line in mlir.splitlines() if "tla.reduce" in line)
    assert "tla.reduce" in reduce_line
    assert "!tla.mask<64>" in reduce_line
    assert 'kind = "add"' in reduce_line


def test_unsigned_reduction_has_no_init_profile_attrs() -> None:
    mlir = vector_unsigned_reduce_kernel.dump_mlir(type_args=(_vector_tensor(tla.UInt16),))

    assert "!tla.ptr<ui16" in mlir
    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_reduce_accepts_none_semantic_keywords() -> None:
    mlir = reduce_none_keywords_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_unsupported_reduction_element_type_is_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce.*unsupported reduction element type i64",
    ):
        vector_reduce_i64_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int64, 32),))


def test_reduce_rejects_non_reduction_op() -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, UnsupportedExecutionLowering), match="expected ReductionOp"
    ):
        reduce_non_reduction_op_kernel.dump_mlir(type_args=(_vector_tensor(),))


@pytest.mark.parametrize(
    "kernel, match",
    (
        (reduce_init_value_keyword_kernel, "only supports init_value=None"),
        (
            reduce_profile_keyword_kernel,
            "only supports reduction_profile=None",
        ),
    ),
)
def test_reduce_hides_unsupported_semantic_keywords(
    kernel: Any,
    match: str,
) -> None:
    with pytest.raises(UnsupportedExecutionLowering, match=match):
        kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_reduce_requires_explicit_mask() -> None:
    with pytest.raises(
        UnsupportedExecutionLowering,
        match=r"missing.*keyword-only.*argument.*'mask'",
    ):
        vector_reduce_no_mask_kernel.dump_mlir(type_args=(_vector_tensor(),))


# --- unsigned reduce signed-twin guard (u16 65535 vs 1) --------------------

_U16_N = 128
_U16_HI = 65535
_U16_LO = 1


@tla.kernel
def u16_reduce_gm_kernel(
    mem_x: tla.Tensor, mem_max: tla.Tensor, mem_min: tla.Tensor
) -> None:
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_U16_N), tla.make_coord(0))
    max_gm = tla.tile_view(mem_max, tla.make_shape(1), tla.make_coord(0))
    min_gm = tla.tile_view(mem_min, tla.make_shape(1), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(_U16_N, tla.UInt16, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    max_ub = tla.make_tensor_like(
        tla.allocate(1, tla.UInt16, tla.AddressSpace.ub, 256),
        max_gm,
        tla.arch.RowMajor,
    )
    min_ub = tla.make_tensor_like(
        tla.allocate(1, tla.UInt16, tla.AddressSpace.ub, 256),
        min_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            va = tla.tile_view(x_ub, tla.make_shape(_U16_N), tla.make_coord(0)).load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            tla.tile_view(max_ub, tla.make_shape(1), tla.make_coord(0)).store(
                va.reduce(tla.ReductionOp.MAX, mask=mask)
            )
            tla.tile_view(min_ub, tla.make_shape(1), tla.make_coord(0)).store(
                va.reduce(tla.ReductionOp.MIN, mask=mask)
            )
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(max_gm, max_ub)
        tla.copy(min_gm, min_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _as_tla_tensor(buf):
    return from_dlpack(
        buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
        origin_shape=tuple(buf.shape),
    )


@pytest.fixture
def npu_device_for_unsigned_reduce(
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


def test_u16_reduce_max_min_65535_vs_1_on_npu(
    npu_device_for_unsigned_reduce: int,
) -> None:
    """Signed twin would yield max→1 / min→65535; unsigned must keep 65535 / 1."""
    import torch

    del npu_device_for_unsigned_reduce
    vals = [_U16_LO] * _U16_N
    vals[0] = _U16_HI
    a = torch.tensor(vals, dtype=torch.uint16, device="cpu").npu()
    out_max = torch.zeros((1,), dtype=torch.uint16, device="npu")
    out_min = torch.zeros((1,), dtype=torch.uint16, device="npu")
    tx, tmax, tmin = (
        _as_tla_tensor(a),
        _as_tla_tensor(out_max),
        _as_tla_tensor(out_min),
    )

    art = tla.compile(u16_reduce_gm_kernel, tx, tmax, tmin, options="--npu-arch 3510")
    art(tx, tmax, tmin, block_num=1)
    torch.npu.synchronize()
    assert int(out_max.cpu().item()) == _U16_HI
    assert int(out_min.cpu().item()) == _U16_LO
