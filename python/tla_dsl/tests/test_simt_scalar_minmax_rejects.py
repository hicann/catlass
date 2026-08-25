"""Unsupported scalar SIMT min/max operand types are rejected at the frontend."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _simt_rejected_minmax_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, 64, nthreads):
                dst[i] = tla.max(src[i], src[i])


@pytest.mark.parametrize(
    "dtype", (tla.Bool, tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64)
)
def test_simt_max_rejects_unsigned_integers_and_bool(dtype) -> None:
    src = make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64,),
    )
    dst = make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64,),
    )
    with pytest.raises(
        Exception, match=r"does not support unsigned integers or Bool operands"
    ):
        _simt_rejected_minmax_kernel.dump_mlir(type_args=(src, dst))
