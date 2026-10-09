# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full license.
"""Ascend950 SYRK: Y = X @ X.T, computing only lower-triangular tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch_npu  # noqa: F401
from catlass import tla
from common import (
    TilingParams,
    SwizzleParams,
    compare,
    create_tla_tensor,
    get_block_num,
)


@tla.kernel
def syrk_kernel(
    x: tla.Tensor,
    xt: tla.Tensor,
    y: tla.Tensor,
    yt: tla.Tensor,
    tiling: tla.Constexpr[TilingParams],
    swizzle: tla.Constexpr[SwizzleParams],
) -> None:
    m = x.origin_shape[0]
    k = x.origin_shape[1]
    dtype = x.ptr.dtype
    l1_available0 = tla.flag("l1_available0", tla.arch.MTE1, tla.arch.MTE2)
    l1_available1 = tla.flag("l1_available1", tla.arch.MTE1, tla.arch.MTE2)
    l1_data_ready0 = tla.flag("l1_data_ready0", tla.arch.MTE2, tla.arch.MTE1)
    l1_data_ready1 = tla.flag("l1_data_ready1", tla.arch.MTE2, tla.arch.MTE1)
    l0_available0 = tla.flag("l0_available0", tla.arch.CUBE, tla.arch.MTE1)
    l0_available1 = tla.flag("l0_available1", tla.arch.CUBE, tla.arch.MTE1)
    l0_data_ready = tla.flag("l0_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)

    # Pair A/B lifetimes: each event protects both copies on the same pipe.
    l1a0 = tla.allocate(tiling.l1_tm * tiling.l1_tk, dtype, tla.AddressSpace.l1, 512)
    l1a1 = tla.allocate(tiling.l1_tm * tiling.l1_tk, dtype, tla.AddressSpace.l1, 512)
    l1b0 = tla.allocate(tiling.l1_tk * tiling.l1_tm, dtype, tla.AddressSpace.l1, 512)
    l1b1 = tla.allocate(tiling.l1_tk * tiling.l1_tm, dtype, tla.AddressSpace.l1, 512)
    l0a0 = tla.allocate(tiling.l1_tm * tiling.l0_tk, dtype, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(tiling.l1_tm * tiling.l0_tk, dtype, tla.AddressSpace.l0a, 512)
    l0b0 = tla.allocate(tiling.l0_tk * tiling.l1_tm, dtype, tla.AddressSpace.l0b, 512)
    l0b1 = tla.allocate(tiling.l0_tk * tiling.l1_tm, dtype, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(
        tiling.l1_tm * tiling.l1_tm, tla.Float32, tla.AddressSpace.l0c, 512
    )
    grid = (m + tiling.l1_tm - 1) // tiling.l1_tm

    with tla.cube():
        tla.set_flag(l1_available0)
        tla.set_flag(l1_available1)
        tla.set_flag(l0_available0)
        tla.set_flag(l0_available1)
        tla.set_flag(l0c_available)
        # Persist parity across output tiles, including odd K-tail iteration counts.
        l1_idx = tla.as_numeric(0)
        l0_idx = tla.as_numeric(0)
        for tile in tla.range(tla.arch.block_idx(), grid * grid, tla.arch.block_num()):
            # Stripe width and direction are compile-time parameters.
            offset = swizzle.SWIZZLE_OFFSET
            stripe = tile // (offset * grid)
            within = tile % (offset * grid)
            stripe_count = (grid + offset - 1) // offset
            width = grid - offset * stripe if stripe == stripe_count - 1 else offset
            row = within // width
            col = stripe * offset + within % width
            if stripe % 2 == 1:
                row = grid - row - 1
            if tla.const_expr(swizzle.SWIZZLE_DIRECTION == 0):
                row, col = col, row
            if row >= col:
                out = tla.tile_view(
                    y,
                    tla.make_shape(tiling.l1_tm, tiling.l1_tm),
                    tla.make_coord(row, col),
                )
                acc = tla.make_tensor_like(l0c_ptr, out, tla.arch.L0Clayout)
                for kk in tla.range(0, (k + tiling.l1_tk - 1) // tiling.l1_tk, 1):
                    a = tla.tile_view(
                        x,
                        tla.make_shape(tiling.l1_tm, tiling.l1_tk),
                        tla.make_coord(row, kk),
                    )
                    b = tla.tile_view(
                        xt,
                        tla.make_shape(tiling.l1_tk, tiling.l1_tm),
                        tla.make_coord(kk, col),
                    )
                    l1a = tla.make_tensor_like(
                        l1a0 if l1_idx == 0 else l1a1, a, tla.arch.zN
                    )
                    l1b = tla.make_tensor_like(
                        l1b0 if l1_idx == 0 else l1b1, b, tla.arch.nZ
                    )
                    if l1_idx == 0:
                        tla.wait_flag(l1_available0)
                    else:
                        tla.wait_flag(l1_available1)
                    tla.copy(l1a, a)
                    tla.copy(l1b, b)
                    if l1_idx == 0:
                        tla.set_flag(l1_data_ready0)
                    else:
                        tla.set_flag(l1_data_ready1)
                    l0_count = (l1a.origin_shape[1] + tiling.l0_tk - 1) // tiling.l0_tk
                    for inner in tla.range(0, l0_count, 1):
                        a_part = tla.tile_view(
                            l1a,
                            tla.make_shape(tiling.l1_tm, tiling.l0_tk),
                            tla.make_coord(0, inner),
                        )
                        b_part = tla.tile_view(
                            l1b,
                            tla.make_shape(tiling.l0_tk, tiling.l1_tm),
                            tla.make_coord(inner, 0),
                        )
                        l0a = tla.make_tensor_like(
                            l0a0 if l0_idx == 0 else l0a1, a_part, tla.arch.zN
                        )
                        l0b = tla.make_tensor_like(
                            l0b0 if l0_idx == 0 else l0b1, b_part, tla.arch.nZ
                        )
                        if inner == 0:
                            if l1_idx == 0:
                                tla.wait_flag(l1_data_ready0)
                            else:
                                tla.wait_flag(l1_data_ready1)
                        if l0_idx == 0:
                            tla.wait_flag(l0_available0)
                        else:
                            tla.wait_flag(l0_available1)
                        tla.copy(l0a, a_part)
                        tla.copy(l0b, b_part)
                        if inner == l0_count - 1:
                            if l1_idx == 0:
                                tla.set_flag(l1_available0)
                            else:
                                tla.set_flag(l1_available1)
                        tla.set_flag(l0_data_ready)
                        tla.wait_flag(l0_data_ready)
                        if kk == 0 and inner == 0:
                            tla.wait_flag(l0c_available)
                        tla.mmad(
                            acc, l0a, l0b, init_c=(kk == 0 and inner == 0), unit_flag=0
                        )
                        if l0_idx == 0:
                            tla.set_flag(l0_available0)
                        else:
                            tla.set_flag(l0_available1)
                        l0_idx = 1 - l0_idx
                    l1_idx = 1 - l1_idx

                # Both Fixpipe stores must finish before another tile reuses L0C.
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                tla.copy(out, acc, tla.params.CopyL0C2DstParams(unit_flag=0))
                if row > col:
                    out_t = tla.tile_view(
                        yt,
                        tla.make_shape(tiling.l1_tm, tiling.l1_tm),
                        tla.make_coord(row, col),
                    )
                    tla.copy(out_t, acc, tla.params.CopyL0C2DstParams(unit_flag=0))
                tla.set_flag(l0c_available)
        tla.wait_flag(l1_available0)
        tla.wait_flag(l1_available1)
        tla.wait_flag(l0_available0)
        tla.wait_flag(l0_available1)
        tla.wait_flag(l0c_available)


def validate_shape(m: int, k: int) -> None:
    """Delivery limits for the 10 GiB host accuracy-test environment."""
    if m <= 0 or k <= 0 or m > 8192 or k > 65536 or m * k > 8192**2:
        raise ValueError(
            "SYRK sample requires 1<=M<=8192, 1<=K<=65536 and M*K<=67108864"
        )


def syrk(x: torch.Tensor, *, block_num: int = -1) -> torch.Tensor:
    """Return X @ X.T for a contiguous, nonempty 2-D f16/bf16 NPU tensor."""
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError("X must have shape (M, K) with positive M and K")
    validate_shape(*x.shape)
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("X must have dtype float16 or bfloat16")
    if x.device.type != "npu" or not x.is_contiguous():
        raise ValueError("X must be a contiguous NPU tensor")
    if block_num != -1 and block_num <= 0:
        raise ValueError("block_num must be -1 (automatic) or positive")

    with torch.npu.device(x.device):
        m = x.shape[0]
        y = torch.empty((m, m), dtype=x.dtype, device=x.device)
        # ColumnMajor wraps the same contiguous storage with reversed logical
        # dimensions; no physical transpose or second copy of X is needed.
        tensors = (
            create_tla_tensor(x, "row"),
            create_tla_tensor(x, "col"),
            create_tla_tensor(y, "row"),
            create_tla_tensor(y, "col"),
        )
        tiling = TilingParams(l0_tk=64)
        swizzle = SwizzleParams(
            SWIZZLE_DIRECTION=1,
            SWIZZLE_OFFSET=3,
        )
        artifact = tla.compile(
            syrk_kernel,
            *tensors,
            tiling,
            swizzle,
            options="--npu-arch 3510",
        )
        blocks = min(
            get_block_num(block_num, x.device.index, kind="cube"),
            ((m + tiling.l1_tm - 1) // tiling.l1_tm) ** 2,
        )
        artifact(*tensors, block_num=blocks)
    return y


def golden(x: torch.Tensor) -> torch.Tensor:
    """Compute the CPU reference with fp32 accumulation on quantized input."""
    x_fp32 = x.float()
    return (x_fp32 @ x_fp32.T).to(x.dtype)


def check(actual: torch.Tensor, expected: torch.Tensor, k: int) -> bool:
    """Check output accuracy and symmetry using the shared tolerances."""
    passed = bool(torch.isfinite(actual).all()) and compare(actual, expected, k)
    symmetric = compare(actual, actual.T, k)
    print(f"passed={passed} symmetric={symmetric}")
    return passed and symmetric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=-1)
    args = parser.parse_args()
    try:
        validate_shape(args.m, args.k)
    except ValueError as error:
        parser.error(str(error))
    if args.block_num != -1 and args.block_num <= 0:
        parser.error("--block-num must be -1 or positive")
    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    torch.manual_seed(0)
    x = (torch.rand(args.m, args.k) * 2 - 1).to(dtype)
    expected = golden(x)
    actual = syrk(x.to(f"npu:{args.device}"), block_num=args.block_num)
    torch.npu.synchronize(args.device)
    print(f"m={args.m} k={args.k} dtype={dtype} blocks={args.block_num}")
    return 0 if check(actual.cpu(), expected, args.k) else 1


if __name__ == "__main__":
    raise SystemExit(main())
