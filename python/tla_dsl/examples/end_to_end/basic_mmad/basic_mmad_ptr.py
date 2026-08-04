# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal ptr/make_tensor MMAD: Kernel + Host in one file.

Static problem sizes. Prefer ``basic_matmul.py`` for dynamic-GM e2e.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla

M_DIM = 64
N_DIM = 64
K_DIM = 64
L1_M_DIM = 32
L1_N_DIM = 32
L1_K_DIM = 32
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def basic_mmad_ptr(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor
) -> None:
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    allocator = tla.utils.LocalmemAllocator()

    l1a_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
    l1a_ptr = tla.recast_ptr(l1a_ptr, dtype=tla.Float32)
    l1b_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
    l1b_ptr = tla.recast_ptr(l1b_ptr, dtype=tla.Float32)
    l0a_ptr = allocator.allocate(L0A_BYTES, 512, tla.AddressSpace.l0a)
    l0a_ptr = tla.recast_ptr(l0a_ptr, dtype=tla.Float32)
    l0b_ptr = allocator.allocate(L0B_BYTES, 512, tla.AddressSpace.l0b)
    l0b_ptr = tla.recast_ptr(l0b_ptr, dtype=tla.Float32)
    l0c_ptr = allocator.allocate(L0C_BYTES, 512, tla.AddressSpace.l0c)
    l0c_ptr = tla.recast_ptr(l0c_ptr, dtype=tla.Float32)

    with tla.cube():
        gm_a = tla.make_tensor(
            32 + lhs.ptr + 32 * K_DIM,
            tla.make_layout(tla.make_shape(L1_M_DIM, L1_K_DIM),
                            tla.make_stride(K_DIM, 1))
        )
        gm_b = tla.tile_view(
            rhs, tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_coord(1, 0)
        )
        gm_c = tla.make_tensor(
            out.ptr,
            tla.make_layout(tla.make_shape(L1_K_DIM, L1_N_DIM),
                            tla.make_stride(L1_N_DIM, 1))
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_by_l0 = tla.tile_view(
            l1_a, tla.make_shape(L1_M_DIM, L1_K_DIM), tla.make_coord(0, 0)
        )
        l1_b_by_l0 = tla.tile_view(
            l1_b, tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_by_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_by_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_by_l0)
        tla.copy(l0_b, l1_b_by_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(l0c_data_ready)
        tla.wait_flag(l0c_data_ready)
        tla.copy(gm_c, l0_c)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"


def golden(lhs, rhs):
    # Kernel uses lhs ptr+offset and rhs tile at (1,0); match that window.
    return lhs[32:, 32:] @ rhs[32:, :32]


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass import runtime as runtime_mod

    def _create_tla_tensor(dev_buf, row: int, col: int):
        contiguous = dev_buf.contiguous()
        with runtime_mod._eager_capture():
            return tla.Tensor(
                tla.make_shape(row, col),
                tla.Float32,
                origin_shape=tla.make_shape(row, col),
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(col, 1),
                data_ptr=int(contiguous.data_ptr()),
            )

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        torch.manual_seed(0)
        lhs = torch.rand(M_DIM, K_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
        rhs = torch.rand(K_DIM, N_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
        out = torch.full((L1_M_DIM, L1_N_DIM), -9.0, dtype=torch.float32, device="npu")
        expected = golden(lhs, rhs)

        tla_lhs = _create_tla_tensor(lhs.npu(), M_DIM, K_DIM)
        tla_rhs = _create_tla_tensor(rhs.npu(), K_DIM, N_DIM)
        tla_out = _create_tla_tensor(out, L1_M_DIM, L1_N_DIM)

        artifact = tla.compile(
            basic_mmad_ptr,
            tla_lhs,
            tla_rhs,
            tla_out,
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
            force_recompile=args.force_recompile,
        )
        artifact(tla_lhs, tla_rhs, tla_out, block_dim=args.block_dim)
        torch.npu.synchronize()

        rtol = (1.0 / 256.0) if K_DIM < 2048 else (1.0 / 128.0)
        got = out.detach().to(device="cpu", dtype=torch.float32)
        exp = expected.detach().to(device="cpu", dtype=torch.float32)
        passed = bool(((got - exp).abs() <= rtol * torch.maximum(torch.ones_like(exp), exp.abs())).all())
        print(f"passed={passed} kernel.o={artifact.kernel_binary_path}")
        return 0 if passed else 1
    finally:
        tla.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal ptr/make_tensor MMAD.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
