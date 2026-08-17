# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT vector add spread over multiple blocks, with a 3-D thread geometry.

Two levels of parallelism at once:

* across blocks -- ``tla.arch.block_idx()`` / ``tla.arch.block_num()``, queried
  *outside* the SIMT region and captured into it as launch arguments;
* within a block -- ``tla.arch.thread_idx()`` / ``tla.arch.thread_block_dim()``,
  queried inside, with a 3-D geometry so the y and z components are exercised.

Each thread strides by the total thread count of the launch, so the work is
partitioned across all blocks and all their threads rather than every block
redundantly doing the whole array.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla

N_ELE = 65536
# 3-D thread geometry: 64 * 4 * 2 = 512 threads per block.
THREADS = (64, 4, 2)
THREADS_TOTAL = THREADS[0] * THREADS[1] * THREADS[2]
_KERNEL_DTYPE = tla.Float32


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def multiple_blocks_simt(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor
) -> None:
    # Block identity, resolved outside the SIMT region and passed in.
    block_id = tla.arch.block_idx()
    nblocks = tla.arch.block_num()

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            # Linearize the 3-D thread geometry.
            tx, ty, tz = tla.arch.thread_idx()
            dx, dy, dz = tla.arch.thread_block_dim()
            tid = tx + ty * dx + tz * dx * dy
            tdim = dx * dy * dz

            # Global id and stride over the whole launch.
            gid = block_id * tdim + tid
            stride = nblocks * tdim

            for i in tla.range(gid, N_ELE, stride):
                gm_c[i] = gm_a[i] + gm_b[i]

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"
_SENTINEL = -999.0


# Arch selection is the only Host compile knob now; caching moved to env vars
# (dsl e745bf10 converged the Host surface). --force-recompile / --no-cache are
# kept as flags and translated here so the runner scripts keep working.
NPU_ARCH = "--npu-arch 3510"


def _apply_cache_env(args) -> None:
    import os

    if getattr(args, "force_recompile", False):
        os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1"
    if getattr(args, "no_cache", False):
        os.environ["CATLASS_DSL_CACHE"] = "0"
    if getattr(args, "cache_dir", None):
        os.environ["CATLASS_DSL_CACHE_DIR"] = str(args.cache_dir)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    def create_tla_tensor(dev_buf):
        return from_dlpack(dev_buf.contiguous(), layout_tag=tla.arch.RowMajor)

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())

    _apply_cache_env(args)

    torch.npu.set_device(args.device)
    # tla.get_aicore_num is gone with the Host-surface convergence; the core
    # counts now come from torch, the way the upstream examples read them. This
    # is a pure-vector kernel, so it wants the AIV count.
    if args.block_dim != -1:
        block_dim = max(1, args.block_dim)
    else:
        block_dim = max(1, int(torch.npu.get_device_properties(args.device).vector_core_num))
    print(
        f"--- multiple_blocks_simt n={N_ELE} blocks={block_dim} "
        f"block={THREADS} ({THREADS_TOTAL} threads/block) ---"
    )

    a = torch.rand(N_ELE, dtype=torch.float32, device="npu") * 10.0 - 5.0
    b = torch.rand(N_ELE, dtype=torch.float32, device="npu") * 10.0 - 5.0
    c = torch.full((N_ELE,), _SENTINEL, dtype=torch.float32, device="npu")
    expected = a + b

    tla_a, tla_b, tla_c = create_tla_tensor(a), create_tla_tensor(b), create_tla_tensor(c)
    artifact = tla.compile(
        multiple_blocks_simt,
        tla_a,
        tla_b,
        tla_c,
    options=NPU_ARCH,
    )
    artifact(tla_a, tla_b, tla_c, block_num=block_dim)
    torch.npu.synchronize()

    # A wrong stride shows up as untouched sentinel values, which is a
    # partitioning failure rather than an arithmetic one -- report it apart.
    untouched = int((c == _SENTINEL).sum())
    passed = bool(torch.isclose(c, expected, rtol=0.0, atol=float(args.atol)).all())
    print(f"untouched={untouched}/{N_ELE}")
    print(f"passed={passed} cache_key={artifact.cache_key}")
    return 0 if passed else 1
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run the multi-block SIMT vadd."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-dim", type=int, default=-1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
