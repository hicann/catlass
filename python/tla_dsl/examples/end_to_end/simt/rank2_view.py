# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A rank-2 UB tensor indexed from a SIMT region.

The SIMT launch passes buffers as bare pointers, so the outlined vector function
gets plain memref parameters. Those must keep the tensor's logical rank: a 2-D
tensor indexed ``matrix[1, 2]`` needs a rank-2 memref parameter, or the access
has more indices than the parameter has dimensions and the kernel does not
compile at all.

Row-major contiguity is what makes this expressible: only the pointer crosses
the launch ABI, so a rank-2 parameter is honest exactly when the rows sit back
to back (``stride == (cols, 1)``). Anything else has to be refused rather than
silently mis-addressed.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

ROWS = 2
COLS = 4
N_UB = ROWS * COLS
SENTINEL = -999.0
VALUE = 7.0
ROW, COL = 1, 2

# Arch selection is the only Host compile knob now; caching moved to env vars
# (dsl e745bf10 converged the Host surface).
NPU_ARCH = "--npu-arch 3510"


@tla.kernel
def rank2_view(out: tla.Tensor) -> None:
    ptr = tla.allocate(N_UB, tla.Float32, tla.AddressSpace.ub, 64)
    matrix = tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(ROWS, COLS),
            tla.make_stride(COLS, 1),
        ),
    )
    flat = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(N_UB), tla.make_stride(1)),
    )

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            # Zero the whole allocation so a mis-addressed write is visible
            # rather than landing on stale memory.
            for i in tla.range(0, N_UB, 1):
                flat[i] = 0.0
            matrix[ROW, COL] = VALUE
            tla.arch.sync_threads()
            # Read back through both views: the 2-D write must land at the
            # linear position row*COLS + col.
            out[0] = matrix[ROW, COL]
            out[1] = flat[ROW * COLS + COL]

        tla.pipe_barrier(tla.pipes.ALL)


def _apply_cache_env(args) -> None:
    import os

    if getattr(args, "force_recompile", False):
        os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1"
    if getattr(args, "no_cache", False):
        os.environ["CATLASS_DSL_CACHE"] = "0"


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    _apply_cache_env(args)
    torch.npu.set_device(args.device)
    print(f"--- rank2_view {ROWS}x{COLS}, writing [{ROW},{COL}] ---")

    out = torch.full((2,), SENTINEL, dtype=torch.float32, device="npu")
    t_out = from_dlpack(out.contiguous(), layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(rank2_view, t_out, options=NPU_ARCH)
    artifact(t_out, block_num=1)
    torch.npu.synchronize()

    via_2d, via_flat = float(out[0]), float(out[1])
    passed = via_2d == VALUE and via_flat == VALUE
    print(f"matrix[{ROW},{COL}]={via_2d} (want {VALUE})   flat[{ROW * COLS + COL}]={via_flat} (want {VALUE})")
    print(f"passed={passed} cache_key={artifact.cache_key}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank-2 tensor indexed from a SIMT region.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
