# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT vector add: one thread per element, straight out of GM.

The SIMD counterpart lives in ``examples/end_to_end/basic_vadd``. The difference
is the whole point of the SIMT mode: no UB staging, no tiles, no vector ops --
each thread loads its own two elements and stores one.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

VECTOR_ELE = 400
_KERNEL_DTYPE = tla.Float32

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_vadd_simt(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=VECTOR_ELE):
            tid, _, _ = tla.arch.thread_idx()
            thread_block_dim, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, VECTOR_ELE, thread_block_dim):
                gm_c[i] = gm_a[i] + gm_b[i]

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from examples.end_to_end.common import get_block_num

    n_ele = VECTOR_ELE

    def create_tla_tensor(dev_buf):
        # No mark_compact_shape_dynamic: a SIMT vector function takes its
        # buffers as statically shaped memrefs, since only a pointer crosses the
        # launch ABI.
        return from_dlpack(dev_buf.contiguous(), layout_tag=tla.arch.RowMajor)

    torch.npu.set_device(args.device)
    block_num = get_block_num(args.block_num, args.device, kind="vector")
    print(f"--- basic_vadd_simt n={n_ele} thread_block_dim={VECTOR_ELE} ---")

    a = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
    b = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
    c = torch.full((n_ele,), -7.0, dtype=torch.float32, device="npu")
    expected = a + b

    tla_a, tla_b, tla_c = (
        create_tla_tensor(a),
        create_tla_tensor(b),
        create_tla_tensor(c),
    )
    artifact = tla.compile(
        basic_vadd_simt,
        tla_a,
        tla_b,
        tla_c,
        options="--npu-arch 3510",
    )
    artifact(tla_a, tla_b, tla_c, block_num=block_num)
    torch.npu.synchronize()

    passed = bool(torch.isclose(c, expected, rtol=0.0, atol=float(args.atol)).all())
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run the SIMT vector add.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="Launch block count; -1 = full vector_core_num (AIV) for this pure-v kernel",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
