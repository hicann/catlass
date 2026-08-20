# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: host Numeric kernel args used directly in arithmetic.

Covers same-type scalar math on a Numeric kernel parameter (no silent promote).

Multi-scalar launch ABI packing is tracked separately; this example uses a
single Numeric arg plus typed literals so it does not depend on that fix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla
import sys


@tla.kernel
def scalar_int_arg_arith_kernel(out: tla.Tensor, a: tla.Int32) -> None:
    """Integer Numeric kernel arg: same-type arithmetic then scalar store."""
    out[0] = a + tla.Int32(6)
    out[1] = a * tla.Int32(6)
    out[2] = a // tla.Int32(6)
    out[3] = a % tla.Int32(6)


@tla.kernel
def scalar_float_arg_arith_kernel(out: tla.Tensor, a: tla.Float32) -> None:
    """Float Numeric kernel arg: same-type arithmetic then scalar store."""
    out[0] = a + tla.Float32(2.0)
    out[1] = a * tla.Float32(2.0)
    out[2] = a / tla.Float32(2.0)


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch.npu.set_device(device)
    return torch


def _gm_vector_contiguous(meta_1d) -> tla.Tensor:
    return tla.from_dlpack(meta_1d.contiguous(), layout_tag=tla.arch.RowMajor)


def _run_int_arg_arith(args: argparse.Namespace, torch, device: str) -> int:
    out = torch.full((4,), -1, dtype=torch.int32, device=device)
    out_t = _gm_vector_contiguous(out)
    a = tla.Int32(20)

    artifact = tla.compile(scalar_int_arg_arith_kernel, out_t, a)
    artifact(out_t, a, block_num=args.block_num)
    torch.npu.synchronize()

    expected = [20 + 6, 20 * 6, 20 // 6, 20 % 6]
    actual = [int(out[i].item()) for i in range(4)]
    if actual != expected:
        print(f"int_arg_arith_failed expected={expected} actual={actual}")
        return 1
    print(f"int_arg_arith_ok=True values={actual}")
    return 0


def _run_float_arg_arith(args: argparse.Namespace, torch, device: str) -> int:
    out = torch.full((3,), -1.0, dtype=torch.float32, device=device)
    out_t = _gm_vector_contiguous(out)
    a = tla.Float32(3.0)

    artifact = tla.compile(scalar_float_arg_arith_kernel, out_t, a)
    artifact(out_t, a, block_num=args.block_num)
    torch.npu.synchronize()

    expected = torch.tensor([5.0, 6.0, 1.5], dtype=torch.float32, device=device)
    if not torch.allclose(out, expected, rtol=0.0, atol=1e-4):
        print(
            f"float_arg_arith_failed expected={expected.tolist()} actual={out.tolist()}"
        )
        return 1
    print(f"float_arg_arith_ok=True values={out.tolist()}")
    return 0


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"
    for runner in (
        lambda: _run_int_arg_arith(args, torch, device),
        lambda: _run_float_arg_arith(args, torch, device),
    ):
        rc = runner()
        if rc != 0:
            return rc
    print("verification_ok=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Host Numeric kernel args used in scalar arithmetic E2E."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
