# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify a tensor, scalar, tensor kernel launch on Ascend."""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla
import sys


SCALAR_VALUE = -1234
TRAILING_VALUE = 2468


@tla.kernel
def scalar_arg_alignment(
    output_tensor: tla.Tensor,
    scalar: tla.Int16,
    trailing_tensor: tla.Tensor,
) -> None:
    """Store the scalar and a value loaded through the trailing pointer."""
    with tla.vector():
        output_tensor[0] = scalar
        output_tensor[1] = trailing_tensor[0]


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch.npu.set_device(device)
    return torch


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"
    output = torch.zeros(2, dtype=torch.int16, device=device)
    trailing = torch.tensor(
        [TRAILING_VALUE], dtype=torch.int16, device=device
    )
    output_tensor = tla.from_dlpack(
        output.contiguous(), layout_tag=tla.arch.RowMajor
    )
    trailing_tensor = tla.from_dlpack(
        trailing.contiguous(), layout_tag=tla.arch.RowMajor
    )
    scalar = tla.Int16(SCALAR_VALUE)

    artifact = tla.compile(
        scalar_arg_alignment,
        output_tensor,
        scalar,
        trailing_tensor,
        options="--npu-arch 3510"
    )
    artifact(output_tensor, scalar, trailing_tensor, block_num=1)
    torch.npu.synchronize()

    actual = [int(value) for value in output.cpu().tolist()]
    expected = [SCALAR_VALUE, TRAILING_VALUE]
    if actual != expected:
        print(f"scalar_arg_alignment_ok=False expected={expected} actual={actual}")
        return 1
    print(
        "scalar_arg_alignment_ok=True "
        f"scalar={actual[0]} trailing={actual[1]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal tensor-scalar-tensor kernel."
    )
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
