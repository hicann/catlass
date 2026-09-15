# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Compile and launch scalar subclasses through direct and structured arguments."""

import argparse
from dataclasses import dataclass
from enum import IntEnum

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


class HostInt(int):
    pass


class HostFloat(float):
    pass


class Choice(IntEnum):
    THREE = 3
    FOUR = 4


@dataclass
class Config:
    value: object


@tla.kernel
def direct(value, output: tla.Tensor):
    output[0] = value


@tla.kernel
def nested(aux, output: tla.Tensor):
    output[0] = aux[0][0]


@tla.kernel
def config(aux, output: tla.Tensor):
    output[0] = aux.value


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    cases = (
        (HostInt(3), HostInt(4), torch.int32),
        (HostFloat(3.5), HostFloat(4.5), torch.float32),
        (Choice.THREE, Choice.FOUR, torch.int32),
        (True, False, torch.bool),
        (tla.Int64(3), tla.Int64(4), torch.int64),
    )
    for sample, value, dtype in cases:
        output = torch.zeros(1, dtype=dtype, device=f"npu:{args.device}")
        tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        for kernel, wrap in (
            (direct, lambda v: v),
            (nested, lambda v: ([v, None],)),
            (config, Config),
        ):
            compiled = tla.compile(
                kernel, wrap(sample), tensor, options="--npu-arch 3510"
            )
            cached = tla.compile(
                kernel, wrap(sample), tensor, options="--npu-arch 3510"
            )
            assert compiled.cache_key == cached.cache_key
            for artifact in (compiled, cached):
                artifact(wrap(value), tensor, block_num=1)
                torch.npu.synchronize()
                expected = value.value if isinstance(value, tla.Numeric) else value
                assert output.cpu().item() == expected
            print(f"PASS {kernel.fn.__name__} / {type(sample).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
