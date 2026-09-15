# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Pass ordinary and Dynamic-GM Tensors through direct and nested arguments."""

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


@tla.kernel
def direct(dynamic: tla.Tensor, bias: tla.Tensor, output: tla.Tensor):
    output[0] = dynamic[0] + bias[0]


@tla.kernel
def nested(aux, output: tla.Tensor):
    output[0] = aux[0][0] + aux[1][0][0] + aux[1][1]


@tla.kernel
def multiple(aux, bias: tla.Tensor, output: tla.Tensor):
    output[0] = aux[0][0] + aux[1][0] + bias[0]


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    for dtype in (torch.float16, torch.float32):
        source = torch.arange(400, dtype=dtype, device=device)
        bias = torch.ones_like(source)
        output = torch.zeros_like(source)
        td, tb, to = [
            from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (source, bias, output)
        ]
        td.mark_compact_shape_dynamic(0)
        second_dynamic = from_dlpack(bias, layout_tag=tla.arch.RowMajor)
        second_dynamic.mark_compact_shape_dynamic(0)
        scale = tla.Float16(2.0) if dtype is torch.float16 else 2.0
        for kernel, values, expected in (
            (direct, (td, tb, to), 1.0),
            (nested, ((td, [tb, scale]), to), 3.0),
            (multiple, ((td, second_dynamic), tb, to), 2.0),
        ):
            compiled = tla.compile(kernel, *values, options="--npu-arch 3510")
            for _ in range(2):
                compiled(*values, block_num=1)
                torch.npu.synchronize()
                assert output[0].item() == expected
            assert torch.count_nonzero(output[1:]).item() == 0
            cached = tla.compile(kernel, *values, options="--npu-arch 3510")
            assert cached.cache_key == compiled.cache_key
            replacement = torch.full_like(bias, 5)
            new_bias = from_dlpack(replacement, layout_tag=tla.arch.RowMajor)
            if kernel is direct:
                rebound = (td, new_bias, to)
            elif kernel is nested:
                rebound = ((td, [new_bias, scale]), to)
            else:
                rebound = ((td, second_dynamic), new_bias, to)
            cached(*rebound, block_num=1)
            torch.npu.synchronize()
            assert output[0].item() == expected + 4
            print(f"PASS {dtype} {kernel.__name__}: launch, repeat, rebind", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
