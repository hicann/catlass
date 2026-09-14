# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Launch Dynamic-GM together with static Tensors and scalar arguments."""

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


@tla.kernel
def mixed(dynamic: tla.Tensor, bias: tla.Tensor, output: tla.Tensor):
    value = dynamic[3]
    output[0] = value + value - bias[5]


@tla.kernel
def static_first(bias: tla.Tensor, dynamic: tla.Tensor, output: tla.Tensor):
    value = dynamic[3]
    output[0] = value + value - bias[5]


@tla.kernel
def mixed_scalar(dynamic: tla.Tensor, scale, bias: tla.Tensor, output: tla.Tensor):
    value = dynamic[3]
    output[0] = value + value - bias[5] + scale


@tla.kernel
def multiple(
    first: tla.Tensor, second: tla.Tensor, bias: tla.Tensor, output: tla.Tensor
):
    left = first[3]
    right = second[7]
    output[0] = left + left - right - right - right + bias[5]


@tla.kernel
def rank2(dynamic: tla.Tensor, bias: tla.Tensor, output: tla.Tensor):
    value = dynamic[3]
    output[1, 2] = value + value - bias[1, 2]


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    for dtype in (torch.float16, torch.float32):
        source = torch.arange(400, dtype=dtype, device=device) + 11
        bias = torch.arange(400, dtype=dtype, device=device) + 31
        second_source = torch.arange(400, dtype=dtype, device=device) + 71
        output = torch.zeros_like(source)
        td, tb, to = [
            from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (source, bias, output)
        ]
        td.mark_compact_shape_dynamic(0)
        second = from_dlpack(second_source, layout_tag=tla.arch.RowMajor)
        second.mark_compact_shape_dynamic(0)
        scale = tla.Float16(2.0) if dtype is torch.float16 else tla.Float32(2.0)
        for kernel, values, bias_index, expected, bias_coefficient in (
            (mixed, (td, tb, to), 1, -8.0, -1),
            (static_first, (tb, td, to), 0, -8.0, -1),
            (mixed_scalar, (td, scale, tb, to), 2, -6.0, -1),
            (multiple, (td, second, tb, to), 2, -170.0, 1),
        ):
            compiled = tla.compile(kernel, *values, options="--npu-arch 3510")
            case = f"{kernel.__name__}/{dtype}"
            for repeat in range(2):
                output.zero_()
                compiled(*values, block_num=1)
                torch.npu.synchronize()
                actual = output[0].item()
                assert actual == expected, (
                    f"{case} repeat {repeat}: expected {expected}, got {actual}"
                )
                assert torch.count_nonzero(output[1:]).item() == 0, (
                    f"{case} repeat {repeat}: unexpected writes outside output[0]"
                )
            recompiled = tla.compile(kernel, *values, options="--npu-arch 3510")
            assert recompiled.cache_key == compiled.cache_key, (
                f"{case}: cache key changed"
            )
            replacement = bias + 4
            rebound = list(values)
            rebound[bias_index] = from_dlpack(replacement, layout_tag=tla.arch.RowMajor)
            output.zero_()
            recompiled(*rebound, block_num=1)
            torch.npu.synchronize()
            actual = output[0].item()
            expected_rebound = expected + bias_coefficient * 4
            assert actual == expected_rebound, (
                f"{case} rebind: expected {expected_rebound}, got {actual}"
            )
            assert torch.count_nonzero(output[1:]).item() == 0, (
                f"{case} rebind: unexpected writes outside output[0]"
            )
            print(
                f"PASS {dtype} {kernel.__name__}: repeat, key stability, rebind",
                flush=True,
            )
        multidim_bias = torch.arange(15, dtype=dtype, device=device).reshape(3, 5)
        result = torch.zeros_like(multidim_bias)
        bm, om = [
            from_dlpack(t, layout_tag=tla.arch.RowMajor)
            for t in (multidim_bias, result)
        ]
        compiled = tla.compile(rank2, td, bm, om, options="--npu-arch 3510")
        expected = torch.zeros_like(result)
        expected[1, 2] = source[3] * 2 - multidim_bias[1, 2]
        for repeat in range(2):
            result.zero_()
            compiled(td, bm, om, block_num=1)
            torch.npu.synchronize()
            torch.testing.assert_close(
                result,
                expected,
                rtol=0,
                atol=0,
                msg=lambda detail: f"rank2/{dtype} repeat {repeat}: {detail}",
            )
        print(
            f"PASS {dtype} rank2: nonzero index, multidimensional strides", flush=True
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
