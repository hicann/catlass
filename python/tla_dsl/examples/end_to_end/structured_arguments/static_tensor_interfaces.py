# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Validate static Tensor interfaces across argument containers on an NPU."""

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import catlass.tla as tla


@dataclass
class Config:
    bias: tla.Tensor


class NamedConfig(NamedTuple):
    bias: tla.Tensor


@tla.kernel
def copy_with_offset(aux, output: tla.Tensor, get_tensor: tla.Constexpr):
    src = get_tensor(aux)
    _ = (src.dtype, src.addrspace, src.layout_tag)
    tla.make_shape(*src.shape)
    tla.make_stride(*src.stride)
    tla.make_coord(*src.coord)
    _ = src.ptr
    output[0] = src[0] + src.element_type(1)


@tla.kernel
def copy_from_self(self: tla.Tensor, output: tla.Tensor, offset: tla.Constexpr):
    output[0] = self[0] + self.element_type(offset)


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    for dtype in (torch.float16, torch.float32):
        src = torch.tensor([3.0], dtype=dtype, device=f"npu:{args.device}")
        out = torch.zeros_like(src)
        ts, to = [tla.from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (src, out)]
        artifact = tla.compile(copy_from_self, ts, to, 1, options="--npu-arch 3510")
        artifact(ts, to, block_num=1)
        torch.npu.synchronize()
        assert out.item() == 4.0
        print(f"PASS {dtype} self: runtime parameter and Constexpr launch signature")
        cases = (
            ("direct", ts, lambda x: x),
            ("tuple", (ts,), lambda x: x[0]),
            ("list", [ts], lambda x: x[0]),
            ("namedtuple", NamedConfig(ts), lambda x: x.bias),
            ("dataclass", Config(ts), lambda x: x.bias),
        )
        for name, aux, getter in cases:
            src.fill_(3)
            artifact = tla.compile(
                copy_with_offset, aux, to, getter, options="--npu-arch 3510"
            )
            cached = tla.compile(
                copy_with_offset, aux, to, getter, options="--npu-arch 3510"
            )
            assert artifact.cache_key == cached.cache_key
            artifact(aux, to, block_num=1)
            torch.npu.synchronize()
            assert out.item() == 4.0
            src.fill_(7)
            cached(aux, to, block_num=1)
            torch.npu.synchronize()
            assert out.item() == 8.0
            print(f"PASS {dtype} {name}: Tensor interfaces and repeated launch")


if __name__ == "__main__":
    raise SystemExit(main())
