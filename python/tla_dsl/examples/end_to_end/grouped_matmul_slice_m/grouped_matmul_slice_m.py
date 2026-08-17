# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse

import catlass.tla as tla
import torch
import torch_npu
from catlass.tla.runtime import from_dlpack
from examples.end_to_end.common import TilingParams


def _group_list_prefix(current_ms: tuple[int, ...]) -> tuple[int, ...]:
    """Length G+1 prefix with leading 0: currentM[g] = prefix[g+1] - prefix[g]."""
    out = [0]
    for current_m in current_ms:
        out.append(out[-1] + current_m)
    return tuple(out)


def _average_current_m(m_val: int, g_val: int, *, l1_m: int) -> tuple[int, ...]:
    """Partition ``[0, m_val)`` into ``g_val`` nearly-equal, L1_M-aligned groups."""
    if g_val <= 0:
        raise ValueError(f"groups must be positive; got {g_val}")
    if m_val % l1_m != 0:
        raise ValueError(f"requires m % l1_m == 0; got m={m_val}, l1_m={l1_m}")
    n_tiles = m_val // l1_m
    base, rem = divmod(n_tiles, g_val)
    out = tuple(((base + 1) if i < rem else base) * l1_m for i in range(g_val))
    if sum(out) != m_val:
        raise RuntimeError(f"internal error: average groups sum to {sum(out)}, expected m={m_val}")
    return out


@tla.kernel
def grouped_matmul_slice_m_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    group_list: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: TilingParams,
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    DTYPE_C = tla.Float32  # L0C accumulator only

    n_dim = gm_c.origin_shape[1]
    k_dim = gm_a.origin_shape[1]
    group_cnt = group_list.origin_shape[0] - 1

    l1a0_copy_end = tla.flag("l1a0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_copy_end = tla.flag("l1a1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_copy_end = tla.flag("l1b0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_copy_end = tla.flag("l1b1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_copy_start = tla.flag("l1a0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_copy_start = tla.flag("l1a1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_copy_start = tla.flag("l1b0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_copy_start = tla.flag("l1b1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_copy_start = tla.flag("l0a0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_copy_start = tla.flag("l0a1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_copy_start = tla.flag("l0b0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_copy_start = tla.flag("l0b1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0_copy_end = tla.flag("l0_copy_end", tla.arch.MTE1, tla.arch.CUBE)

    l1a0_ptr = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    grid_n = (n_dim + _tiling.l1_tn - 1) // _tiling.l1_tn

    with tla.cube():
        tla.set_flag(l1a0_copy_start)
        tla.set_flag(l1a1_copy_start)
        tla.set_flag(l1b0_copy_start)
        tla.set_flag(l1b1_copy_start)
        tla.set_flag(l0a0_copy_start)
        tla.set_flag(l0a1_copy_start)
        tla.set_flag(l0b0_copy_start)
        tla.set_flag(l0b1_copy_start)

        l1_buf_idx = c0
        l0_buf_idx = c0

        for g in tla.range(c0, group_cnt, c1):
            m_start = group_list[g]
            m_end = group_list[g + 1]
            current_m = m_end - m_start

            if current_m > 0:
                # L1-aligned groups: tile index = element offset / _tiling.l1_tm.
                m_tile_base = m_start // _tiling.l1_tm
                grid_m = current_m // _tiling.l1_tm
                mn_blocks = grid_m * grid_n

                gm_b_group = tla.tile_view(
                    gm_b, tla.make_shape(k_dim, n_dim), tla.make_coord(g, c0)
                )

                block_range = tla.range(
                    tla.arch.block_idx(), mn_blocks, tla.arch.block_num()
                )
                for loop_idx in block_range:
                    block_row = loop_idx // grid_n
                    block_col = loop_idx % grid_n
                    abs_row = m_tile_base + block_row

                    gm_a_by_core = tla.tile_view(
                        gm_a, tla.make_shape(_tiling.l1_tm, k_dim), tla.make_coord(abs_row, c0)
                    )
                    gm_b_by_core = tla.tile_view(
                        gm_b_group, tla.make_shape(k_dim, _tiling.l1_tn), tla.make_coord(c0, block_col)
                    )
                    gm_c_by_core = tla.tile_view(
                        gm_c,
                        tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                        tla.make_coord(abs_row, block_col),
                    )

                    k_block = gm_a_by_core.origin_shape[1]
                    k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
                    k_l1_range = tla.range(c0, k_l1_count, c1)

                    l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                    for k_l1 in k_l1_range:
                        gm_a_l1 = tla.tile_view(
                            gm_a_by_core, tla.make_shape(_tiling.l1_tm, _tiling.l1_tk), tla.make_coord(c0, k_l1)
                        )
                        gm_b_l1 = tla.tile_view(
                            gm_b_by_core, tla.make_shape(_tiling.l1_tk, _tiling.l1_tn), tla.make_coord(k_l1, c0)
                        )

                        l1_a = tla.make_tensor_like(
                            l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                        )
                        l1_b = tla.make_tensor_like(
                            l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                        )
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_copy_start)
                        else:
                            tla.wait_flag(l1a1_copy_start)
                        tla.copy(l1_a, gm_a_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_copy_end)
                        else:
                            tla.set_flag(l1a1_copy_end)

                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_copy_start)
                        else:
                            tla.wait_flag(l1b1_copy_start)
                        tla.copy(l1_b, gm_b_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_copy_end)
                        else:
                            tla.set_flag(l1b1_copy_end)

                        k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                        k_l0_range = tla.range(c0, k_l0_count, c1)
                        for k_l0 in k_l0_range:
                            l1_a_l0 = tla.tile_view(
                                l1_a, tla.make_shape(_tiling.l0_tm, _tiling.l0_tk), tla.make_coord(c0, k_l0)
                            )
                            l1_b_l0 = tla.tile_view(
                                l1_b, tla.make_shape(_tiling.l0_tk, _tiling.l0_tn), tla.make_coord(k_l0, c0)
                            )
                            l0_a = tla.make_tensor_like(
                                l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                            )
                            l0_b = tla.make_tensor_like(
                                l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                            )
                            if k_l0 == 0:
                                if l1_buf_idx == c0:
                                    tla.wait_flag(l1a0_copy_end)
                                else:
                                    tla.wait_flag(l1a1_copy_end)

                            if l0_buf_idx == c0:
                                tla.wait_flag(l0a0_copy_start)
                            else:
                                tla.wait_flag(l0a1_copy_start)
                            tla.copy(l0_a, l1_a_l0)
                            if k_l0 == k_l0_count - 1:
                                if l1_buf_idx == c0:
                                    tla.set_flag(l1a0_copy_start)
                                else:
                                    tla.set_flag(l1a1_copy_start)

                            if k_l0 == 0:
                                if l1_buf_idx == c0:
                                    tla.wait_flag(l1b0_copy_end)
                                else:
                                    tla.wait_flag(l1b1_copy_end)
                            if l0_buf_idx == c0:
                                tla.wait_flag(l0b0_copy_start)
                            else:
                                tla.wait_flag(l0b1_copy_start)
                            tla.copy(l0_b, l1_b_l0)
                            if k_l0 == k_l0_count - 1:
                                if l1_buf_idx == c0:
                                    tla.set_flag(l1b0_copy_start)
                                else:
                                    tla.set_flag(l1b1_copy_start)

                            tla.set_flag(l0_copy_end)
                            tla.wait_flag(l0_copy_end)

                            unit_flag = (
                                0b11
                                if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                                else 0b10
                            )
                            init_c = True if k_l1 == 0 and k_l0 == 0 else False
                            tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0a0_copy_start)
                                tla.set_flag(l0b0_copy_start)
                            else:
                                tla.set_flag(l0a1_copy_start)
                                tla.set_flag(l0b1_copy_start)
                            l0_buf_idx = c1 - l0_buf_idx
                        l1_buf_idx = c1 - l1_buf_idx

                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)


def run(args: argparse.Namespace) -> int:
    from examples.end_to_end.common import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- groups=({args.groups}) mnk=({args.m},{args.n},{args.k}) "
        f"layout={args.layout_a}/{args.layout_b} "
        f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} ---"
    )
    torch.manual_seed(0)
    dtypes = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype_a = dtypes[args.dtype_a]
    dtype_b = dtypes[args.dtype_b]
    dtype_c = dtypes[args.dtype_c]

    tiling = TilingParams()
    current_ms = _average_current_m(args.m, args.groups, l1_m=tiling.l1_tm)
    prefix = _group_list_prefix(current_ms)

    a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
    b = torch.rand(args.groups * args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
    c = torch.rand(args.m, args.n, dtype=dtype_c, device="cpu") * 10.0 - 5.0
    ref = torch.zeros(args.m, args.n, dtype=torch.float32)
    offset = 0
    for g, current_m in enumerate(current_ms):
        if current_m > 0:
            ref[offset : offset + current_m] = (
                a[offset : offset + current_m].float()
                @ b[g * args.k : (g + 1) * args.k].float()
            )
        offset += current_m
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    a = (
        a.contiguous() if args.layout_a == "row" else a.permute(1, 0).contiguous()
    ).npu()
    b = (
        b.contiguous() if args.layout_b == "row" else b.permute(1, 0).contiguous()
    ).npu()
    c = c.contiguous().npu()
    group_list = torch.tensor(prefix, dtype=torch.int32).npu()

    a_tensor = create_tla_tensor(a, args.layout_a)
    b_tensor = create_tla_tensor(b, args.layout_b)
    c_tensor = create_tla_tensor(c, "row")
    gl_tensor = from_dlpack(
        group_list, layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)

    artifact = tla.compile(
        grouped_matmul_slice_m_kernel,
        a_tensor,
        b_tensor,
        gl_tensor,
        c_tensor,
        tiling,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_tensor, b_tensor, gl_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(c.detach().cpu(), ref, args.k)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f32")
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
