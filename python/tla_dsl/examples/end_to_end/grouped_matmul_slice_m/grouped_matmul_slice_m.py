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
from pathlib import Path
from typing import Any, Literal

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

# ---------------------------------------------------------------------------
# Device kernel + compile-time knobs (formerly grouped_matmul_slice_m_kernels.py)
# Host mutates the globals below before ``tla.compile``.
# ---------------------------------------------------------------------------
M_DIM = 1024
N_DIM = 256
K_DIM = 256
GROUPS = 4

L1_TM = 256
L1_TN = 256
L1_TK = 256
L0_TM = 256
L0_TN = 256
L0_TK = 64

# Match C++ GemmIdentityBlockSwizzle<3, *> (example 60). Host sets DIRECTION
# before compile (Zn=0 when m/G >= n, else Nz=1). Keep as const_expr — a
# runtime if in the tile loop regresses aic_time under dynamic GM.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 0

DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float16
ENABLE_UNIT_FLAG = True


@tla.kernel
def grouped_matmul_slice_m_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    group_list: tla.Tensor,
    mem_c: tla.Tensor,
) -> None:
    c0 = 0
    c1 = 1
    # Runtime extents from dynamic GM (host marks mark_layout_dynamic /
    # mark_compact_shape_dynamic). Module-level M_DIM/N_DIM/K_DIM/GROUPS are
    # only for host tensor allocation and compile-time type_args.
    n_dim = mem_c.origin_shape[1]
    k_dim = mem_a.origin_shape[1]
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
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    fix_done = tla.flag("fix_done", tla.arch.FIX, tla.arch.CUBE)

    l1a0_ptr = tla.allocate(L1_TM * L1_TK, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(L1_TM * L1_TK, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(L1_TK * L1_TN, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(L1_TK * L1_TN, DTYPE_B, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(L0_TM * L0_TK, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(L0_TM * L0_TK, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(L0_TK * L0_TN, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(L0_TK * L0_TN, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0_TM * L0_TN, DTYPE_C, tla.AddressSpace.l0c, 512)

    grid_n = (n_dim + L1_TN - 1) // L1_TN

    with tla.cube():
        tla.set_flag(l1a0_copy_start)
        tla.set_flag(l1a1_copy_start)
        tla.set_flag(l1b0_copy_start)
        tla.set_flag(l1b1_copy_start)
        tla.set_flag(l0a0_copy_start)
        tla.set_flag(l0a1_copy_start)
        tla.set_flag(l0b0_copy_start)
        tla.set_flag(l0b1_copy_start)
        tla.set_flag(fix_done)

        l1_buf_idx = c0
        l0_buf_idx = c0
        # Carry core offset across groups (C++ startCoreIdx) for load balance.
        start_core_idx = c0
        core_num = tla.arch.block_dim()
        core_idx = tla.arch.block_idx()

        for g in tla.range(c0, group_cnt, c1):
            # 1D GM Int32 prefix: currentM = end - start (no Python list).
            m_start = group_list[g]
            m_end = group_list[g + 1]
            current_m = m_end - m_start

            if current_m > 0:
                # L1-aligned groups: tile index = element offset / L1_TM.
                m_tile_base = m_start // L1_TM
                grid_m = current_m // L1_TM
                mn_blocks = grid_m * grid_n

                gm_b_group = tla.tile_view(
                    mem_b, tla.make_shape(k_dim, n_dim), tla.make_coord(g, c0)
                )

                # Rotate which core owns loop 0 of this group.
                # (core_idx - start_core_idx) mod core_num — avoid dynamic if new defs.
                start_loop = (core_idx + core_num - start_core_idx) % core_num
                # Tail band width for GemmIdentityBlockSwizzle (equals OFFSET when exact).
                last_n_row = grid_m - SWIZZLE_OFFSET * (
                    (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
                )
                last_n_col = grid_n - SWIZZLE_OFFSET * (
                    (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
                )
                block_range = tla.range(start_loop, mn_blocks, core_num)
                for loop_idx in block_range:
                    # GemmIdentityBlockSwizzle<SWIZZLE_OFFSET, SWIZZLE_DIRECTION>
                    # DIRECTION is host/compile-time (ex60: Zn when m/G >= n_dim).
                    if tla.const_expr(SWIZZLE_DIRECTION == 0):
                        # Zn: bands along M, serpentine N
                        tile_block_idx = loop_idx // (SWIZZLE_OFFSET * grid_n)
                        in_tile = loop_idx % (SWIZZLE_OFFSET * grid_n)
                        tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                        n_row = (
                            last_n_row
                            if tile_block_idx == tile_block_loop - 1
                            else SWIZZLE_OFFSET
                        )
                        block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_row
                        block_col = in_tile // n_row
                        odd = tile_block_idx % 2
                        block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
                    else:
                        # Nz: bands along N, serpentine M
                        tile_block_idx = loop_idx // (SWIZZLE_OFFSET * grid_m)
                        in_tile = loop_idx % (SWIZZLE_OFFSET * grid_m)
                        tile_block_loop = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                        n_col = (
                            last_n_col
                            if tile_block_idx == tile_block_loop - 1
                            else SWIZZLE_OFFSET
                        )
                        block_row = in_tile // n_col
                        block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                        odd = tile_block_idx % 2
                        block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

                    abs_row = m_tile_base + block_row

                    gm_a_by_core = tla.tile_view(
                        mem_a, tla.make_shape(L1_TM, k_dim), tla.make_coord(abs_row, c0)
                    )
                    gm_b_by_core = tla.tile_view(
                        gm_b_group, tla.make_shape(k_dim, L1_TN), tla.make_coord(c0, block_col)
                    )
                    gm_c_by_core = tla.tile_view(
                        mem_c,
                        tla.make_shape(L1_TM, L1_TN),
                        tla.make_coord(abs_row, block_col),
                    )

                    k_block = gm_a_by_core.origin_shape[1]
                    k_l1_count = (k_block + L1_TK - 1) // L1_TK
                    k_l1_range = tla.range(c0, k_l1_count, c1)
                    # Soft-pipeline K-L1 like C++ BlockMmadPingpongTla:
                    # load tile0, then each iteration prefetches the next tile
                    # into the alternate L1 buffer while computing the current.

                    l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                    if not ENABLE_UNIT_FLAG:
                        tla.wait_flag(fix_done)

                    # Prologue: GM→L1 for k_l1 = 0 into the current buffer.
                    gm_a_l1 = tla.tile_view(
                        gm_a_by_core,
                        tla.make_shape(L1_TM, L1_TK),
                        tla.make_coord(c0, c0),
                    )
                    gm_b_l1 = tla.tile_view(
                        gm_b_by_core,
                        tla.make_shape(L1_TK, L1_TN),
                        tla.make_coord(c0, c0),
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

                    for k_l1 in k_l1_range:
                        l1_next = c1 - l1_buf_idx

                        # Prefetch next L1 A/B into the free buffer (overlaps L0).
                        if k_l1 < k_l1_count - 1:
                            k_next = k_l1 + 1
                            gm_a_next = tla.tile_view(
                                gm_a_by_core,
                                tla.make_shape(L1_TM, L1_TK),
                                tla.make_coord(c0, k_next),
                            )
                            gm_b_next = tla.tile_view(
                                gm_b_by_core,
                                tla.make_shape(L1_TK, L1_TN),
                                tla.make_coord(k_next, c0),
                            )
                            l1_a_next = tla.make_tensor_like(
                                l1a0_ptr if (l1_next == c0) else l1a1_ptr, gm_a_next
                            )
                            l1_b_next = tla.make_tensor_like(
                                l1b0_ptr if (l1_next == c0) else l1b1_ptr, gm_b_next
                            )
                            if l1_next == c0:
                                tla.wait_flag(l1a0_copy_start)
                            else:
                                tla.wait_flag(l1a1_copy_start)
                            tla.copy(l1_a_next, gm_a_next)
                            if l1_next == c0:
                                tla.set_flag(l1a0_copy_end)
                            else:
                                tla.set_flag(l1a1_copy_end)
                            if l1_next == c0:
                                tla.wait_flag(l1b0_copy_start)
                            else:
                                tla.wait_flag(l1b1_copy_start)
                            tla.copy(l1_b_next, gm_b_next)
                            if l1_next == c0:
                                tla.set_flag(l1b0_copy_end)
                            else:
                                tla.set_flag(l1b1_copy_end)

                        # Compute current L1 tile already resident in l1_buf_idx.
                        gm_a_l1 = tla.tile_view(
                            gm_a_by_core,
                            tla.make_shape(L1_TM, L1_TK),
                            tla.make_coord(c0, k_l1),
                        )
                        gm_b_l1 = tla.tile_view(
                            gm_b_by_core,
                            tla.make_shape(L1_TK, L1_TN),
                            tla.make_coord(k_l1, c0),
                        )
                        l1_a = tla.make_tensor_like(
                            l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                        )
                        l1_b = tla.make_tensor_like(
                            l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                        )

                        k_l0_count = (l1_a.origin_shape[1] + L0_TK - 1) // L0_TK
                        k_l0_range = tla.range(c0, k_l0_count, c1)
                        for k_l0 in k_l0_range:
                            l1_a_l0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(L0_TM, L0_TK),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_l0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(L0_TK, L0_TN),
                                tla.make_coord(k_l0, c0),
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

                            unit_flag = 0
                            if ENABLE_UNIT_FLAG:
                                if (k_l1 == k_l1_count - 1) and (
                                    k_l0 == k_l0_count - 1
                                ):
                                    unit_flag = 0b11
                                else:
                                    unit_flag = 0b10
                            init_c = True if k_l1 == 0 and k_l0 == 0 else False
                            tla.mmad(
                                l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag
                            )
                            if l0_buf_idx == c0:
                                tla.set_flag(l0a0_copy_start)
                                tla.set_flag(l0b0_copy_start)
                            else:
                                tla.set_flag(l0a1_copy_start)
                                tla.set_flag(l0b1_copy_start)
                            l0_buf_idx = c1 - l0_buf_idx
                        l1_buf_idx = l1_next

                    if not ENABLE_UNIT_FLAG:
                        tla.set_flag(mmad_done)
                        tla.wait_flag(mmad_done)
                        tla.copy(gm_c_by_core, l0_c)
                        tla.set_flag(fix_done)
                    else:
                        tla.copy(
                            gm_c_by_core,
                            l0_c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )

                start_core_idx = (start_core_idx + mn_blocks) % core_num

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
        tla.wait_flag(fix_done)

# Host-facing aliases (kept in sync by ``_apply_problem_size``).
m = M_DIM
n = N_DIM
k = K_DIM
problem_count = GROUPS



def _parse_layout_choice(name: str) -> LayoutChoice:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, LayoutChoice] = {
        "row": "row",
        "rowmajor": "row",
        "col": "col",
        "columnmajor": "col",
        "colmajor": "col",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown layout {name!r}; expected one of row, col"
        )
    return mapping[key]


def _gm_layout_tag(choice: LayoutChoice) -> Any:
    if choice == "row":
        return tla.arch.RowMajor
    return tla.arch.ColumnMajor


def _parse_elem_dtype(name: str) -> ElemDType:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, ElemDType] = {
        "f16": "f16",
        "float16": "f16",
        "fp16": "f16",
        "half": "f16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f32": "f32",
        "float32": "f32",
        "fp32": "f32",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown dtype {name!r}; expected f16, bf16, or f32"
        )
    return mapping[key]


def _tla_elem_dtype(token: ElemDType) -> Any:
    if token == "f16":
        return tla.Float16
    if token == "bf16":
        return tla.BFloat16
    return tla.Float32


def _validate_mmad_dtype_triple(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    if dtype_a != dtype_b:
        raise ValueError("dtype-a and dtype-b must match for tla.mmad.")
    allowed = {
        ("f16", "f16", "f32"),
        ("f16", "f16", "f16"),
        ("bf16", "bf16", "f32"),
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
    }
    if (dtype_a, dtype_b, dtype_c) not in allowed:
        raise ValueError(
            "unsupported dtype triple; allowed: "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32"
        )


def _apply_kernel_dtypes(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    global DTYPE_A, DTYPE_B, DTYPE_C, DTYPE_GM_C
    DTYPE_A = _tla_elem_dtype(dtype_a)
    DTYPE_B = _tla_elem_dtype(dtype_b)
    DTYPE_GM_C = _tla_elem_dtype(dtype_c)
    DTYPE_C = tla.Float32


def _apply_problem_size(m_val: int, n_val: int, k_val: int, g_val: int) -> None:
    global m, n, k, problem_count, M_DIM, N_DIM, K_DIM, GROUPS, SWIZZLE_DIRECTION
    if min(m_val, n_val, k_val, g_val) <= 0:
        raise ValueError(
            f"m,n,k,groups must be positive; got m={m_val} n={n_val} k={k_val} g={g_val}"
        )
    M_DIM, N_DIM, K_DIM, GROUPS = m_val, n_val, k_val, g_val
    # Match C++ ex60: Zn (0) when m/problemCount >= n, else Nz (1).
    SWIZZLE_DIRECTION = 0 if (m_val // g_val) >= n_val else 1
    m, n, k, problem_count = m_val, n_val, k_val, g_val


def _group_list_prefix(current_ms: tuple[int, ...]) -> tuple[int, ...]:
    """Length G+1 prefix with leading 0: currentM[g] = prefix[g+1] - prefix[g]."""
    out = [0]
    for current_m in current_ms:
        out.append(out[-1] + current_m)
    return tuple(out)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Host-side tensors require PyTorch.") from exc
    return torch


def _torch_dtype(token: ElemDType) -> Any:
    torch = _require_torch()
    if token == "f16":
        return torch.float16
    if token == "bf16":
        return torch.bfloat16
    return torch.float32


def _require_torch_npu(device_id: int) -> Any:
    torch = _require_torch()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("This example requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()


def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    """Wrap one device buffer as a dynamic-layout ``tla.Tensor`` (schema v4)."""
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


def generate_average_current_m(
    m_val: int, g_val: int, *, l1_tm: int
) -> tuple[int, ...]:
    """Partition ``[0, m_val)`` into ``g_val`` nearly-equal P0-aligned groups.

    Old code used ``m // g`` for every group and **dropped** ``m % g`` rows, so
    ``sum(current_ms) < m`` whenever ``m`` was not divisible by ``g``.

    P0 needs heights that are multiples of ``l1_tm``. Split the ``m // l1_tm``
    tiles as evenly as possible: the first ``n_tiles % g`` groups get one extra
    tile. Empty groups appear only when ``g > n_tiles``.
    """
    if g_val <= 0:
        raise ValueError(f"groups must be positive; got {g_val}")
    if m_val % l1_tm != 0:
        raise ValueError(
            f"P0 average groups require m % l1_tm == 0; got m={m_val}, l1_tm={l1_tm}"
        )
    n_tiles = m_val // l1_tm
    base, rem = divmod(n_tiles, g_val)
    out = tuple(((base + 1) if i < rem else base) * l1_tm for i in range(g_val))
    if sum(out) != m_val:
        raise RuntimeError(
            f"internal error: average groups sum to {sum(out)}, expected m={m_val}"
        )
    return out


def generate_random_current_m(
    m_val: int, g_val: int, *, seed: int = 0, l1_tm: int
) -> tuple[int, ...]:
    """Randomly partition ``[0, m_val)`` into ``g_val`` groups for P0.

    Old bug: sampled ``g_val`` endpoints in ``[0, m]`` and diffed them, so
    ``sum(out) == prefix[-1]`` (often ``< m_val``) and heights were almost never
    ``l1_tm``-aligned — ``_validate_l1_aligned_groups`` would reject nearly always.

    P0 fix: only cut on L1_M tile boundaries. Distribute ``m_val // l1_tm`` tiles
    across ``g_val`` bags (empty groups allowed, same as C++ zero-height groups).
    """
    import random

    if g_val <= 0:
        raise ValueError(f"groups must be positive; got {g_val}")
    if m_val % l1_tm != 0:
        raise ValueError(
            f"P0 random groups require m % l1_tm == 0; got m={m_val}, l1_tm={l1_tm}"
        )
    n_tiles = m_val // l1_tm
    rng = random.Random(seed)
    # g_val-1 cut points on the tile grid [0, n_tiles], then force ends 0 and n_tiles.
    cuts = sorted(rng.randint(0, n_tiles) for _ in range(g_val - 1))
    points = [0, *cuts, n_tiles]
    out = tuple((points[i + 1] - points[i]) * l1_tm for i in range(g_val))
    if sum(out) != m_val:
        raise RuntimeError(
            f"internal error: random groups sum to {sum(out)}, expected m={m_val}"
        )
    return out


def _validate_l1_aligned_groups(
    current_ms: tuple[int, ...], l1_tm: int, *, expected_m: int | None = None
) -> None:
    total = 0
    for i, current_m in enumerate(current_ms):
        if current_m < 0:
            raise ValueError(f"negative currentM at group {i}")
        if current_m == 0:
            continue
        if total % l1_tm != 0:
            raise ValueError(
                f"P0 requires group start % l1_tm == 0; group {i} start={total}"
            )
        if current_m % l1_tm != 0:
            raise ValueError(
                f"P0 requires currentM % l1_tm == 0; group {i} currentM={current_m}, "
                f"l1_tm={l1_tm}."
            )
        total += current_m
    # Empty leading/middle groups skip the loop body but still occupy no rows;
    # include zeros in the coverage check via sum(current_ms).
    covered = sum(current_ms)
    if expected_m is not None and covered != expected_m:
        raise ValueError(
            f"group heights must cover all m rows; sum(currentM)={covered}, m={expected_m}, "
            f"currentM={current_ms}"
        )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


# Relative tol for torch.isclose (Catlass CompareData uses 1/128 for large K).
DEFAULT_RTOL = 1.0 / 128.0


def _comparison_rtol(args: argparse.Namespace) -> float:
    return float(args.rtol)


def _comparison_atol(dtype_c: ElemDType, args: argparse.Namespace) -> float:
    """Same absolute floor as ``basic_matmul._comparison_atol``."""
    if dtype_c in ("f16", "bf16"):
        return max(float(args.atol), 5e-3)
    return float(args.atol)


def _first_mismatch_torch(
    actual: Any, expected: Any, *, rtol: float, atol: float
) -> dict[str, Any] | None:
    torch = _require_torch()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    if bool(close.all()):
        return None
    flat = close.logical_not().nonzero(as_tuple=False)[0]
    row, col = (int(v) for v in flat)
    av = float(actual[row, col].item())
    ev = float(expected[row, col].item())
    return {
        "index": [row, col],
        "actual": av,
        "expected": ev,
        "abs_err": abs(av - ev),
        "scaled_err": abs(av - ev) / max(1.0, abs(ev)),
    }


def _grouped_golden(
    torch: Any,
    a: Any,
    b_packed: Any,
    current_ms: tuple[int, ...],
    *,
    n_val: int,
    k_val: int,
) -> Any:
    c = torch.zeros((a.shape[0], n_val), dtype=torch.float32, device=a.device)
    offset = 0
    for g, current_m in enumerate(current_ms):
        if current_m > 0:
            a_g = a[offset : offset + current_m].to(torch.float32)
            b_g = b_packed[g * k_val : (g + 1) * k_val].to(torch.float32)
            c[offset : offset + current_m] = a_g @ b_g
        offset += current_m
    return c


def run_single_case(
    args: argparse.Namespace,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    current_ms: tuple[int, ...],
) -> int:
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    torch = _require_torch_npu(args.device)
    device = "npu"
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)

    # Inputs: uniform in [-5, 5] (matches basic_mmad / Catlass FillRandomData range).
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.data_seed))
    a = (
        (
            torch.rand((m, k), generator=gen, device=device, dtype=torch.float32)
            * 10.0
            - 5.0
        ).to(torch_dtype_a)
    )
    b_packed = (
        (
            torch.rand(
                (problem_count * k, n), generator=gen, device=device, dtype=torch.float32
            )
            * 10.0
            - 5.0
        ).to(torch_dtype_b)
    )
    c = torch.full((m, n), args.sentinel, dtype=torch_dtype_c, device=device)
    prefix = _group_list_prefix(current_ms)
    group_list = torch.tensor(prefix, dtype=torch.int32, device=device)

    expected_f32 = _grouped_golden(
        torch, a, b_packed, current_ms, n_val=n, k_val=k
    )
    valid_rows = sum(current_ms)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32[:valid_rows].to(torch_dtype_c).to(torch.float32)
    else:
        expected = expected_f32[:valid_rows]
    rtol = _comparison_rtol(args)
    atol = _comparison_atol(dtype_c, args)

    # Single launch: A, B, group_list, C (matches Ascend950 torch op order).
    tla_a = _create_tla_tensor(a, layout_a)
    tla_b = _create_tla_tensor(b_packed, layout_b)
    tla_gl = from_dlpack(
        group_list.contiguous(), layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)
    tla_c = _create_tla_tensor(c, "row")
    artifact = tla.compile(
        grouped_matmul_slice_m_kernel,
        tla_a,
        tla_b,
        tla_gl,
        tla_c,
        **_runtime_kwargs(args),
    )
    artifact(tla_a, tla_b, tla_gl, tla_c, block_dim=args.block)
    torch.npu.synchronize()

    actual = c[:valid_rows].to(torch.float32)
    sentinel_f32 = torch.full_like(actual, args.sentinel)
    unchanged = torch.isclose(actual, sentinel_f32, rtol=0.0, atol=atol)
    expected_match = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    first_mismatch = _first_mismatch_torch(actual, expected, rtol=rtol, atol=atol)

    print(
        "compile_ok=True "
        f"host=torch_npu layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c} "
        f"groups={problem_count} launches=1"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print(f"cache_key={artifact.cache_key}")
    print("launch_ok=True")
    print(f"GROUP_CURRENT_M={current_ms}")
    print(f"GROUP_LIST_PREFIX={prefix}")
    print(f"data=uniform([-5,5]) data_seed={args.data_seed}")
    print(f"isclose rtol={rtol} atol={atol}")
    print(f"C[:{valid_rows}] unchanged? {bool(unchanged.all())}")
    print(f"C equals grouped golden? {bool(expected_match.all())}")
    print(f"C changed count={int((~unchanged).sum().item())}")
    print(f"first mismatch={first_mismatch}")
    return 0 if first_mismatch is None else 1


MMAD_DTYPE_TRIPLES: tuple[tuple[ElemDType, ElemDType, ElemDType], ...] = (
    ("f16", "f16", "f32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "f32"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
)


def _layout_pairs(
    args: argparse.Namespace,
) -> list[tuple[LayoutChoice, LayoutChoice]]:
    if args.all_layouts:
        return [(la, lb) for la in ("row", "col") for lb in ("row", "col")]
    return [(args.layout_a, args.layout_b)]


def _dtype_triples(
    args: argparse.Namespace,
) -> list[tuple[ElemDType, ElemDType, ElemDType]]:
    if args.all_dtypes:
        return list(MMAD_DTYPE_TRIPLES)
    return [(args.dtype_a, args.dtype_b, args.dtype_c)]


def _resolve_current_ms(args: argparse.Namespace) -> tuple[int, ...]:
    if args.group_mode == "average":
        current_ms = generate_average_current_m(
            m, problem_count, l1_tm=L1_TM
        )
    else:
        current_ms = generate_random_current_m(
            m, problem_count, seed=args.group_seed, l1_tm=L1_TM
        )
    _validate_l1_aligned_groups(current_ms, L1_TM, expected_m=m)
    return current_ms


def run(args: argparse.Namespace, current_ms: tuple[int, ...]) -> int:
    tla.initialize(device=args.device)
    try:
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"m={m}",
                    f"n={n}",
                    f"k={k}",
                    f"groups={problem_count}",
                    f"dtype_a={dtype_a}",
                    f"dtype_b={dtype_b}",
                    f"dtype_c={dtype_c}",
                    f"layout_a={layout_a}",
                    f"layout_b={layout_b}",
                    "---",
                )
                failed += run_single_case(
                    args,
                    layout_a,
                    layout_b,
                    dtype_a,
                    dtype_b,
                    dtype_c,
                    current_ms,
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grouped matmul slice-M. Single launch over all M-groups; "
            "device reads Int32 group_list prefix (len G+1). B packed as (G*K, N). "
            "P0 requires L1_M-aligned group sizes."
        )
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--k", type=int, default=K_DIM)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--sentinel", type=float, default=-7.0)
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for torch.isclose (default 1/128).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance base; f16/bf16 use max(atol, 5e-3) like basic_matmul.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="RNG seed for uniform inputs in [-5, 5].",
    )
    parser.add_argument("--layout-a", type=_parse_layout_choice, default="row")
    parser.add_argument("--layout-b", type=_parse_layout_choice, default="row")
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run all four (layout-a, layout-b) combinations sequentially.",
    )
    parser.add_argument("--dtype-a", type=_parse_elem_dtype, default="f16")
    parser.add_argument("--dtype-b", type=_parse_elem_dtype, default="f16")
    parser.add_argument("--dtype-c", type=_parse_elem_dtype, default="f16")
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help=(
            "Run all supported (dtype-a, dtype-b, dtype-c) triples sequentially "
            "(with the chosen layout pair or all layout pairs when --all-layouts is set)."
        ),
    )
    parser.add_argument(
        "--group-mode", choices=("average", "random"), default="random"
    )
    parser.add_argument("--group-seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _apply_problem_size(args.m, args.n, args.k, args.groups)
    current_ms = _resolve_current_ms(args)
    print(f"current_ms: {current_ms}")
    print(f"group_list_prefix: {_group_list_prefix(current_ms)}")
    if not args.all_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    return run(args, current_ms)


if __name__ == "__main__":
    raise SystemExit(main())
