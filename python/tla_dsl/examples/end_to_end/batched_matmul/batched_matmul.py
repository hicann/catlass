# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host driver for batched matmul (Catlass example 01).

Semantics: for each batch ``b``, ``C[b] = A[b] @ B[b]`` with the same ``(m, n, k)``.
Storage matches C++ strides: A ``B*M*K``, B ``B*K*N``, C ``B*M*N``.
"""

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
# Device kernel + compile-time knobs (formerly batched_matmul_kernels.py)
# Host mutates the globals below before ``tla.compile``.
# ---------------------------------------------------------------------------
batch_count = 5
m = 256
n = 512
k = 1024

l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

# Match C++ GemmIdentityBlockSwizzle. Host may override OFFSET/DIRECTION.
# Default OFFSET=3 (ex01/ex67). When grid_* % OFFSET != 0, tail uses select.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1

DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float16
# Unit-flag fuses L0C→GM with the K-loop (matches C++ enableUnitFlag=true).
# Soft-flag path remains for hosts that set this False (e.g. multi-layout
# in-process sweeps where residual cube/FIX state can break a later launch).
ENABLE_UNIT_FLAG = True


@tla.kernel
def batched_matmul_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
) -> None:
    """Cube batched GEMM：对每个 batch 做 C = A @ B（L1/L0 双缓冲 + flag）。

    mem_a/b/c 为展平 2D（schema v4 dynamic GM；工作尺寸取自 origin_shape）：
      A: (batch_count * m, k)
      B: (batch_count * k, n)
      C: (batch_count * m, n)
    """
    c0 = 0
    c1 = 1

    # Dynamic GM: extents from launch memref, not host module globals.
    k_dim = mem_a.origin_shape[1]
    n_dim = mem_c.origin_shape[1]
    batch_cnt = mem_b.origin_shape[0] // k_dim
    m_dim = mem_a.origin_shape[0] // batch_cnt

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

    l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(l0_tm * l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    grid_m = (m_dim + l1_tm - 1) // l1_tm
    grid_n = (n_dim + l1_tn - 1) // l1_tn
    mn_blocks = grid_m * grid_n
    # 与 C++ 一致：coreLoops = batchCount * GetCoreLoops()
    total_blocks = batch_cnt * mn_blocks
    # Tail band width (equals SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - SWIZZLE_OFFSET * ((grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1)
    last_n_col = grid_n - SWIZZLE_OFFSET * ((grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1)
    # Hoist CeilDiv(band) / K-L1 trip count out of the MN-tile loop.
    tile_block_loop_m = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
    tile_block_loop_n = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
    k_l1_count = (k_dim + l1_tk - 1) // l1_tk

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

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for loop_idx in block_range:
            batch_idx = loop_idx // mn_blocks
            mn_linear = loop_idx % mn_blocks

            # GemmIdentityBlockSwizzle<SWIZZLE_OFFSET, SWIZZLE_DIRECTION>
            # DIRECTION stays host/compile-time (Zn when M>N); band tails are dynamic.
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_idx = mn_linear // (SWIZZLE_OFFSET * grid_n)
                in_tile = mn_linear % (SWIZZLE_OFFSET * grid_n)
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop_m - 1
                    else SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                # Serpentine: branchless (odd band flips N)
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                # Nz: bands along N, serpentine M (C++ when M<=N)
                tile_block_idx = mn_linear // (SWIZZLE_OFFSET * grid_m)
                in_tile = mn_linear % (SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop_n - 1
                    else SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # A/C: nested batch→core (needed when m_dim % l1_tm != 0).
            # B: one-level tile_view (batch along K tiles) — same as C++ batchOffset+GetTile.
            gm_a_batch = tla.tile_view(
                mem_a, tla.make_shape(m_dim, k_dim), tla.make_coord(batch_idx, c0)
            )
            gm_c_batch = tla.tile_view(
                mem_c, tla.make_shape(m_dim, n_dim), tla.make_coord(batch_idx, c0)
            )

            gm_a_by_core = tla.tile_view(
                gm_a_batch, tla.make_shape(l1_tm, k_dim), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(k_dim, l1_tn), tla.make_coord(batch_idx, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c_batch,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            if not ENABLE_UNIT_FLAG:
                tla.wait_flag(fix_done)
            for k_l1 in k_l1_range:
                gm_a_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk), tla.make_coord(c0, k_l1)
                )
                gm_b_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn), tla.make_coord(k_l1, c0)
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

                k_l0_count = (l1_a.origin_shape[1] + l0_tk - 1) // l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)
                for k_l0 in k_l0_range:
                    l1_a_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk), tla.make_coord(c0, k_l0)
                    )
                    l1_b_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn), tla.make_coord(k_l0, c0)
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
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(
                        l0_c, l0_a, l0_b,
                        init_c=init_c,
                        unit_flag=unit_flag,
                        compute_order=tla.params.ComputeOrder.N_FIRST,
                    )
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_copy_start)
                        tla.set_flag(l0b0_copy_start)
                    else:
                        tla.set_flag(l0a1_copy_start)
                        tla.set_flag(l0b1_copy_start)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            if not ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                tla.copy(gm_c_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                tla.copy(
                    gm_c_by_core, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11)
                )

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
        tla.wait_flag(fix_done)



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


def _apply_problem_size(
    batch_val: int, m_val: int, n_val: int, k_val: int
) -> None:
    global batch_count, m, n, k, SWIZZLE_DIRECTION
    if min(batch_val, m_val, n_val, k_val) <= 0:
        raise ValueError(
            f"batch,m,n,k must be positive; got "
            f"batch={batch_val} m={m_val} n={n_val} k={k_val}"
        )
    batch_count, m, n, k = batch_val, m_val, n_val, k_val
    # Match C++ ex01/ex67: Zn (0) when M>N, else Nz (1). Compile-time only.
    SWIZZLE_DIRECTION = 0 if m_val > n_val else 1


def _apply_unit_flag_policy(layout_a: LayoutChoice, layout_b: LayoutChoice) -> None:
    """Prefer unit-flag for row/row (C++ path); soft-flag otherwise for stability."""
    global ENABLE_UNIT_FLAG
    ENABLE_UNIT_FLAG = layout_a == "row" and layout_b == "row"


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
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


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
    idx = close.logical_not().nonzero(as_tuple=False)[0]
    coords = [int(v) for v in idx]
    gather = actual
    gather_e = expected
    for c in coords:
        gather = gather[c]
        gather_e = gather_e[c]
    av = float(gather.item())
    ev = float(gather_e.item())
    return {
        "index": coords,
        "actual": av,
        "expected": ev,
        "abs_err": abs(av - ev),
        "scaled_err": abs(av - ev) / max(1.0, abs(ev)),
    }


def _batched_golden(torch: Any, a: Any, b: Any) -> Any:
    """a,b: (B,M,K), (B,K,N) → (B,M,N) float32."""
    return torch.matmul(a.to(torch.float32), b.to(torch.float32))


def run_single_case(
    args: argparse.Namespace,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> int:
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    _apply_unit_flag_policy(layout_a, layout_b)
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
            torch.rand(
                (batch_count, m, k), generator=gen, device=device, dtype=torch.float32
            )
            * 10.0
            - 5.0
        ).to(torch_dtype_a)
    )
    b = (
        (
            torch.rand(
                (batch_count, k, n), generator=gen, device=device, dtype=torch.float32
            )
            * 10.0
            - 5.0
        ).to(torch_dtype_b)
    )
    c = torch.full(
        (batch_count, m, n), args.sentinel, dtype=torch_dtype_c, device=device
    )

    expected_f32 = _batched_golden(torch, a, b)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32.to(torch_dtype_c).to(torch.float32)
    else:
        expected = expected_f32
    rtol = _comparison_rtol(args)
    atol = _comparison_atol(dtype_c, args)

    # Flatten to 2D for TLA (stride = one batch matrix).
    a_flat = a.reshape(batch_count * m, k)
    b_flat = b.reshape(batch_count * k, n)
    c_flat = c.reshape(batch_count * m, n)

    tla_a = _create_tla_tensor(a_flat, layout_a)
    tla_b = _create_tla_tensor(b_flat, layout_b)
    tla_c = _create_tla_tensor(c_flat, "row")

    artifact = tla.compile(
        batched_matmul_kernel,
        tla_a,
        tla_b,
        tla_c,
        **_runtime_kwargs(args),
    )
    print(f"cache_key={artifact.cache_key}")
    artifact(tla_a, tla_b, tla_c, block_dim=args.block)
    torch.npu.synchronize()

    actual = c.to(torch.float32)
    sentinel_f32 = torch.full_like(actual, args.sentinel)
    unchanged = torch.isclose(actual, sentinel_f32, rtol=0.0, atol=atol)
    expected_match = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    first_mismatch = _first_mismatch_torch(actual, expected, rtol=rtol, atol=atol)

    print(
        "compile_ok=True "
        f"host=torch_npu layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c} "
        f"batch={batch_count} m={m} n={n} k={k}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"data=uniform([-5,5]) data_seed={args.data_seed}")
    print(f"isclose rtol={rtol} atol={atol}")
    print(f"C unchanged? {bool(unchanged.all())}")
    print(f"C equals batched golden? {bool(expected_match.all())}")
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


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"batch={batch_count}",
                    f"m={m}",
                    f"n={n}",
                    f"k={k}",
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
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batched matmul (example 01). Single cube launch over "
            "batch*MN tiles; A/B/C packed with per-batch strides m*k / k*n / m*n."
        )
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--batch", type=int, default=batch_count)
    parser.add_argument("--m", type=int, default=m)
    parser.add_argument("--n", type=int, default=n)
    parser.add_argument("--k", type=int, default=k)
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
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _apply_problem_size(args.batch, args.m, args.n, args.k)
    if not args.all_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
