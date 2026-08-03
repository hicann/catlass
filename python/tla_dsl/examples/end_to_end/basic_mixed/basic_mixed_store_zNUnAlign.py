# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end mixed kernel using the ``zNUnAlign`` layout tag.

Mirrors ``basic_mixed_store_zN.py`` but stores the UB tile through a
``zNUnAlign`` dest tensor (``BlockStoreParams`` with ``block_stride``).
Unlike ``zN``, ``zNUnAlign`` does not fractal-block the M axis, so the dest
leaf[0] is the runtime row count and stride[1] is runtime-varying -- this is
the layout to use when M is not a multiple of the fractal factor (e.g. 60).

M_DIM / N_DIM / K_DIM are exposed as CLI arguments (``--m`` / ``--n``
/ ``--k``) so the GM problem shape can be varied without editing source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catlass as tla
from catlass.params import BlockStoreParams

def ceil_div(a:int, b:int):
    return (a + b - 1) // b

# Default GM problem shape (overridable via CLI). M_DIM=60 is intentionally not
# a multiple of the fractal factor (16) to exercise the zNUnAlign M axis.
DEFAULT_M_DIM = 60
DEFAULT_N_DIM = 64
DEFAULT_K_DIM = 128

# Tile / blocking factors consumed by the kernel (compile-time constants).
M_L1 = 64
N_L1 = 64
K_L1 = 128

ELE_NUM_PER_BLK = 32 // 4
UB_A_SIZE = M_L1 // 2 * K_L1
UB_A_ZN_SIZE = M_L1 // 2 * K_L1

VF_LEN = 256 // 4

L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1

L0A_SIZE = M_L1 * K_L1
L0B_SIZE = K_L1 * N_L1
L0C_SIZE = M_L1 * N_L1

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache" / "basic_mixed_store_zNUnAlign"


@tla.kernel
def basic_mixed_store_zNUnAlign(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1b_done = tla.flag("l1b_done", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)

    store_done = tla.flag("store_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub2l1_done = tla.cross_flag("ub2l1_done")

    l1a_ptr = tla.allocate(L1A_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_SIZE, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_SIZE, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_SIZE, tla.Float32, tla.AddressSpace.l0c, 512)

    ub_a_ptr = tla.allocate(UB_A_SIZE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_a_zN_ptr = tla.allocate(UB_A_ZN_SIZE, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        # 提前copy l1b/l0b
        tla.copy(l1_b, gm_b)
        tla.set_flag(l1b_done)

        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0)
        )
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        tla.wait_flag(l1b_done)
        tla.copy(l0_b, l1_b_l0)

        # l1a and l0a
        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)

        tla.cross_core_wait_flag(ub2l1_done, tla.arch.MTE1) # wait uba->l1a
        tla.copy(l0_a, l1_a_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        gm_c = tla.tile_view(out, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)
        tla.copy(gm_c, l0_c)

        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_a_tile = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_a = tla.tile_view(
            gm_a_tile,
            tla.make_shape(gm_a_tile.origin_shape[0]//2, K_L1),
            tla.make_coord(vec_idx, 0)
        )

        ub_a_tile = tla.make_tensor(
            ub_a_ptr,
            tla.make_layout(tla.make_shape(M_L1//2, K_L1), tla.make_stride(K_L1, 1)),
            tla.make_coord(0, 0)
        )
        ub_a = tla.tile_view(
            ub_a_tile,
            tla.make_shape(gm_a.origin_shape[0], gm_a.origin_shape[1]),
            tla.make_coord(0, 0)
        )

        tla.copy(ub_a, gm_a)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        # zNUnAlign dest: M axis is not fractal-blocked, so leaf[0] is the runtime
        # row count and stride[1]/stride[3] are runtime-varying (rows * ele_num_per_c0).
        ub_a_zN_full = tla.make_tensor_like(ub_a_zN_ptr, ub_a_tile, tla.arch.zNUnAlign)
        ub_a_zN = tla.tile_view(
            ub_a_zN_full,
            tla.make_shape(gm_a.origin_shape[0], gm_a.origin_shape[1]),
            tla.make_coord(0, 0)
        )

        vf_row_loops = ub_a.origin_shape[0]
        vf_col_loops = ceil_div(K_L1, VF_LEN)
        block_stride = ub_a_zN.stride[1][1] // ub_a_zN.shape[1][0]
        with tla.vec.func(mode="simd"):
            for row_tile_idx in tla.range(vf_row_loops):
                for col_tile_idx in tla.range(vf_col_loops):
                    a_chunk = tla.tile_view(
                        ub_a,
                        tla.make_shape(1, VF_LEN),
                        tla.make_coord(row_tile_idx, col_tile_idx)
                    )
                    a_zN_chunk = tla.tile_view(
                        ub_a_zN,
                        tla.make_shape(1, VF_LEN),
                        tla.make_coord(row_tile_idx, col_tile_idx)
                    )
                    a_zN_chunk.store(a_chunk.load(), params=BlockStoreParams(block_stride=block_stride))

        tla.set_flag(store_done)
        tla.wait_flag(store_done)

        l1_a_tile = tla.make_tensor_like(l1a_ptr, gm_a_tile, tla.arch.zN)
        l1_a = tla.tile_view(
            l1_a_tile,
            tla.make_shape(l1_a_tile.origin_shape[0]//2, K_L1),
            tla.make_coord(vec_idx, 0)
        )

        # UB zNUnAlign -> L1 zN copy (reuses the copy_ub_zN_to_l1_zN route).
        tla.copy(l1_a, ub_a_zN)
        tla.cross_core_set_flag(ub2l1_done, tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


def _compile_only_type_args(m_dim: int, n_dim: int, k_dim: int) -> tuple[Any, Any, Any]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        lhs_shape = tla.make_shape(m_dim, k_dim)
        rhs_shape = tla.make_shape(k_dim, n_dim)
        out_shape = tla.make_shape(m_dim, n_dim)
        out_stride = tla.make_stride(n_dim, 1)
        return (
            tla.Tensor(
                lhs_shape,
                tla.Float32,
                origin_shape=lhs_shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(k_dim, 1),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                rhs_shape,
                tla.Float32,
                origin_shape=rhs_shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(n_dim, 1),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                out_shape,
                tla.Float32,
                origin_shape=out_shape,
                coord=tla.make_coord(0, 0),
                stride=out_stride,
                layout_tag=tla.arch.RowMajor,
            ),
        )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def dump_tlair(args: argparse.Namespace) -> str:
    return basic_mixed_store_zNUnAlign.dump_mlir(
        type_args=_compile_only_type_args(args.m_dim, args.n_dim, args.k_dim)
    )


def build_only(args: argparse.Namespace) -> int:
    artifact = tla.compile(
        basic_mixed_store_zNUnAlign,
        mlir_print_ir_after_all=True,
        *_compile_only_type_args(args.m_dim, args.n_dim, args.k_dim),
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0


def _require_torch_npu(device_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("basic_mixed_store_zNUnAlign --run requires PyTorch.") from exc
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("basic_mixed_store_zNUnAlign --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _create_tla_tensor(dev_buf: Any) -> Any:
    from catlass import runtime as runtime_mod

    shape0, shape1 = dev_buf.shape

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(shape0, shape1),
            tla.Float32,
            origin_shape=tla.make_shape(shape0, shape1),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(shape1, 1),
            data_ptr=int(contiguous.data_ptr()),
        )
    tensor._external_binding = True
    return tensor


def run(args: argparse.Namespace) -> int:
    m_dim, n_dim, k_dim = args.m, args.n, args.k
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        device = "npu"
        lhs = torch.randn(m_dim * k_dim, dtype=torch.float32, device="cpu").reshape(
            m_dim, k_dim
        )
        rhs = torch.randn(k_dim * n_dim, dtype=torch.float32, device="cpu").reshape(
            k_dim, n_dim
        )
        out = torch.full((m_dim, n_dim), -9.0, dtype=torch.float32, device="cpu").to(device)
        expected = lhs @ rhs
        lhs = lhs.to(device)
        rhs = rhs.to(device)

        tla_lhs = _create_tla_tensor(lhs)
        tla_rhs = _create_tla_tensor(rhs)
        tla_out = _create_tla_tensor(out)

        artifact = tla.compile(
            basic_mixed_store_zNUnAlign,
            tla_lhs,
            tla_rhs,
            tla_out,
            **_runtime_kwargs(args),
        )
        artifact(tla_lhs, tla_rhs, tla_out, block_dim=args.block_dim)

        torch.npu.synchronize()
        out = out.cpu()

        expected_match = torch.isclose(out, expected, rtol=0.0, atol=args.atol)
        mismatch = expected_match.logical_not().nonzero(as_tuple=False)
        first_mismatch: dict[str, Any] | None = None
        if mismatch.numel():
            i, j = (int(v) for v in mismatch[0].tolist())
            first_mismatch = {
                "index": [i, j],
                "actual": out[i, j].item(),
                "expected": expected[i, j].item(),
            }

        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"out equals expected mixed result? {bool(expected_match.all())}")
        print(f"first mismatch={first_mismatch}")
        return 0 if first_mismatch is None else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run a mixed kernel using zNUnAlign store "
        "(matrix A gm->ub(row)->zNUnAlign(store)->l1)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--m", type=int, default=DEFAULT_M_DIM, help="GM M dimension (rows of lhs/out).")
    parser.add_argument("--n", type=int, default=DEFAULT_N_DIM, help="GM N dimension (cols of rhs/out).")
    parser.add_argument("--k", type=int, default=DEFAULT_K_DIM, help="GM K dimension (lhs cols / rhs rows).")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.dump_tlair:
        print(dump_tlair(args))
        return 0
    if args.build_only:
        return build_only(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
