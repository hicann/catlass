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
from typing import Any

import catlass as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode

# unaligned M_DIM/N_DIM is OK
M_DIM = 60
N_DIM = 100
K_DIM = 64

M_L1 = 64
N_L1 = 128
K_L1 = 64

L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1

L0A_SIZE = M_L1 * K_L1
L0B_SIZE = K_L1 * N_L1
L0C_SIZE = M_L1 * N_L1

UB_C_SIZE = L0C_SIZE

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache" / "basic_mixed_fixpipe_nz2dn"


@tla.kernel
def basic_mixed_fixpipe_nz2dn(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1b_done = tla.flag("l1b_done", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    fix_done = tla.cross_flag("fix_done", mode=4)

    l1a_ptr = tla.allocate(L1A_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_SIZE, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_SIZE, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_SIZE, tla.Float32, tla.AddressSpace.l0c, 512)

    ub_c_ptr = tla.allocate(UB_C_SIZE, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1b_done)
        tla.wait_flag(l1b_done)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)

        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        out_col = tla.make_tensor(
            out.ptr,
            tla.make_layout(
                tla.make_shape(M_DIM, N_DIM), tla.make_stride(1, M_DIM), layoutTag=tla.arch.ColumnMajor
            ),
            coord=tla.make_coord(0, 0)
        )
        gm_c = tla.tile_view(out_col, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))

        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        ub_c = tla.make_tensor_like(ub_c_ptr, l0_c, tla.arch.ColumnMajor)
        tla.copy(ub_c, l0_c, params=CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0))

        tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_c = tla.tile_view(out, tla.make_shape(N_L1, M_L1), tla.make_coord(0, 0))
        ub_c = tla.make_tensor_like(ub_c_ptr, gm_c, tla.arch.RowMajor)

        if vec_idx == 0:
            tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=0)
            tla.copy(gm_c, ub_c)  # copy to gm row->row

        tla.pipe_barrier(tla.pipes.ALL)


def _compile_only_type_args() -> tuple[Any, Any, Any]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        lhs_shape = tla.make_shape(M_DIM, K_DIM)
        rhs_shape = tla.make_shape(K_DIM, N_DIM)
        out_shape = tla.make_shape(N_DIM, M_DIM)
        out_stride = tla.make_stride(M_DIM, 1)
        return (
            tla.Tensor(
                lhs_shape,
                tla.Float32,
                origin_shape=lhs_shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(K_DIM, 1),
                layout_tag=tla.arch.RowMajor,
            ),
            tla.Tensor(
                rhs_shape,
                tla.Float32,
                origin_shape=rhs_shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(N_DIM, 1),
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


def dump_tlair() -> str:
    return basic_mixed_fixpipe_nz2dn.dump_mlir(type_args=_compile_only_type_args())


def build_only(args: argparse.Namespace) -> int:
    artifact = tla.compile(
        basic_mixed_fixpipe_nz2dn,
        mlir_print_ir_after_all=True,
        *_compile_only_type_args(),
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0


def _require_torch_npu(device_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("basic_mixed_fixpipe_nz2dn --run requires PyTorch.") from exc
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("basic_mixed_fixpipe_nz2dn --run requires torch_npu.") from exc
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
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        device = "npu"
        lhs = torch.randn(M_DIM * K_DIM, dtype=torch.float32, device="cpu").reshape(M_DIM, K_DIM)
        rhs = torch.randn(K_DIM * N_DIM, dtype=torch.float32, device="cpu").reshape(K_DIM, N_DIM)
        # original out is ColumnMajor(M_DIM, N_DIM), same as RowMajor(N_DIM, M_DIM)
        out = torch.full((N_DIM, M_DIM), -9.0, dtype=torch.float32, device="cpu").to(device)
        expected = lhs @ rhs
        expected = expected.transpose(1, 0).contiguous() # (N_DIM, M_DIM)
        lhs, rhs = lhs.to(device), rhs.to(device)

        tla_lhs = _create_tla_tensor(lhs)
        tla_rhs = _create_tla_tensor(rhs)
        tla_out = _create_tla_tensor(out)

        artifact = tla.compile(
            basic_mixed_fixpipe_nz2dn,
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
        description="Compile and run a minimal mixed kernel, matrix l0c->ub(column_major)->gm."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
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
        print(dump_tlair())
        return 0
    if args.build_only:
        return build_only(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
