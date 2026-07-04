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

M_DIM = 32
N_DIM = 32
K_DIM = 32
VECTOR_TILE_M = 16
VECTOR_TILE_N = 32
VECTOR_REG_TILE_M = 2
ROW_TILE_BYTES = VECTOR_REG_TILE_M * VECTOR_TILE_N * 4
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"


@tla.kernel
def basic_mixed(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    addend: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    fix_done = tla.cross_flag("fix_done", tla.arch.FIX, tla.arch.SCALAR)

    allocator = tla.utils.LocalmemAllocator()

    l1a_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
    l1a_ptr = tla.recast_ptr(l1a_ptr, dtype=tla.Float32)
    l1b_ptr = allocator.allocate(L1_STAGE_BYTES, 512, tla.AddressSpace.l1)
    l1b_ptr = tla.recast_ptr(l1b_ptr, dtype=tla.Float32)
    l0a_ptr = allocator.allocate(L0A_BYTES, 512, tla.AddressSpace.l0a)
    l0a_ptr = tla.recast_ptr(l0a_ptr, dtype=tla.Float32)
    l0b_ptr = allocator.allocate(L0B_BYTES, 512, tla.AddressSpace.l0b)
    l0b_ptr = tla.recast_ptr(l0b_ptr, dtype=tla.Float32)
    l0c_ptr = allocator.allocate(L0C_BYTES, 512, tla.AddressSpace.l0c)
    l0c_ptr = tla.recast_ptr(l0c_ptr, dtype=tla.Float32)

    out_ub_ptr = allocator.allocate(ROW_TILE_BYTES, 256, tla.AddressSpace.ub)
    out_ub_ptr = tla.recast_ptr(out_ub_ptr, dtype=tla.Float32)
    addend_ub_ptr = allocator.allocate(ROW_TILE_BYTES, 256, tla.AddressSpace.ub)
    addend_ub_ptr = tla.recast_ptr(addend_ub_ptr, dtype=tla.Float32)
    result_ub_ptr = allocator.allocate(ROW_TILE_BYTES, 256, tla.AddressSpace.ub)
    result_ub_ptr = tla.recast_ptr(result_ub_ptr, dtype=tla.Float32)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(M_DIM, N_DIM), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)
        tla.copy(gm_c, l0_c)
        tla.cross_core_set_flag(fix_done)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        tla.cross_core_wait_flag(fix_done)

        vector_tile_row = tla.arch.sub_block_idx()

        for row_idx in tla.range_constexpr(0, VECTOR_TILE_M, VECTOR_REG_TILE_M):
            row_tile_idx = vector_tile_row * 8 + row_idx // 2
            out_gm_chunk = tla.tile_view(
                out,
                tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                tla.make_coord(row_tile_idx, 0),
            )
            addend_gm_chunk = tla.tile_view(
                addend,
                tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                tla.make_coord(row_tile_idx, 0),
            )
            out_chunk = tla.make_tensor_like(out_ub_ptr, out_gm_chunk, tla.arch.RowMajor)
            addend_chunk = tla.make_tensor_like(
                addend_ub_ptr, addend_gm_chunk, tla.arch.RowMajor
            )
            result_chunk = tla.make_tensor_like(
                result_ub_ptr, out_gm_chunk, tla.arch.RowMajor
            )

            tla.set_flag(ub_load_ready)
            tla.wait_flag(ub_load_ready)
            tla.copy(out_chunk, out_gm_chunk)
            tla.copy(addend_chunk, addend_gm_chunk)
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)

            with tla.vec.func(mode="simd"):
                result_chunk.store(out_chunk.load() + addend_chunk.load())

            tla.set_flag(vec_done)
            tla.wait_flag(vec_done)
            tla.copy(out_gm_chunk, result_chunk)

        tla.pipe_barrier(tla.pipes.ALL)


def _compile_only_type_args() -> tuple[Any, Any, Any, Any]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        lhs_shape = tla.make_shape(M_DIM, K_DIM)
        rhs_shape = tla.make_shape(K_DIM, N_DIM)
        out_shape = tla.make_shape(M_DIM, N_DIM)
        out_stride = tla.make_stride(N_DIM, 1)
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
    return basic_mixed.dump_mlir(type_args=_compile_only_type_args())


def build_only(args: argparse.Namespace) -> int:
    artifact = tla.compile(
        basic_mixed,
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
        raise SystemExit("basic_mixed --run requires PyTorch.") from exc
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("basic_mixed --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _create_tla_tensor(dev_buf: Any) -> Any:
    from catlass import runtime as runtime_mod

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(M_DIM, N_DIM),
            tla.Float32,
            origin_shape=tla.make_shape(M_DIM, N_DIM),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(N_DIM, 1),
            data_ptr=int(contiguous.data_ptr()),
        )
    tensor._external_binding = True
    return tensor


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        device = "npu"
        lhs = torch.arange(M_DIM * K_DIM, dtype=torch.float32, device=device).reshape(
            M_DIM, K_DIM
        )
        rhs = torch.arange(K_DIM * N_DIM, dtype=torch.float32, device=device).reshape(
            K_DIM, N_DIM
        )
        addend = torch.full((M_DIM, N_DIM), 3.0, dtype=torch.float32, device=device)
        out = torch.full((M_DIM, N_DIM), -9.0, dtype=torch.float32, device=device)
        expected = lhs @ rhs + addend

        tla_lhs = _create_tla_tensor(lhs)
        tla_rhs = _create_tla_tensor(rhs)
        tla_out = _create_tla_tensor(out)
        tla_addend = _create_tla_tensor(addend)

        artifact = tla.compile(
            basic_mixed,
            tla_lhs,
            tla_rhs,
            tla_out,
            tla_addend,
            **_runtime_kwargs(args),
        )
        artifact(tla_lhs, tla_rhs, tla_out, tla_addend, block=args.block)

        torch.npu.synchronize()
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
    parser = argparse.ArgumentParser(description="Compile and run a minimal mixed kernel.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block", type=int, default=1)
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
