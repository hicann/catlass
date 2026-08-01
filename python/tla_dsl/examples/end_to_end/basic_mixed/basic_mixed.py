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
from contextlib import redirect_stdout
from io import StringIO
import re
from pathlib import Path
from typing import Any

import catlass as tla

M_DIM = 32
N_DIM = 32
K_DIM = 32
VECTOR_TILE_M = 16
VECTOR_TILE_N = 32
VECTOR_REG_TILE_M = 2
UB_TILE_BYTES = VECTOR_TILE_M * VECTOR_TILE_N * 4
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4

# Minimal dynamic-GM smoke for mix kernel: M/N stay 32 (AIV tile split), vary K.
DEFAULT_MNK_SHAPES = (
    (M_DIM, N_DIM, K_DIM),
    (M_DIM, N_DIM, 16),
)

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"
_PRINT_RECORD = re.compile(
    r"^tla\.print dtype=float32 position=(?P<position>[A-Z0-9]+) "
    r"subblock=(?P<subblock>[01]) "
    r"shape=\[(?P<shape>[0-9,]+)\] count=(?P<count>[0-9]+) "
    r"values=\[(?P<values>.*)\]$"
)


@tla.kernel
def basic_mixed(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    addend: tla.Tensor,
) -> None:
    m = lhs.origin_shape[0]
    k = lhs.origin_shape[1]
    n = rhs.origin_shape[1]

    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    fix_done = tla.cross_flag("fix_done")

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 512)

    c_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    addend_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    result_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(m, k), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(k, n), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(m, n), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(m, k), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(k, n), tla.make_coord(0, 0)
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

        ub_c = tla.make_tensor_like(c_ub_ptr, l0_c, tla.arch.RowMajor)
        tla.copy(ub_c, l0_c, tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
        ))

        tla.cross_core_set_flag(fix_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_result = tla.tile_view(
            out,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0)
        )
        gm_addend = tla.tile_view(
            addend,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0)
        )
        ub_result = tla.make_tensor_like(result_ub_ptr, gm_result, tla.arch.RowMajor)
        ub_addend = tla.make_tensor_like(addend_ub_ptr, gm_addend, tla.arch.RowMajor)

        tla.set_flag(ub_load_ready)
        tla.wait_flag(ub_load_ready)
        tla.copy(ub_addend, gm_addend)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        tla.print(ub_addend, 16)

        ub_c = tla.make_tensor_like(c_ub_ptr, gm_result, tla.arch.RowMajor)
        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)

        for row_tile_idx in tla.range(0, VECTOR_TILE_M//VECTOR_REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                c_chunk = tla.tile_view(
                    ub_c,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                addend_chunk = tla.tile_view(
                    ub_addend,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                result_chunk.store(c_chunk.load() + addend_chunk.load())

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_result, ub_result)

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_mixed_mutex(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    addend: tla.Tensor,
) -> None:
    m = lhs.origin_shape[0]
    k = lhs.origin_shape[1]
    n = rhs.origin_shape[1]

    mutex_l1a = tla.mutex(resource="l1a", id=0)
    mutex_l1b = tla.mutex(resource="l1b", id=1)
    mutex_l0a = tla.mutex(resource="l0a", id=2)
    mutex_l0b = tla.mutex(resource="l0b", id=3)
    mutex_l0c = tla.mutex(resource="l0c", id=4)
    mutex_c_ub = tla.mutex(resource="c_ub", id=5)
    mutex_addend_ub = tla.mutex(resource="addend_ub", id=6)
    mutex_result_ub = tla.mutex(resource="result_ub", id=7)

    fix_done = tla.cross_flag("fix_done")

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 512)

    c_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    addend_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    result_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(m, k), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(k, n), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(m, n), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        mutex_l1a.lock(pipe=tla.arch.MTE2)
        tla.copy(l1_a, gm_a)
        mutex_l1a.unlock(pipe=tla.arch.MTE2)

        mutex_l1b.lock(pipe=tla.arch.MTE2)
        tla.copy(l1_b, gm_b)
        mutex_l1b.unlock(pipe=tla.arch.MTE2)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(m, k), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(k, n), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)

        mutex_l1a.lock(pipe=tla.arch.MTE1)
        mutex_l0a.lock(pipe=tla.arch.MTE1)
        tla.copy(l0_a, l1_a_l0)
        mutex_l0a.unlock(pipe=tla.arch.MTE1)
        mutex_l1a.unlock(pipe=tla.arch.MTE1)

        mutex_l1b.lock(pipe=tla.arch.MTE1)
        mutex_l0b.lock(pipe=tla.arch.MTE1)
        tla.copy(l0_b, l1_b_l0)
        mutex_l0b.unlock(pipe=tla.arch.MTE1)
        mutex_l1b.unlock(pipe=tla.arch.MTE1)

        mutex_l0a.lock(pipe=tla.arch.CUBE)
        mutex_l0b.lock(pipe=tla.arch.CUBE)
        mutex_l0c.lock(pipe=tla.arch.CUBE)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        mutex_l0c.unlock(pipe=tla.arch.CUBE)
        mutex_l0b.unlock(pipe=tla.arch.CUBE)
        mutex_l0a.unlock(pipe=tla.arch.CUBE)

        ub_c = tla.make_tensor_like(c_ub_ptr, l0_c, tla.arch.RowMajor)
        mutex_l0c.lock(pipe=tla.arch.FIX)
        mutex_c_ub.lock(pipe=tla.arch.FIX)
        tla.copy(ub_c, l0_c, tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
        ))
        mutex_c_ub.unlock(pipe=tla.arch.FIX)
        mutex_l0c.unlock(pipe=tla.arch.FIX)

        tla.cross_core_set_flag(fix_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_result = tla.tile_view(
            out,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0)
        )
        gm_addend = tla.tile_view(
            addend,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0)
        )
        ub_result = tla.make_tensor_like(result_ub_ptr, gm_result, tla.arch.RowMajor)
        ub_addend = tla.make_tensor_like(addend_ub_ptr, gm_addend, tla.arch.RowMajor)

        mutex_addend_ub.lock(pipe=tla.arch.MTE2)
        tla.copy(ub_addend, gm_addend)
        mutex_addend_ub.unlock(pipe=tla.arch.MTE2)

        mutex_addend_ub.lock(pipe=tla.arch.VECTOR)
        tla.print(ub_addend, 16)
        mutex_addend_ub.unlock(pipe=tla.arch.VECTOR)

        ub_c = tla.make_tensor_like(c_ub_ptr, gm_result, tla.arch.RowMajor)
        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)

        for row_tile_idx in tla.range(0, VECTOR_TILE_M//VECTOR_REG_TILE_M, 1):
            mutex_c_ub.lock(pipe=tla.arch.VECTOR)
            mutex_addend_ub.lock(pipe=tla.arch.VECTOR)
            mutex_result_ub.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                c_chunk = tla.tile_view(
                    ub_c,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                addend_chunk = tla.tile_view(
                    ub_addend,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0)
                )
                result_chunk.store(c_chunk.load() + addend_chunk.load())
            mutex_result_ub.unlock(pipe=tla.arch.VECTOR)
            mutex_addend_ub.unlock(pipe=tla.arch.VECTOR)
            mutex_c_ub.unlock(pipe=tla.arch.VECTOR)

        mutex_result_ub.lock(pipe=tla.arch.MTE3)
        tla.copy(gm_result, ub_result)
        mutex_result_ub.unlock(pipe=tla.arch.MTE3)

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
            ).mark_layout_dynamic(),
            tla.Tensor(
                rhs_shape,
                tla.Float32,
                origin_shape=rhs_shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(N_DIM, 1),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
            tla.Tensor(
                out_shape,
                tla.Float32,
                origin_shape=out_shape,
                coord=tla.make_coord(0, 0),
                stride=out_stride,
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
            tla.Tensor(
                out_shape,
                tla.Float32,
                origin_shape=out_shape,
                coord=tla.make_coord(0, 0),
                stride=out_stride,
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
        )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _select_kernel(args: argparse.Namespace) -> Any:
    if getattr(args, "use_mutex", False):
        return basic_mixed_mutex
    return basic_mixed


def dump_tlair(args: argparse.Namespace | None = None) -> str:
    return _select_kernel(args).dump_mlir(type_args=_compile_only_type_args())


def build_only(args: argparse.Namespace) -> int:
    kernel = _select_kernel(args)
    artifact = tla.compile(
        kernel,
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


def _create_tla_tensor(dev_buf: Any, rows: int, cols: int) -> Any:
    from catlass import runtime as runtime_mod

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(rows, cols),
            tla.Float32,
            origin_shape=tla.make_shape(rows, cols),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(cols, 1),
            data_ptr=int(contiguous.data_ptr()),
        ).mark_layout_dynamic()
    tensor._external_binding = True
    return tensor


def _verify_mixed_print_output(output: str) -> list[str]:
    records = [
        line
        for line in output.splitlines()
        if line.startswith("tla.print ")
    ]
    if len(records) != 2:
        raise tla.TlaExecutionError(
            "mixed tensor tla.print validation failed: "
            f"expected two AIV records, got {records!r}"
        )
    records_by_subblock: dict[int, str] = {}
    for record in records:
        match = _PRINT_RECORD.fullmatch(record)
        if match is None:
            raise tla.TlaExecutionError(
                f"mixed tensor tla.print validation failed: malformed record {record!r}"
            )
        shape = tuple(int(extent) for extent in match.group("shape").split(","))
        try:
            values = [
                float(value.strip())
                for value in match.group("values").split(",")
            ]
        except ValueError as exc:
            raise tla.TlaExecutionError(
                f"mixed tensor tla.print validation failed: malformed record {record!r}"
            ) from exc
        subblock = int(match.group("subblock"))
        if subblock in records_by_subblock:
            raise tla.TlaExecutionError(
                "mixed tensor tla.print validation failed: "
                f"duplicate subblock={subblock}"
            )
        records_by_subblock[subblock] = record
        if (
            match.group("position") != "UB"
            or shape != (VECTOR_TILE_M, VECTOR_TILE_N)
            or int(match.group("count")) != 16
            or values != [3.0] * 16
        ):
            raise tla.TlaExecutionError(
                "mixed tensor tla.print validation failed: expected position=UB, "
                f"shape=[{VECTOR_TILE_M},{VECTOR_TILE_N}], count=16, and sixteen "
                f"3.0 values; got {record!r}"
            )
    if set(records_by_subblock) != {0, 1}:
        raise tla.TlaExecutionError(
            "mixed tensor tla.print validation failed: expected records from "
            f"subblocks 0 and 1, got {sorted(records_by_subblock)}"
        )
    return records


def _run_single_case(
    args: argparse.Namespace, torch: Any, m: int, n: int, k: int
) -> int:
    if (m, n) != (M_DIM, N_DIM):
        raise ValueError(
            f"basic_mixed AIV path expects m=={M_DIM}, n=={N_DIM}; got m={m}, n={n}"
        )
    if k <= 0 or k > K_DIM:
        raise ValueError(f"k must be in 1..{K_DIM}; got {k}")

    device = "npu"
    lhs = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k)
    rhs = torch.arange(k * n, dtype=torch.float32, device=device).reshape(k, n)
    addend = torch.full((m, n), 3.0, dtype=torch.float32, device=device)
    out = torch.full((m, n), -9.0, dtype=torch.float32, device=device)
    expected = lhs @ rhs + addend

    tla_lhs = _create_tla_tensor(lhs, m, k)
    tla_rhs = _create_tla_tensor(rhs, k, n)
    tla_out = _create_tla_tensor(out, m, n)
    tla_addend = _create_tla_tensor(addend, m, n)

    kernel = _select_kernel(args)
    artifact = tla.compile(
        kernel,
        tla_lhs,
        tla_rhs,
        tla_out,
        tla_addend,
        **_runtime_kwargs(args),
    )
    block = max(1, args.block if args.block != -1 else tla.get_aicore_num(args.device))
    captured = StringIO()
    with redirect_stdout(captured):
        artifact(tla_lhs, tla_rhs, tla_out, tla_addend, block=block)
    print_records = _verify_mixed_print_output(captured.getvalue())

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

    print(f"compile_ok=True mnk={m}x{n}x{k}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print(f"cache_key={artifact.cache_key}")
    for record in print_records:
        print(record)
    print("launch_ok=True")
    print(f"out equals expected mixed result? {bool(expected_match.all())}")
    print(f"first mismatch={first_mismatch}")
    return 0 if first_mismatch is None else 1


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        failed = 0
        for m, n, k in DEFAULT_MNK_SHAPES:
            print("---", f"mnk={m}x{n}x{k}", "---")
            failed += _run_single_case(args, torch, m, n, k)
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and run a minimal mixed kernel.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block", type=int, default=-1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--use-mutex",
        action="store_true",
        help="Use explicit mutex lock/unlock for local on-chip synchronization.",
    )
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
