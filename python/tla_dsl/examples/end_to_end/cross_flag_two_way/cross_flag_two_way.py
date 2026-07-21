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

ROWS = 2
COLS = 64
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"


@tla.kernel
def cross_flag_two_way(
    x: tla.Tensor,
    y: tla.Tensor,
    out: tla.Tensor,
) -> None:
    aic_to_aiv = tla.cross_flag("aic_to_aiv", mode=4)
    aiv_to_aic = tla.cross_flag("aiv_to_aic", mode=4)

    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_ub_ptr = tla.allocate(COLS, tla.Float32, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(COLS, tla.Float32, tla.AddressSpace.ub, 256)
    out_ub_ptr = tla.allocate(COLS, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        # Phase 1: make one token available to each physical AIV.
        tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=0)
        tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=1)

        # Phase 2: do not release either AIV until both have acknowledged.
        tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=1)

        # Phase 3: a second counter increment releases each AIV to store.
        tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=0)
        tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=1)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        aiv_id = tla.arch.sub_block_idx()

        # Compiler-generated guards make AIV0 consume only id 0 and AIV1 only id 1.
        tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=1)

        x_gm = tla.tile_view(x, tla.make_shape(1, COLS), tla.make_coord(aiv_id, 0))
        y_gm = tla.tile_view(y, tla.make_shape(1, COLS), tla.make_coord(aiv_id, 0))
        out_gm = tla.tile_view(out, tla.make_shape(1, COLS), tla.make_coord(aiv_id, 0))
        x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
        y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
        out_ub = tla.make_tensor_like(out_ub_ptr, out_gm, tla.arch.RowMajor)

        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            out_ub.store(x_ub.load() + y_ub.load())

        # Each AIV independently acknowledges that its row is ready.
        tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)

        # Consume the second token only after AIC has observed both acknowledgements.
        tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=1)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _tensor_type() -> Any:
    return tla.Tensor(
        tla.make_shape(ROWS, COLS),
        tla.Float32,
        origin_shape=tla.make_shape(ROWS, COLS),
        coord=tla.make_coord(0, 0),
        stride=tla.make_stride(COLS, 1),
        layout_tag=tla.arch.RowMajor,
    )


def _compile_only_type_args() -> tuple[Any, Any, Any]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        return (_tensor_type(), _tensor_type(), _tensor_type())


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def dump_tlair() -> str:
    return cross_flag_two_way.dump_mlir(type_args=_compile_only_type_args())


def build_only(args: argparse.Namespace) -> int:
    artifact = tla.compile(
        cross_flag_two_way,
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
        raise SystemExit("cross_flag_two_way --run requires PyTorch.") from exc
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("cross_flag_two_way --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _create_tla_tensor(dev_buf: Any) -> Any:
    from catlass import runtime as runtime_mod

    contiguous = dev_buf.contiguous()
    with runtime_mod._eager_capture():
        tensor = tla.Tensor(
            tla.make_shape(ROWS, COLS),
            tla.Float32,
            origin_shape=tla.make_shape(ROWS, COLS),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(COLS, 1),
            data_ptr=int(contiguous.data_ptr()),
        )
    tensor._external_binding = True
    return tensor


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        x = torch.arange(ROWS * COLS, dtype=torch.float32, device="npu").reshape(
            ROWS, COLS
        )
        y = (x * 2.0) + 3.0
        out = torch.full_like(x, -99.0)
        expected = x + y

        tla_x = _create_tla_tensor(x)
        tla_y = _create_tla_tensor(y)
        tla_out = _create_tla_tensor(out)
        artifact = tla.compile(
            cross_flag_two_way,
            tla_x,
            tla_y,
            tla_out,
            **_runtime_kwargs(args),
        )
        artifact(tla_x, tla_y, tla_out, block=args.block)
        torch.npu.synchronize()

        matches = torch.isclose(out, expected, rtol=0.0, atol=args.atol)
        row_matches = [bool(matches[row].all()) for row in range(ROWS)]
        mismatch = matches.logical_not().nonzero(as_tuple=False)
        first_mismatch: dict[str, Any] | None = None
        if mismatch.numel():
            row, col = (int(v) for v in mismatch[0].tolist())
            first_mismatch = {
                "index": [row, col],
                "actual": out[row, col].item(),
                "expected": expected[row, col].item(),
            }

        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"aiv0_row_ok={row_matches[0]}")
        print(f"aiv1_row_ok={row_matches[1]}")
        print(f"out_equals_expected={bool(matches.all())}")
        print(f"first_mismatch={first_mismatch}")
        return 0 if first_mismatch is None else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run a two-way mode-4 AIC/AIV handshake."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
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
