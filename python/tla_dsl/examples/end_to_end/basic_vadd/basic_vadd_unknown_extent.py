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
import re
from typing import Any

import catlass as tla


ROWS = 2
COLS = 128
ELEMENTS = ROWS * COLS
PTR_OFFSET = 0
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "runtime-cache"
    / "basic_vadd_unknown_extent"
)


@tla.kernel
def basic_vadd_unknown_extent(
    mem_x: tla.Tensor,
    mem_y: tla.Tensor,
    mem_z: tla.Tensor,
    rows: tla.Int32,
    ptr_offset: tla.Int32,
) -> None:
    """Add f16 rows through ptr-backed UB memrefs with unknown capacity."""
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    shape = tla.make_shape(rows, COLS)
    layout = tla.make_layout(
        shape,
        tla.make_stride(COLS, 1),
        origin_shape=shape,
    )
    x_gm = tla.make_tensor(mem_x.ptr, layout)
    y_gm = tla.make_tensor(mem_y.ptr, layout)
    z_gm = tla.make_tensor(mem_z.ptr, layout)

    x_ub_ptr = tla.allocate((ROWS, COLS), tla.Float16, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate((ROWS, COLS), tla.Float16, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate((ROWS, COLS), tla.Float16, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor(x_ub_ptr + ptr_offset, layout)
    y_ub = tla.make_tensor(y_ub_ptr + ptr_offset, layout)
    z_ub = tla.make_tensor(z_ub_ptr + ptr_offset, layout)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for row in tla.range(0, rows, 1):
                x_tile = tla.tile_view(
                    x_ub, tla.make_shape(1, COLS), tla.make_coord(row, 0)
                )
                y_tile = tla.tile_view(
                    y_ub, tla.make_shape(1, COLS), tla.make_coord(row, 0)
                )
                z_tile = tla.tile_view(
                    z_ub, tla.make_shape(1, COLS), tla.make_coord(row, 0)
                )
                z_tile.store(tla.add(x_tile.load(), y_tile.load()))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _scalar_args() -> tuple[Any, Any]:
    return tla.Int32(ROWS), tla.Int32(PTR_OFFSET)


def _compile_only_type_args() -> tuple[Any, Any, Any, Any, Any]:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        shape = tla.make_shape(ROWS, COLS)
        tensor_args = tuple(
            tla.Tensor(
                shape,
                tla.Float16,
                origin_shape=shape,
                coord=tla.make_coord(0, 0),
                stride=tla.make_stride(COLS, 1),
                layout_tag=tla.arch.RowMajor,
            )
            for _ in range(3)
        )
    return (*tensor_args, *_scalar_args())


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aiv.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _is_zero_index_value(lowered_mlir: str, value: str) -> bool:
    escaped_value = re.escape(value)
    if re.search(
        rf"{escaped_value} = arith\.constant 0 : index", lowered_mlir
    ):
        return True
    if re.search(
        rf"{escaped_value} = llvm\.mlir\.constant\(0 : index\) : i64",
        lowered_mlir,
    ):
        return True

    cast = re.search(
        rf"{escaped_value} = builtin\.unrealized_conversion_cast "
        r"(%[A-Za-z0-9_]+) : i64 to index",
        lowered_mlir,
    )
    if cast is None:
        return False
    source = re.escape(cast.group(1))
    return bool(
        re.search(
            rf"{source} = llvm\.mlir\.constant\(0 : index\) : i64",
            lowered_mlir,
        )
    )


def _verify_unknown_extent_ir(lowered_mlir: str) -> None:
    dynamic_ub_casts = re.findall(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[(%[A-Za-z0-9_]+)\] "
        r": memref<\?xf16, #hivm\.address_space<ub>>",
        lowered_mlir,
    )
    if not dynamic_ub_casts:
        raise RuntimeError("missing zero-sized dynamic UB pointer_cast")
    for extent in set(dynamic_ub_casts):
        if not _is_zero_index_value(lowered_mlir, extent):
            raise RuntimeError(f"dynamic UB extent is not zero: {extent}")

    helper_signatures = [
        line
        for line in lowered_mlir.splitlines()
        if "func.func private @vector_region_" in line
    ]
    dynamic_ub_type = "memref<?xf16, #hivm.address_space<ub>>"
    if not any(line.count(dynamic_ub_type) >= 3 for line in helper_signatures):
        raise RuntimeError("vector helper does not keep all UB base capacities unknown")

    if re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\).*"
        r"memref<128xf16, #hivm\.address_space<ub>>",
        lowered_mlir,
    ):
        raise RuntimeError("static tile size was incorrectly used as UB base capacity")
    if "memref<0xf16" in lowered_mlir:
        raise RuntimeError("unknown capacity must not be represented as memref<0xf16>")

    helper_ir = lowered_mlir.split("func.func private @vector_region_", maxsplit=1)[-1]
    has_row_width = (
        "arith.constant 128 : index" in helper_ir
        or "llvm.mlir.constant(128 : index)" in helper_ir
    )
    has_row_offset = "arith.muli" in helper_ir or "llvm.mul" in helper_ir
    if not has_row_width or not has_row_offset:
        raise RuntimeError("missing row * 128 tile offset in vector helper")
    if not re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) "
        r"\[%[A-Za-z0-9_]+, %[A-Za-z0-9_]+\] "
        r": memref<\?x\?xf16, #hivm\.address_space<gm>>",
        lowered_mlir,
    ):
        raise RuntimeError("dynamic GM pointer_cast does not carry real rank-2 sizes")


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    artifact = tla.compile(
        basic_vadd_unknown_extent,
        *type_args,
        **_runtime_kwargs(args),
    )
    _verify_unknown_extent_ir(artifact.lowered_llvm)
    print("compile_ok=True")
    print("zero_extent_ir_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return artifact


def dump_tlair() -> str:
    return basic_vadd_unknown_extent.dump_mlir(type_args=_compile_only_type_args())


def build_only(args: argparse.Namespace) -> int:
    _compile(args, *_compile_only_type_args())
    return 0


def _require_torch_npu(device_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("basic_vadd_unknown_extent --run requires PyTorch.") from exc
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("basic_vadd_unknown_extent --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        torch = _require_torch_npu(args.device)
        device = f"npu:{args.device}"
        values = torch.arange(ELEMENTS, dtype=torch.float32, device=device)
        x = ((values % 17) - 8).to(torch.float16).reshape(ROWS, COLS)
        y = ((values % 13) - 6).to(torch.float16).reshape(ROWS, COLS)
        z = torch.full((ROWS, COLS), -7.0, dtype=torch.float16, device=device)
        expected = x + y

        tla_x = tla.from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
        tla_y = tla.from_dlpack(y.contiguous(), layout_tag=tla.arch.RowMajor)
        tla_z = tla.from_dlpack(z.contiguous(), layout_tag=tla.arch.RowMajor)
        rows, ptr_offset = _scalar_args()
        artifact = _compile(args, tla_x, tla_y, tla_z, rows, ptr_offset)
        artifact(tla_x, tla_y, tla_z, rows, ptr_offset, block=args.block)

        torch.npu.synchronize()
        expected_match = torch.isclose(z, expected, rtol=0.0, atol=args.atol)
        mismatch = expected_match.logical_not().nonzero(as_tuple=False)
        first_mismatch: dict[str, Any] | None = None
        if mismatch.numel():
            row, col = (int(value) for value in mismatch[0].tolist())
            first_mismatch = {
                "index": [row, col],
                "actual": z[row, col].item(),
                "expected": expected[row, col].item(),
            }

        print("launch_ok=True")
        print(f"Z equals expected add? {bool(expected_match.all())}")
        print(f"first mismatch={first_mismatch}")
        return 0 if first_mismatch is None else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run dynamic RowMajor f16 vadd with zero UB extents."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
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
