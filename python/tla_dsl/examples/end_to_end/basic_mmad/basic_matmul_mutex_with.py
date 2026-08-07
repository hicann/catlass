"""Basic MMAD (mutex_guard): Kernel + Host in one file.

Dynamic GM; mnk/dtype/layout from CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla
from catlass.runtime import from_dlpack

l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

DESCRIPTION = "Basic MMAD mutex_guard; dynamic GM."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def basic_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
) -> None:
    c0 = 0
    c1 = 1

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    mutex_l1a0  = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1  = tla.mutex(resource="l1a1", id=1)
    mutex_l1b0  = tla.mutex(resource="l1b0", id=2)
    mutex_l1b1  = tla.mutex(resource="l1b1", id=3)

    mutex_l0a0  = tla.mutex(resource="l0a0", id=4)
    mutex_l0a1  = tla.mutex(resource="l0a1", id=5)
    mutex_l0b0  = tla.mutex(resource="l0b0", id=6)
    mutex_l0b1  = tla.mutex(resource="l0b1", id=7)

    mutex_l0c  = tla.mutex(resource="l0c", id=8)

    l1a0_ptr = tla.allocate((l1_tm, l1_tk), DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate((l1_tm, l1_tk), DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate((l1_tk, l1_tn), DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate((l1_tk, l1_tn), DTYPE_B, tla.AddressSpace.l1, 512)

    l0a0_ptr = tla.allocate((l0_tm, l0_tk), DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate((l0_tm, l0_tk), DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate((l0_tk, l0_tn), DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate((l0_tk, l0_tn), DTYPE_B, tla.AddressSpace.l0b, 512)

    l0c_ptr = tla.allocate((l0_tm, l0_tn), tla.Float32, tla.AddressSpace.l0c, 512)

    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n


    with tla.cube():
        l1_buf_idx = c0
        l0_buf_idx = c0

        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            for k_l1 in k_l1_range:
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk), tla.make_coord(c0, k_l1)
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn), tla.make_coord(k_l1, c0)
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )

                mutex_l1a = mutex_l1a0 if (l1_buf_idx == c0) else mutex_l1a1
                with tla.mutex_guard(mutex_l1a):
                    tla.copy(l1_a, gm_a_by_l1)

                mutex_l1b = mutex_l1b0 if (l1_buf_idx == c0) else mutex_l1b1
                with tla.mutex_guard(mutex_l1b):
                    tla.copy(l1_b, gm_b_by_l1)

                k_l0_count = (l1_a.origin_shape[1] + l0_tk - 1) // l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)

                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk), tla.make_coord(c0, k_l0)
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn), tla.make_coord(k_l0, c0)
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )

                    mutex_l0a = mutex_l0a0 if (l0_buf_idx == c0) else mutex_l0a1
                    with tla.mutex_guard(mutex_l1a, mutex_l0a):
                        tla.copy(l0_a, l1_a_by_l0)


                    mutex_l0b = mutex_l0b0 if (l0_buf_idx == c0) else mutex_l0b1
                    with tla.mutex_guard(mutex_l1b, mutex_l0b):
                        tla.copy(l0_b, l1_b_by_l0)

                    with tla.mutex_guard(mutex_l0a, mutex_l0b, mutex_l0c):
                        tla.mmad(
                            l0_c, l0_a, l0_b, init_c=True if k_l1 == 0 and k_l0 == 0 else False
                        )

                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            with tla.mutex_guard(mutex_l0c):
                tla.copy(gm_c_by_core, l0_c)

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"

def golden(a, b, out_dtype):
    import torch

    expected = a.to(torch.float32) @ b.to(torch.float32)
    if out_dtype in (torch.float16, torch.bfloat16):
        expected = expected.to(out_dtype).to(torch.float32)
    return expected


def run(args: argparse.Namespace) -> int:
    import sys
    import torch
    import torch_npu  # noqa: F401

    mod = sys.modules[__name__]
    tla_of = {"f16": tla.Float16, "bf16": tla.BFloat16, "f32": tla.Float32}
    torch_of = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    da, db, dc = args.dtype_a, args.dtype_b, args.dtype_c
    la, lb = args.layout_a, args.layout_b
    mi, ni, ki = int(args.m), int(args.n), int(args.k)
    mod.DTYPE_A = tla_of[da]
    mod.DTYPE_B = tla_of[db]

    def create_tla_tensor(buf, layout: str):
        storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
        tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
        return from_dlpack(storage, layout_tag=tag).mark_layout_dynamic()

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        print(f"--- mnk=({mi},{ni},{ki}) layout={la}/{lb} dtype={da}/{db}/{dc} ---")
        torch.npu.manual_seed(0)
        a = torch.rand(mi, ki, dtype=torch_of[da], device="npu") * 10.0 - 5.0
        b = torch.rand(ki, ni, dtype=torch_of[db], device="npu") * 10.0 - 5.0
        c = torch.full((mi, ni), args.sentinel, dtype=torch_of[dc], device="npu")
        expected = golden(a, b, torch_of[dc])

        ta, tb, tc = create_tla_tensor(a, la), create_tla_tensor(b, lb), create_tla_tensor(c, "row")
        artifact = tla.compile(
            basic_mmad_kernel,
            ta,
            tb,
            tc,
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=cache_dir,
            force_recompile=args.force_recompile,
        )
        block_dim = max(
            1,
            args.block_dim if args.block_dim != -1 else tla.get_aicore_num(args.device),
        )
        artifact(ta, tb, tc, block_dim=block_dim)
        torch.npu.synchronize()

        # Match catlass examples/common/golden/compare_data.hpp:
        # bf16: rtol 1/128 (K<2048) or 1/64, floor 1/256; else f16/f32: rtol 1/256 or 1/128, floor 1.
        if dc == "bf16":
            rtol = (1.0 / 128.0) if ki < 2048 else (1.0 / 64.0)
            floor = 1.0 / 256.0
        else:
            rtol = (1.0 / 256.0) if ki < 2048 else (1.0 / 128.0)
            floor = 1.0
        got = c.detach().to(device="cpu", dtype=torch.float32)
        exp = expected.detach().to(device="cpu", dtype=torch.float32)
        passed = bool(((got - exp).abs() <= rtol * torch.maximum(torch.full_like(exp, floor), exp.abs())).all())
        print(f"passed={passed} cache_key={artifact.cache_key}")
        print(f"kernel.o={artifact.kernel_binary_path}")
        return 0 if passed else 1
    finally:
        tla.finalize()


def main() -> int:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--layout-a", choices=("row", "col"), default="row")
    p.add_argument("--layout-b", choices=("row", "col"), default="row")
    p.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    p.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    p.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--block-dim", type=int, default=-1)
    p.add_argument("--sentinel", type=float, default=-7.0)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--force-recompile", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
