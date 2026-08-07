"""SIMT vector add: one thread per element, straight out of GM.

The SIMD counterpart lives in ``examples/end_to_end/basic_vadd``. The difference
is the whole point of the SIMT mode: no UB staging, no tiles, no vector ops --
each thread loads its own two elements and stores one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla

VECTOR_ELE = 400
_KERNEL_DTYPE = tla.Float32


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def basic_vadd_simt(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=VECTOR_ELE):
            tid, _, _ = tla.arch.thread_idx()
            thread_block_dim, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, VECTOR_ELE, thread_block_dim):
                gm_c[i] = gm_a[i] + gm_b[i]

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.runtime import from_dlpack

    n_ele = VECTOR_ELE

    def create_tla_tensor(dev_buf):
        # No mark_compact_shape_dynamic: a SIMT vector function takes its
        # buffers as statically shaped memrefs, since only a pointer crosses the
        # launch ABI.
        return from_dlpack(dev_buf.contiguous(), layout_tag=tla.arch.RowMajor)

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        block_dim = max(
            1, args.block_dim if args.block_dim != -1 else tla.get_aicore_num(args.device)
        )
        print(f"--- basic_vadd_simt n={n_ele} thread_block_dim={VECTOR_ELE} ---")

        a = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
        b = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
        c = torch.full((n_ele,), -7.0, dtype=torch.float32, device="npu")
        expected = a + b

        tla_a, tla_b, tla_c = create_tla_tensor(a), create_tla_tensor(b), create_tla_tensor(c)
        artifact = tla.compile(
            basic_vadd_simt,
            tla_a,
            tla_b,
            tla_c,
            arch_scope="aiv.c310",
            cache=not args.no_cache,
            cache_dir=cache_dir,
            force_recompile=args.force_recompile,
        )
        artifact(tla_a, tla_b, tla_c, block_dim=block_dim)
        torch.npu.synchronize()

        passed = bool(torch.isclose(c, expected, rtol=0.0, atol=float(args.atol)).all())
        print(f"passed={passed} cache_key={artifact.cache_key}")
        print(f"kernel.o={artifact.kernel_binary_path}")
        return 0 if passed else 1
    finally:
        tla.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run the SIMT vector add.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-dim", type=int, default=-1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
