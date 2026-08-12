"""SIMT block reduction through a UB buffer shared between the threads.

Each thread sums a strided slice of the GM input into a private accumulator,
publishes that partial into a UB buffer, and after a barrier thread 0 folds the
partials into a single GM output.

The UB buffer is allocated outside the region with ``tla.allocate`` and viewed
as a tensor with ``tla.make_tensor``; the SIMT outlining passes it to the vector
function as a second buffer parameter, alongside the GM ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla

N_ELE = 1024
THREADS = 128
_KERNEL_DTYPE = tla.Float32


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def reduction_simt(
    gm_in: tla.Tensor,
    gm_out: tla.Tensor
) -> None:
    # Shared scratch: one f32 partial per thread, in UB.
    ub_ptr = tla.allocate(THREADS, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_partial = tla.make_tensor(
        ub_ptr, tla.make_layout(tla.make_shape(THREADS), tla.make_stride(1))
    )

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()

            acc = _KERNEL_DTYPE(0.0)
            for i in tla.range(tid, N_ELE, nthreads):
                acc = acc + gm_in[i]
            ub_partial[tid] = acc

            tla.arch.sync_threads()

            if tid == 0:
                total = _KERNEL_DTYPE(0.0)
                for j in tla.range(0, THREADS, 1):
                    total = total + ub_partial[j]
                gm_out[0] = total

        tla.pipe_barrier(tla.pipes.ALL)


# Arch selection is the only Host compile knob now; caching moved to env vars
# (dsl e745bf10 converged the Host surface). --force-recompile / --no-cache are
# kept as flags and translated here so the runner scripts keep working.
NPU_ARCH = "--npu-arch 3510"


def _apply_cache_env(args: argparse.Namespace) -> None:
    import os

    if getattr(args, "force_recompile", False):
        os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1"
    if getattr(args, "no_cache", False):
        os.environ["CATLASS_DSL_CACHE"] = "0"
    if getattr(args, "cache_dir", None):
        os.environ["CATLASS_DSL_CACHE_DIR"] = str(args.cache_dir)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    def create_tla_tensor(dev_buf):
        return from_dlpack(dev_buf.contiguous(), layout_tag=tla.arch.RowMajor)

    _apply_cache_env(args)

    torch.npu.set_device(args.device)
    if True:
        print(f"--- reduction_simt n={N_ELE} block={THREADS} ---")

        src = torch.arange(N_ELE, dtype=torch.float32, device="npu")
        dst = torch.full((1,), -999.0, dtype=torch.float32, device="npu")
        expected = src.sum()

        tla_in, tla_out = create_tla_tensor(src), create_tla_tensor(dst)
        artifact = tla.compile(reduction_simt, tla_in, tla_out, options=NPU_ARCH)
        # One block: the reduction is block-local, so a multi-core launch would
        # have every core redundantly write the same output.
        artifact(tla_in, tla_out, block_num=1)
        torch.npu.synchronize()

        got = float(dst[0].item())
        want = float(expected.item())
        passed = abs(got - want) <= float(args.atol) * max(1.0, abs(want))
        print(f"got={got} expected={want}")
        print(f"passed={passed} cache_key={artifact.cache_key}")
        return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run the SIMT block reduction.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
