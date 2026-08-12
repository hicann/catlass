"""Two views of one UB allocation, one of them offset, read from a SIMT region.

``full`` covers the whole 16-element allocation; ``tail`` is the same allocation
viewed from element 4. Writing ``tail[0]`` must be observable as ``full[4]``.

The SIMT launch ABI passes buffers as bare pointers -- there is no descriptor on
the other side -- so a view's starting coordinate has to be folded into the
pointer at the launch site. If it is not, both views collapse to the same base
address, ``tail[0] = 7.0`` lands on element 0, and the kernel silently returns
the wrong value with no diagnostic anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla

N_UB = 16
OFFSET = 4
SENTINEL = -999.0
VALUE = 7.0


@tla.kernel
def ub_offset_view(out: tla.Tensor) -> None:
    ptr = tla.allocate(N_UB, tla.Float32, tla.AddressSpace.ub, 64)

    # The whole allocation, starting at 0.
    full = tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(N_UB),
            tla.make_stride(1),
            origin_shape=tla.make_shape(N_UB),
        ),
        coord=tla.make_coord(0),
    )

    # The same allocation, viewed from element OFFSET.
    tail = tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(N_UB - OFFSET),
            tla.make_stride(1),
            origin_shape=tla.make_shape(N_UB),
        ),
        coord=tla.make_coord(OFFSET),
    )

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            # Clear the slot the offset view must NOT touch, so hitting element 0
            # instead of element 4 is visible rather than reading stale memory.
            full[0] = 0.0
            tail[0] = VALUE
            tla.arch.sync_threads()
            out[0] = full[OFFSET]
            out[1] = full[0]

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


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    _apply_cache_env(args)

    torch.npu.set_device(args.device)
    if True:
        print(f"--- ub_offset_view n={N_UB} offset={OFFSET} ---")

        out = torch.full((2,), SENTINEL, dtype=torch.float32, device="npu")
        t_out = from_dlpack(out.contiguous(), layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(ub_offset_view, t_out, options=NPU_ARCH)
        artifact(t_out, block_num=1)
        torch.npu.synchronize()

        at_offset, at_zero = float(out[0]), float(out[1])
        # tail[0] must land on full[OFFSET], and must NOT land on full[0].
        passed = at_offset == VALUE and at_zero == 0.0
        print(f"full[{OFFSET}]={at_offset} (want {VALUE})   full[0]={at_zero} (want 0.0)")
        if at_zero == VALUE:
            print("        the offset view wrote element 0: the view's coord was dropped")
        print(f"passed={passed} cache_key={artifact.cache_key}")
        return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Offset UB view read from a SIMT region.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
