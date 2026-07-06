from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catlass as tla
from catlass import runtime as runtime_mod

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

VECTOR_ELE = 64
ELEMENT_BYTES = 4
_IDX_ELEMENT_BYTES = 4


@tla.kernel
def gather_op(mem_src: tla.Tensor, mem_idx: tla.Tensor, mem_dst: tla.Tensor) -> None:
    x_loaded = tla.flag("x_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    idx_loaded = tla.flag("idx_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    allocator = tla.utils.LocalmemAllocator()

    src_gm = tla.tile_view(mem_src, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    idx_gm = tla.tile_view(mem_idx, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    src_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    idx_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * _IDX_ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Int32,
    )
    dst_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )

    src_ub = tla.make_tensor_like(src_ptr, src_gm, tla.arch.RowMajor)
    idx_ub = tla.make_tensor_like(idx_ptr, idx_gm, tla.arch.RowMajor)
    dst_ub = tla.make_tensor_like(dst_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(idx_ub, idx_gm)

        tla.set_flag(x_loaded)
        tla.wait_flag(x_loaded)

        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(src_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
            idx_tile = tla.tile_view(idx_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
            dst_tile = tla.tile_view(dst_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

            indices = idx_tile.load()
            gathered = tla.gather(x_tile, indices)
            dst_tile.store(gathered)

        tla.set_flag(done)
        tla.wait_flag(done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _tensor(shape: tuple[int, ...], dtype: type[tla.Numeric], data_ptr: int | None = None) -> Any:
    with runtime_mod._eager_capture():
        tla_shape = tla.make_shape(*shape)
        return tla.Tensor(
            tla_shape,
            dtype,
            origin_shape=tla_shape,
            coord=tla.make_coord(*(0 for _ in shape)),
            stride=tla.make_stride(1),
            data_ptr=data_ptr,
        )


def _runtime_tensor(dev_buf: Any, shape: tuple[int, ...], dtype: type[tla.Numeric]) -> Any:
    tensor = _tensor(shape, dtype, int(dev_buf.contiguous().data_ptr()))
    tensor._external_binding = True
    return tensor


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        gather_op,
        *type_args,
        arch_scope="aiv.c310",
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.build_only:
        artifact = _compile(
            args,
            _tensor((VECTOR_ELE,), tla.Float32),
            _tensor((VECTOR_ELE,), tla.Int32),
            _tensor((VECTOR_ELE,), tla.Float32),
        )
        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0

    tla.initialize(device=args.device)
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        src = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
        idx = torch.arange(VECTOR_ELE - 1, -1, -1, dtype=torch.int32, device="npu")
        dst = torch.full((VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu")

        tla_src = _runtime_tensor(src, (VECTOR_ELE,), tla.Float32)
        tla_idx = _runtime_tensor(idx, (VECTOR_ELE,), tla.Int32)
        tla_dst = _runtime_tensor(dst, (VECTOR_ELE,), tla.Float32)

        artifact = _compile(args, tla_src, tla_idx, tla_dst)
        artifact(tla_src, tla_idx, tla_dst, block=1)
        torch.npu.synchronize()

        expected = src[idx.to(torch.long)]
        ok = bool(torch.isclose(dst, expected, rtol=0.0, atol=1e-4).all())
        print(f"compile_ok=True host=torch_npu op=gather dtype=f32 layout=row")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"output equals expected gather? {ok}")
        return 0 if ok else 1
    finally:
        tla.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
