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
_REDUCE_OP = tla.ReductionOp.ADD


@tla.kernel
def reduction_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    allocator = tla.utils.LocalmemAllocator()

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(1), tla.make_coord(0))
    x_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    z_ptr = tla.recast_ptr(
        allocator.allocate(ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    tla.copy(x_ub, x_gm)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    with tla.vec.func(mode="simd"):
        x_tile = tla.tile_view(x_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
        z_tile = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(0))
        z_tile.store(x_tile.load().reduce(_REDUCE_OP))
    tla.set_flag(done)
    tla.wait_flag(done)
    tla.copy(z_gm, z_ub)
    tla.pipe_barrier(tla.pipes.ALL)


def _set_op(op: str) -> None:
    global _REDUCE_OP
    _REDUCE_OP = {
        "add": tla.ReductionOp.ADD,
        "max": tla.ReductionOp.MAX,
        "min": tla.ReductionOp.MIN,
    }[op]


def _tensor(shape: tuple[int, ...], data_ptr: int | None = None) -> Any:
    with runtime_mod._eager_capture():
        tla_shape = tla.make_shape(*shape)
        return tla.Tensor(
            tla_shape,
            tla.Float32,
            origin_shape=tla_shape,
            coord=tla.make_coord(*(0 for _ in shape)),
            stride=tla.make_stride(1),
            data_ptr=data_ptr,
        )


def _runtime_tensor(dev_buf: Any, shape: tuple[int, ...]) -> Any:
    tensor = _tensor(shape, int(dev_buf.contiguous().data_ptr()))
    tensor._external_binding = True
    return tensor


def _expected(op: str, x: Any) -> Any:
    if op == "add":
        return x.sum().reshape(1)
    if op == "max":
        return x.max().reshape(1)
    return x.min().reshape(1)


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        reduction_op,
        *type_args,
        arch_scope="aiv.c310",
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("add", "max", "min"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    _set_op(args.op)
    if args.build_only:
        artifact = _compile(args, _tensor((VECTOR_ELE,)), _tensor((1,)))
        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0

    tla.initialize(device=args.device)
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        x = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
        z = torch.full((1,), -999.0, dtype=torch.float32, device="npu")
        tla_x = _runtime_tensor(x, (VECTOR_ELE,))
        tla_z = _runtime_tensor(z, (1,))
        artifact = _compile(args, tla_x, tla_z)
        artifact(tla_x, tla_z, block=1)
        torch.npu.synchronize()
        ok = bool(torch.isclose(z, _expected(args.op, x), rtol=0.0, atol=1e-4).all())
        print(f"compile_ok=True host=torch_npu op={args.op} dtype=f32 layout=row")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"output equals expected reduction? {ok}")
        return 0 if ok else 1
    finally:
        tla.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
