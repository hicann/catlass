from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catlass as tla
from catlass import runtime as runtime_mod
from catlass.params import UnalignStoreParams

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

VECTOR_ELE = 128
ELEMENT_BYTES = 4
_REDUCE_OP = tla.ReductionOp.ADD


@tla.kernel
def reduction_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Reduce 128 fp32 → two 1-element results.

    Tile 0 (first 64 ele): result at z coord 0 (aligned) → normal store.
    Tile 1 (last  64 ele): result at z coord 1 (unaligned) → UnalignStoreParams.
    """
    TILE_ELE = 64
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(2), tla.make_coord(0))
    x_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(2, tla.Float32, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            # tile 0: first 64 elements, result at z coord 0 (aligned)
            x0 = tla.tile_view(x_ub, tla.make_shape(TILE_ELE), tla.make_coord(0))
            z0 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(0))
            z0.store(x0.load().reduce(_REDUCE_OP))
            # tile 1: last 64 elements, result at z coord 1 (NOT aligned)
            x1 = tla.tile_view(x_ub, tla.make_shape(TILE_ELE), tla.make_coord(1))
            z1 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(1))
            z1.store(x1.load().reduce(_REDUCE_OP), params=UnalignStoreParams())
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
        artifact = _compile(args, _tensor((VECTOR_ELE,)), _tensor((2,)))
        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0

    tla.initialize(device=args.device)
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)

        TILE_ELE = 64
        x = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
        z = torch.full((2,), -999.0, dtype=torch.float32, device="npu")
        tla_x = _runtime_tensor(x, (VECTOR_ELE,))
        tla_z = _runtime_tensor(z, (2,))
        artifact = _compile(args, tla_x, tla_z)
        artifact(tla_x, tla_z, block=1)
        torch.npu.synchronize()

        exp0 = _expected(args.op, x[:TILE_ELE])
        exp1 = _expected(args.op, x[TILE_ELE:])
        z0_cpu = z[0].cpu()
        z1_cpu = z[1].cpu()
        exp0_cpu = exp0.cpu()
        exp1_cpu = exp1.cpu()
        print(f"z[0] actual  = {z0_cpu.item():.6f}")
        print(f"z[0] expected= {exp0_cpu.item():.6f}")
        ok0 = bool(torch.isclose(z[0].reshape(1), exp0, rtol=0.0, atol=1e-4).all())
        print(f"z[1] actual  = {z1_cpu.item():.6f}")
        print(f"z[1] expected= {exp1_cpu.item():.6f}")
        ok1 = bool(torch.isclose(z[1].reshape(1), exp1, rtol=0.0, atol=1e-4).all())
        ok = ok0 and ok1
        print(f"tile0 (coord=0, aligned)   match? {ok0}")
        print(f"tile1 (coord=1, unalign)   match? {ok1}")

        print(f"compile_ok=True host=torch_npu op={args.op} dtype=f32 layout=row")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"output equals expected reduction? {ok}")
        return 0 if ok else 1
    finally:
        tla.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
