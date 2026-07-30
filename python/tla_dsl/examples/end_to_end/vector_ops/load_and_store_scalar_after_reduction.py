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
VEC_FUNC_STORE_INDEX = 7
VECTOR_STORE_INDEX = 8


@tla.kernel
def load_and_store_scalar_after_reduction(
    mem_x: tla.Tensor,
    mem_stored: tla.Tensor,
    mem_reduced: tla.Tensor,
) -> None:
    """Exercise UB scalar accesses both inside and outside ``tla.vec.func``."""
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_func_done = tla.flag(
        "vec_func_done", tla.arch.VECTOR, tla.arch.SCALAR
    )
    scalar_to_mte3 = tla.flag("scalar_to_mte3", tla.arch.SCALAR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    stored_gm = tla.tile_view(mem_stored, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    reduced_gm = tla.tile_view(mem_reduced, tla.make_shape(1), tla.make_coord(0))

    x_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    stored_ptr = tla.recast_ptr(
        allocator.allocate(VECTOR_ELE * ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    reduced_ptr = tla.recast_ptr(
        allocator.allocate(ELEMENT_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    stored_ub = tla.make_tensor_like(stored_ptr, stored_gm, tla.arch.RowMajor)
    reduced_ub = tla.make_tensor_like(reduced_ptr, reduced_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(stored_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x_vec_tile = tla.tile_view(
                x_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            reduced_vec_tile = tla.tile_view(
                reduced_ub, tla.make_shape(1), tla.make_coord(0)
            )
            stored_vec_tile = tla.tile_view(
                stored_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            reduce_mask = tla.create_mask(
                pattern=tla.mask.ALL,
                dtype=tla.Float32,
            )
            reduced = x_vec_tile.load().reduce(
                tla.ReductionOp.ADD,
                mask=reduce_mask,
            )
            reduced_vec_tile.store(reduced)

            # Keep the reduction store, scalar load, and scalar store in the
            # same outlined helper. The barrier makes the reduction slot
            # visible to the scalar pipe before it is read.
            tla.local_mem_bar(
                tla.params.MemType.VEC_STORE,
                tla.params.MemType.SCALAR_LOAD,
            )
            vec_func_scalar = reduced_vec_tile[0]
            stored_vec_tile[VEC_FUNC_STORE_INDEX] = vec_func_scalar

        # local_mem_bar only orders accesses inside the helper. The outer
        # scalar pipe must also wait until the complete vector helper,
        # including its scalar store, has finished.
        tla.set_flag(vec_func_done)
        tla.wait_flag(vec_func_done)

        # Exercise a second UB scalar load/store pair directly in tla.vector,
        # outside tla.vec.func.
        reduced_scalar_tile = tla.tile_view(
            reduced_ub, tla.make_shape(1), tla.make_coord(0)
        )
        stored_scalar_tile = tla.tile_view(
            stored_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
        )
        vector_scalar = reduced_scalar_tile[0]
        stored_scalar_tile[VECTOR_STORE_INDEX] = vector_scalar

        # MTE3 must not read stored_ub before either scalar store is visible.
        tla.set_flag(scalar_to_mte3)
        tla.wait_flag(scalar_to_mte3)
        tla.copy(stored_gm, stored_ub)
        tla.copy(reduced_gm, reduced_ub)
        tla.pipe_barrier(tla.pipes.ALL)


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


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        load_and_store_scalar_after_reduction,
        *type_args,
        arch_scope="aiv.c310",
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce an f32 UB vector, then load/store the result as scalar SSA "
            "both inside tla.vec.func and directly inside tla.vector."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dump-tla", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    type_args = (
        _tensor((VECTOR_ELE,)),
        _tensor((VECTOR_ELE,)),
        _tensor((1,)),
    )
    if args.dump_tla:
        print(
            load_and_store_scalar_after_reduction.dump_mlir(type_args=type_args)
        )
        return 0
    if args.build_only:
        artifact = _compile(args, *type_args)
        print("compile_ok=True")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0

    tla.initialize(device=args.device)
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        x = torch.linspace(
            -17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu"
        )
        stored_out = torch.full(
            (VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu"
        )
        reduced_out = torch.full((1,), -999.0, dtype=torch.float32, device="npu")

        tla_x = _runtime_tensor(x, (VECTOR_ELE,))
        tla_stored = _runtime_tensor(stored_out, (VECTOR_ELE,))
        tla_reduced = _runtime_tensor(reduced_out, (1,))
        artifact = _compile(args, tla_x, tla_stored, tla_reduced)
        artifact(tla_x, tla_stored, tla_reduced, block=1)
        torch.npu.synchronize()

        expected_scalar = x.sum()
        expected_stored = x.clone()
        expected_stored[VEC_FUNC_STORE_INDEX] = expected_scalar
        expected_stored[VECTOR_STORE_INDEX] = expected_scalar

        reduced_ok = bool(
            torch.isclose(reduced_out[0], expected_scalar, rtol=0.0, atol=1e-4)
        )
        vec_func_store_ok = bool(
            torch.isclose(
                stored_out[VEC_FUNC_STORE_INDEX],
                expected_scalar,
                rtol=0.0,
                atol=1e-4,
            )
        )
        vector_store_ok = bool(
            torch.isclose(
                stored_out[VECTOR_STORE_INDEX],
                expected_scalar,
                rtol=0.0,
                atol=1e-4,
            )
        )
        stored_ok = bool(
            torch.isclose(stored_out, expected_stored, rtol=0.0, atol=1e-4).all()
        )

        print(
            "compile_ok=True host=torch_npu "
            "op=load_and_store_scalar_after_reduction dtype=f32"
        )
        print(f"kernel.o path={artifact.kernel_binary_path}")
        print("launch_ok=True")
        print(f"reduction UB slot equals expected scalar? {reduced_ok}")
        print(
            "tla.vec.func UB scalar load/store wrote index "
            f"{VEC_FUNC_STORE_INDEX}? {vec_func_store_ok}"
        )
        print(
            "tla.vector UB scalar load/store wrote index "
            f"{VECTOR_STORE_INDEX}? {vector_store_ok}"
        )
        print(f"complete stored output matches expected? {stored_ok}")
        print(f"expected scalar={float(expected_scalar.cpu())}")
        print(f"reduced scalar={float(reduced_out[0].cpu())}")
        print(
            "stored scalars="
            f"({float(stored_out[VEC_FUNC_STORE_INDEX].cpu())}, "
            f"{float(stored_out[VECTOR_STORE_INDEX].cpu())})"
        )
        print(f"stored_out[:9]={stored_out[:9].cpu()}")
        return (
            0
            if reduced_ok and vec_func_store_ok and vector_store_ok and stored_ok
            else 1
        )
    finally:
        tla.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
