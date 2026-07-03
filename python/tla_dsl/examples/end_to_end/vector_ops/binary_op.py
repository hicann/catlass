from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import catlass as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_type_args,
    shape_label,
    vector_kernel_config,
)

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i8", "i16", "i32", "f16", "f32", "bf16")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_BINARY_OP: Callable[[Any, Any], Any] | None = None


@tla.kernel
def binary_op(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    y_ub_ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    z_ub_ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    x_ub_ptr = tla.recast_ptr(x_ub_ptr, dtype=_KERNEL_DTYPE)
    y_ub_ptr = tla.recast_ptr(y_ub_ptr, dtype=_KERNEL_DTYPE)
    z_ub_ptr = tla.recast_ptr(z_ub_ptr, dtype=_KERNEL_DTYPE)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    tla.copy(x_ub, x_gm)
    tla.copy(y_ub, y_gm)

    tla.set_flag(ub_loaded)
    tla.wait_flag(ub_loaded)
    with tla.vec.func(mode="simd"):
        for i in tla.range(LOOPS):
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(i))
            y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(i))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(i))

            z_tile.store(_BINARY_OP(x_tile.load(), y_tile.load()))

    tla.set_flag(vec_done)
    tla.wait_flag(vec_done)

    tla.copy(z_gm, z_ub)
    tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "add": {
            "op": lambda lhs, rhs: lhs + rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "sub": {
            "op": lambda lhs, rhs: lhs - rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "mul": {
            "op": lambda lhs, rhs: lhs * rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "div": {
            "op": lambda lhs, rhs: lhs / rhs,
            "default_atol": 1e-3,
            "nonzero_rhs": True,
        },
        "max": {
            "op": lambda lhs, rhs: tla.max(lhs, rhs),
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "min": {
            "op": lambda lhs, rhs: tla.min(lhs, rhs),
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    if dtype_name == "i8" and op_name in {"mul", "div"}:
        return True
    # AVE vdiv rejects vector<128xbf16> (padded bf16 layout).
    if dtype_name == "bf16" and op_name == "div":
        return True
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    if dtype_name == "i8" and op_name in {"mul", "div"}:
        reason = "i8 is not supported for mul/div"
    elif dtype_name == "bf16" and op_name == "div":
        reason = "bf16 vector div is not supported by AVE vdiv"
    else:
        reason = "unsupported case"
    print(
        f"skip op={op_name} dtype={dtype_name} shape={shape_label(shape)}: {reason}"
    )


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _BINARY_OP
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"unknown binary operator {op_name!r}; expected one of: {choices}")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _BINARY_OP = specs[op_name]["op"]
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, Any, Any]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 3)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    dtype = vector_kernel_config(dtype_name, args.shape, ALL_DTYPES).torch_dtype
    device = "npu"
    if dtype_name in {"i8", "i16", "i32"}:
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        x = ((arange % 97) - 48).to(dtype)
        y = ((arange % 31) + (1 if args.op == "div" else -15)).to(dtype)
        if args.op == "div":
            y = y.clamp_min(1)
        return x, y
    x = torch.arange(VECTOR_ELE, dtype=dtype, device=device) + 1
    y = (torch.arange(VECTOR_ELE, dtype=dtype, device=device) * 0.5) + 2
    return x, y


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    x, y = inputs
    if op_name == "add":
        return x + y
    if op_name == "sub":
        return x - y
    if op_name == "mul":
        return x * y
    if op_name == "div":
        if not x.is_floating_point():
            return x.div(y, rounding_mode="trunc")
        return x / y
    if op_name == "max":
        return x.maximum(y)
    if op_name == "min":
        return x.minimum(y)
    raise AssertionError(op_name)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a vector binary op.",
        kernel=binary_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        compile_only_type_args=_compile_only_type_args,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        env_compile_jobs="BINARY_OP_COMPILE_JOBS",
        float_dtypes=frozenset({"f32", "f16", "bf16"}),
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
