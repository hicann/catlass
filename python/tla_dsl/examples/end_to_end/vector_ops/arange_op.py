from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_type_args,
    vector_kernel_config,
)

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i8", "i16", "i32")

_KERNEL_DTYPE = tla.Int32
_KERNEL_TORCH_DTYPE = None
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_ARANGE_ORDER = "increase"


@tla.kernel
def arange_op(mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_ub_ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    out_ub_ptr = tla.recast_ptr(out_ub_ptr, dtype=_KERNEL_DTYPE)
    out_ub = tla.make_tensor_like(out_ub_ptr, out_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                out_tile = tla.tile_view(
                    out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                chunk_start = i * VL_ELE
                out_tile.store(
                    tla.arange(chunk_start, order=_ARANGE_ORDER, dtype=_KERNEL_DTYPE)
                )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "increase": {
            "default_atol": 0,
        },
        "decrease": {
            "default_atol": 0,
        },
    }


def _set_kernel_config(
    op_name: str,
    dtype_name: str,
    shape: tuple[int, ...] | None = None,
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_TORCH_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _ARANGE_ORDER
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"unknown arange variant {op_name!r}; expected one of: {choices}")
    _ARANGE_ORDER = op_name
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_TORCH_DTYPE = config.torch_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str,
    dtype_name: str,
    shape: tuple[int, ...] | None = None,
) -> tuple[Any, ...]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 1)


def _make_inputs(args: Any, dtype_name: str, _torch: Any) -> tuple[Any, ...]:
    """Arange is output-only; sync module globals and return no GM input tensors."""
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    return tuple()


def _expected(op_name: str, _inputs: tuple[Any, ...]) -> Any:
    import torch

    if op_name == "decrease":
        result = torch.empty(VECTOR_ELE, dtype=torch.int64)
        for i in range(LOOPS):
            start = i * VL_ELE
            end = min((i + 1) * VL_ELE, VECTOR_ELE)
            block_len = end - start
            result[start:end] = torch.arange(
                start + VL_ELE - 1, start + VL_ELE - 1 - block_len, -1
            )
        idx = result
    elif op_name == "increase":
        idx = torch.arange(VECTOR_ELE, dtype=torch.int64, device="cpu")
    else:
        raise ValueError(f"mode can only be 'increase' or 'decrease' for tla.arange")
    return idx.to(dtype=_KERNEL_TORCH_DTYPE, device="npu")


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector arange kernels.",
        kernel=arange_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        compile_only_type_args=_compile_only_type_args,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=lambda _op, _dtype: False,
        print_skip=lambda _op, _dtype, _shape: None,
        script_path=Path(__file__).resolve(),
        env_compile_jobs="TLA_DSL_ARANGE_COMPILE_JOBS",
        float_dtypes=frozenset(),
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
