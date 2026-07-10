from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import catlass as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_type_args,
    vector_kernel_config,
)

VECTOR_ELE = 512
VL_ELE = 64
LOOPS = 8
ALL_DTYPES = ("i8", "i16", "i32", "f16", "f32", "bf16")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_OP: Literal["interleave", "deinterleave"] = "interleave"


@tla.kernel
def interleave_op(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_out0: tla.Tensor,
    mem_out1: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out0_gm = tla.tile_view(mem_out0, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out1_gm = tla.tile_view(mem_out1, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(allocator, a_gm)
    b_ub = _make_ub_tensor(allocator, b_gm)
    out0_ub = _make_ub_tensor(allocator, out0_gm)
    out1_ub = _make_ub_tensor(allocator, out1_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                a_t = _chunk(a_ub, i)
                b_t = _chunk(b_ub, i)
                out0_t = _chunk(out0_ub, i)
                out1_t = _chunk(out1_ub, i)

                if tla.const_expr(_KERNEL_OP == "interleave"):
                    r0, r1 = tla.interleave(a_t.load(), b_t.load())
                else:
                    r0, r1 = tla.deinterleave(a_t.load(), b_t.load())
                out0_t.store(r0)
                out1_t.store(r1)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out0_gm, out0_ub)
        tla.copy(out1_gm, out1_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(allocator: Any, like_tensor: Any) -> Any:
    ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    ptr = tla.recast_ptr(ptr, dtype=_KERNEL_DTYPE)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "interleave": {
            "default_atol": 0.0,
        },
        "deinterleave": {
            "default_atol": 0.0,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE
    global _KERNEL_OP
    if op_name not in _operator_specs():
        raise SystemExit("unknown interleave operator")

    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _KERNEL_OP = op_name
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, ...]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 4)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"

    if dtype_name in {"i8", "i16", "i32"}:
        idx = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        a = ((idx % 97) - 48).to(dtype)
        b = ((idx % 53) + 3).to(dtype)
        return a, b

    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    a = ((idx % 37.0) * 0.25 + 1.0).to(dtype)
    b = ((idx % 29.0) * 0.5 + 2.0).to(dtype)
    return a, b


def _expected_interleave(a: Any, b: Any) -> tuple[Any, Any]:
    import torch

    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    half = VL_ELE // 2

    for base in range(0, VECTOR_ELE, VL_ELE):
        a_chunk = a[base : base + VL_ELE]
        b_chunk = b[base : base + VL_ELE]

        out0_chunk = torch.empty_like(a_chunk)
        out1_chunk = torch.empty_like(a_chunk)

        out0_chunk[0::2] = a_chunk[:half]
        out0_chunk[1::2] = b_chunk[:half]
        out1_chunk[0::2] = a_chunk[half:]
        out1_chunk[1::2] = b_chunk[half:]

        out0[base : base + VL_ELE] = out0_chunk
        out1[base : base + VL_ELE] = out1_chunk

    return out0, out1


def _expected_deinterleave(a: Any, b: Any) -> tuple[Any, Any]:
    import torch

    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    half = VL_ELE // 2

    for base in range(0, VECTOR_ELE, VL_ELE):
        a_chunk = a[base : base + VL_ELE]
        b_chunk = b[base : base + VL_ELE]

        out0_chunk = torch.empty_like(a_chunk)
        out1_chunk = torch.empty_like(a_chunk)

        out0_chunk[:half] = a_chunk[0::2]
        out0_chunk[half:] = b_chunk[0::2]
        out1_chunk[:half] = a_chunk[1::2]
        out1_chunk[half:] = b_chunk[1::2]

        out0[base : base + VL_ELE] = out0_chunk
        out1[base : base + VL_ELE] = out1_chunk

    return out0, out1


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    a, b = inputs
    if op_name == "interleave":
        return _expected_interleave(a, b)
    if op_name == "deinterleave":
        return _expected_deinterleave(a, b)
    raise AssertionError(op_name)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector interleave/deinterleave ops.",
        kernel=interleave_op,
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
        env_compile_jobs="INTERLEAVE_OP_COMPILE_JOBS",
        float_dtypes=frozenset({"f32", "f16", "bf16"}),
        output_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
