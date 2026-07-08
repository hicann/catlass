from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import catlass as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_type_args,
    vector_kernel_config,
)

# End-to-end coverage for ``tla.cmp`` / compare masks:
#   - all six vector-vector modes (lt/le/gt/ge/eq/ne) via ``tla.where``
#   - vector-scalar compares against 0.0 (gt/ge)
#   - input mask on ``tla.cmp`` (pattern H)
#   - cmp result driving masked ``tla.add`` plus masked ``tla.store``
#   - static ``tile_view(coord=0)`` reference vs dynamic loop chunks (shape metadata
#     may differ while tile width stays VL-wide)

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_CMP_OP: Callable[[Any, Any], Any] | None = None
_OP_NAME: str = ""


@tla.kernel
def compare_mask(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(allocator, a_gm)
    b_ub = _make_ub_tensor(allocator, b_gm)
    out_ub = _make_ub_tensor(allocator, out_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            a_static = tla.tile_view(a_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            a_ref = a_static.load()
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                a_tile = _chunk(a_ub, i)
                b_tile = _chunk(b_ub, i)
                out_tile = _chunk(out_ub, i)

                av = a_tile.load()
                bv = b_tile.load()
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                zero = tla.sub(av, av)

                if _OP_NAME == "static_dynamic_lt":
                    m = tla.cmp(a_ref, bv, "lt")
                    out_tile.store(tla.where(m, bv, zero), mask=tail)
                elif _OP_NAME == "cmp_masked_fused":
                    m_lt = tla.cmp(av, bv, "lt")
                    m_ge = tla.cmp(av, 0.0, "ge")
                    fused = tla.add(av, bv, mask=m_lt)
                    out_tile.store(tla.where(m_ge, fused, zero), mask=tail)
                else:
                    out_tile.store(_CMP_OP(av, bv), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(allocator: Any, like_tensor: Any) -> Any:
    ptr = allocator.allocate(VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub)
    return tla.make_tensor_like(
        tla.recast_ptr(ptr, dtype=_KERNEL_DTYPE), like_tensor, tla.arch.RowMajor
    )


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _vector_vector_where(mode: str) -> Callable[[Any, Any], Any]:
    def op(lhs: Any, rhs: Any) -> Any:
        return tla.where(tla.cmp(lhs, rhs, mode), lhs, rhs)

    return op


def _operator_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "vector_scalar_gt": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(lhs, 0.0, "gt"), lhs, tla.sub(rhs, rhs)
            ),
            "default_atol": 1e-5,
        },
        "vector_scalar_ge": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(lhs, 0.0, "ge"), lhs, tla.sub(rhs, rhs)
            ),
            "default_atol": 1e-5,
        },
        "masked_vector_vector_lt": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(
                    lhs,
                    rhs,
                    "lt",
                    mask=tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE),
                ),
                lhs,
                rhs,
            ),
            "default_atol": 1e-5,
        },
        "cmp_masked_fused": {
            "op": None,
            "default_atol": 1e-5,
        },
        "static_dynamic_lt": {
            "op": None,
            "default_atol": 1e-5,
        },
    }
    for mode in ("lt", "le", "gt", "ge", "eq", "ne"):
        specs[f"vector_vector_{mode}"] = {
            "op": _vector_vector_where(mode),
            "default_atol": 1e-5,
        }
    return specs


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _CMP_OP, _OP_NAME
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"unknown compare-mask operator {op_name!r}; expected one of: {choices}")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _OP_NAME = op_name
    _CMP_OP = specs[op_name]["op"]
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, Any, Any]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 3)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    idx = torch.arange(VECTOR_ELE, dtype=dtype, device=device)
    a = ((idx % 31) - 15.0) * 0.5
    b = torch.flip(a, dims=[0]) + 0.25
    return a, b


def _chunk_expected(a: Any, b: Any, op_name: str, torch: Any) -> Any:
    if op_name.startswith("vector_vector_"):
        mode = op_name.removeprefix("vector_vector_")
        if mode == "lt":
            return torch.where(a < b, a, b)
        if mode == "le":
            return torch.where(a <= b, a, b)
        if mode == "gt":
            return torch.where(a > b, a, b)
        if mode == "ge":
            return torch.where(a >= b, a, b)
        if mode == "eq":
            return torch.where(a == b, a, b)
        if mode == "ne":
            return torch.where(a != b, a, b)
    if op_name == "vector_scalar_gt":
        return torch.where(a > 0.0, a, torch.zeros_like(a))
    if op_name == "vector_scalar_ge":
        return torch.where(a >= 0.0, a, torch.zeros_like(a))
    if op_name == "masked_vector_vector_lt":
        lane = torch.arange(a.numel(), device=a.device) % VL_ELE
        active = lane < (VL_ELE // 2)
        return torch.where((a < b) & active, a, b)
    if op_name == "cmp_masked_fused":
        m_lt = a < b
        partial = torch.where(m_lt, a + b, torch.zeros_like(a))
        m_ge = a >= 0.0
        return torch.where(m_ge, partial, torch.zeros_like(a))
    if op_name == "static_dynamic_lt":
        n = b.numel()
        mask = a[:n] < b
        return torch.where(mask, b, torch.zeros_like(b))
    raise AssertionError(op_name)


def _iter_chunks(tensor: Any, chunk_size: int) -> list[Any]:
    return [tensor[i : i + chunk_size] for i in range(0, tensor.numel(), chunk_size)]


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    import torch

    a_full, b_full = inputs
    if op_name != "static_dynamic_lt":
        a_chunks = _iter_chunks(a_full, VL_ELE)
        b_chunks = _iter_chunks(b_full, VL_ELE)
        out_chunks = [
            _chunk_expected(a_chunks[i], b_chunks[i], op_name, torch)
            for i in range(len(a_chunks))
        ]
        return torch.cat(out_chunks, dim=0)

    a_ref = _iter_chunks(a_full, VL_ELE)[0]
    out_chunks = [
        _chunk_expected(a_ref, chunk, op_name, torch)
        for chunk in _iter_chunks(b_full, VL_ELE)
    ]
    return torch.cat(out_chunks, dim=0)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector compare-mask kernels.",
        kernel=compare_mask,
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
        env_compile_jobs="COMPARE_MASK_COMPILE_JOBS",
        float_dtypes=frozenset({"f32"}),
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
