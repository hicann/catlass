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
ALL_DTYPES = ("i16", "i32", "f16", "f32")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)


@tla.kernel
def mask_logic(
    mem_a: tla.Tensor,
    mem_rnot: tla.Tensor,
    mem_rand: tla.Tensor,
    mem_ror: tla.Tensor,
    mem_rxor: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rnot_gm = tla.tile_view(mem_rnot, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rand_gm = tla.tile_view(mem_rand, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    ror_gm = tla.tile_view(mem_ror, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rxor_gm = tla.tile_view(mem_rxor, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(allocator, a_gm)
    rnot_ub = _make_ub_tensor(allocator, rnot_gm)
    rand_ub = _make_ub_tensor(allocator, rand_gm)
    ror_ub = _make_ub_tensor(allocator, ror_gm)
    rxor_ub = _make_ub_tensor(allocator, rxor_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                a_t = _chunk(a_ub, i)
                rnot_t = _chunk(rnot_ub, i)
                rand_t = _chunk(rand_ub, i)
                ror_t = _chunk(ror_ub, i)
                rxor_t = _chunk(rxor_ub, i)

                av = a_t.load()
                zero = tla.sub(av, av)
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                m_h = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                m_q = tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                m_m4 = tla.create_mask(pattern=tla.mask.M4, dtype=_KERNEL_DTYPE)

                m_not = tla.not_(m_q, tail)
                m_and = tla.and_(m_h, m_m4, tail)
                m_or = tla.or_(m_q, m_m4, tail)
                m_xor = tla.xor(m_h, m_m4, tail)

                rnot_t.store(tla.where(m_not, av, zero), mask=tail)
                rand_t.store(tla.where(m_and, av, zero), mask=tail)
                ror_t.store(tla.where(m_or, av, zero), mask=tail)
                rxor_t.store(tla.where(m_xor, av, zero), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(rnot_gm, rnot_ub)
        tla.copy(rand_gm, rand_ub)
        tla.copy(ror_gm, ror_ub)
        tla.copy(rxor_gm, rxor_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(allocator: Any, like_tensor: Any) -> Any:
    ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 32, tla.AddressSpace.ub
    )
    return tla.make_tensor_like(
        tla.recast_ptr(ptr, dtype=_KERNEL_DTYPE), like_tensor, tla.arch.RowMajor
    )


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"mask_logic": {"default_atol": 1e-3}}


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
    global _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown mask-logic operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, ...]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 5)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    if dtype_name in {"i16", "i32"}:
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        return (((arange % 29) - 11).to(dtype),)
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    return (((idx % 17.0) * 0.5 - 3.0).to(dtype),)


def _select(torch: Any, x: Any, keep: Any) -> Any:
    return torch.where(keep, x, torch.zeros_like(x))


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    del op_name
    import torch

    (a,) = inputs
    lane = torch.arange(VECTOR_ELE, device=a.device) % VL_ELE
    h = lane < (VL_ELE // 2)
    q = lane < (VL_ELE // 4)
    m4 = (lane % 4) == 0
    return (
        _select(torch, a, ~q),
        _select(torch, a, h & m4),
        _select(torch, a, q | m4),
        _select(torch, a, h ^ m4),
    )


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a predicate mask logic kernel.",
        kernel=mask_logic,
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
        script_path=Path(__file__),
        env_compile_jobs="TLADSL_MASK_LOGIC_COMPILE_JOBS",
        float_dtypes=frozenset({"f16", "f32"}),
        output_count=4,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
