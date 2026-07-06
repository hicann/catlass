from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

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
_FLOAT_UNARY_DTYPES = frozenset({"f16", "f32"})
_INTEGER_DTYPES = frozenset({"i8", "i16", "i32"})
_KERNEL_OUTPUTS = 4

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_DEFAULT_SENTINEL: float | int = -7
_UNARY_MODE: Literal["unmasked_unary", "masked_unary", "masked_abs"] = "unmasked_unary"
_UNARY_OP: Callable[[Any], Any] | None = None

_UNMASKED_UNARY_OPS = ("exp", "log", "sqrt", "abs")
_OPERATOR_SPECS: dict[str, dict[str, Any]] = {
    "exp": {"op": tla.exp, "default_atol": 1e-3, "kind": "float"},
    "log": {"op": tla.log, "default_atol": 1e-4, "kind": "float"},
    "sqrt": {"op": tla.sqrt, "default_atol": 1e-4, "kind": "float"},
    "abs": {"op": tla.abs, "default_atol": 1e-4, "kind": "numeric"},
    "masked_unary": {"default_atol": 1e-3},
    "masked_abs": {"default_atol": 1e-4},
}
_SCRIPT_PATH = Path(__file__).resolve()


def _make_ub_tensor(allocator: Any, gm_tile: Any) -> Any:
    ptr = allocator.allocate(
        VECTOR_ELE * _KERNEL_ELEMENT_BYTES, 256, tla.AddressSpace.ub
    )
    ptr = tla.recast_ptr(ptr, dtype=_KERNEL_DTYPE)
    return tla.make_tensor_like(ptr, gm_tile, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


@tla.kernel
def vector_unary(
    mem_x: tla.Tensor,
    mem_z: tla.Tensor,
    mem_rlog: tla.Tensor,
    mem_rsqrt: tla.Tensor,
    mem_rabs: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    allocator = tla.utils.LocalmemAllocator()

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rlog_gm = tla.tile_view(mem_rlog, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rsqrt_gm = tla.tile_view(mem_rsqrt, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rabs_gm = tla.tile_view(mem_rabs, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub = _make_ub_tensor(allocator, x_gm)
    z_ub = _make_ub_tensor(allocator, z_gm)
    rlog_ub = _make_ub_tensor(allocator, rlog_gm)
    rsqrt_ub = _make_ub_tensor(allocator, rsqrt_gm)
    rabs_ub = _make_ub_tensor(allocator, rabs_gm)

    with tla.vector():
        tla.copy(x_ub, x_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            if _UNARY_MODE == "unmasked_unary":
                for i in tla.range(LOOPS):
                    x_t = _chunk(x_ub, i)
                    z_t = _chunk(z_ub, i)
                    z_t.store(_UNARY_OP(x_t.load()))
            elif _UNARY_MODE == "masked_abs":
                remaining = VECTOR_ELE
                for i in tla.range(LOOPS):
                    x_t = _chunk(x_ub, i)
                    z_t = _chunk(z_ub, i)
                    tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                    lane_mask = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                    z_t.store(tla.abs(x_t.load(), mask=lane_mask), mask=tail)
            else:
                remaining = VECTOR_ELE
                for i in tla.range(LOOPS):
                    x_t = _chunk(x_ub, i)
                    rexp_t = _chunk(z_ub, i)
                    rlog_t = _chunk(rlog_ub, i)
                    rsqrt_t = _chunk(rsqrt_ub, i)
                    rabs_t = _chunk(rabs_ub, i)

                    tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                    m_exp = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                    m_log = tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                    m_sqrt = tla.create_mask(pattern=tla.mask.M4, dtype=_KERNEL_DTYPE)
                    m_abs = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)

                    xv = x_t.load()
                    rexp_t.store(tla.exp(xv, mask=m_exp), mask=tail)
                    rlog_t.store(tla.log(xv, mask=m_log), mask=tail)
                    rsqrt_t.store(tla.sqrt(xv, mask=m_sqrt), mask=tail)
                    rabs_t.store(tla.abs(xv, mask=m_abs), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(z_gm, z_ub)
        if _UNARY_MODE == "masked_unary":
            tla.copy(rlog_gm, rlog_ub)
            tla.copy(rsqrt_gm, rsqrt_ub)
            tla.copy(rabs_gm, rabs_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return _OPERATOR_SPECS


def _expected_unary(op_name: str, x: Any) -> Any:
    if op_name == "exp":
        return x.exp()
    if op_name == "log":
        return x.log()
    if op_name == "sqrt":
        return x.sqrt()
    if op_name == "abs":
        return x.abs()
    raise AssertionError(op_name)


def _pad_with_sentinel(
    outputs: tuple[Any, ...], reference: Any, *, total: int = _KERNEL_OUTPUTS
) -> tuple[Any, ...]:
    import torch

    padded = list(outputs)
    while len(padded) < total:
        padded.append(torch.full_like(reference, _DEFAULT_SENTINEL))
    return tuple(padded)


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    if op_name == "masked_unary":
        return dtype_name not in _FLOAT_UNARY_DTYPES
    if op_name == "masked_abs":
        return dtype_name not in _INTEGER_DTYPES
    kind = _OPERATOR_SPECS[op_name]["kind"]
    if kind == "float":
        return dtype_name not in _FLOAT_UNARY_DTYPES
    return dtype_name not in _INTEGER_DTYPES | _FLOAT_UNARY_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    if op_name == "masked_unary":
        print(
            f"skip op={op_name} dtype={dtype_name}: "
            "masked float unary ops require f16 or f32"
        )
        return
    if op_name in _UNMASKED_UNARY_OPS:
        kind = _OPERATOR_SPECS[op_name]["kind"]
        if kind == "float":
            reason = "float unary ops require f16 or f32 (bf16 unsupported by AVE intrinsics)"
        elif dtype_name == "bf16":
            reason = "bf16 abs is not supported by AVE vabs intrinsics"
        else:
            reason = "unsupported case"
        print(
            f"skip op={op_name} dtype={dtype_name} shape={shape_label(shape)}: {reason}"
        )
        return
    print(f"skip op={op_name} dtype={dtype_name} shape={shape_label(shape)}")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _UNARY_OP, _UNARY_MODE, _DEFAULT_SENTINEL
    if op_name not in _OPERATOR_SPECS:
        raise SystemExit(f"unknown unary operator: {op_name}")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _DEFAULT_SENTINEL = config.default_sentinel
    if op_name in _UNMASKED_UNARY_OPS:
        _UNARY_MODE = "unmasked_unary"
        _UNARY_OP = _OPERATOR_SPECS[op_name]["op"]
    elif op_name == "masked_unary":
        _UNARY_MODE = "masked_unary"
        _UNARY_OP = None
    else:
        _UNARY_MODE = "masked_abs"
        _UNARY_OP = None
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, ...]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 1 + _KERNEL_OUTPUTS)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    if args.op == "masked_unary" or (
        args.op in _UNMASKED_UNARY_OPS
        and _OPERATOR_SPECS[args.op]["kind"] == "float"
    ):
        idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
        return (((idx % 32.0) + 1.0).to(dtype),)
    arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
    return (((arange % 31) - 15).to(dtype),)


def _merge(torch: Any, res: Any, background: Any, keep: Any) -> Any:
    return torch.where(keep, res, background)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    if op_name in _UNMASKED_UNARY_OPS:
        (x,) = inputs
        return _pad_with_sentinel((_expected_unary(op_name, x),), x)
    if op_name == "masked_unary":
        import torch

        (x,) = inputs
        exp_r = x.exp()
        log_r = x.log()
        sqrt_r = x.sqrt()
        abs_r = x.abs()

        lane = torch.arange(VECTOR_ELE, device=x.device) % VL_ELE
        keep_exp = lane < (VL_ELE // 2)
        keep_log = lane < (VL_ELE // 4)
        keep_sqrt = (lane % 4) == 0
        keep_abs = lane < (VL_ELE // 2)

        zero = torch.zeros_like(x)
        return (
            _merge(torch, exp_r, zero, keep_exp),
            _merge(torch, log_r, zero, keep_log),
            _merge(torch, sqrt_r, zero, keep_sqrt),
            _merge(torch, abs_r, zero, keep_abs),
        )
    if op_name == "masked_abs":
        import torch

        (x,) = inputs
        abs_r = x.abs()
        lane = torch.arange(VECTOR_ELE, device=x.device) % VL_ELE
        keep = lane < (VL_ELE // 2)
        zero = torch.zeros_like(x)
        return _pad_with_sentinel((torch.where(keep, abs_r, zero),), x)
    raise SystemExit(f"unknown unary operator: {op_name}")


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector unary ops.",
        kernel=vector_unary,
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
        script_path=_SCRIPT_PATH,
        env_compile_jobs="UNARY_OP_COMPILE_JOBS",
        float_dtypes=frozenset({"f16", "f32"}),
        output_count=_KERNEL_OUTPUTS,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
