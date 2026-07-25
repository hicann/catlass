from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass as tla
from catlass.params import MaskLoadParams, MaskStoreParams

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_type_args,
    vector_kernel_config,
)

# Single-VL MaskSSA load/store round-trip only:
#   create_mask(H) → store(..., MaskStoreParams) → load(MaskLoadParams)
#   → masked store of src
# Expected: first half of the VL holds src; second half stays 0.
# Companion vector dtype is fixed to f32; this case does not sweep add/dtypes.

VECTOR_ELE = 64
VL_ELE = 64
MASK_BYTES = VL_ELE // 8
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)
_MASK_BYTES = MASK_BYTES


@tla.kernel
def load_store_mask(mem_src: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    src_ub = _make_ub_tensor(mem_src)
    out_ub = _make_ub_tensor(mem_out)
    mask_ub = _make_mask_ub_tensor()

    with tla.vector():
        tla.copy(src_ub, mem_src)
        # Masked-out lanes must stay deterministic (0).
        tla.copy(out_ub, mem_out)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            value = src_ub.load()

            pattern = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
            mask_ub.store(pattern, MaskStoreParams())
            loaded = mask_ub.load(MaskLoadParams())

            out_ub.store(value, mask=loaded)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(mem_out, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 32)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _make_mask_ub_tensor() -> Any:
    ptr = tla.allocate(_MASK_BYTES, tla.Int8, tla.AddressSpace.ub, 32)
    layout = tla.make_layout(
        shape=tla.make_shape(_MASK_BYTES),
        stride=tla.make_stride(1),
    )
    return tla.make_tensor(ptr, layout)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "load_store_mask": {
            "default_atol": 1e-5,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VECTOR_ELE, VL_ELE, _KERNEL_DTYPE, _KERNEL_SHAPE, _MASK_BYTES, MASK_BYTES
    del shape
    if op_name not in _operator_specs():
        raise SystemExit("unknown load_store_mask operator")
    config = vector_kernel_config(dtype_name, None, ALL_DTYPES)
    VL_ELE = config.lanes
    VECTOR_ELE = VL_ELE
    MASK_BYTES = VL_ELE // 8
    _MASK_BYTES = MASK_BYTES
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SHAPE = (VECTOR_ELE,)
    return config.tla_dtype, config.torch_dtype, 0.0


def _compile_only_type_args(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[Any, ...]:
    tla_dtype, _, _ = _set_kernel_config(op_name, dtype_name, shape)
    return make_type_args(tla_dtype, _KERNEL_SHAPE, 2)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    src = torch.arange(VECTOR_ELE, dtype=dtype, device="npu") - (VECTOR_ELE // 2)
    return (src,)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    (src,) = inputs
    expected = src.new_zeros(src.shape)
    expected[: VL_ELE // 2] = src[: VL_ELE // 2]
    return expected


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run MaskLoadParams/MaskStoreParams round-trip kernels.",
        kernel=load_store_mask,
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
        env_compile_jobs="LOAD_STORE_MASK_COMPILE_JOBS",
        float_dtypes=frozenset({"f32"}),
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
