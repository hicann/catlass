# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# Four end-to-end cases shift every active lane by exactly four bits:
# reg-reg shift_left/shift_right and reg-scalar shift_lefts/shift_rights.
VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
SHIFT_BITS = 4
ALL_DTYPES = ("i32", "i16", "i8")

_KERNEL_DTYPE = tla.Int32
_KERNEL_SHAPE = (VECTOR_ELE,)
_SHIFT_OP = "shift_left"


@tla.kernel
def shift_op(
    mem_source: tla.Tensor,
    mem_shift: tla.Tensor,
    mem_output: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    source_gm = tla.tile_view(mem_source, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    shift_gm = tla.tile_view(mem_shift, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    output_gm = tla.tile_view(mem_output, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    source_ub = _make_ub_tensor(source_gm)
    shift_ub = _make_ub_tensor(shift_gm)
    output_ub = _make_ub_tensor(output_gm)

    with tla.vector():
        tla.copy(source_ub, source_gm)
        tla.copy(shift_ub, shift_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                source_tile = _chunk(source_ub, i)
                shift_tile = _chunk(shift_ub, i)
                output_tile = _chunk(output_ub, i)
                source_reg = source_tile.load()
                shift_reg = shift_tile.load()
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)

                # The public API dispatches to reg-reg or reg-scalar TLA ops
                # according to the type of its second argument.
                if tla.const_expr(_SHIFT_OP == "shift_left"):
                    result = tla.shift_left(source_reg, shift_reg, mask=tail)
                elif tla.const_expr(_SHIFT_OP == "shift_right"):
                    result = tla.shift_right(source_reg, shift_reg, mask=tail)
                elif tla.const_expr(_SHIFT_OP == "shift_lefts"):
                    result = tla.shift_left(source_reg, SHIFT_BITS, mask=tail)
                else:
                    result = tla.shift_right(source_reg, SHIFT_BITS, mask=tail)
                output_tile.store(result, mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(output_gm, output_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    # Keep the underlying four TLA op names as CLI case names. The scalar
    # cases still exercise the intentionally unified tla.shift_left/right API.
    return {
        "shift_left": {"default_atol": 0},
        "shift_right": {"default_atol": 0},
        "shift_lefts": {"default_atol": 0},
        "shift_rights": {"default_atol": 0},
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
    global VECTOR_ELE, VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_SHAPE, _SHIFT_OP
    if op_name not in _operator_specs():
        choices = ", ".join(sorted(_operator_specs()))
        raise SystemExit(
            f"unknown shift operator {op_name!r}; expected one of: {choices}"
        )
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    _SHIFT_OP = op_name
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    index = torch.arange(VECTOR_ELE, dtype=torch.int32, device="npu")
    if args.op.startswith("shift_left"):
        # Keep i8 left shifts representable as well, avoiding overflow in the
        # reference calculation while still producing distinct lane values.
        source = (index % 8).to(dtype)
    else:
        source = ((index % 127) - 63).to(dtype)
    shift = torch.full((VECTOR_ELE,), SHIFT_BITS, dtype=dtype, device="npu")
    return source, shift


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    source, shift = inputs
    if op_name.startswith("shift_left"):
        return source << shift
    return source >> shift


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector left/right shifts by four bits.",
        kernel=shift_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset(),
        input_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
