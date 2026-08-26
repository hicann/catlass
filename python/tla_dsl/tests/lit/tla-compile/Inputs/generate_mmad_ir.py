"""Generate lit input from legal Python ``@tla.kernel`` programs."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import argparse

import catlass.tla as tla
import catlass.runtime as runtime_mod


@tla.kernel
def _static_f16_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.Float16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Float16, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 512),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_bf16_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.BFloat16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.BFloat16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.BFloat16, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.BFloat16, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 512),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_f32_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_i8_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    # The integer route accumulates in i32, not fp32.
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l0c, 512),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _dynamic_init_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=3)


@tla.kernel
def _dynamic_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=unit_flag)


@tla.kernel
def _dynamic_init_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)


def _f32_mmad_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _fp8_mmad_kernel(elem_a, elem_b):
    """Build a cube kernel with the given fp8 operand formats."""

    @tla.kernel
    def _kernel() -> None:
        lhs_parent = tla.make_tensor(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        rhs_parent = tla.make_tensor(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
        )
        acc_parent = tla.make_tensor(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
        )
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l0a, 512),
            lhs_parent,
            tla.arch.zN,
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l0b, 512),
            rhs_parent,
            tla.arch.nZ,
        )
        acc = tla.make_tensor_like(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 512),
            acc_parent,
            tla.arch.L0Clayout,
        )
        with tla.cube():
            tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)

    return _kernel


@tla.kernel
def _fp8_vector_copy_kernel(mem_a: tla.Tensor) -> None:
    """fp8 staged through UB on the vector path -- no such route exists.

    fp8 is a cube operand format: the bc layer implements it for GM->L1 and
    L1->L0A/L0B only. Nothing about the tile itself says so, which is why the
    route resolution has to reject it rather than name a symbol that was never
    registered.
    """
    ub = tla.make_tensor(
        tla.allocate((32, 32), tla.Float8E4M3FN, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    with tla.vector():
        tla.copy(ub, mem_a)


def _fp8_vector_copy_args() -> tuple[tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float8E4M3FN,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _i8_fixpipe_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    """GM operands / result for the integer fixpipe kernel: i8 in, i32 out."""
    return (
        make_fake_tensor(
            tla.Int8, (128, 64), (64, 1), origin_shape=(128, 64), layout_tag=tla.arch.RowMajor
        ),
        make_fake_tensor(
            tla.Int8, (64, 128), (128, 1), origin_shape=(64, 128), layout_tag=tla.arch.RowMajor
        ),
        make_fake_tensor(
            tla.Int32, (128, 128), (128, 1), origin_shape=(128, 128), layout_tag=tla.arch.RowMajor
        ),
    )


@tla.kernel
def _i8_fixpipe_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    """Every way out of an i32 accumulator: L0C -> GM, L0C -> UB, L0C -> L1.

    The i32 accumulator has no narrowing path, so each copy must resolve to an
    int32_t -> int32_t runtime callee rather than one of the fp32 ones.
    """
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l0c, 512), mem_c, tla.arch.L0Clayout
    )
    ub_c = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.ub, 256), mem_c, tla.arch.RowMajor
    )
    l1_c = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l1, 512), mem_c, tla.arch.zN
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)
        tla.copy(mem_c, acc)
        tla.copy(ub_c, acc, tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
        ))
        tla.copy(l1_c, acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=(
            "f16",
            "bf16",
            "f32",
            "i8",
            "i8-fixpipe",
            "fp8",
            "fp8-mixed",
            "fp8-vector-copy",
            "dynamic-init",
            "dynamic-unit",
            "dynamic-both",
        ),
    )
    case = parser.parse_args().case
    if case == "f16":
        kernel, type_args = _static_f16_mmad_kernel, ()
    elif case == "bf16":
        kernel, type_args = _static_bf16_mmad_kernel, ()
    elif case == "f32":
        kernel, type_args = _static_f32_mmad_kernel, ()
    elif case == "i8":
        kernel, type_args = _static_i8_mmad_kernel, ()
    elif case == "i8-fixpipe":
        kernel, type_args = _i8_fixpipe_kernel, _i8_fixpipe_args()
    elif case == "fp8":
        kernel, type_args = _fp8_mmad_kernel(tla.Float8E4M3FN, tla.Float8E4M3FN), ()
    elif case == "fp8-mixed":
        kernel, type_args = _fp8_mmad_kernel(tla.Float8E4M3FN, tla.Float8E5M2), ()
    elif case == "fp8-vector-copy":
        kernel, type_args = _fp8_vector_copy_kernel, _fp8_vector_copy_args()
    else:
        kernels = {
            "dynamic-init": _dynamic_init_mmad_kernel,
            "dynamic-unit": _dynamic_unit_mmad_kernel,
            "dynamic-both": _dynamic_init_unit_mmad_kernel,
        }
        kernel, type_args = kernels[case], _f32_mmad_args()
    print(kernel.dump_mlir(type_args=type_args))


if __name__ == "__main__":
    main()
