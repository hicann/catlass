"""Generate lit input from legal Python ``@tla.kernel`` programs."""

from __future__ import annotations

import argparse

import catlass as tla
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
def _dynamic_init_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(32, 32), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(32, 32), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=3)


@tla.kernel
def _dynamic_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(32, 32), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(32, 32), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=unit_flag)


@tla.kernel
def _dynamic_init_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(32, 32), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(32, 32), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)


def _f32_mmad_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape((16, 2), (8, 4)),
                tla.Float32,
                addrspace=tla.AddressSpace.l0a,
                origin_shape=tla.make_shape(32, 32),
                layout_tag=tla.arch.zN,
            ),
            tla.Tensor(
                tla.make_shape((8, 4), (16, 2)),
                tla.Float32,
                addrspace=tla.AddressSpace.l0b,
                origin_shape=tla.make_shape(32, 32),
                layout_tag=tla.arch.nZ,
            ),
            tla.Tensor(
                tla.make_shape((16, 2), (16, 2)),
                tla.Float32,
                addrspace=tla.AddressSpace.l0c,
                origin_shape=tla.make_shape(32, 32),
                layout_tag=tla.arch.L0Clayout,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=("f16", "bf16", "f32", "dynamic-init", "dynamic-unit", "dynamic-both"),
    )
    case = parser.parse_args().case
    if case == "f16":
        kernel, type_args = _static_f16_mmad_kernel, ()
    elif case == "bf16":
        kernel, type_args = _static_bf16_mmad_kernel, ()
    elif case == "f32":
        kernel, type_args = _static_f32_mmad_kernel, ()
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
