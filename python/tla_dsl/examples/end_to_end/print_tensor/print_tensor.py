"""Compile and run the GM or UB tensor form of ``tla.print``."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import catlass as tla


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"
EXPECTED_VALUES = [float(value) for value in range(16)]
SOURCE_SHAPE = (8, 4)
UB_SHAPE = (4, 8)
CAPACITY_SHAPE = (262_112,)


def _ub_row_major_layout() -> Any:
    return tla.make_layout(
        shape=tla.make_shape(*UB_SHAPE),
        stride=tla.make_stride(UB_SHAPE[1], 1),
    )


def _dynamic_ub_row_major_layout(rows: Any) -> Any:
    return tla.make_layout(
        shape=tla.make_shape(rows, SOURCE_SHAPE[1]),
        stride=tla.make_stride(SOURCE_SHAPE[1], 1),
    )


@tla.kernel
def print_tensor_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)


@tla.kernel
def print_tensor_aic_kernel(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value, 16)


@tla.kernel
def print_tensor_dynamic_aiv_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    tensor = tla.make_tensor(value.ptr, _dynamic_ub_row_major_layout(rows))
    with tla.vector():
        tla.print(tensor, length)


@tla.kernel
def print_tensor_dynamic_aic_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    tensor = tla.make_tensor(value.ptr, _dynamic_ub_row_major_layout(rows))
    with tla.cube():
        tla.print(tensor, length)


@tla.kernel
def print_tensor_capacity_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, CAPACITY_SHAPE[0])


@tla.kernel
def print_tensor_capacity_aic_kernel(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value, CAPACITY_SHAPE[0])


@tla.kernel
def print_tensor_ub_base_kernel(value: tla.Tensor) -> None:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(32, tla.Float32, tla.AddressSpace.ub, 256)
    layout = _ub_row_major_layout()
    gm = tla.make_tensor(value.ptr, layout)
    ub = tla.make_tensor(ptr, layout)
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(ub, 16)


@tla.kernel
def print_tensor_ub_aligned_offset_kernel(value: tla.Tensor) -> None:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    allocation = tla.allocate(40, tla.Float32, tla.AddressSpace.ub, 256)
    # Eight float32 elements make a non-zero, 32-byte-aligned effective address.
    layout = _ub_row_major_layout()
    gm = tla.make_tensor(value.ptr, layout)
    ub = tla.make_tensor(allocation + 8, layout)
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(ub, 16)


@tla.kernel
def print_tensor_ub_dynamic_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(32, tla.Float32, tla.AddressSpace.ub, 256)
    copy_layout = _ub_row_major_layout()
    gm = tla.make_tensor(value.ptr, copy_layout)
    ub = tla.make_tensor(ptr, copy_layout)
    dynamic_ub = tla.make_tensor(ptr, _dynamic_ub_row_major_layout(rows))
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(dynamic_ub, length)


def _kernel(args: argparse.Namespace) -> Callable[[Any], None]:
    storage = getattr(args, "storage", "gm")
    case = getattr(args, "case", "base")
    if storage == "ub":
        if args.arch_scope != "aiv.c310":
            raise tla.TlaExecutionError("UB tensor tla.print requires --arch-scope aiv.c310")
        if case == "base":
            return print_tensor_ub_base_kernel
        if case == "aligned-offset":
            return print_tensor_ub_aligned_offset_kernel
        if case == "dynamic":
            return print_tensor_ub_dynamic_kernel
        raise tla.TlaExecutionError(f"unsupported UB tensor case {case!r}")
    if case == "capacity":
        if args.arch_scope == "aic.c310":
            return print_tensor_capacity_aic_kernel
        if args.arch_scope == "aiv.c310":
            return print_tensor_capacity_aiv_kernel
    if args.arch_scope == "aic.c310":
        return (
            print_tensor_dynamic_aic_kernel
            if args.dynamic_shape
            else print_tensor_aic_kernel
        )
    if args.arch_scope == "aiv.c310":
        return (
            print_tensor_dynamic_aiv_kernel
            if args.dynamic_shape
            else print_tensor_aiv_kernel
        )
    raise tla.TlaExecutionError(
        "tensor tla.print supports --arch-scope aic.c310 or aiv.c310"
    )


def _verify_public_output(
    output: str,
    *,
    values: list[float] = EXPECTED_VALUES,
    shape: tuple[int, ...] = SOURCE_SHAPE,
) -> str:
    expected = _format_record(values, shape=shape)
    records = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("tla.print ")
    ]
    if records != [expected]:
        raise tla.TlaExecutionError(
            "tensor tla.print native initialization or decoding failed: "
            f"expected exactly {expected!r}, got {records!r}"
        )
    return expected


def _format_record(
    values: list[float], *, shape: tuple[int, ...] = SOURCE_SHAPE
) -> str:
    from catlass.execution import _format_print_tensor_record

    return _format_print_tensor_record(values, shape=shape)


def _compile(
    args: argparse.Namespace, kernel: Callable[..., None], kernel_args: tuple[Any, ...]
) -> Any:
    return tla.compile(
        kernel,
        *kernel_args,
        arch_scope=args.arch_scope,
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def run(args: argparse.Namespace) -> int:
    kernel = _kernel(args)
    if args.block != 1:
        raise tla.TlaExecutionError("tensor tla.print requires --block 1")
    if args.case == "capacity" and args.storage != "gm":
        raise tla.TlaExecutionError("the capacity case requires --storage gm")
    if args.case == "capacity" and args.dynamic_shape:
        raise tla.TlaExecutionError("the capacity case does not use --dynamic-shape")
    if args.layout == "column-major" and (
        args.storage != "gm" or args.case != "base" or args.dynamic_shape
    ):
        raise tla.TlaExecutionError(
            "the column-major case requires static GM base printing"
        )

    import torch
    import torch_npu  # noqa: F401

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        source_shape = CAPACITY_SHAPE if args.case == "capacity" else SOURCE_SHAPE
        logical_source = torch.arange(
            source_shape[0] if len(source_shape) == 1 else 32,
            dtype=torch.float32,
            device="npu",
        ).reshape(source_shape).contiguous()
        source = (
            logical_source.permute(1, 0).contiguous()
            if args.layout == "column-major"
            else logical_source
        )
        layout_tag = (
            tla.arch.ColumnMajor
            if args.layout == "column-major"
            else tla.arch.RowMajor
        )
        value = tla.from_dlpack(source, layout_tag=layout_tag)
        kernel_args = (
            (value, tla.Int32(SOURCE_SHAPE[0]), tla.Int32(16))
            if (args.storage == "ub" and args.case == "dynamic")
            or (args.storage == "gm" and args.dynamic_shape)
            else (value,)
        )
        executor = _compile(args, kernel, kernel_args)
        captured = StringIO()
        with redirect_stdout(captured):
            executor(*kernel_args, block=args.block)
        output_shape = (
            UB_SHAPE
            if args.storage == "ub" and args.case != "dynamic"
            else source_shape
        )
        expected_values = (
            [float(value) for value in range(CAPACITY_SHAPE[0])]
            if args.case == "capacity"
            else [
                float(value)
                for value in source.flatten()[: len(EXPECTED_VALUES)].tolist()
            ]
        )
        print(
            _verify_public_output(
                captured.getvalue(), values=expected_values, shape=output_shape
            )
        )
        print("compile_ok=True")
        print("launch_ok=True")
        print("output_ok=True")
        return 0
    finally:
        tla.finalize()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--arch-scope", default="aiv.c310")
    parser.add_argument("--storage", choices=("gm", "ub"), default="gm")
    parser.add_argument(
        "--layout",
        choices=("row-major", "column-major"),
        default="row-major",
    )
    parser.add_argument(
        "--case",
        choices=("base", "aligned-offset", "dynamic", "capacity"),
        default="base",
    )
    parser.add_argument(
        "--dynamic-shape",
        action="store_true",
        help="construct the printed GM tensor with a scalar runtime first extent",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
