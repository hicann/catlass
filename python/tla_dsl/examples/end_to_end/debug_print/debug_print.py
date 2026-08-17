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

import argparse
import ctypes
import os
import struct
import tempfile
from pathlib import Path
import re
from typing import Any, Callable, NamedTuple

import catlass.tla as tla
import sys


class ScalarSpec(NamedTuple):
    scalar_type: str
    parser: str
    value: int | float
    expected: str


DTYPE_SPECS = {
    "i8": ScalarSpec("Int8", "signed8", -128, "-128"),
    "i16": ScalarSpec("Int16", "signed16", -32768, "-32768"),
    "i32": ScalarSpec("Int32", "signed32", -37, "-37"),
    "u8": ScalarSpec("UInt8", "unsigned8", 255, "255"),
    "u16": ScalarSpec("UInt16", "unsigned16", 65535, "65535"),
    "u32": ScalarSpec("UInt32", "unsigned32", 4294967295, "4294967295"),
    "f16": ScalarSpec("Float16", "float16", 1.25, "1.250000"),
    "f32": ScalarSpec("Float32", "float32", 1.25, "1.250000"),
}


class ExpressionSpec(NamedTuple):
    lhs: int | float
    rhs: int | float
    expected: str


EXPRESSION_SPECS = {
    "i8": ExpressionSpec(-37, 5, "-32"),
    "i16": ExpressionSpec(-30000, 123, "-29877"),
    "i32": ExpressionSpec(-37, 5, "-32"),
    "f16": ExpressionSpec(1.25, 0.75, "2.000000"),
    "f32": ExpressionSpec(1.25, 0.75, "2.000000"),
}
EXPRESSION_DTYPES = tuple(EXPRESSION_SPECS)


@tla.kernel
def debug_print_aiv_kernel(value: object) -> None:
    with tla.vector():
        tla.print(value)



@tla.kernel
def debug_print_expression_aiv_kernel(lhs: object, rhs: object) -> None:
    with tla.vector():
        tla.print(lhs + rhs)



def _kernel(args: argparse.Namespace) -> Any:
    if args.expression:
        if args.dtype not in EXPRESSION_DTYPES:
            supported = ", ".join(EXPRESSION_DTYPES)
            raise ValueError(
                f"--expression does not support {args.dtype}; expected one of {supported}"
            )
        return debug_print_expression_aiv_kernel
    return debug_print_aiv_kernel


def dump_tlair(args: argparse.Namespace) -> str:
    if not args.all_dtypes:
        return _kernel(args).dump_mlir(type_args=_type_args(args))
    dumps = []
    for dtype, value, rhs in _selected_cases(args):
        case_args = _case_args(args, dtype=dtype, value=value, rhs=rhs)
        dumps.append(
            f"// dtype={dtype}\n"
            f"{_kernel(case_args).dump_mlir(type_args=_type_args(case_args))}"
        )
    return "\n".join(dumps)


def _scalar_value(args: argparse.Namespace, value: int | float) -> Any:
    scalar_type = getattr(tla, DTYPE_SPECS[args.dtype].scalar_type)
    return scalar_type(value)


def _type_args(args: argparse.Namespace) -> tuple[Any, ...]:
    values = [_scalar_value(args, args.value)]
    if args.expression:
        values.append(_scalar_value(args, args.rhs))
    return tuple(values)


def _compile(args: argparse.Namespace) -> Any:
    return tla.compile(
        _kernel(args),
        *_type_args(args),
        options="--npu-arch 3510"
    )


def _capture_c_stdout(launch: Callable[[], None]) -> str:
    libc = ctypes.CDLL(None)
    fflush = libc.fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    fflush(None)
    saved_stdout = os.dup(1)
    try:
        with tempfile.TemporaryFile(mode="w+b") as captured:
            os.dup2(captured.fileno(), 1)
            try:
                launch()
            finally:
                fflush(None)
                os.dup2(saved_stdout, 1)
            captured.seek(0)
            return captured.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_stdout)


def _verify_debug_output(
    output: str, *, dtype: str, expected_value: str, expect_count: int
) -> None:
    tag = "v" if dtype in {"f16", "f32"} else "x"
    pattern = re.compile(
        rf"^TLA printf: core=[0-9]+ block=([0-9]+) {tag}={re.escape(expected_value)}$"
    )
    diagnostic = re.compile(
        r"^TLA printf: core=[0-9]+ block=[0-9]+ "
        r"(?:\[WARNING\]: CANN TimeStamp is invalid.*|"
        r"\[(?:AIC|AIV) Block [0-9]+/[0-9]+\]\s*)$"
    )
    matches = []
    unexpected = []
    for line in output.splitlines():
        match = pattern.fullmatch(line)
        if match:
            matches.append(match)
        elif line.startswith("TLA printf:") and not diagnostic.fullmatch(line):
            unexpected.append(line)
    if unexpected or len(matches) != expect_count:
        raise RuntimeError(
            f"expected {expect_count} {dtype} debug line(s); got {output!r}"
        )
    if expect_count > 1 and len({match.group(1) for match in matches}) != expect_count:
        raise RuntimeError(
            f"expected records from {expect_count} distinct blocks; got {output!r}"
        )
    if "malformed" in output or "no records captured" in output:
        raise RuntimeError(f"invalid device debug output: {output!r}")


def _expected_value(dtype: str, value: int | float) -> str:
    if dtype == "f16":
        value = _round_to_f16(float(value))
    return f"{value:.6f}" if dtype in {"f16", "f32"} else str(value)


def _case_args(
    args: argparse.Namespace, *, dtype: str, value: int | float, rhs: int | float
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(dtype=dtype, value=value, rhs=rhs)
    return argparse.Namespace(**values)


def _selected_cases(
    args: argparse.Namespace,
) -> list[tuple[str, int | float, int | float]]:
    if not args.all_dtypes:
        return [(args.dtype, args.value, args.rhs)]
    if args.expression:
        return [
            (dtype, EXPRESSION_SPECS[dtype].lhs, EXPRESSION_SPECS[dtype].rhs)
            for dtype in EXPRESSION_DTYPES
        ]
    return [(dtype, spec.value, 0) for dtype, spec in DTYPE_SPECS.items()]


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    kernel_paths = []
    for dtype, value, rhs in _selected_cases(args):
        case_args = _case_args(args, dtype=dtype, value=value, rhs=rhs)
        executor = _compile(case_args)
        output = _capture_c_stdout(
            lambda: executor(*_type_args(case_args), block_num=args.block_num)
        )
        result = value + rhs if args.expression else value
        if args.all_dtypes:
            expected_value = (
                EXPRESSION_SPECS[dtype].expected
                if args.expression
                else DTYPE_SPECS[dtype].expected
            )
        else:
            expected_value = _expected_value(dtype, result)
        _verify_debug_output(
            output,
            dtype=dtype,
            expected_value=expected_value,
            expect_count=args.expect_count,
        )
        print(output, end="" if output.endswith("\n") else "\n")
        kernel_paths.append(executor.kernel_binary_path)
    print("compile_ok=True")
    for path in kernel_paths:
        print(f"kernel.o path={path}")
    print("launch_ok=True")
    print("output_ok=True")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run direct or computed tla.print values."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    parser.add_argument("--all-dtypes", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("i8", "i16", "i32", "u8", "u16", "u32", "f16", "f32"),
        default="i32",
    )
    parser.add_argument("--value", default="3")
    parser.add_argument("--expression", action="store_true")
    parser.add_argument("--rhs", default="0")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--expect-count", type=int, default=1)
    return parser


def _signed_integer(text: str, width: int) -> int:
    value = int(text, 0)
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise argparse.ArgumentTypeError(f"expected a signed {width}-bit integer")
    return value


def _unsigned_integer(text: str, width: int) -> int:
    value = int(text, 0)
    if not 0 <= value < (1 << width):
        raise argparse.ArgumentTypeError(f"expected an unsigned {width}-bit integer")
    return value


def _f32(text: str) -> float:
    return float(text)


def _round_to_f16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def _f16(text: str) -> float:
    try:
        return _round_to_f16(float(text))
    except OverflowError as error:
        raise argparse.ArgumentTypeError("expected an f16 value") from error


def _parse_scalar(dtype: str, text: str) -> int | float:
    parser = DTYPE_SPECS[dtype].parser
    if parser.startswith("signed"):
        return _signed_integer(text, int(parser.removeprefix("signed")))
    if parser.startswith("unsigned"):
        return _unsigned_integer(text, int(parser.removeprefix("unsigned")))
    if parser == "float16":
        return _f16(text)
    return _f32(text)


def main() -> int:
    args = _parser().parse_args()
    if not args.all_dtypes:
        args.value = _parse_scalar(args.dtype, args.value)
        args.rhs = _parse_scalar(args.dtype, args.rhs)
    if args.dump_tlair:
        print(dump_tlair(args))
        return 0
    if not args.run:
        raise SystemExit("pass --run or --dump-tlair")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
