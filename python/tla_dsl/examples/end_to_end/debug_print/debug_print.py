from __future__ import annotations

import argparse
import ctypes
import os
import tempfile
from pathlib import Path
import re
from typing import Any, Callable

import catlass as tla


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"


@tla.kernel
def debug_print_aiv_kernel(value: object) -> None:
    with tla.vector():
        tla.debug_print(value)


@tla.kernel
def debug_print_aic_kernel(value: object) -> None:
    with tla.cube():
        tla.debug_print(value)


@tla.kernel
def debug_print_expression_aiv_kernel(lhs: object, rhs: object) -> None:
    with tla.vector():
        tla.debug_print(lhs + rhs)


@tla.kernel
def debug_print_expression_aic_kernel(lhs: object, rhs: object) -> None:
    with tla.cube():
        tla.debug_print(lhs + rhs)


def _kernel(args: argparse.Namespace) -> Any:
    if args.expression:
        if args.arch_scope.startswith("aic."):
            return debug_print_expression_aic_kernel
        return debug_print_expression_aiv_kernel
    if args.arch_scope.startswith("aic."):
        return debug_print_aic_kernel
    return debug_print_aiv_kernel


def dump_tlair(args: argparse.Namespace) -> str:
    return _kernel(args).dump_mlir(type_args=_type_args(args))


def _scalar_value(args: argparse.Namespace, value: int | float) -> Any:
    return tla.Int32(value) if args.dtype == "i32" else tla.Float32(value)


def _type_args(args: argparse.Namespace) -> tuple[Any, ...]:
    values = [_scalar_value(args, args.value)]
    if args.expression:
        values.append(_scalar_value(args, args.rhs))
    return tuple(values)


def _compile(args: argparse.Namespace) -> Any:
    return tla.compile(
        _kernel(args),
        *_type_args(args),
        arch_scope=args.arch_scope,
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
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
    tag = "x" if dtype == "i32" else "v"
    pattern = re.compile(
        rf"^TLA printf: core=[0-9]+ block=([0-9]+) {tag}={re.escape(expected_value)}$"
    )
    lines = output.splitlines()
    framed = [line for line in lines if line.startswith("TLA printf:")]
    matches = []
    for line in framed:
        match = pattern.fullmatch(line)
        if match:
            matches.append(match)
    if len(framed) != expect_count or len(matches) != expect_count:
        raise RuntimeError(
            f"expected {expect_count} {dtype} debug line(s); got {output!r}"
        )
    if expect_count > 1 and len({match.group(1) for match in matches}) != expect_count:
        raise RuntimeError(
            f"expected records from {expect_count} distinct blocks; got {output!r}"
        )
    if "malformed" in output or "no records captured" in output:
        raise RuntimeError(f"invalid device debug output: {output!r}")


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        executor = _compile(args)
        output = _capture_c_stdout(
            lambda: executor(*_type_args(args), block=args.block)
        )
        result = args.value + args.rhs if args.expression else args.value
        expected_value = str(result) if args.dtype == "i32" else f"{result:.6f}"
        _verify_debug_output(
            output,
            dtype=args.dtype,
            expected_value=expected_value,
            expect_count=args.expect_count,
        )
        print(output, end="" if output.endswith("\n") else "\n")
        print("compile_ok=True")
        print(f"kernel.o path={executor.kernel_binary_path}")
        print("launch_ok=True")
        print("output_ok=True")
        return 0
    finally:
        tla.finalize()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run direct or computed tla.debug_print values."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    parser.add_argument("--dtype", choices=("i32", "f32"), default="i32")
    parser.add_argument("--value", default="3")
    parser.add_argument("--expression", action="store_true")
    parser.add_argument("--rhs", default="0")
    parser.add_argument(
        "--arch-scope", choices=("aic.c310", "aiv.c310"), default="aiv.c310"
    )
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--expect-count", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def _i32(text: str) -> int:
    value = int(text, 0)
    if not -(1 << 31) <= value < (1 << 31):
        raise argparse.ArgumentTypeError("expected a signed 32-bit integer")
    return value


def _f32(text: str) -> float:
    return float(text)


def main() -> int:
    args = _parser().parse_args()
    parse_scalar = _i32 if args.dtype == "i32" else _f32
    args.value = parse_scalar(args.value)
    args.rhs = parse_scalar(args.rhs)
    if args.dump_tlair:
        print(dump_tlair(args))
        return 0
    if not args.run:
        raise SystemExit("pass --run or --dump-tlair")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
