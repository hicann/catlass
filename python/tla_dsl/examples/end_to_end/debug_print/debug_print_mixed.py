from __future__ import annotations

import argparse
import ctypes
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

import catlass as tla

from debug_print import DTYPE_SPECS


DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "mixed-runtime-cache"
)


@tla.kernel
def debug_print_mixed_cube_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.print(tla.Int32(-37))
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def debug_print_mixed_vector_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.print(tla.Float32(1.25))


@tla.kernel
def debug_print_mixed_both_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.print(tla.Int32(-37))
    with tla.vector():
        tla.print(tla.Float32(1.25))


_KERNELS = {
    "cube": debug_print_mixed_cube_kernel,
    "vector": debug_print_mixed_vector_kernel,
    "both": debug_print_mixed_both_kernel,
}


@tla.kernel
def debug_print_matrix_cube_kernel(value: object) -> None:
    with tla.cube():
        tla.print(value)
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def debug_print_matrix_vector_kernel(value: object) -> None:
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.print(value)


@tla.kernel
def debug_print_matrix_both_kernel(value: object) -> None:
    with tla.cube():
        tla.print(value)
    with tla.vector():
        tla.print(value)


_MATRIX_KERNELS = {
    "cube": debug_print_matrix_cube_kernel,
    "vector": debug_print_matrix_vector_kernel,
    "both": debug_print_matrix_both_kernel,
}


def _kernel(args: argparse.Namespace):
    kernels = _MATRIX_KERNELS if args.all_dtypes else _KERNELS
    return kernels[args.print_region]


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


def _verify_mixed_debug_output(
    output: str,
    *,
    print_region: str,
    dtype: str | None = None,
    expected_value: str | None = None,
) -> None:
    """Check native C310 mixed-region records without constraining their order.

    The Cube callsite executes once for the logical block.  The Vector callsite
    executes on both C310 AIV sub-cores, which share the logical block index,
    so its two exact frames must come from distinct AIV cores rather than being
    collapsed by a MIX-only guard.
    """
    framed = [line for line in output.splitlines() if line.startswith("TLA printf:")]
    expected_counts = {
        "cube": (1, 1, 0),
        "vector": (2, 0, 2),
        "both": (3, 1, 2),
    }
    total, cube_count, vector_count = expected_counts[print_region]
    if dtype is not None:
        if expected_value is None:
            raise ValueError("expected_value is required for a typed matrix check")
        tag = "v" if dtype in {"f16", "f32"} else "x"
        pattern = re.compile(
            rf"^TLA printf: core=(?P<core>[0-9]+) block=0 "
            rf"{tag}={re.escape(expected_value)}$"
        )
        matches = [pattern.fullmatch(line) for line in framed]
        cores = {match.group("core") for match in matches if match is not None}
        if (
            len(framed) != total
            or any(match is None for match in matches)
            or len(cores) != total
        ):
            raise RuntimeError(
                f"expected {print_region} {dtype} records from {total} "
                f"distinct cores; got {output!r}"
            )
        if "malformed" in output or "no records captured" in output:
            raise RuntimeError(f"invalid mixed device debug output: {output!r}")
        return

    if dtype is None:
        expected = {
            "cube": re.compile(r"^TLA printf: core=[0-9]+ block=0 x=-37$"),
            "vector": re.compile(
                r"^TLA printf: core=(?P<core>[0-9]+) block=0 v=1\.250000$"
            ),
        }
    matching = {
        region: [line for line in framed if pattern.fullmatch(line)]
        for region, pattern in expected.items()
    }
    vector_cores = set()
    for line in matching["vector"]:
        match = expected["vector"].fullmatch(line)
        if match is not None:
            vector_cores.add(match.group("core"))
    if (
        len(framed) != total
        or len(matching["cube"]) != cube_count
        or len(matching["vector"]) != vector_count
        or (vector_count and len(vector_cores) != vector_count)
    ):
        raise RuntimeError(
            f"expected {print_region} records with {cube_count} Cube x and "
            f"{vector_count} distinct Vector-core v records; "
            f"got {output!r}"
        )
    if "malformed" in output or "no records captured" in output:
        raise RuntimeError(f"invalid mixed device debug output: {output!r}")


def dump_tlair(args: argparse.Namespace) -> str:
    if args.all_dtypes:
        dumps = []
        for dtype, spec in DTYPE_SPECS.items():
            scalar = getattr(tla, spec.scalar_type)(spec.value)
            dumps.append(
                f"// dtype={dtype}\n"
                f"{_kernel(args).dump_mlir(type_args=(scalar,))}"
            )
        return "\n".join(dumps)
    return _kernel(args).dump_mlir(
        type_args=(tla.Float32(1.0), tla.Float32(0.25))
    )


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        kernel_paths = []
        cases = DTYPE_SPECS.items() if args.all_dtypes else ((None, None),)
        for dtype, spec in cases:
            if spec is None:
                type_args = (tla.Float32(1.0), tla.Float32(0.25))
                expected_value = None
            else:
                type_args = (getattr(tla, spec.scalar_type)(spec.value),)
                expected_value = spec.expected
            executor = tla.compile(
                _kernel(args),
                *type_args,
                arch_scope="aic.c310",
                cache=not args.no_cache,
                cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
                force_recompile=args.force_recompile,
            )
            output = _capture_c_stdout(
                lambda: executor(*type_args, block=1)
            )
            _verify_mixed_debug_output(
                output,
                print_region=args.print_region,
                dtype=dtype,
                expected_value=expected_value,
            )
            print(output, end="" if output.endswith("\n") else "\n")
            kernel_paths.append(executor.kernel_binary_path)
        print("compile_ok=True")
        for path in kernel_paths:
            print(f"kernel.o path={path}")
        print("launch_ok=True")
        print("output_ok=True")
        return 0
    finally:
        tla.finalize()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run mixed tla.print."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    parser.add_argument("--all-dtypes", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--print-region", choices=("cube", "vector", "both"), default="both"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dump_tlair:
        print(dump_tlair(args))
        return 0
    if not args.run:
        raise SystemExit("pass --run or --dump-tlair")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
