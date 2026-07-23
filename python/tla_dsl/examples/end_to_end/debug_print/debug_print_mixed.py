from __future__ import annotations

import argparse
import ctypes
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

import catlass as tla


DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "mixed-runtime-cache"
)


@tla.kernel
def debug_print_mixed_cube_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.debug_print(tla.Int32(-37))
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def debug_print_mixed_vector_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.debug_print(tla.Float32(1.25))


@tla.kernel
def debug_print_mixed_both_kernel(x: object, y: object) -> None:
    with tla.cube():
        tla.debug_print(tla.Int32(-37))
    with tla.vector():
        tla.debug_print(tla.Float32(1.25))


_KERNELS = {
    "cube": debug_print_mixed_cube_kernel,
    "vector": debug_print_mixed_vector_kernel,
    "both": debug_print_mixed_both_kernel,
}


def _kernel(args: argparse.Namespace):
    return _KERNELS[args.print_region]


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


def _verify_mixed_debug_output(output: str, *, print_region: str) -> None:
    """Check native C310 mixed-region records without constraining their order.

    The Cube callsite executes once for the logical block.  The Vector callsite
    executes on both C310 AIV sub-cores, which share the logical block index,
    so its two exact frames must come from distinct AIV cores rather than being
    collapsed by a MIX-only guard.
    """
    expected = {
        "x": re.compile(r"^TLA printf: core=[0-9]+ block=0 x=-37$"),
        "v": re.compile(
            r"^TLA printf: core=(?P<core>[0-9]+) block=0 v=1\.250000$"
        ),
    }
    framed = [line for line in output.splitlines() if line.startswith("TLA printf:")]
    matching = {
        tag: [line for line in framed if pattern.fullmatch(line)]
        for tag, pattern in expected.items()
    }
    vector_cores = set()
    for line in matching["v"]:
        match = expected["v"].fullmatch(line)
        if match is not None:
            vector_cores.add(match.group("core"))
    expected_counts = {
        "cube": (1, 1, 0),
        "vector": (2, 0, 2),
        "both": (3, 1, 2),
    }
    total, cube_count, vector_count = expected_counts[print_region]
    if (
        len(framed) != total
        or len(matching["x"]) != cube_count
        or len(matching["v"]) != vector_count
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
    return _kernel(args).dump_mlir(
        type_args=(tla.Float32(1.0), tla.Float32(0.25))
    )


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        kernel = _kernel(args)
        executor = tla.compile(
            kernel,
            tla.Float32(1.0),
            tla.Float32(0.25),
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
            force_recompile=args.force_recompile,
        )
        output = _capture_c_stdout(
            lambda: executor(tla.Float32(1.0), tla.Float32(0.25), block=1)
        )
        _verify_mixed_debug_output(output, print_region=args.print_region)
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
        description="Compile and run mixed tla.debug_print."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
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
