from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

import catlass as tla


DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "format-runtime-cache"
)


@tla.kernel
def debug_print_format_aiv_kernel() -> None:
    with tla.vector():
        tla.print("hello")
        tla.print("value={}", 7)
        tla.print("i={} f={}", 7, 1.25)
        tla.print("start")
        tla.print("i={} f={}", 7, 1.25)
        tla.print(
            "all={} {} {} {} {} {} {} {}",
            tla.Int8(-37),
            tla.Int16(-30000),
            tla.Int32(0),
            tla.UInt8(255),
            tla.UInt16(65535),
            tla.UInt32(4294967295),
            tla.Float16(1.25),
            tla.Float32(-2.5),
        )


@tla.kernel
def debug_print_format_aic_kernel() -> None:
    with tla.cube():
        tla.print("hello")
        tla.print("value={}", 7)
        tla.print("i={} f={}", 7, 1.25)
        tla.print("start")
        tla.print("i={} f={}", 7, 1.25)
        tla.print(
            "all={} {} {} {} {} {} {} {}",
            tla.Int8(-37),
            tla.Int16(-30000),
            tla.Int32(0),
            tla.UInt8(255),
            tla.UInt16(65535),
            tla.UInt32(4294967295),
            tla.Float16(1.25),
            tla.Float32(-2.5),
        )


_PAYLOADS = (
    "hello",
    "value=7",
    "i=7 f=1.250000",
    "start",
    "i=7 f=1.250000",
    "all=-37 -30000 0 255 65535 4294967295 1.250000 -2.500000",
)
_FRAMED_LINE = re.compile(
    r"^TLA printf: core=[0-9]+ block=([0-9]+) (.*)$"
)


def _kernel(args: argparse.Namespace) -> Any:
    if args.arch_scope.startswith("aic."):
        return debug_print_format_aic_kernel
    return debug_print_format_aiv_kernel


def _compile(args: argparse.Namespace) -> Any:
    return tla.compile(
        _kernel(args),
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


def _verify_case_output(output: str, *, payloads: tuple[str, ...], block: int) -> None:
    actual: Counter[tuple[int, str]] = Counter()
    for line in output.splitlines():
        if not line.startswith("TLA printf:"):
            continue
        match = _FRAMED_LINE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"malformed formatted debug record: {line!r}")
        block_id = int(match.group(1))
        if block_id not in range(block):
            raise RuntimeError(f"unexpected block id {block_id} in {line!r}")
        actual[(block_id, match.group(2))] += 1

    expected = Counter(
        (block_id, payload)
        for block_id in range(block)
        for payload in payloads
    )
    if actual != expected:
        raise RuntimeError(
            f"formatted debug records differ: expected {expected!r}, "
            f"got {actual!r}; output={output!r}"
        )
    if "malformed" in output or "no records captured" in output:
        raise RuntimeError(f"invalid formatted debug output: {output!r}")


def _run(args: argparse.Namespace) -> None:
    executor = _compile(args)
    output = _capture_c_stdout(lambda: executor(block_dim=args.block_dim))
    _verify_case_output(output, payloads=_PAYLOADS, block=args.block_dim)
    print(output, end="" if output.endswith("\n") else "\n")
    print("compile_ok=True")
    print(f"kernel.o path={executor.kernel_binary_path}")
    print("launch_ok=True")
    print("output_ok=True")


def dump_tlair(args: argparse.Namespace) -> str:
    return _kernel(args).dump_mlir(type_args=())


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        _run(args)
        return 0
    finally:
        tla.finalize()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and run formatted tla.print examples."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    parser.add_argument(
        "--arch-scope", choices=("aic.c310", "aiv.c310"), default="aiv.c310"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
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
