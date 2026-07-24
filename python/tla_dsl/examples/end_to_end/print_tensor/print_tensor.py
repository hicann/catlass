"""Compile and run the tensor form of ``tla.print`` on AIC or AIV."""

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


@tla.kernel
def print_tensor_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)


@tla.kernel
def print_tensor_aic_kernel(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value, 16)


def _kernel(args: argparse.Namespace) -> Callable[[Any], None]:
    if args.arch_scope == "aic.c310":
        return print_tensor_aic_kernel
    if args.arch_scope == "aiv.c310":
        return print_tensor_aiv_kernel
    raise tla.TlaExecutionError(
        "tensor tla.print supports --arch-scope aic.c310 or aiv.c310"
    )


def _verify_public_output(output: str) -> str:
    expected = _format_record(EXPECTED_VALUES)
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


def _format_record(values: list[float]) -> str:
    from catlass.execution import _format_print_tensor_record

    return _format_print_tensor_record(values, shape=SOURCE_SHAPE)


def _compile(
    args: argparse.Namespace, kernel: Callable[[Any], None], value: Any
) -> Any:
    return tla.compile(
        kernel,
        value,
        arch_scope=args.arch_scope,
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def run(args: argparse.Namespace) -> int:
    kernel = _kernel(args)
    if args.block != 1:
        raise tla.TlaExecutionError("tensor tla.print requires --block 1")

    import torch
    import torch_npu  # noqa: F401

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        source = (
            torch.arange(32, dtype=torch.float32, device="npu")
            .reshape(SOURCE_SHAPE)
            .contiguous()
        )
        value = tla.from_dlpack(source, layout_tag=tla.arch.RowMajor)
        executor = _compile(args, kernel, value)
        captured = StringIO()
        with redirect_stdout(captured):
            executor(value, block=args.block)
        print(_verify_public_output(captured.getvalue()))
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
