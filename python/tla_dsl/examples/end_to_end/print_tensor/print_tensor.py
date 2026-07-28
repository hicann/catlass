"""Compile and run one typed GM or UB tensor ``tla.print`` C310 case."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, NamedTuple

import catlass as tla


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"
UB_SHAPE = (1, 32)
SHAPE = (4, 4)
COUNT = 16


class _DTypeSpec(NamedTuple):
    token: str
    torch_dtype: str
    tla_dtype: str
    values: tuple[float | int, ...]


class _RuntimeInput(NamedTuple):
    value: Any
    owner: Any


_FLOAT_VALUES = (
    0.0,
    -0.0,
    1.0,
    -2.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    3.25,
) * 2


def _integer_values(minimum: int, maximum: int) -> tuple[int, ...]:
    return (minimum, maximum, 0, -1 if minimum < 0 else 1) * 4


DTYPE_SPECS = {
    "f16": _DTypeSpec("f16", "float16", "Float16", _FLOAT_VALUES),
    "f32": _DTypeSpec("f32", "float32", "Float32", _FLOAT_VALUES),
    "i8": _DTypeSpec("i8", "int8", "Int8", _integer_values(-128, 127)),
    "i16": _DTypeSpec("i16", "int16", "Int16", _integer_values(-32768, 32767)),
    "i32": _DTypeSpec(
        "i32", "int32", "Int32", _integer_values(-2147483648, 2147483647)
    ),
    "u8": _DTypeSpec("u8", "uint8", "UInt8", _integer_values(0, 255)),
    "u16": _DTypeSpec("u16", "uint16", "UInt16", _integer_values(0, 65535)),
    "u32": _DTypeSpec(
        "u32", "uint32", "UInt32", _integer_values(0, 4294967295)
    ),
}
_UNSIGNED_ITEMSIZE = {"u16": 2, "u32": 4}
_ELEMENT_BYTES = {
    "f16": 2,
    "f32": 4,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "u8": 1,
    "u16": 2,
    "u32": 4,
}
_SIGNED_STAGING_DTYPES = {"u8": "Int8", "u16": "Int16", "u32": "Int32"}
_KERNEL_DTYPE: Any = None
_KERNEL_COPY_DTYPE: Any = None
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_UNSIGNED = False


def _ub_row_major_layout() -> Any:
    return tla.make_layout(
        shape=tla.make_shape(*UB_SHAPE),
        stride=tla.make_stride(UB_SHAPE[1], 1),
    )


def _configure_ub_kernel(spec: _DTypeSpec) -> None:
    global _KERNEL_COPY_DTYPE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_UNSIGNED
    _KERNEL_DTYPE = getattr(tla, spec.tla_dtype)
    staging_dtype = _SIGNED_STAGING_DTYPES.get(spec.token, spec.tla_dtype)
    _KERNEL_COPY_DTYPE = getattr(tla, staging_dtype)
    _KERNEL_ELEMENT_BYTES = _ELEMENT_BYTES[spec.token]
    _KERNEL_UNSIGNED = spec.token in _SIGNED_STAGING_DTYPES


@tla.kernel
def print_tensor_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def print_tensor_aic_kernel(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value)


@tla.kernel
def print_tensor_ub_base_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(value, 0)
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub)


def _prepare_ub_tensors(
    value: tla.Tensor, element_offset: int
) -> tuple[Any, Any, Any, Any]:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    allocation = tla.allocate(
        32 + element_offset, _KERNEL_COPY_DTYPE, tla.AddressSpace.ub, 256
    )
    copy_ptr = allocation + element_offset
    gm_ptr = value.ptr
    print_ptr = copy_ptr
    if _KERNEL_UNSIGNED:
        gm_ptr = tla.recast_ptr(gm_ptr, dtype=_KERNEL_COPY_DTYPE)
        print_ptr = tla.recast_ptr(print_ptr, dtype=_KERNEL_DTYPE)
    layout = _ub_row_major_layout()
    gm = tla.make_tensor(gm_ptr, layout)
    copy_ub = tla.make_tensor(copy_ptr, layout)
    print_ub = tla.make_tensor(print_ptr, layout)
    return loaded, copy_ub, gm, print_ub


def _print_ub_tensor(
    loaded: Any, copy_ub: Any, gm: Any, print_ub: Any
) -> None:
    tla.copy(copy_ub, gm)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.print(print_ub, 16)


@tla.kernel
def print_tensor_ub_aligned_offset_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(
        value, 32 // _KERNEL_ELEMENT_BYTES
    )
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub)


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
        raise tla.TlaExecutionError(f"unsupported UB tensor case {case!r}")
    kernels = {
        "aic.c310": print_tensor_aic_kernel,
        "aiv.c310": print_tensor_aiv_kernel,
    }
    try:
        return kernels[args.arch_scope]
    except KeyError as exc:
        raise tla.TlaExecutionError(
            "tensor tla.print supports --arch-scope aic.c310 or aiv.c310"
        ) from exc


def _format_record(
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...] | None = None,
    shape: tuple[int, ...] = SHAPE,
) -> str:
    from catlass.execution import _format_print_tensor_record

    return _format_print_tensor_record(
        spec.values if values is None else values,
        shape=shape,
        dtype=spec.token,
    )


def _verify_public_output(
    output: str,
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...] | None = None,
    shape: tuple[int, ...] = SHAPE,
) -> str:
    expected = _format_record(spec, values=values, shape=shape)
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


def _compile(
    args: argparse.Namespace,
    kernel: Callable[[Any], None],
    value: Any,
    spec: _DTypeSpec,
) -> Any:
    core = args.arch_scope.split(".", maxsplit=1)[0]
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        / core
        / args.storage
        / args.case
        / spec.token
    )
    return tla.compile(
        kernel,
        value,
        arch_scope=args.arch_scope,
        cache=not args.no_cache,
        cache_dir=str(cache_dir),
        force_recompile=args.force_recompile,
    )


def _make_external_unsigned_input(
    torch: Any,
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...],
    shape: tuple[int, ...],
) -> _RuntimeInput:
    import numpy as np

    byte_view = np.asarray(
        values, dtype=np.dtype(f"<u{_UNSIGNED_ITEMSIZE[spec.token]}")
    ).view(np.uint8)
    owner = (
        torch.from_numpy(byte_view.copy())
        .to(device="npu", dtype=torch.uint8)
        .contiguous()
    )

    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        shape_value = tla.make_shape(*shape)
        value = tla.Tensor(
            shape_value,
            getattr(tla, spec.tla_dtype),
            origin_shape=shape_value,
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(shape[1], 1),
            layout_tag=tla.arch.RowMajor,
            data_ptr=int(owner.data_ptr()),
        )
    value._external_binding = True
    return _RuntimeInput(value, owner)


def _make_runtime_input(
    torch: Any,
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...] | None = None,
    shape: tuple[int, ...] = SHAPE,
) -> _RuntimeInput:
    input_values = spec.values if values is None else values
    torch_dtype = getattr(torch, spec.torch_dtype, None)
    if torch_dtype is not None:
        try:
            owner = (
                torch.tensor(input_values, dtype=torch_dtype, device="npu")
                .reshape(shape)
                .contiguous()
            )
            return _RuntimeInput(
                tla.from_dlpack(owner, layout_tag=tla.arch.RowMajor), owner
            )
        except (AttributeError, RuntimeError, TypeError):
            if spec.token not in _UNSIGNED_ITEMSIZE:
                raise
    if spec.token not in _UNSIGNED_ITEMSIZE:
        raise tla.TlaExecutionError(
            f"torch does not expose the required {spec.torch_dtype} dtype"
        )
    return _make_external_unsigned_input(
        torch, spec, values=input_values, shape=shape
    )


def _run_case(args: argparse.Namespace, torch: Any, spec: _DTypeSpec) -> None:
    runtime_input = _make_runtime_input(torch, spec)
    executor = _compile(args, _kernel(args), runtime_input.value, spec)
    print(f"case dtype={spec.token} core={args.arch_scope} compile_ok=True")
    captured = StringIO()
    with redirect_stdout(captured):
        executor(runtime_input.value, block=args.block)
    print(_verify_public_output(captured.getvalue(), spec))
    print(f"case dtype={spec.token} core={args.arch_scope} launch_ok=True")
    print(f"case dtype={spec.token} core={args.arch_scope} output_ok=True")


def _run_ub_case(
    args: argparse.Namespace, torch: Any, spec: _DTypeSpec
) -> None:
    _configure_ub_kernel(spec)
    runtime_input = _make_runtime_input(
        torch, spec, values=spec.values * 2, shape=UB_SHAPE
    )
    executor = _compile(args, _kernel(args), runtime_input.value, spec)
    print(
        f"case dtype={spec.token} core={args.arch_scope} "
        "storage=ub compile_ok=True"
    )
    captured = StringIO()
    with redirect_stdout(captured):
        executor(runtime_input.value, block=args.block)
    print(
        _verify_public_output(
            captured.getvalue(), spec, shape=UB_SHAPE
        )
    )
    print(
        f"case dtype={spec.token} core={args.arch_scope} "
        "storage=ub launch_ok=True"
    )
    print(
        f"case dtype={spec.token} core={args.arch_scope} "
        "storage=ub output_ok=True"
    )


def run(args: argparse.Namespace) -> int:
    if args.block != 1:
        raise tla.TlaExecutionError("tensor tla.print requires --block 1")

    import torch
    import torch_npu  # noqa: F401

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        if args.storage == "ub":
            specs = (
                DTYPE_SPECS.values()
                if args.all_dtypes
                else (DTYPE_SPECS[args.dtype],)
            )
            for spec in specs:
                _run_ub_case(args, torch, spec)
        else:
            specs = (
                DTYPE_SPECS.values()
                if args.all_dtypes
                else (DTYPE_SPECS[args.dtype],)
            )
            for spec in specs:
                _run_case(args, torch, spec)
    finally:
        tla.finalize()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    dtype = parser.add_mutually_exclusive_group()
    dtype.add_argument("--dtype", choices=tuple(DTYPE_SPECS), default="f32")
    dtype.add_argument("--all-dtypes", action="store_true")
    parser.add_argument("--arch-scope", default="aiv.c310")
    parser.add_argument("--storage", choices=("gm", "ub"), default="gm")
    parser.add_argument(
        "--case", choices=("base", "aligned-offset"), default="base"
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
