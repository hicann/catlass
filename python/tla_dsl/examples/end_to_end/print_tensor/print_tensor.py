"""Compile and run typed GM or UB tensor ``tla.print`` C310 cases."""

from __future__ import annotations

import argparse
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, NamedTuple

import catlass as tla


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "runtime-cache"
SOURCE_SHAPE = (8, 4)
UB_SHAPE = (4, 8)
CAPACITY_SHAPE = (262_112,)


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
    "u32": _DTypeSpec("u32", "uint32", "UInt32", _integer_values(0, 4294967295)),
}
EXPECTED_VALUES = list(DTYPE_SPECS["f32"].values)
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


def _ub_copy_layout() -> Any:
    return tla.make_layout(
        shape=tla.make_shape(1, 32),
        stride=tla.make_stride(32, 1),
    )


def _configure_ub_kernel(spec: _DTypeSpec) -> None:
    global _KERNEL_COPY_DTYPE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_UNSIGNED
    _KERNEL_DTYPE = getattr(tla, spec.tla_dtype)
    staging_dtype = _SIGNED_STAGING_DTYPES.get(spec.token, spec.tla_dtype)
    _KERNEL_COPY_DTYPE = getattr(tla, staging_dtype)
    _KERNEL_ELEMENT_BYTES = _ELEMENT_BYTES[spec.token]
    _KERNEL_UNSIGNED = spec.token in _SIGNED_STAGING_DTYPES


def _dynamic_row_major_layout(rows: Any) -> Any:
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
def print_tensor_aiv_two_calls_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)
        tla.print(value, 8)


@tla.kernel
def print_tensor_aic_two_calls_kernel(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value, 16)
        tla.print(value, 8)


@tla.kernel
def print_tensor_dynamic_aiv_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    tensor = tla.make_tensor(value.ptr, _dynamic_row_major_layout(rows))
    with tla.vector():
        tla.print(tensor, length)


@tla.kernel
def print_tensor_dynamic_aic_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    tensor = tla.make_tensor(value.ptr, _dynamic_row_major_layout(rows))
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
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(value, 0)
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=1)


@tla.kernel
def print_tensor_ub_base_two_calls_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(value, 0)
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=2)


@tla.kernel
def print_tensor_ub_aligned_offset_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(
        value, 32 // _KERNEL_ELEMENT_BYTES
    )
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=1)


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
    copy_layout = _ub_copy_layout()
    gm = tla.make_tensor(gm_ptr, copy_layout)
    copy_ub = tla.make_tensor(copy_ptr, copy_layout)
    print_ub = tla.make_tensor(print_ptr, _ub_row_major_layout())
    return loaded, copy_ub, gm, print_ub


def _print_ub_tensor(
    loaded: Any, copy_ub: Any, gm: Any, print_ub: Any, *, calls: int
) -> None:
    tla.copy(copy_ub, gm)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.print(print_ub, 16)
    if calls == 2:
        tla.print(print_ub, 8)


@tla.kernel
def print_tensor_ub_dynamic_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(32, tla.Float32, tla.AddressSpace.ub, 256)
    copy_layout = _ub_row_major_layout()
    gm = tla.make_tensor(value.ptr, copy_layout)
    ub = tla.make_tensor(ptr, copy_layout)
    dynamic_ub = tla.make_tensor(ptr, _dynamic_row_major_layout(rows))
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(dynamic_ub, length)


def _kernel(args: argparse.Namespace) -> Callable[..., None]:
    if args.storage == "ub":
        if args.arch_scope != "aiv.c310":
            raise tla.TlaExecutionError(
                "UB tensor tla.print requires --arch-scope aiv.c310"
            )
        kernels = {
            ("base", 1): print_tensor_ub_base_kernel,
            ("base", 2): print_tensor_ub_base_two_calls_kernel,
            ("aligned-offset", 1): print_tensor_ub_aligned_offset_kernel,
            ("dynamic", 1): print_tensor_ub_dynamic_kernel,
        }
        try:
            return kernels[(args.case, args.calls)]
        except KeyError as exc:
            raise tla.TlaExecutionError(
                "unsupported UB tensor case/call combination"
            ) from exc
    if args.case == "capacity":
        if args.calls != 1:
            raise tla.TlaExecutionError("the capacity case supports one call")
        if args.arch_scope == "aic.c310":
            return print_tensor_capacity_aic_kernel
        if args.arch_scope == "aiv.c310":
            return print_tensor_capacity_aiv_kernel
    if args.dynamic_shape:
        if args.calls != 1:
            raise tla.TlaExecutionError("the dynamic GM case supports one call")
        if args.arch_scope == "aic.c310":
            return print_tensor_dynamic_aic_kernel
        if args.arch_scope == "aiv.c310":
            return print_tensor_dynamic_aiv_kernel
    kernels = {
        ("aic.c310", 1): print_tensor_aic_kernel,
        ("aiv.c310", 1): print_tensor_aiv_kernel,
        ("aic.c310", 2): print_tensor_aic_two_calls_kernel,
        ("aiv.c310", 2): print_tensor_aiv_two_calls_kernel,
    }
    try:
        return kernels[(args.arch_scope, args.calls)]
    except KeyError as exc:
        raise tla.TlaExecutionError(
            "tensor tla.print supports --arch-scope aic.c310 or aiv.c310"
        ) from exc


def _format_record(
    spec_or_values: _DTypeSpec | list[float | int] = DTYPE_SPECS["f32"],
    *,
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    arch_scope: str = "aiv.c310",
) -> str:
    from catlass.execution import _format_print_tensor_record

    if isinstance(spec_or_values, _DTypeSpec):
        spec = spec_or_values
        record_values = spec.values if values is None else values
    else:
        spec = DTYPE_SPECS["f32"]
        record_values = spec_or_values if values is None else values
    return _format_print_tensor_record(
        record_values,
        shape=shape,
        dtype=spec.token,
        subblock=0 if arch_scope.startswith("aiv.") else None,
    )


def _public_records(output: str) -> list[str]:
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("tla.print ")
    ]


def _verify_public_output(
    output: str,
    spec: _DTypeSpec = DTYPE_SPECS["f32"],
    *,
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    arch_scope: str = "aiv.c310",
) -> str:
    expected = _format_record(spec, values=values, shape=shape, arch_scope=arch_scope)
    records = _public_records(output)
    if records != [expected]:
        raise tla.TlaExecutionError(
            "tensor tla.print native initialization or decoding failed: "
            f"expected exactly {expected!r}, got {records!r}"
        )
    return expected


def _verify_multi_record_public_output(
    output: str,
    *,
    calls: int,
    block_count: int,
    spec: _DTypeSpec = DTYPE_SPECS["f32"],
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    arch_scope: str = "aiv.c310",
) -> str:
    from catlass.execution import _format_print_tensor_record

    record_values = spec.values if values is None else values
    records = _public_records(output)
    subblocks: tuple[int | None, ...] = (
        (0,) if arch_scope.startswith("aiv.") else (None,)
    )
    expected_identities = {
        (call, block, subblock)
        for call in range(calls)
        for block in range(block_count)
        for subblock in subblocks
    }
    seen: set[tuple[int, int, int | None]] = set()
    expected = []
    for record in records:
        match = re.match(
            r"^tla\.print call=(\d+) block=(\d+) "
            r"dtype=\S+(?: position=\S+)?(?: subblock=(\d+))? ",
            record,
        )
        if match is None:
            raise tla.TlaExecutionError(
                f"tensor tla.print has malformed record {record!r}"
            )
        call = int(match.group(1))
        block = int(match.group(2))
        subblock = int(match.group(3)) if match.group(3) is not None else None
        identity = (call, block, subblock)
        if identity not in expected_identities or identity in seen:
            raise tla.TlaExecutionError(
                f"tensor tla.print has unexpected record identity {identity!r}"
            )
        seen.add(identity)
        count = 16 if call == 0 else 8
        expected.append(
            _format_print_tensor_record(
                record_values[:count],
                shape=shape,
                dtype=spec.token,
                call=call,
                block=block,
                subblock=subblock,
            )
        )
    if seen != expected_identities or records != expected:
        raise tla.TlaExecutionError(
            f"tensor tla.print expected {expected_identities!r}, got {records!r}"
        )
    return "\n".join(expected)


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
    values: tuple[float | int, ...],
    shape: tuple[int, ...],
) -> _RuntimeInput:
    torch_dtype = getattr(torch, spec.torch_dtype, None)
    if torch_dtype is not None:
        try:
            owner = (
                torch.tensor(values, dtype=torch_dtype, device="npu")
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
    return _make_external_unsigned_input(torch, spec, values=values, shape=shape)


def _compile(
    args: argparse.Namespace,
    kernel: Callable[..., None],
    kernel_args: tuple[Any, ...],
) -> Any:
    return tla.compile(
        kernel,
        *kernel_args,
        arch_scope=args.arch_scope,
        cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        force_recompile=args.force_recompile,
    )


def _run_spec(args: argparse.Namespace, torch: Any, spec: _DTypeSpec) -> None:
    _configure_ub_kernel(spec)
    kernel = _kernel(args)
    source_shape = CAPACITY_SHAPE if args.case == "capacity" else SOURCE_SHAPE
    if args.case == "capacity":
        source = (
            torch.arange(CAPACITY_SHAPE[0], dtype=torch.float32, device="npu")
            .reshape(CAPACITY_SHAPE)
            .contiguous()
        )
        value = tla.from_dlpack(source, layout_tag=tla.arch.RowMajor)
        expected_values: list[float | int] = [
            float(value)
            for value in range(CAPACITY_SHAPE[0])
        ]
    else:
        runtime_input = _make_runtime_input(
            torch,
            spec,
            values=spec.values * 2,
            shape=SOURCE_SHAPE,
        )
        source = runtime_input.owner
        value = runtime_input.value
        expected_values = list(spec.values)
        if args.layout == "column-major":
            source = source.permute(1, 0).contiguous()
            value = tla.from_dlpack(source, layout_tag=tla.arch.ColumnMajor)
            expected_values = [
                value
                for value in source.flatten()[: len(spec.values)].tolist()
            ]
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
        UB_SHAPE if args.storage == "ub" and args.case != "dynamic" else source_shape
    )
    if args.calls == 1 and args.block == 1:
        rendered = _verify_public_output(
            captured.getvalue(),
            spec,
            values=expected_values,
            shape=output_shape,
            arch_scope=args.arch_scope,
        )
    else:
        rendered = _verify_multi_record_public_output(
            captured.getvalue(),
            calls=args.calls,
            block_count=args.block,
            spec=spec,
            values=expected_values,
            shape=output_shape,
            arch_scope=args.arch_scope,
        )
    print(rendered)
    print(f"case dtype={spec.token} compile_ok=True")
    print(f"case dtype={spec.token} launch_ok=True")
    print(f"case dtype={spec.token} output_ok=True")


def run(args: argparse.Namespace) -> int:
    if args.case == "capacity" and (
        args.storage != "gm" or args.dynamic_shape or args.block != 1
    ):
        raise tla.TlaExecutionError(
            "the capacity case requires static GM printing with --block 1"
        )
    if args.layout == "column-major" and (
        args.storage != "gm" or args.case != "base" or args.dynamic_shape
    ):
        raise tla.TlaExecutionError(
            "the column-major case requires static GM base printing"
        )
    if (
        args.case in ("capacity", "dynamic")
        or args.dynamic_shape
        or args.layout == "column-major"
    ) and (args.all_dtypes or args.dtype != "f32"):
        raise tla.TlaExecutionError(
            "capacity, dynamic-shape, and column-major cases require --dtype f32"
        )

    import torch
    import torch_npu  # noqa: F401

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        specs = DTYPE_SPECS.values() if args.all_dtypes else (DTYPE_SPECS[args.dtype],)
        for spec in specs:
            _run_spec(args, torch, spec)
        return 0
    finally:
        tla.finalize()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    dtype = parser.add_mutually_exclusive_group()
    dtype.add_argument("--dtype", choices=tuple(DTYPE_SPECS), default="f32")
    dtype.add_argument("--all-dtypes", action="store_true")
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
    parser.add_argument("--dynamic-shape", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--calls", type=int, choices=(1, 2), default=1)
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
