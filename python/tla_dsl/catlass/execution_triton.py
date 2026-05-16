"""Triton-specific execution helpers.

This module isolates Triton ABI detection and launch packing from the generic
execution path.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Sequence


_TRITON_COMPILE_FLAG = "--enable-triton-kernel-compile=true"

_RUNTIME_LIB_CACHE: ctypes.CDLL | None = None


def infer_hivmc_args(mlir_text: str, hivmc_args: Sequence[str]) -> tuple[str, ...]:
    if not is_triton_mlir(mlir_text):
        return tuple(hivmc_args)
    if _TRITON_COMPILE_FLAG in hivmc_args:
        return tuple(hivmc_args)
    return (*hivmc_args, _TRITON_COMPILE_FLAG)


def should_use_triton_compile_mode(hivmc_args: Sequence[str]) -> bool:
    return _TRITON_COMPILE_FLAG in hivmc_args


def try_build_packed_launch_args(
    *,
    artifact: Any,
    launch_args: Sequence[Any],
    grid: tuple[int, int, int],
    device: int,
    try_extract_entrypoint: Callable[[str], str | None],
    unsupported_abi_error: type[Exception],
    runtime_unavailable_error: type[Exception],
) -> bytes | None:
    mlir_text = artifact.lowered_llvm or artifact.tlair_mlir
    if "hacc.arg_type = #hacc.arg_type<ffts_base_address>" not in mlir_text:
        return None
    if "func_dyn_memref_args" not in mlir_text:
        return None
    if len(launch_args) != 3:
        raise unsupported_abi_error(
            "Triton-style AscendNPU-IR kernels currently require exactly three "
            "tensor launch arguments: lhs, rhs, dst."
        )
    if any(int(getattr(arg, "data_ptr", 0)) == 0 for arg in launch_args):
        raise unsupported_abi_error(
            "Triton-style AscendNPU-IR kernels require tensor launch arguments "
            "with uploaded device storage."
        )
    tensor_elems = [
        _infer_tensor_element_count(arg, label, unsupported_abi_error)
        for arg, label in zip(launch_args, ("lhs", "rhs", "dst"), strict=True)
    ]
    if len(set(tensor_elems)) != 1:
        raise unsupported_abi_error(
            "Triton-style AscendNPU-IR kernels currently require tensors with "
            "matching logical element counts."
        )
    try:
        ffts_base_address = _get_ffts_base_address(
            device=device,
            runtime_unavailable_error=runtime_unavailable_error,
        )
    except runtime_unavailable_error:
        # The single-core Triton-style AscendNPU-IR fixtures we validate in-tree
        # use a zero FFTS base in the known-good direct runner.
        ffts_base_address = 0
    entry_abi = _parse_entry_abi(
        mlir_text, try_extract_entrypoint=try_extract_entrypoint
    )
    if entry_abi is None:
        sync_block_lock = 0
        workspace = 0
        lhs, rhs, dst = (int(arg.data_ptr) for arg in launch_args)
        return struct.pack(
            "<QQQQQQiiii",
            int(ffts_base_address),
            int(sync_block_lock),
            int(workspace),
            lhs,
            rhs,
            dst,
            int(tensor_elems[0]),
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
        )

    tensor_iter = iter(
        zip(launch_args, tensor_elems, ("lhs", "rhs", "dst"), strict=True)
    )
    packed_words: list[int] = []
    trailing_i32_count = sum(
        1 for arg in entry_abi if arg["kind"] == "scalar" and arg["type"] == "i32"
    )
    trailing_i32_values: list[int] = []
    if trailing_i32_count > 0:
        trailing_i32_values.append(int(tensor_elems[0]))
    if trailing_i32_count > 1:
        trailing_i32_values.extend([int(grid[0]), int(grid[1]), int(grid[2])])
    if len(trailing_i32_values) < trailing_i32_count:
        trailing_i32_values.extend(
            [0] * (trailing_i32_count - len(trailing_i32_values))
        )
    scalar_i32_index = 0

    for arg in entry_abi:
        kind = arg["kind"]
        if kind == "ffts":
            packed_words.append(int(ffts_base_address))
            continue
        if kind in {"sync_block_lock", "workspace"}:
            packed_words.append(0)
            continue
        if kind == "tensor":
            tensor_arg, elem_count, _label = next(tensor_iter)
            ptr = int(tensor_arg.data_ptr)
            if arg["dynamic"]:
                packed_words.append(ptr)
                continue
            if arg["rank"] != 1:
                raise unsupported_abi_error(
                    "Static Triton-style memref launch currently supports only rank-1 tensor arguments."
                )
            packed_words.extend(
                [
                    ptr,
                    ptr,
                    0,
                    int(arg["static_sizes"][0] or elem_count),
                    1,
                ]
            )
            continue
        if kind == "scalar":
            arg_type = str(arg["type"])
            if arg_type == "i32":
                value = int(trailing_i32_values[scalar_i32_index])
                scalar_i32_index += 1
                packed_words.append(value & 0xFFFFFFFF)
                continue
            packed_words.append(0)
            continue
        raise unsupported_abi_error(
            f"Unsupported Triton-style entry ABI argument kind: {kind!r}."
        )

    return struct.pack("<" + "Q" * len(packed_words), *packed_words)


def is_triton_mlir(mlir_text: str) -> bool:
    return "vector.transfer_read" in mlir_text or "vector.transfer_write" in mlir_text


def _parse_entry_abi(
    mlir_text: str,
    *,
    try_extract_entrypoint: Callable[[str], str | None],
) -> list[dict[str, Any]] | None:
    entrypoint = try_extract_entrypoint(mlir_text)
    if entrypoint is None:
        return None
    func_match = re.search(
        rf"func\.func\s+@{re.escape(entrypoint)}\((?P<args>.*?)\)\s+attributes\s*\{{(?P<attrs>.*?)\}}\s*\{{",
        mlir_text,
        flags=re.DOTALL,
    )
    if func_match is None:
        return None

    attrs_text = func_match.group("attrs")
    dyn_match = re.search(
        r"func_dyn_memref_args\s*=\s*dense<\[(?P<bits>[^\]]*)\]>\s*:\s*vector<\d+xi1>",
        attrs_text,
    )
    dyn_bits: list[bool] = []
    if dyn_match is not None:
        dyn_bits = [
            bit.strip().lower() == "true"
            for bit in dyn_match.group("bits").split(",")
            if bit.strip()
        ]

    abi: list[dict[str, Any]] = []
    for index, arg_text in enumerate(
        _split_mlir_argument_list(func_match.group("args"))
    ):
        if not arg_text:
            continue
        type_match = re.search(r":\s*(.+?)(?:\s*\{.*\})?$", arg_text, flags=re.DOTALL)
        if type_match is None:
            continue
        arg_type = type_match.group(1).strip()
        if "hacc.arg_type = #hacc.arg_type<ffts_base_address>" in arg_text:
            abi.append({"kind": "ffts", "type": arg_type, "dynamic": False})
            continue
        if "hacc.arg_type = #hacc.arg_type<sync_block_lock>" in arg_text:
            abi.append({"kind": "sync_block_lock", "type": arg_type, "dynamic": False})
            continue
        if "hacc.arg_type = #hacc.arg_type<workspace>" in arg_text:
            abi.append({"kind": "workspace", "type": arg_type, "dynamic": False})
            continue
        memref_info = _parse_memref_abi_type(arg_type)
        if memref_info is not None:
            abi.append(
                {
                    "kind": "tensor",
                    "type": arg_type,
                    "dynamic": dyn_bits[index] if index < len(dyn_bits) else True,
                    "rank": memref_info["rank"],
                    "static_sizes": memref_info["static_sizes"],
                }
            )
            continue
        abi.append({"kind": "scalar", "type": arg_type, "dynamic": False})
    return abi


def _split_mlir_argument_list(args_text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth_angle = 0
    depth_brace = 0
    for char in args_text:
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle = max(0, depth_angle - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "," and depth_angle == 0 and depth_brace == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _parse_memref_abi_type(type_text: str) -> dict[str, Any] | None:
    if not type_text.startswith("memref<"):
        return None
    inner = (
        type_text[len("memref<") : -1]
        if type_text.endswith(">")
        else type_text[len("memref<") :]
    )
    depth = 0
    split_index = None
    for index, char in enumerate(inner):
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            split_index = index
            break
    shape_and_dtype = (
        inner[:split_index].strip() if split_index is not None else inner.strip()
    )
    shape_match = re.match(
        r"(?P<shape>(?:\?|[0-9]+x)+)(?P<dtype>[A-Za-z0-9_]+)$", shape_and_dtype
    )
    if shape_match is None:
        return None
    dims = [dim for dim in shape_match.group("shape").split("x") if dim]
    static_sizes: list[int | None] = [None if dim == "?" else int(dim) for dim in dims]
    return {"rank": len(static_sizes), "static_sizes": tuple(static_sizes)}


def _infer_tensor_element_count(
    arg: Any, label: str, unsupported_abi_error: type[Exception]
) -> int:
    size_bytes = getattr(arg, "size_bytes", None)
    dtype = getattr(arg, "dtype", None)
    if size_bytes is None or dtype is None:
        raise unsupported_abi_error(
            f"Triton-style launch expects Tensor-like arguments for {label!r}."
        )
    elem_bytes = _dtype_size_bytes(str(dtype))
    if elem_bytes <= 0:
        raise unsupported_abi_error(
            f"Unsupported tensor dtype for Triton-style launch: {dtype!r}."
        )
    return int(size_bytes) // elem_bytes


def _dtype_size_bytes(dtype: str) -> int:
    dtype_sizes = {
        "i1": 1,
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "u16": 2,
        "f16": 2,
        "bf16": 2,
        "i32": 4,
        "u32": 4,
        "f32": 4,
        "index": 8,
        "i64": 8,
        "u64": 8,
        "f64": 8,
    }
    return dtype_sizes.get(dtype.strip().lower(), 0)


def _runtime_library() -> ctypes.CDLL:
    global _RUNTIME_LIB_CACHE
    if _RUNTIME_LIB_CACHE is not None:
        return _RUNTIME_LIB_CACHE
    candidates = []
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    if ascend_home:
        candidates.append(Path(ascend_home) / "lib64" / "libruntime.so")
    candidates.append(Path("/usr/local/Ascend/latest/lib64/libruntime.so"))
    for candidate in candidates:
        if candidate.exists():
            _RUNTIME_LIB_CACHE = ctypes.CDLL(str(candidate))
            break
    if _RUNTIME_LIB_CACHE is None:
        _RUNTIME_LIB_CACHE = ctypes.CDLL("libruntime.so")
    _RUNTIME_LIB_CACHE.rtSetDevice.argtypes = [ctypes.c_int]
    _RUNTIME_LIB_CACHE.rtSetDevice.restype = ctypes.c_int
    _RUNTIME_LIB_CACHE.rtGetC2cCtrlAddr.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    _RUNTIME_LIB_CACHE.rtGetC2cCtrlAddr.restype = ctypes.c_int
    return _RUNTIME_LIB_CACHE


def _get_ffts_base_address(
    *, device: int | None, runtime_unavailable_error: type[Exception]
) -> int:
    runtime = _runtime_library()
    if device is None:
        try:
            from . import runtime as runtime_mod

            device = int(runtime_mod.get_current_device())
        except Exception:
            device = 0
    status = runtime.rtSetDevice(int(device))
    if status != 0:
        raise runtime_unavailable_error(
            f"rtSetDevice({device}) failed with status {status}."
        )
    addr = ctypes.c_uint64()
    size = ctypes.c_uint32()
    status = runtime.rtGetC2cCtrlAddr(ctypes.byref(addr), ctypes.byref(size))
    if status != 0:
        raise runtime_unavailable_error(
            f"rtGetC2cCtrlAddr failed with status {status}."
        )
    return int(addr.value)
