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
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack

import grouped_matmul_slice_m_kernels as _kernels

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

m = 1024
n = _kernels.N_DIM
k = _kernels.K_DIM
problem_count = 4
grouped_matmul_slice_m_kernel = _kernels.grouped_matmul_slice_m_kernel


def _parse_layout_choice(name: str) -> LayoutChoice:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, LayoutChoice] = {
        "row": "row",
        "rowmajor": "row",
        "col": "col",
        "columnmajor": "col",
        "colmajor": "col",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown layout {name!r}; expected one of row, col"
        )
    return mapping[key]


def _gm_layout_tag(choice: LayoutChoice) -> Any:
    if choice == "row":
        return tla.arch.RowMajor
    return tla.arch.ColumnMajor


def _parse_elem_dtype(name: str) -> ElemDType:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, ElemDType] = {
        "f16": "f16",
        "float16": "f16",
        "fp16": "f16",
        "half": "f16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f32": "f32",
        "float32": "f32",
        "fp32": "f32",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown dtype {name!r}; expected f16, bf16, or f32"
        )
    return mapping[key]


def _tla_elem_dtype(token: ElemDType) -> Any:
    if token == "f16":
        return tla.Float16
    if token == "bf16":
        return tla.BFloat16
    return tla.Float32


def _validate_mmad_dtype_triple(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    if dtype_a != dtype_b:
        raise ValueError("dtype-a and dtype-b must match for tla.mmad.")
    allowed = {
        ("f16", "f16", "f32"),
        ("f16", "f16", "f16"),
        ("bf16", "bf16", "f32"),
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
    }
    if (dtype_a, dtype_b, dtype_c) not in allowed:
        raise ValueError(
            "unsupported dtype triple; allowed: "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32"
        )


def _apply_kernel_dtypes(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    _kernels.DTYPE_A = _tla_elem_dtype(dtype_a)
    _kernels.DTYPE_B = _tla_elem_dtype(dtype_b)
    _kernels.DTYPE_GM_C = _tla_elem_dtype(dtype_c)
    _kernels.DTYPE_C = tla.Float32


def _apply_problem_size(m_val: int, n_val: int, k_val: int, g_val: int) -> None:
    global m, n, k, problem_count
    if min(m_val, n_val, k_val, g_val) <= 0:
        raise ValueError(
            f"m,n,k,groups must be positive; got m={m_val} n={n_val} k={k_val} g={g_val}"
        )
    _kernels.M_DIM = m_val
    _kernels.N_DIM = n_val
    _kernels.K_DIM = k_val
    _kernels.GROUPS = g_val
    m, n, k, problem_count = m_val, n_val, k_val, g_val


def _group_list_prefix(current_ms: tuple[int, ...]) -> tuple[int, ...]:
    """Length G+1 prefix with leading 0: currentM[g] = prefix[g+1] - prefix[g]."""
    out = [0]
    for current_m in current_ms:
        out.append(out[-1] + current_m)
    return tuple(out)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Host-side tensors require PyTorch.") from exc
    return torch


def _torch_dtype(token: ElemDType) -> Any:
    torch = _require_torch()
    if token == "f16":
        return torch.float16
    if token == "bf16":
        return torch.bfloat16
    return torch.float32


def _require_torch_npu(device_id: int) -> Any:
    torch = _require_torch()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("This example requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()


def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    """Wrap one device buffer as a dynamic-layout ``tla.Tensor`` (schema v4)."""
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


def generate_average_current_m(
    m_val: int, g_val: int, *, l1_tm: int
) -> tuple[int, ...]:
    """Partition ``[0, m_val)`` into ``g_val`` nearly-equal P0-aligned groups.

    Old code used ``m // g`` for every group and **dropped** ``m % g`` rows, so
    ``sum(current_ms) < m`` whenever ``m`` was not divisible by ``g``.

    P0 needs heights that are multiples of ``l1_tm``. Split the ``m // l1_tm``
    tiles as evenly as possible: the first ``n_tiles % g`` groups get one extra
    tile. Empty groups appear only when ``g > n_tiles``.
    """
    if g_val <= 0:
        raise ValueError(f"groups must be positive; got {g_val}")
    if m_val % l1_tm != 0:
        raise ValueError(
            f"P0 average groups require m % l1_tm == 0; got m={m_val}, l1_tm={l1_tm}"
        )
    n_tiles = m_val // l1_tm
    base, rem = divmod(n_tiles, g_val)
    out = tuple(((base + 1) if i < rem else base) * l1_tm for i in range(g_val))
    if sum(out) != m_val:
        raise RuntimeError(
            f"internal error: average groups sum to {sum(out)}, expected m={m_val}"
        )
    return out


def generate_random_current_m(
    m_val: int, g_val: int, *, seed: int = 0, l1_tm: int
) -> tuple[int, ...]:
    """Randomly partition ``[0, m_val)`` into ``g_val`` groups for P0.

    Old bug: sampled ``g_val`` endpoints in ``[0, m]`` and diffed them, so
    ``sum(out) == prefix[-1]`` (often ``< m_val``) and heights were almost never
    ``l1_tm``-aligned — ``_validate_l1_aligned_groups`` would reject nearly always.

    P0 fix: only cut on L1_M tile boundaries. Distribute ``m_val // l1_tm`` tiles
    across ``g_val`` bags (empty groups allowed, same as C++ zero-height groups).
    """
    import random

    if g_val <= 0:
        raise ValueError(f"groups must be positive; got {g_val}")
    if m_val % l1_tm != 0:
        raise ValueError(
            f"P0 random groups require m % l1_tm == 0; got m={m_val}, l1_tm={l1_tm}"
        )
    n_tiles = m_val // l1_tm
    rng = random.Random(seed)
    # g_val-1 cut points on the tile grid [0, n_tiles], then force ends 0 and n_tiles.
    cuts = sorted(rng.randint(0, n_tiles) for _ in range(g_val - 1))
    points = [0, *cuts, n_tiles]
    out = tuple((points[i + 1] - points[i]) * l1_tm for i in range(g_val))
    if sum(out) != m_val:
        raise RuntimeError(
            f"internal error: random groups sum to {sum(out)}, expected m={m_val}"
        )
    return out


def _validate_l1_aligned_groups(
    current_ms: tuple[int, ...], l1_tm: int, *, expected_m: int | None = None
) -> None:
    total = 0
    for i, current_m in enumerate(current_ms):
        if current_m < 0:
            raise ValueError(f"negative currentM at group {i}")
        if current_m == 0:
            continue
        if total % l1_tm != 0:
            raise ValueError(
                f"P0 requires group start % l1_tm == 0; group {i} start={total}"
            )
        if current_m % l1_tm != 0:
            raise ValueError(
                f"P0 requires currentM % l1_tm == 0; group {i} currentM={current_m}, "
                f"l1_tm={l1_tm}."
            )
        total += current_m
    # Empty leading/middle groups skip the loop body but still occupy no rows;
    # include zeros in the coverage check via sum(current_ms).
    covered = sum(current_ms)
    if expected_m is not None and covered != expected_m:
        raise ValueError(
            f"group heights must cover all m rows; sum(currentM)={covered}, m={expected_m}, "
            f"currentM={current_ms}"
        )


def _compile_only_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any, Any]:
    from catlass import runtime as runtime_mod

    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    gl_len = problem_count + 1
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(m, k),
                ta,
                origin_shape=tla.make_shape(m, k),
                layout_tag=_gm_layout_tag(layout_a),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(problem_count * k, n),
                tb,
                origin_shape=tla.make_shape(problem_count * k, n),
                layout_tag=_gm_layout_tag(layout_b),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(gl_len),
                tla.Int32,
                origin_shape=tla.make_shape(gl_len),
                layout_tag=tla.arch.RowMajor,
            ).mark_compact_shape_dynamic(0),
            tla.Tensor(
                tla.make_shape(m, n),
                tc,
                origin_shape=tla.make_shape(m, n),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
        )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


# Relative tol for torch.isclose (Catlass CompareData uses 1/128 for large K).
DEFAULT_RTOL = 1.0 / 128.0


def _comparison_rtol(args: argparse.Namespace) -> float:
    return float(args.rtol)


def _comparison_atol(dtype_c: ElemDType, args: argparse.Namespace) -> float:
    """Same absolute floor as ``basic_matmul._comparison_atol``."""
    if dtype_c in ("f16", "bf16"):
        return max(float(args.atol), 5e-3)
    return float(args.atol)


def _first_mismatch_torch(
    actual: Any, expected: Any, *, rtol: float, atol: float
) -> dict[str, Any] | None:
    torch = _require_torch()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    if bool(close.all()):
        return None
    flat = close.logical_not().nonzero(as_tuple=False)[0]
    row, col = (int(v) for v in flat)
    av = float(actual[row, col].item())
    ev = float(expected[row, col].item())
    return {
        "index": [row, col],
        "actual": av,
        "expected": ev,
        "abs_err": abs(av - ev),
        "scaled_err": abs(av - ev) / max(1.0, abs(ev)),
    }


def _grouped_golden(
    torch: Any,
    a: Any,
    b_packed: Any,
    current_ms: tuple[int, ...],
    *,
    n_val: int,
    k_val: int,
) -> Any:
    c = torch.zeros((a.shape[0], n_val), dtype=torch.float32, device=a.device)
    offset = 0
    for g, current_m in enumerate(current_ms):
        if current_m > 0:
            a_g = a[offset : offset + current_m].to(torch.float32)
            b_g = b_packed[g * k_val : (g + 1) * k_val].to(torch.float32)
            c[offset : offset + current_m] = a_g @ b_g
        offset += current_m
    return c


def dump_tlair(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> str:
    return grouped_matmul_slice_m_kernel.dump_mlir(
        type_args=_compile_only_type_args(
            layout_a, layout_b, dtype_a, dtype_b, dtype_c
        )
    )


def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
    artifact = tla.compile(
        grouped_matmul_slice_m_kernel,
        *_compile_only_type_args(
            args.layout_a,
            args.layout_b,
            args.dtype_a,
            args.dtype_b,
            args.dtype_c,
        ),
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0


def run_single_case(
    args: argparse.Namespace,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    current_ms: tuple[int, ...],
) -> int:
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    torch = _require_torch_npu(args.device)
    device = "npu"
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)

    # Inputs: randn then clamp to [-5, 5] (matches Catlass FillRandomData range).
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.data_seed))
    a = (
        torch.randn((m, k), generator=gen, device=device, dtype=torch.float32)
        .clamp(-5.0, 5.0)
        .to(torch_dtype_a)
    )
    b_packed = (
        torch.randn(
            (problem_count * k, n), generator=gen, device=device, dtype=torch.float32
        )
        .clamp(-5.0, 5.0)
        .to(torch_dtype_b)
    )
    c = torch.full((m, n), args.sentinel, dtype=torch_dtype_c, device=device)
    prefix = _group_list_prefix(current_ms)
    group_list = torch.tensor(prefix, dtype=torch.int32, device=device)

    expected_f32 = _grouped_golden(
        torch, a, b_packed, current_ms, n_val=n, k_val=k
    )
    valid_rows = sum(current_ms)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32[:valid_rows].to(torch_dtype_c).to(torch.float32)
    else:
        expected = expected_f32[:valid_rows]
    rtol = _comparison_rtol(args)
    atol = _comparison_atol(dtype_c, args)

    # Single launch: A, B, group_list, C (matches Ascend950 torch op order).
    tla_a = _create_tla_tensor(a, layout_a)
    tla_b = _create_tla_tensor(b_packed, layout_b)
    tla_gl = from_dlpack(
        group_list.contiguous(), layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)
    tla_c = _create_tla_tensor(c, "row")
    artifact = tla.compile(
        grouped_matmul_slice_m_kernel,
        tla_a,
        tla_b,
        tla_gl,
        tla_c,
        **_runtime_kwargs(args),
    )
    artifact(tla_a, tla_b, tla_gl, tla_c, block=args.block)
    torch.npu.synchronize()

    actual = c[:valid_rows].to(torch.float32)
    sentinel_f32 = torch.full_like(actual, args.sentinel)
    unchanged = torch.isclose(actual, sentinel_f32, rtol=0.0, atol=atol)
    expected_match = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    first_mismatch = _first_mismatch_torch(actual, expected, rtol=rtol, atol=atol)

    print(
        "compile_ok=True "
        f"host=torch_npu layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c} "
        f"groups={problem_count} launches=1"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print(f"cache_key={artifact.cache_key}")
    print("launch_ok=True")
    print(f"GROUP_CURRENT_M={current_ms}")
    print(f"GROUP_LIST_PREFIX={prefix}")
    print(f"data=randn.clamp([-5,5]) data_seed={args.data_seed}")
    print(f"isclose rtol={rtol} atol={atol}")
    print(f"C[:{valid_rows}] unchanged? {bool(unchanged.all())}")
    print(f"C equals grouped golden? {bool(expected_match.all())}")
    print(f"C changed count={int((~unchanged).sum().item())}")
    print(f"first mismatch={first_mismatch}")
    return 0 if first_mismatch is None else 1


MMAD_DTYPE_TRIPLES: tuple[tuple[ElemDType, ElemDType, ElemDType], ...] = (
    ("f16", "f16", "f32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "f32"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
)


def _layout_pairs(
    args: argparse.Namespace,
) -> list[tuple[LayoutChoice, LayoutChoice]]:
    if args.all_layouts:
        return [(la, lb) for la in ("row", "col") for lb in ("row", "col")]
    return [(args.layout_a, args.layout_b)]


def _dtype_triples(
    args: argparse.Namespace,
) -> list[tuple[ElemDType, ElemDType, ElemDType]]:
    if args.all_dtypes:
        return list(MMAD_DTYPE_TRIPLES)
    return [(args.dtype_a, args.dtype_b, args.dtype_c)]


def _resolve_current_ms(args: argparse.Namespace) -> tuple[int, ...]:
    if args.group_mode == "average":
        current_ms = generate_average_current_m(
            m, problem_count, l1_tm=_kernels.L1_TM
        )
    else:
        current_ms = generate_random_current_m(
            m, problem_count, seed=args.group_seed, l1_tm=_kernels.L1_TM
        )
    _validate_l1_aligned_groups(current_ms, _kernels.L1_TM, expected_m=m)
    return current_ms


def run(args: argparse.Namespace, current_ms: tuple[int, ...]) -> int:
    tla.initialize(device=args.device)
    try:
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"m={m}",
                    f"n={n}",
                    f"k={k}",
                    f"groups={problem_count}",
                    f"dtype_a={dtype_a}",
                    f"dtype_b={dtype_b}",
                    f"dtype_c={dtype_c}",
                    f"layout_a={layout_a}",
                    f"layout_b={layout_b}",
                    "---",
                )
                failed += run_single_case(
                    args,
                    layout_a,
                    layout_b,
                    dtype_a,
                    dtype_b,
                    dtype_c,
                    current_ms,
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grouped matmul slice-M. Single launch over all M-groups; "
            "device reads Int32 group_list prefix (len G+1). B packed as (G*K, N). "
            "P0 requires L1_M-aligned group sizes."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=_kernels.N_DIM)
    parser.add_argument("--k", type=int, default=_kernels.K_DIM)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--sentinel", type=float, default=-7.0)
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for torch.isclose (default 1/128).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance base; f16/bf16 use max(atol, 5e-3) like basic_matmul.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="RNG seed for randn inputs clamped to [-5, 5].",
    )
    parser.add_argument("--layout-a", type=_parse_layout_choice, default="row")
    parser.add_argument("--layout-b", type=_parse_layout_choice, default="row")
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run all four (layout-a, layout-b) combinations sequentially.",
    )
    parser.add_argument("--dtype-a", type=_parse_elem_dtype, default="f16")
    parser.add_argument("--dtype-b", type=_parse_elem_dtype, default="f16")
    parser.add_argument("--dtype-c", type=_parse_elem_dtype, default="f16")
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help=(
            "Run all supported (dtype-a, dtype-b, dtype-c) triples sequentially "
            "(with the chosen layout pair or all layout pairs when --all-layouts is set)."
        ),
    )
    parser.add_argument(
        "--group-mode", choices=("average", "random"), default="random"
    )
    parser.add_argument("--group-seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _apply_problem_size(args.m, args.n, args.k, args.groups)
    current_ms = _resolve_current_ms(args)
    print(f"current_ms: {current_ms}")
    print(f"group_list_prefix: {_group_list_prefix(current_ms)}")
    if not args.all_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    if args.dump_tlair:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--dump-tlair requires a single layout and dtype triple.")
        _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
        print(
            dump_tlair(
                args.layout_a,
                args.layout_b,
                args.dtype_a,
                args.dtype_b,
                args.dtype_c,
            )
        )
        return 0
    if args.build_only:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--build-only requires a single layout and dtype triple.")
        return build_only(args)
    return run(args, current_ms)


if __name__ == "__main__":
    raise SystemExit(main())
