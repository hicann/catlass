"""Host for multi-core split-K matmul (example 68).

Modes: --dump-tlair | --build-only | --run (default).

CLI:
  --dtype              : f16 | bf16 | f32  (A/B/C same element type; L0C/W fp32)
  --layout-a/--layout-b: row | col         (C/W RowMajor)
  --m/--n/--k          : positive GEMM shape (tail tiles handled in kernel)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack

import splitk_mmad_kernels as _kernels

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

# Default problem shape (host-only; kernel reads M/N/K from origin_shape).
DEFAULT_M = 256
DEFAULT_N = 512
DEFAULT_K = 1024

m = DEFAULT_M
n = DEFAULT_N
k = DEFAULT_K
splitk_factor = 2

l1_tm = _kernels.l1_tm
l1_tn = _kernels.l1_tn
l1_tk = _kernels.l1_tk
multi_core_splitk_mmad_kernel = _kernels.multi_core_splitk_mmad_kernel

SUPPORTED_DTYPES: tuple[ElemDType, ...] = ("f16", "bf16", "f32")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _parse_layout_choice(name: str) -> LayoutChoice:
    key = name.strip().lower()
    if key in ("row", "col"):
        return key  # type: ignore[return-value]
    raise argparse.ArgumentTypeError(f"unknown layout {name!r}; expected row or col")


def _parse_elem_dtype(name: str) -> ElemDType:
    key = name.strip().lower()
    if key in SUPPORTED_DTYPES:
        return key  # type: ignore[return-value]
    raise argparse.ArgumentTypeError(
        f"unknown dtype {name!r}; expected one of {', '.join(SUPPORTED_DTYPES)}"
    )


def _gm_layout_tag(choice: LayoutChoice) -> Any:
    return tla.arch.RowMajor if choice == "row" else tla.arch.ColumnMajor


def _tla_elem_dtype(token: ElemDType) -> Any:
    if token == "f16":
        return tla.Float16
    if token == "bf16":
        return tla.BFloat16
    return tla.Float32


def _torch_dtype(token: ElemDType) -> Any:
    torch = _require_torch()
    if token == "f16":
        return torch.float16
    if token == "bf16":
        return torch.bfloat16
    return torch.float32


def validate_dtype_triple(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    """Require A/B/C to use the same element type."""
    if dtype_a != dtype_b or dtype_a != dtype_c:
        raise SystemExit(
            "unsupported configuration:\n  - dtype-a, dtype-b, and dtype-c must match "
            f"(got {dtype_a}/{dtype_b}/{dtype_c}); allowed: f16 | bf16 | f32"
        )
    if dtype_a not in SUPPORTED_DTYPES:
        raise SystemExit(f"unsupported dtype {dtype_a!r}")


def validate_shape(m_val: int, n_val: int, k_val: int) -> None:
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise SystemExit(f"m, n, k must be positive; got ({m_val},{n_val},{k_val})")


def get_splitk_factor(
    m_val: int,
    n_val: int,
    k_val: int,
    aic_core_num: int,
    *,
    l1_m: int = l1_tm,
    l1_n: int = l1_tn,
    l1_k: int = l1_tk,
) -> int:
    """Compute split-K factor from problem shape and AIC core count."""
    factor = 2
    block_num = ceil_div(m_val, l1_m) * ceil_div(n_val, l1_n)
    k_tile_num = ceil_div(k_val, l1_k)
    if aic_core_num // block_num > 0:
        factor = aic_core_num // block_num
    return min(factor, k_tile_num)


def workspace_elems(m_val: int, n_val: int, factor: int) -> int:
    min_elems = (2 * 1024 * 1024) // 4
    if factor * m_val * n_val >= min_elems:
        return factor * m_val * n_val
    rows = max(factor * m_val, ceil_div(min_elems, n_val))
    return rows * n_val


def workspace_shape(m_val: int, n_val: int, factor: int) -> tuple[int, int]:
    elems = workspace_elems(m_val, n_val, factor)
    rows = max(factor * m_val, ceil_div(elems, n_val))
    if rows % m_val != 0:
        rows = ceil_div(rows, m_val) * m_val
    return rows, n_val


def _apply_kernel_dtypes(dtype: ElemDType) -> None:
    elem = _tla_elem_dtype(dtype)
    _kernels.DTYPE_A = elem
    _kernels.DTYPE_B = elem
    _kernels.DTYPE_C = tla.Float32  # L0C always fp32
    _kernels.DTYPE_W = tla.Float32
    _kernels.DTYPE_GM_C = elem


def compute_reduce_tiling(
    m_val: int,
    n_val: int,
    factor: int,
    aic_core_num: int,
    *,
    compute_length: int = 192 * 1024 // 4,
    ele_per_vector_block: int = 64,
    ele_align: int = 8,
    max_vl_loops: int = 512,
) -> dict[str, int]:
    """Compute AIV ReduceAdd tiling for flat M×N output.

    Spread element_count across 2× aic_core_num AIVs. Each AIV processes
    task_per_aiv elements per reduce_loops iteration. ub_row_stride is padded
    to ele_align; reduce_vl_loops = task_per_aiv // ele_per_vector_block.
    task_per_aiv is capped by UB budget and max_vl_loops.
    """
    if aic_core_num <= 0:
        raise ValueError(f"aic_core_num must be positive; got {aic_core_num}")
    elem_count = m_val * n_val
    aiv_num = aic_core_num * 2
    per_aiv = ceil_div(elem_count, aiv_num)
    task = ceil_div(per_aiv, ele_per_vector_block) * ele_per_vector_block

    # Cap task to fit in UB: factor slices × ub_stride ≤ compute_length
    task_vl_cap = max_vl_loops * ele_per_vector_block
    task_stage_cap = compute_length // factor // ele_per_vector_block * ele_per_vector_block
    task = min(task, task_stage_cap, task_vl_cap)
    if task == 0:
        task = ele_per_vector_block

    ub_stride = ceil_div(task, ele_align) * ele_align
    if factor * ub_stride > compute_length:
        raise ValueError(
            f"splitk reduce UB overflow: factor={factor} ub_row_stride={ub_stride} "
            f"compute_length={compute_length}"
        )
    loops = ceil_div(elem_count, task)
    return {
        "element_count": elem_count,
        "task_per_aiv": task,
        "reduce_loops": loops,
        "ub_row_stride": ub_stride,
        "reduce_vl_loops": ceil_div(task, ele_per_vector_block),
    }


def _inject_kernel_compile_params(
    factor: int,
    reduce: dict[str, int],
) -> None:
    """Push host-computed scheduling into the kernel module before compile."""
    _kernels.splitk_factor = factor
    _kernels.element_count = reduce["element_count"]
    _kernels.task_per_aiv = reduce["task_per_aiv"]
    _kernels.reduce_loops = reduce["reduce_loops"]
    _kernels.ub_row_stride = reduce["ub_row_stride"]
    _kernels.reduce_vl_loops = reduce["reduce_vl_loops"]


def _apply_problem(
    m_val: int,
    n_val: int,
    k_val: int,
    factor: int,
    *,
    aic_core_num: int = 28,
) -> None:
    global m, n, k, splitk_factor
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise ValueError(f"m, n, k must be positive; got {m_val}, {n_val}, {k_val}")
    if factor <= 0:
        raise ValueError(f"splitk_factor must be positive; got {factor}")
    m, n, k, splitk_factor = m_val, n_val, k_val, factor

    reduce = compute_reduce_tiling(m_val, n_val, factor, aic_core_num)
    _inject_kernel_compile_params(factor, reduce)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Host-side tensors require PyTorch. Install with ``pip install torch``."
        ) from exc
    return torch


def _require_torch_npu(device_id: int) -> Any:
    torch = _require_torch()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This example requires torch_npu for device DLPack bindings."
        ) from exc
    torch.npu.set_device(device_id)
    return torch


def _compile_only_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype: ElemDType,
) -> tuple[Any, Any, Any, Any]:
    from catlass import runtime as runtime_mod

    elem = _tla_elem_dtype(dtype)
    ws_rows, ws_cols = workspace_shape(m, n, splitk_factor)
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(m, k),
                elem,
                origin_shape=tla.make_shape(m, k),
                layout_tag=_gm_layout_tag(layout_a),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(k, n),
                elem,
                origin_shape=tla.make_shape(k, n),
                layout_tag=_gm_layout_tag(layout_b),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(m, n),
                elem,
                origin_shape=tla.make_shape(m, n),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(ws_rows, ws_cols),
                tla.Float32,
                origin_shape=tla.make_shape(ws_rows, ws_cols),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
        )


def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()


def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    # from_dlpack only stores data_ptr; keep the physical buffer alive for the
    # lifetime of the TLA tensor (col path allocates a permute().contiguous() temp).
    phys = _device_buffer_for_layout(dev_buf, layout)
    tensor = from_dlpack(phys, layout_tag=_gm_layout_tag(layout)).mark_layout_dynamic()
    tensor._host_storage = phys
    return tensor


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _compare_data_params(
    dtype: ElemDType, compute_num: int
) -> tuple[float, float]:
    """Relative tolerance thresholds for result comparison; compute_num is K."""
    threshold = 2048
    if dtype == "bf16":
        rtol = (1.0 / 128.0) if compute_num < threshold else (1.0 / 64.0)
        return rtol, 1.0 / 256.0
    rtol = (1.0 / 256.0) if compute_num < threshold else (1.0 / 128.0)
    return rtol, 1.0


def _mismatch_ratio_budget(dtype: ElemDType) -> float:
    """Maximum allowed fraction of elements outside tolerance."""
    return 1.0 / 10000.0 if dtype == "f32" else 1.0 / 1000.0


def _results_match_compare_data(
    actual: Any, expected: Any, *, dtype: ElemDType, compute_num: int
) -> dict[str, Any]:
    """Check actual vs expected with rtol/floor and mismatch-ratio budget."""
    torch = _require_torch()
    rtol, floor = _compare_data_params(dtype, compute_num)
    actual_f = actual.to(torch.float32)
    expect_f = expected.to(torch.float32)
    diff = (actual_f - expect_f).abs()
    thr = rtol * torch.maximum(torch.full_like(expect_f, floor), expect_f.abs())
    # NaN/Inf compare as False for `diff > thr`; count them as mismatches.
    bad = (diff > thr) | torch.isnan(actual_f) | torch.isinf(actual_f)
    n_total = int(expect_f.numel())
    n_bad = int(bad.sum().item())
    mismatch_ratio = (n_bad / n_total) if n_total else 0.0
    budget = _mismatch_ratio_budget(dtype)
    return {
        "ok": mismatch_ratio <= budget,
        "n_bad": n_bad,
        "n_total": n_total,
        "mismatch_ratio": mismatch_ratio,
        "mismatch_pct": 100.0 * mismatch_ratio,
        "budget_ratio": budget,
        "budget_pct": 100.0 * budget,
        "rtol": rtol,
        "floor": floor,
    }


def dump_tlair(
    layout_a: LayoutChoice, layout_b: LayoutChoice, dtype: ElemDType
) -> str:
    return multi_core_splitk_mmad_kernel.dump_mlir(
        type_args=_compile_only_type_args(layout_a, layout_b, dtype)
    )


def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype)
    tla.compile(
        multi_core_splitk_mmad_kernel,
        *_compile_only_type_args(args.layout_a, args.layout_b, args.dtype),
        **_runtime_kwargs(args),
    )
    print(f"compile_ok=True splitk_factor={splitk_factor}")
    return 0


def _run_one(
    args: argparse.Namespace,
    *,
    m_val: int,
    n_val: int,
    k_val: int,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    label: str = "",
    quiet: bool = False,
) -> dict[str, Any]:
    """Compile and launch one case; caller owns tla.initialize."""
    validate_dtype_triple(dtype_a, dtype_b, dtype_c)
    validate_shape(m_val, n_val, k_val)
    dtype = dtype_a
    factor = get_splitk_factor(m_val, n_val, k_val, args.block)
    _apply_problem(m_val, n_val, k_val, factor, aic_core_num=args.block)
    _apply_kernel_dtypes(dtype)

    torch = _require_torch_npu(args.device)
    device = "npu"
    torch_dtype = _torch_dtype(dtype)

    # Random inputs in fp32, then cast to target dtype.
    torch_a = (
        torch.rand(m, k, dtype=torch.float32, device=device) * 10.0 - 5.0
    ).to(torch_dtype)
    torch_b = (
        torch.rand(k, n, dtype=torch.float32, device=device) * 10.0 - 5.0
    ).to(torch_dtype)
    torch_c = torch.full((m, n), args.sentinel, dtype=torch_dtype, device=device)
    ws_rows, ws_cols = workspace_shape(m, n, splitk_factor)
    torch_w = torch.zeros((ws_rows, ws_cols), dtype=torch.float32, device=device)

    expected = (torch_a.to(torch.float32) @ torch_b.to(torch.float32)).to(torch_dtype)

    tla_a = _create_tla_tensor(torch_a, layout_a)
    tla_b = _create_tla_tensor(torch_b, layout_b)
    tla_c = _create_tla_tensor(torch_c, "row")
    tla_w = _create_tla_tensor(torch_w, "row")
    # NPU fills / col permute().contiguous() are async; launch must not race them.
    torch.npu.synchronize()

    kwargs = _runtime_kwargs(args)
    artifact = tla.compile(
        multi_core_splitk_mmad_kernel, tla_a, tla_b, tla_c, tla_w, **kwargs
    )
    artifact(tla_a, tla_b, tla_c, tla_w, block=args.block)
    torch.npu.synchronize()

    cmp = _results_match_compare_data(
        torch_c, expected, dtype=dtype, compute_num=k_val
    )
    ok = bool(cmp["ok"])

    tag = f" [{label}]" if label else ""
    if not quiet:
        print(
            f"case{tag} shape=({m},{n},{k}) dtype={dtype} "
            f"layout={layout_a}/{layout_b} "
            f"rtol={cmp['rtol']:.6g} floor={cmp['floor']:.6g} "
            f"mismatch={cmp['mismatch_pct']:.4f}% "
            f"(budget={cmp['budget_pct']:.4f}%) ok={ok}"
        )

    return {
        "label": label,
        "shape": (m, n, k),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "layout_a": layout_a,
        "layout_b": layout_b,
        "splitk_factor": splitk_factor,
        "ok": ok,
        "mismatch_pct": cmp["mismatch_pct"],
        "budget_pct": cmp["budget_pct"],
    }


def _layout_pairs(
    args: argparse.Namespace,
) -> list[tuple[LayoutChoice, LayoutChoice]]:
    if args.all_layouts:
        return [(la, lb) for la in ("row", "col") for lb in ("row", "col")]
    return [(args.layout_a, args.layout_b)]


def _dtype_list(args: argparse.Namespace) -> list[ElemDType]:
    if args.all_dtypes:
        return list(SUPPORTED_DTYPES)
    return [args.dtype]


def run(args: argparse.Namespace) -> int:
    validate_shape(args.m, args.n, args.k)
    tla.initialize(device=args.device)
    failed = 0
    try:
        for dtype in _dtype_list(args):
            validate_dtype_triple(dtype, dtype, dtype)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    f"dtype={dtype}",
                    f"layout_a={layout_a}",
                    f"layout_b={layout_b}",
                    "---",
                )
                result = _run_one(
                    args,
                    m_val=args.m,
                    n_val=args.n,
                    k_val=args.k,
                    layout_a=layout_a,
                    layout_b=layout_b,
                    dtype_a=dtype,
                    dtype_b=dtype,
                    dtype_c=dtype,
                    label=f"{dtype}_{layout_a}{layout_b}",
                )
                if not result["ok"]:
                    failed += 1
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-core split-K matmul (example 68). "
            "Same dtype on A/B/C (f16|bf16|f32); A/B layout row|col; "
            "L0C/workspace fp32. Single mixed kernel: AIC workspace + AIV ReduceAdd."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Compile and exit after generating kernel.o.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compile, launch, and compare against torch matmul (default).",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id.")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="GEMM M.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="GEMM N.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="GEMM K.")
    parser.add_argument(
        "--dtype",
        type=_parse_elem_dtype,
        default="f32",
        help="Element type for A/B/C (same dtype in/out): f16 | bf16 | f32.",
    )
    parser.add_argument(
        "--layout-a",
        type=_parse_layout_choice,
        default="row",
        help="GM layout of A: row or col.",
    )
    parser.add_argument(
        "--layout-b",
        type=_parse_layout_choice,
        default="row",
        help="GM layout of B: row or col.",
    )
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run all four (layout-a, layout-b) combinations sequentially.",
    )
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help="Run f16, bf16, and f32 (same A/B/C dtype) sequentially.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=28,
        help="AIC core count (also used for GetSplitkFactor).",
    )
    parser.add_argument("--sentinel", type=float, default=-7.0, help="Initial C value.")
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Unused; accuracy check uses rtol/floor comparison.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Compile cache directory.",
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Force kernel recompile (ignore on-disk / in-memory cache).",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--dump-tlair",
        action="store_true",
        help="Print TLA MLIR and exit.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    factor = get_splitk_factor(args.m, args.n, args.k, args.block)
    _apply_problem(args.m, args.n, args.k, factor, aic_core_num=args.block)
    print(
        f"problem=({m},{n},{k}) factor={splitk_factor} "
        f"dtype={args.dtype} layout={args.layout_a}/{args.layout_b}"
    )

    if args.dump_tlair:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--dump-tlair requires a single dtype and layout pair.")
        validate_shape(m, n, k)
        validate_dtype_triple(args.dtype, args.dtype, args.dtype)
        _apply_kernel_dtypes(args.dtype)
        print(dump_tlair(args.layout_a, args.layout_b, args.dtype))
        return 0
    if args.build_only:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--build-only requires a single dtype and layout pair.")
        validate_shape(m, n, k)
        validate_dtype_triple(args.dtype, args.dtype, args.dtype)
        return build_only(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
