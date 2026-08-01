"""Host for tail multi-core split-K matmul (example 69).

Modes: --dump-tlair | --build-only | --run (default).

CLI:
  --dtype              : f16 | bf16 | f32  (A/B/C same element type; L0C/W fp32)
  --layout-a/--layout-b: row | col         (C/W RowMajor)
  --m/--n/--k          : positive GEMM shape (MN/K tails via origin_shape)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack

import tail_splitk_mmad_kernels as _kernels

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

DEFAULT_M = 256
DEFAULT_N = 512
DEFAULT_K = 1024

m = DEFAULT_M
n = DEFAULT_N
k = DEFAULT_K

l1_tm = _kernels.l1_tm
l1_tn = _kernels.l1_tn
l1_tk = _kernels.l1_tk
tail_multi_core_splitk_mmad_kernel = _kernels.tail_multi_core_splitk_mmad_kernel

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


def compute_tail_scheduler(
    m_val: int, n_val: int, k_val: int, core_num: int
) -> dict[str, int]:
    """Compute tail split-K scheduling parameters.

    Split the M×N block grid into normal blocks (full-K MMAD, one tile per AIC)
    and tail blocks (split-K across AICs when mn_blocks is not divisible by
    core_num).

    Returns scheduling fields: grid_m, grid_n, k_tile_num, mn_blocks,
    tail_block_num, normal_block_num, splitk_factor, core_loops, aic_core_num.
    """
    if core_num <= 0:
        raise ValueError(
            f"core_num must be positive; got {core_num}"
        )
    grid_m = ceil_div(m_val, l1_tm)
    grid_n = ceil_div(n_val, l1_tn)
    k_tile_num = ceil_div(k_val, l1_tk)
    mn_blocks = grid_m * grid_n
    tail_block_num = mn_blocks % core_num
    normal_block_num = mn_blocks - tail_block_num
    splitk_factor = 1
    if tail_block_num > 0:
        splitk_factor = core_num // tail_block_num
    splitk_factor = min(splitk_factor, k_tile_num)
    core_loops = normal_block_num + tail_block_num * splitk_factor
    return {
        "grid_m": grid_m,
        "grid_n": grid_n,
        "k_tile_num": k_tile_num,
        "mn_blocks": mn_blocks,
        "tail_block_num": tail_block_num,
        "normal_block_num": normal_block_num,
        "splitk_factor": splitk_factor,
        "core_loops": core_loops,
        "aic_core_num": core_num,
    }


def workspace_shape(aic: int) -> tuple[int, int]:
    """Compute workspace shape as (rows, l1_tn) with a ≥10 MB floor.

    Each AIC gets an L1_M×L1_N slot in the workspace on contiguous rows.
    min_elems = 10 * 1024 * 1024 / sizeof(fp32) = ~2.5M elements.
    """
    min_elems = (10 * 1024 * 1024) // 4
    need = aic * l1_tm * l1_tn
    elems = max(min_elems, need)
    rows = max(aic * l1_tm, ceil_div(elems, l1_tn))
    return rows, l1_tn


def validate_shape(m_val: int, n_val: int, k_val: int) -> None:
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise SystemExit(f"m, n, k must be positive; got ({m_val},{n_val},{k_val})")


def validate_dtype_triple(
    dtype_a: ElemDType | str, dtype_b: ElemDType | str, dtype_c: ElemDType | str
) -> None:
    """Require A/B/C to use the same element type."""
    if dtype_a != dtype_b or dtype_a != dtype_c:
        raise SystemExit(
            "unsupported configuration:\n  - dtype-a, dtype-b, and dtype-c must match "
            f"(got {dtype_a}/{dtype_b}/{dtype_c}); allowed: f16 | bf16 | f32"
        )
    if dtype_a not in SUPPORTED_DTYPES:
        raise SystemExit(f"unsupported dtype {dtype_a!r}")


def _apply_kernel_dtypes(dtype: ElemDType) -> None:
    elem = _tla_elem_dtype(dtype)
    _kernels.DTYPE_A = elem
    _kernels.DTYPE_B = elem
    _kernels.DTYPE_C = tla.Float32  # L0C always fp32
    _kernels.DTYPE_W = tla.Float32
    _kernels.DTYPE_GM_C = elem


def compute_tail_reduce_tiling(
    factor: int,
    *,
    l1_m: int = l1_tm,
    l1_n: int = l1_tn,
    compute_length: int = 192 * 1024 // 4,
    ele_per_vector_block: int = 64,
    ele_align: int = 8,
) -> dict[str, int]:
    """Compute AIV ReduceAdd row-chunk tiling for tail blocks.

    Each AIV processes tile_per_core rows per iteration. ub_row_stride is
    padded to ele_align; reduce_vl_loops = chunk_elems // ele_per_vector_block.
    Chunk size is reduced until factor × chunk_elems fits in compute_length.
    """
    labor = factor * 2
    tile_len_align = ceil_div(l1_n, ele_align) * ele_align
    tile_per_core_max = (compute_length // labor) // tile_len_align
    if tile_per_core_max == 0:
        tile_per_core_max = 1
    tile_per_core = ceil_div(l1_m, labor)
    if tile_per_core > tile_per_core_max:
        tile_per_core = tile_per_core_max
    if tile_per_core > l1_m:
        tile_per_core = l1_m
    if tile_per_core == 0:
        tile_per_core = 1
    ub_stride = tile_len_align
    chunk_elems = tile_per_core * ub_stride
    while factor * chunk_elems > compute_length and tile_per_core > 1:
        tile_per_core -= 1
        chunk_elems = tile_per_core * ub_stride
    if factor * chunk_elems > compute_length:
        raise ValueError(
            f"tail reduce UB overflow: factor={factor} chunk={chunk_elems} "
            f"compute_length={compute_length}"
        )
    return {
        "tile_per_core": tile_per_core,
        "ub_row_stride": ub_stride,
        "reduce_vl_loops": ceil_div(chunk_elems, ele_per_vector_block),
        "chunk_elems": chunk_elems,
    }


def _inject_kernel_compile_params(
    sched: dict[str, int],
    reduce: dict[str, int],
) -> None:
    """Push host-computed scheduling into the kernel module before compile."""
    _kernels.aic_core_num = sched["aic_core_num"]
    _kernels.normal_block_num = sched["normal_block_num"]
    _kernels.tail_block_num = sched["tail_block_num"]
    _kernels.splitk_factor = sched["splitk_factor"]
    _kernels.core_loops = sched["core_loops"]
    _kernels.tile_per_core = reduce["tile_per_core"]
    _kernels.ub_row_stride = reduce["ub_row_stride"]
    _kernels.reduce_vl_loops = reduce["reduce_vl_loops"]
    _kernels.chunk_elems = reduce["chunk_elems"]


def _apply_problem(m_val: int, n_val: int, k_val: int, sched: dict[str, int]) -> None:
    """Update host problem size and inject kernel compile-time scheduling."""
    global m, n, k
    m, n, k = m_val, n_val, k_val

    reduce = compute_tail_reduce_tiling(sched["splitk_factor"])
    _inject_kernel_compile_params(sched, reduce)


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
    sched: dict[str, int],
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype: ElemDType,
) -> tuple[Any, ...]:
    from catlass import runtime as runtime_mod

    elem = _tla_elem_dtype(dtype)
    ws_rows, ws_cols = workspace_shape(sched["aic_core_num"])
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
    sched: dict[str, int],
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype: ElemDType,
) -> str:
    return tail_multi_core_splitk_mmad_kernel.dump_mlir(
        type_args=_compile_only_type_args(sched, layout_a, layout_b, dtype)
    )


def build_only(args: argparse.Namespace, sched: dict[str, int]) -> int:
    _apply_kernel_dtypes(args.dtype)
    tla.compile(
        tail_multi_core_splitk_mmad_kernel,
        *_compile_only_type_args(sched, args.layout_a, args.layout_b, args.dtype),
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
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
    dtype = dtype_a  # type: ignore[assignment]
    sched = compute_tail_scheduler(m_val, n_val, k_val, args.block)
    _apply_problem(m_val, n_val, k_val, sched)
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
    ws_rows, ws_cols = workspace_shape(sched["aic_core_num"])
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
        tail_multi_core_splitk_mmad_kernel,
        tla_a,
        tla_b,
        tla_c,
        tla_w,
        **kwargs,
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
            f"normal={sched['normal_block_num']} tail={sched['tail_block_num']} "
            f"factor={sched['splitk_factor']} "
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
        "sched": sched,
        "ok": ok,
        "mismatch_pct": cmp["mismatch_pct"],
        "budget_pct": cmp["budget_pct"],
        "normal_block_num": sched["normal_block_num"],
        "tail_block_num": sched["tail_block_num"],
        "splitk_factor": sched["splitk_factor"],
        "core_loops": sched["core_loops"],
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
                r = _run_one(
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
                if not r["ok"]:
                    failed += 1
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tail multi-core split-K matmul (example 69). "
            "dtype/layout/shape surface matches example 68 (unaligned MN/K ok). "
            "Normal MN blocks write C; tail MN blocks split-K + AIV ReduceAdd."
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
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="GEMM M (any positive).")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="GEMM N (any positive).")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="GEMM K (any positive).")
    parser.add_argument(
        "--dtype",
        type=_parse_elem_dtype,
        default="f32",
        help="A/B/C element type: f16 | bf16 | f32 (same for all three).",
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
        default=24,
        help="AIC core count (drives tail split-K factor).",
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

    validate_shape(args.m, args.n, args.k)
    validate_dtype_triple(args.dtype, args.dtype, args.dtype)
    sched = compute_tail_scheduler(args.m, args.n, args.k, args.block)
    _apply_problem(args.m, args.n, args.k, sched)
    _apply_kernel_dtypes(args.dtype)
    print(
        f"problem=({m},{n},{k}) "
        f"normal={sched['normal_block_num']} tail={sched['tail_block_num']} "
        f"factor={sched['splitk_factor']} dtype={args.dtype} "
        f"layout={args.layout_a}/{args.layout_b}"
    )
    if args.dump_tlair:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--dump-tlair requires a single dtype and layout pair.")
        print(dump_tlair(sched, args.layout_a, args.layout_b, args.dtype))
        return 0
    if args.build_only:
        if args.all_layouts or args.all_dtypes:
            raise SystemExit("--build-only requires a single dtype and layout pair.")
        return build_only(args, sched)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
