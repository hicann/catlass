"""Host driver for batched matmul (Catlass example 01).

Semantics: for each batch ``b``, ``C[b] = A[b] @ B[b]`` with the same ``(m, n, k)``.
Storage matches C++ strides: A ``B*M*K``, B ``B*K*N``, C ``B*M*N``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack

import batched_matmul_kernels as _kernels

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

batch_count = _kernels.batch_count
m = _kernels.m
n = _kernels.n
k = _kernels.k
batched_matmul_kernel = _kernels.batched_matmul_kernel


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


def _apply_problem_size(
    batch_val: int, m_val: int, n_val: int, k_val: int
) -> None:
    global batch_count, m, n, k
    if min(batch_val, m_val, n_val, k_val) <= 0:
        raise ValueError(
            f"batch,m,n,k must be positive; got "
            f"batch={batch_val} m={m_val} n={n_val} k={k_val}"
        )
    _kernels.batch_count = batch_val
    _kernels.m = m_val
    _kernels.n = n_val
    _kernels.k = k_val
    # Match C++ ex01/ex67: Zn (0) when M>N, else Nz (1).
    _kernels.SWIZZLE_DIRECTION = 0 if m_val > n_val else 1
    batch_count, m, n, k = batch_val, m_val, n_val, k_val


def _apply_unit_flag_policy(layout_a: LayoutChoice, layout_b: LayoutChoice) -> None:
    """Prefer unit-flag for row/row (C++ path); soft-flag otherwise for stability."""
    _kernels.ENABLE_UNIT_FLAG = layout_a == "row" and layout_b == "row"


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
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


def _compile_only_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any]:
    from catlass import runtime as runtime_mod

    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(batch_count * m, k),
                ta,
                origin_shape=tla.make_shape(batch_count * m, k),
                layout_tag=_gm_layout_tag(layout_a),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(batch_count * k, n),
                tb,
                origin_shape=tla.make_shape(batch_count * k, n),
                layout_tag=_gm_layout_tag(layout_b),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(batch_count * m, n),
                tc,
                origin_shape=tla.make_shape(batch_count * m, n),
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
    idx = close.logical_not().nonzero(as_tuple=False)[0]
    coords = [int(v) for v in idx]
    gather = actual
    gather_e = expected
    for c in coords:
        gather = gather[c]
        gather_e = gather_e[c]
    av = float(gather.item())
    ev = float(gather_e.item())
    return {
        "index": coords,
        "actual": av,
        "expected": ev,
        "abs_err": abs(av - ev),
        "scaled_err": abs(av - ev) / max(1.0, abs(ev)),
    }


def _batched_golden(torch: Any, a: Any, b: Any) -> Any:
    """a,b: (B,M,K), (B,K,N) → (B,M,N) float32."""
    return torch.matmul(a.to(torch.float32), b.to(torch.float32))


def dump_tlair(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> str:
    return batched_matmul_kernel.dump_mlir(
        type_args=_compile_only_type_args(
            layout_a, layout_b, dtype_a, dtype_b, dtype_c
        )
    )


def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
    _apply_unit_flag_policy(args.layout_a, args.layout_b)
    artifact = tla.compile(
        batched_matmul_kernel,
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
) -> int:
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    _apply_unit_flag_policy(layout_a, layout_b)
    torch = _require_torch_npu(args.device)
    device = "npu"
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)

    # Inputs: randn then clamp to [-5, 5].
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.data_seed))
    a = (
        torch.randn(
            (batch_count, m, k), generator=gen, device=device, dtype=torch.float32
        )
        .clamp(-5.0, 5.0)
        .to(torch_dtype_a)
    )
    b = (
        torch.randn(
            (batch_count, k, n), generator=gen, device=device, dtype=torch.float32
        )
        .clamp(-5.0, 5.0)
        .to(torch_dtype_b)
    )
    c = torch.full(
        (batch_count, m, n), args.sentinel, dtype=torch_dtype_c, device=device
    )

    expected_f32 = _batched_golden(torch, a, b)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32.to(torch_dtype_c).to(torch.float32)
    else:
        expected = expected_f32
    rtol = _comparison_rtol(args)
    atol = _comparison_atol(dtype_c, args)

    # Flatten to 2D for TLA (stride = one batch matrix).
    a_flat = a.reshape(batch_count * m, k)
    b_flat = b.reshape(batch_count * k, n)
    c_flat = c.reshape(batch_count * m, n)

    tla_a = _create_tla_tensor(a_flat, layout_a)
    tla_b = _create_tla_tensor(b_flat, layout_b)
    tla_c = _create_tla_tensor(c_flat, "row")

    artifact = tla.compile(
        batched_matmul_kernel,
        tla_a,
        tla_b,
        tla_c,
        **_runtime_kwargs(args),
    )
    print(f"cache_key={artifact.cache_key}")
    artifact(tla_a, tla_b, tla_c, block=args.block)
    torch.npu.synchronize()

    actual = c.to(torch.float32)
    sentinel_f32 = torch.full_like(actual, args.sentinel)
    unchanged = torch.isclose(actual, sentinel_f32, rtol=0.0, atol=atol)
    expected_match = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    first_mismatch = _first_mismatch_torch(actual, expected, rtol=rtol, atol=atol)

    print(
        "compile_ok=True "
        f"host=torch_npu layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c} "
        f"batch={batch_count} m={m} n={n} k={k}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"data=randn.clamp([-5,5]) data_seed={args.data_seed}")
    print(f"isclose rtol={rtol} atol={atol}")
    print(f"C unchanged? {bool(unchanged.all())}")
    print(f"C equals batched golden? {bool(expected_match.all())}")
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


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"batch={batch_count}",
                    f"m={m}",
                    f"n={n}",
                    f"k={k}",
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
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batched matmul (example 01). Single cube launch over "
            "batch*MN tiles; A/B/C packed with per-batch strides m*k / k*n / m*n."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--batch", type=int, default=_kernels.batch_count)
    parser.add_argument("--m", type=int, default=_kernels.m)
    parser.add_argument("--n", type=int, default=_kernels.n)
    parser.add_argument("--k", type=int, default=_kernels.k)
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
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _apply_problem_size(args.batch, args.m, args.n, args.k)
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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
