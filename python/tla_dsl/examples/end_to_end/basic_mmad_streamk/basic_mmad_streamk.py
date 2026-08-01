"""Host entry for the StreamK MMAD demo.

Builds and launches the single mixed kernel: the AIC cube runs the full-K normal
tiles and the StreamK partial sums, the AIV vector section reduces the workspace
into GM C. Supports ``--run`` / ``--build-only`` / ``--dump-tlair`` plus the
``--all-layouts`` / ``--all-mmad-dtypes`` sweeps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack

import basic_mmad_streamk_kernels as _kernels
import streamk_config as _cfg

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

# Relative tolerance for result compare: tighter when K is below this threshold.
_COMPARE_RTOL_K_THRESHOLD = 2048
_COMPARE_RTOL_NUMERATOR = 1.0
_COMPARE_RTOL_DENOM_SMALL_K = 256
_COMPARE_RTOL_DENOM_LARGE_K = 128
# Extra pass gate: allow a small fraction of elements outside atol/rtol.
_COMPARE_MISMATCH_RATIO_NARROW = 0.001  # f16 / bf16: <= 0.1%
_COMPARE_MISMATCH_RATIO_F32 = 0.0001  # f32: <= 0.01%

m = _cfg.m
n = _cfg.n
k = _cfg.k
streamk_mmad_kernel = _kernels.streamk_mmad_kernel


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
            f"unknown dtype {name!r}; expected f16, bf16, or f32 "
            "(aliases e.g. float16, fp16, half / bfloat16 / float32, fp32)"
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
        raise ValueError(
            "dtype-a and dtype-b must match (tla.mmad requires lhs and rhs element types equal)."
        )
    allowed = {
        ("f16", "f16", "f32"),
        ("f16", "f16", "f16"),
        ("bf16", "bf16", "f32"),
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
    }
    triple = (dtype_a, dtype_b, dtype_c)
    if triple not in allowed:
        raise ValueError(
            "unsupported (dtype-a, dtype-b, dtype-c); allowed: "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32 "
            "(L0C is fp32; dtype-c is GM C element type, including narrowed f16/bf16)."
        )


def _apply_kernel_dtypes(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    _cfg.DTYPE_A = _tla_elem_dtype(dtype_a)
    _cfg.DTYPE_B = _tla_elem_dtype(dtype_b)
    _cfg.DTYPE_GM_C = _tla_elem_dtype(dtype_c)
    _cfg.DTYPE_C = tla.Float32


def _apply_problem_size(m_val: int, n_val: int, k_val: int, block: int) -> None:
    global m, n, k
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise ValueError(f"m, n, k must be positive; got m={m_val}, n={n_val}, k={k_val}")
    if block <= 0:
        raise ValueError(f"block must be positive; got {block}")
    _cfg.m = m_val
    _cfg.n = n_val
    _cfg.k = k_val
    _cfg.BLOCK_DIM = block
    m, n, k = m_val, n_val, k_val


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Host-side tensors in this example require PyTorch. "
            "Install it with ``pip install torch``."
        ) from exc
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
        import torch_npu
    except ImportError as exc:
        raise SystemExit(
            "This example requires torch_npu for device DLPack bindings."
        ) from exc
    torch.npu.set_device(device_id)
    return torch


def _default_aic_block_dim(device_id: int) -> int:
    """Host launch block count matching runtime ``tla.arch.block_dim()``.

    Uses ``tla.get_aicore_num`` after ``tla.initialize``. Falls back to
    ``BLOCK_DIM`` only if the runtime query is unavailable.
    """
    try:
        return max(1, int(tla.get_aicore_num(device_id)))
    except Exception:
        return max(1, int(_cfg.BLOCK_DIM))


def _l1_padded_mn(m_val: int, n_val: int) -> tuple[int, int]:
    """Round MN up to L1 tile multiples so full-tile AIC GM access stays in bounds.

    The AIC addresses MN with SSA coords and always moves whole ``l1_tm x l1_tn``
    tiles, so a residual problem size needs zero-padded GM behind it.
    """
    pm = _kernels._ceil_div(m_val, _cfg.l1_tm) * _cfg.l1_tm
    pn = _kernels._ceil_div(n_val, _cfg.l1_tn) * _cfg.l1_tn
    return pm, pn


def _compile_only_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any, Any]:
    """Metadata-only tensors for mixed kernel ``dump_mlir`` / ``--build-only``."""
    from catlass import runtime as runtime_mod

    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    pm, pn = _l1_padded_mn(m, n)
    ws_rows = _kernels.workspace_rows()
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(pm, k),
                ta,
                origin_shape=tla.make_shape(pm, k),
                layout_tag=_gm_layout_tag(layout_a),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(k, pn),
                tb,
                origin_shape=tla.make_shape(k, pn),
                layout_tag=_gm_layout_tag(layout_b),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(pm, pn),
                tc,
                origin_shape=tla.make_shape(pm, pn),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(ws_rows, _cfg.l1_tn),
                tla.Float32,
                origin_shape=tla.make_shape(ws_rows, _cfg.l1_tn),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
        )


def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()


def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


def dump_tlair(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> str:
    return streamk_mmad_kernel.dump_mlir(
        type_args=_compile_only_type_args(
            layout_a, layout_b, dtype_a, dtype_b, dtype_c
        )
    )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _comparison_atol(dtype_c: ElemDType, args: argparse.Namespace) -> float:
    if dtype_c in ("f16", "bf16"):
        return max(float(args.atol), 5e-3)
    return float(args.atol)


def _comparison_rtol(k_val: int) -> float:
    """Pick relative tolerance from K: ``1/256`` if ``k < 2048``, else ``1/128``."""
    if k_val < _COMPARE_RTOL_K_THRESHOLD:
        return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_SMALL_K
    return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_LARGE_K


def _mismatch_ratio_budget(dtype_c: ElemDType) -> float:
    """Max fraction of out-of-tolerance elements still counted as pass."""
    if dtype_c in ("f16", "bf16"):
        return _COMPARE_MISMATCH_RATIO_NARROW
    return _COMPARE_MISMATCH_RATIO_F32


def _compare_expected_torch(
    actual: Any, expected: Any, *, rtol: float, atol: float, dtype_c: ElemDType
) -> dict[str, Any]:
    """Compare against golden with atol/rtol plus a small mismatch-ratio budget."""
    torch = _require_torch()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    total = int(actual.numel())
    mismatch_count = int((~close).sum().item())
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    budget = _mismatch_ratio_budget(dtype_c)
    all_close = mismatch_count == 0
    within_budget = mismatch_ratio <= budget
    ok = all_close or within_budget
    first_mismatch: dict[str, Any] | None = None
    if mismatch_count > 0:
        row, col = (
            int(value)
            for value in close.logical_not().nonzero(as_tuple=False)[0]
        )
        first_mismatch = {
            "index": [row, col],
            "actual": float(actual[row, col].item()),
            "expected": float(expected[row, col].item()),
        }
    return {
        "ok": ok,
        "all_close": all_close,
        "within_budget": within_budget,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_ratio,
        "mismatch_budget": budget,
        "total": total,
        "first_mismatch": first_mismatch,
    }


def _print_case_result(
    *,
    host: str,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    artifact: Any,
    unchanged_all: bool | None = None,
    expected_match_all: bool | None = None,
    changed_count: int | None = None,
    first_mismatch: dict[str, Any] | None = None,
    mismatch_count: int | None = None,
    mismatch_ratio: float | None = None,
    mismatch_budget: float | None = None,
    verify: bool = True,
) -> None:
    print(
        "compile_ok=True "
        f"host={host} layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    if verify:
        print(f"C unchanged? {unchanged_all}")
        print(f"C equals expected matmul? {expected_match_all}")
        print(f"C changed count={changed_count}")
        if mismatch_count is not None and mismatch_ratio is not None:
            budget_s = (
                f"{mismatch_budget:.6f}" if mismatch_budget is not None else "n/a"
            )
            print(
                f"mismatch count={mismatch_count} "
                f"ratio={mismatch_ratio:.8f} budget={budget_s}"
            )
        print(f"first mismatch={first_mismatch}")


def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
    artifact = tla.compile(
        streamk_mmad_kernel,
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
    torch = _require_torch_npu(args.device)
    device = "npu"
    print("m: ", m)
    print("n: ", n)
    print("k: ", k)
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)
    # Pad MN to L1 multiples so StreamK AIC full-tile GM DMA stays in-bounds.
    # Schedule / AIV still use the logical (m, n); only GM backing is padded.
    pad_m, pad_n = _l1_padded_mn(m, n)
    torch_tensor_a = torch.zeros((pad_m, k), dtype=torch_dtype_a, device=device)
    torch_tensor_b = torch.zeros((k, pad_n), dtype=torch_dtype_b, device=device)
    torch_tensor_a[:m, :] = (
        torch.empty((m, k), dtype=torch.float32, device=device).uniform_(-5.0, 5.0)
    ).to(torch_dtype_a)
    torch_tensor_b[:, :n] = (
        torch.empty((k, n), dtype=torch.float32, device=device).uniform_(-5.0, 5.0)
    ).to(torch_dtype_b)
    torch_tensor_c = torch.full(
        (pad_m, pad_n), args.sentinel, dtype=torch_dtype_c, device=device
    )
    ws_rows = _kernels.workspace_rows(args.block)
    torch_workspace = torch.zeros(
        (ws_rows, _cfg.l1_tn), dtype=torch.float32, device=device
    )
    verify = not args.no_verify
    expected = None
    atol = _comparison_atol(dtype_c, args)
    if verify:
        expected_f32 = torch_tensor_a[:m, :].to(torch.float32) @ torch_tensor_b[
            :, :n
        ].to(torch.float32)
        if dtype_c in ("f16", "bf16"):
            expected = expected_f32.to(torch_dtype_c).to(torch.float32)
        else:
            expected = expected_f32

    tla_tensor_a = _create_tla_tensor(torch_tensor_a, layout_a)
    tla_tensor_b = _create_tla_tensor(torch_tensor_b, layout_b)
    tla_tensor_c = _create_tla_tensor(torch_tensor_c, "row")
    tla_workspace = _create_tla_tensor(torch_workspace, "row")

    artifact = tla.compile(
        streamk_mmad_kernel,
        tla_tensor_a,
        tla_tensor_b,
        tla_tensor_c,
        tla_workspace,
        **_runtime_kwargs(args),
    )

    artifact(
        tla_tensor_a,
        tla_tensor_b,
        tla_tensor_c,
        tla_workspace,
        block=args.block,
    )
    torch.npu.synchronize()

    if verify:
        rtol = _comparison_rtol(k)
        print("rtol: ", rtol)
        actual = torch_tensor_c[:m, :n].to(torch.float32)
        sentinel_f32 = torch.full_like(actual, args.sentinel)
        unchanged = torch.isclose(actual, sentinel_f32, rtol=rtol, atol=atol)
        cmp = _compare_expected_torch(
            actual, expected, rtol=rtol, atol=atol, dtype_c=dtype_c
        )
        _print_case_result(
            host="torch_npu",
            layout_a=layout_a,
            layout_b=layout_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            artifact=artifact,
            unchanged_all=bool(unchanged.all()),
            expected_match_all=bool(cmp["ok"]),
            changed_count=int((~unchanged).sum().item()),
            first_mismatch=cmp["first_mismatch"],
            mismatch_count=int(cmp["mismatch_count"]),
            mismatch_ratio=float(cmp["mismatch_ratio"]),
            mismatch_budget=float(cmp["mismatch_budget"]),
            verify=True,
        )
        return 0 if cmp["ok"] else 1

    _print_case_result(
        host="torch_npu",
        layout_a=layout_a,
        layout_b=layout_b,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        artifact=artifact,
        verify=False,
    )
    return 0


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
    if args.all_mmad_dtypes:
        return list(MMAD_DTYPE_TRIPLES)
    return [(args.dtype_a, args.dtype_b, args.dtype_c)]


def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        # Resolve auto block after ACL init so schedule BLOCK_DIM matches launch.
        if getattr(args, "auto_block", False):
            args.block = _default_aic_block_dim(args.device)
            _apply_problem_size(args.m, args.n, args.k, args.block)
            print(
                f"auto_block=True block={args.block} "
                "(tla.get_aicore_num / tla.arch.block_dim)"
            )
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"dtype_a={dtype_a}",
                    f"dtype_b={dtype_b}",
                    f"dtype_c={dtype_c}",
                    f"layout_a={layout_a}",
                    f"layout_b={layout_b}",
                    "---",
                )
                failed += run_single_case(
                    args, layout_a, layout_b, dtype_a, dtype_b, dtype_c
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, launch, and validate StreamK MMAD (separate AIC cube + AIV vector). "
            "Tail-round StreamK on AIC + AIV workspace reduce. GM layouts for A/B are "
            "selectable; A/B must match; allowed (dtype-a, dtype-b, dtype-c): "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32. "
            "dtype-c is GM C element type; L0C stays fp32. Output C is GM row_major."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Compile the example and exit after generating kernel.o.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compile, launch, and compare the full output matrix. This is the default.",
    )
    parser.add_argument("--device", type=int, default=2, help="NPU device id.")
    parser.add_argument(
        "--m",
        type=int,
        default=_cfg.m,
        help=f"GEMM M dimension (default: {_cfg.m}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_cfg.n,
        help=f"GEMM N dimension (default: {_cfg.n}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_cfg.k,
        help=f"GEMM K dimension (default: {_cfg.k}).",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help=(
            "Launch block count (AIC cores); also used as StreamK BLOCK_DIM. "
            "Default: tla.get_aicore_num (matches kernel tla.arch.block_dim), else "
            f"{_cfg.BLOCK_DIM}."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip golden matmul and accuracy checks after launch "
            "(compile/launch only; useful when measuring kernel runtime)."
        ),
    )
    parser.add_argument("--sentinel", type=float, default=-7.0, help="Initial C value.")
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Comparison tolerance."
    )
    parser.add_argument(
        "--layout-a",
        type=_parse_layout_choice,
        default="row",
        help="GM layout for A (M×K): row or col.",
    )
    parser.add_argument(
        "--layout-b",
        type=_parse_layout_choice,
        default="row",
        help="GM layout for B (K×N): row or col.",
    )
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run all four (layout-a, layout-b) combinations sequentially.",
    )
    parser.add_argument(
        "--dtype-a",
        type=_parse_elem_dtype,
        default="f16",
        help="GM element type for A (M×K); must equal --dtype-b for tla.mmad.",
    )
    parser.add_argument(
        "--dtype-b",
        type=_parse_elem_dtype,
        default="f16",
        help="GM element type for B (K×N); must equal --dtype-a.",
    )
    parser.add_argument(
        "--dtype-c",
        type=_parse_elem_dtype,
        default="f32",
        help="GM element type for C (M×N): f32, or narrowed f16/bf16 with f16/f16 or bf16/bf16 inputs.",
    )
    parser.add_argument(
        "--all-mmad-dtypes",
        action="store_true",
        help=(
            "Run all supported (dtype-a, dtype-b, dtype-c) triples sequentially "
            "(with the chosen layout pair or all layout pairs when --all-layouts is set)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for compile cache and generated kernel.o files.",
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Ignore any existing compile cache entry.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable compile cache reuse.",
    )
    parser.add_argument(
        "--dump-tlair",
        action="store_true",
        help="Print TLA MLIR (tla dialect) and exit without compiling or launching.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    # None / -1 means take the block count from the device once ACL is up.
    args.auto_block = args.block is None or args.block < 0
    if args.auto_block:
        args.block = _cfg.BLOCK_DIM
    _apply_problem_size(args.m, args.n, args.k, args.block)
    if not args.all_mmad_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    if args.dump_tlair:
        if args.all_layouts or args.all_mmad_dtypes:
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
        if args.all_layouts or args.all_mmad_dtypes:
            raise SystemExit("--build-only requires a single layout and dtype triple.")
        return build_only(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
