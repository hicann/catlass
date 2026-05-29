from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from typing import Any, Literal

import catlass as tla
from catlass import runtime as runtime_mod

import basic_mmad_kernels as _kernels

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

m = _kernels.m
n = _kernels.n
k = _kernels.k
basic_mmad_kernel = _kernels.basic_mmad_kernel


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


def _validate_mmad_dtype_triple(dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType) -> None:
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


def _apply_kernel_dtypes(dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType) -> None:
    _kernels.DTYPE_A = _tla_elem_dtype(dtype_a)
    _kernels.DTYPE_B = _tla_elem_dtype(dtype_b)
    _kernels.DTYPE_GM_C = _tla_elem_dtype(dtype_c)
    _kernels.DTYPE_C = tla.Float32


def _apply_problem_size(m_val: int, n_val: int, k_val: int) -> None:
    global m, n, k
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise ValueError(f"m, n, k must be positive; got m={m_val}, n={n_val}, k={k_val}")
    _kernels.m = m_val
    _kernels.n = n_val
    _kernels.k = k_val
    m, n, k = m_val, n_val, k_val


def _np_elem_dtype(token: ElemDType) -> Any:
    if token == "f16":
        return np.float16
    if token == "f32":
        return np.float32
    # NumPy often does not register ``bfloat16`` until ``ml_dtypes`` is imported.
    try:
        return np.dtype("bfloat16")
    except (TypeError, ValueError):
        try:
            import ml_dtypes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "bf16 host tensors need the ``bfloat16`` NumPy dtype. "
                "Install ``ml_dtypes`` (``pip install ml_dtypes``) or use a NumPy "
                "build that supports ``np.dtype('bfloat16')``."
            ) from exc
        try:
            return np.dtype("bfloat16")
        except (TypeError, ValueError):
            from ml_dtypes import bfloat16

            return np.dtype(bfloat16)


def _make_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any]:
    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(m, k),
                ta,
                origin_shape=tla.make_shape(m, k),
                layout_tag=_gm_layout_tag(layout_a),
            ),
            tla.Tensor(
                tla.make_shape(k, n),
                tb,
                origin_shape=tla.make_shape(k, n),
                layout_tag=_gm_layout_tag(layout_b),
            ),
            tla.Tensor(
                tla.make_shape(m, n),
                tc,
                origin_shape=tla.make_shape(m, n),
                layout_tag=tla.arch.RowMajor,
            ),
        )


def _runtime_tensors(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any]:
    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(m, k),
            ta,
            origin_shape=tla.make_shape(m, k),
            layout_tag=_gm_layout_tag(layout_a),
        )
        mem_b = tla.Tensor(
            tla.make_shape(k, n),
            tb,
            origin_shape=tla.make_shape(k, n),
            layout_tag=_gm_layout_tag(layout_b),
        )
        mem_c = tla.Tensor(
            tla.make_shape(m, n),
            tc,
            origin_shape=tla.make_shape(m, n),
            layout_tag=tla.arch.RowMajor,
        )
    return mem_a, mem_b, mem_c


def _fill_gm_from_dense(mem: Any, dense: Any, choice: LayoutChoice) -> None:
    if choice == "row":
        mem.data = dense
        return
    # Golden matrices stay row-major dense; GM ``column_major`` expects the device linear
    # order (Fortran order of the logical ``(rows, cols)`` matrix). Host ``Tensor`` uses
    # C-order ``tobytes`` traversal: column-major element order equals
    # ``transpose(dense)`` flattened in C order, then viewed as an ``(rows, cols)`` row-major
    # buffer (same as ``dense.flatten(order="F").reshape(rows, cols)``).
    rows, cols = int(dense.shape[0]), int(dense.shape[1])
    mem.data = np.transpose(dense).flatten().reshape(rows, cols)


def dump_tlair(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> str:
    return basic_mmad_kernel.dump_mlir(
        type_args=_make_type_args(layout_a, layout_b, dtype_a, dtype_b, dtype_c)
    )


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _first_mismatch(
    actual: Any, expected: Any, *, atol: float
) -> dict[str, Any] | None:
    import numpy as np

    indices = np.argwhere(~np.isclose(actual, expected, rtol=0.0, atol=atol))
    if indices.size == 0:
        return None
    row, col = (int(value) for value in indices[0])
    return {
        "index": [row, col],
        "actual": float(actual[row, col]),
        "expected": float(expected[row, col]),
    }


def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
    artifact = tla.compile(
        basic_mmad_kernel,
        *_make_type_args(
            args.layout_a, args.layout_b, args.dtype_a, args.dtype_b, args.dtype_c
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
    mem_a, mem_b, mem_c = _runtime_tensors(
        layout_a, layout_b, dtype_a, dtype_b, dtype_c
    )
    np_a = _np_elem_dtype(dtype_a)
    np_b = _np_elem_dtype(dtype_b)
    np_c = _np_elem_dtype(dtype_c)
    a_host = ((np.arange(m * k, dtype=np.float32).reshape(m, k) % 7.0) - 3.0).astype(
        np_a
    )
    b_host = ((np.arange(k * n, dtype=np.float32).reshape(k, n) % 5.0) - 2.0).astype(
        np_b
    )
    sentinel = np.full((m, n), args.sentinel, dtype=np_c)
    expected_f32 = a_host.astype(np.float32) @ b_host.astype(np.float32)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32.astype(np_c).astype(np.float32)
        atol = max(float(args.atol), 5e-3)
    else:
        expected = expected_f32
        atol = float(args.atol)

    _fill_gm_from_dense(mem_a, a_host, layout_a)
    _fill_gm_from_dense(mem_b, b_host, layout_b)
    mem_c.data = sentinel

    runtime_kwargs = _runtime_kwargs(args)
    artifact = tla.compile(basic_mmad_kernel, mem_a, mem_b, mem_c, **runtime_kwargs)
    print(
        "compile_ok=True "
        f"m={m} n={n} k={k} "
        f"layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    artifact(mem_a, mem_b, mem_c, block=args.block)
    print("launch_ok=True")

    mem_c.download_data()
    actual = np.asarray(mem_c.data, dtype=np.float32)
    unchanged = np.isclose(actual, sentinel.astype(np.float32), rtol=0.0, atol=atol)
    expected_match = np.isclose(actual, expected, rtol=0.0, atol=atol)
    first_mismatch = _first_mismatch(actual, expected, atol=atol)

    print(f"C unchanged? {bool(np.all(unchanged))}")
    print(f"C equals expected matmul? {bool(np.all(expected_match))}")
    print(f"C changed count={int(np.count_nonzero(~unchanged))}")
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
    if args.all_mmad_dtypes:
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
            "Compile, launch, and validate K-tile MMAD. GM layouts for A/B are selectable; "
            "A/B must match; allowed (dtype-a, dtype-b, dtype-c): "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32. "
            "dtype-c is GM C element type (fp32 or narrowed fp16/bf16); L0C stays fp32 and "
            "tla.copy lowers to copy_cc_to_gm_row_major_float | _half | _bf16. "
            "GM row_major→L1 zN and GM column_major→L1 nZ only. Output C is GM row_major."
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
        default=_kernels.m,
        help=f"GEMM M dimension (default: {_kernels.m}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_kernels.n,
        help=f"GEMM N dimension (default: {_kernels.n}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_kernels.k,
        help=f"GEMM K dimension (default: {_kernels.k}).",
    )
    parser.add_argument("--block", type=int, default=1, help="Launch block count.")
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
    _apply_problem_size(args.m, args.n, args.k)
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
