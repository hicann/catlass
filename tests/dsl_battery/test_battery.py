"""The end-to-end DSL validation battery -- the RDV gate, as pytest cases.

Every case replays an example's own main() with a given argv, inside the one
process, so the device bring-up and the torch_npu import are paid once. The
examples are not modified in any way; a case is just an argv.

Adding coverage means adding to the tables below. Keep them as data: the point of
this file is that the whole gate is readable in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest

pytestmark = pytest.mark.npu

MMAD_SHAPES = (("333", "444", "555"), ("1", "2", "3"))
MMAD_LAYOUTS = (("row", "row"), ("row", "col"), ("col", "row"), ("col", "col"))
#: (dtype-a/b, dtype-c)
MMAD_TRIPLES = (
    ("f16", "f32"),
    ("f16", "f16"),
    ("bf16", "f32"),
    ("bf16", "bf16"),
    ("f32", "f32"),
)
MMAD_MNK = ("--m", "333", "--n", "444", "--k", "555")
MMAD_ROW_ROW = ("--layout-a", "row", "--layout-b", "row")

VADD_DTYPES = ("i8", "i16", "i32", "f16", "f32")
VADD_MODES = ("", "--use-mutex", "--use-mutex-with", "--use-atomic-add")

COMPARE_MASK_OPS = (
    "vector_vector_lt", "vector_vector_le", "vector_vector_gt", "vector_vector_ge",
    "vector_vector_eq", "vector_vector_ne", "vector_scalar_gt", "vector_scalar_ge",
    "masked_vector_vector_lt", "cmp_masked_fused", "static_dynamic_lt",
)

#: (block-num, calls, dtype-args)
PRINT_TENSOR_VARIANTS = (
    ("1", "1", ("--all-dtypes",)),
    ("2", "1", ("--dtype", "f32")),
    ("1", "2", ("--dtype", "f32")),
    ("2", "2", ("--all-dtypes",)),
)
#: (case, block-num, calls, dtype-args)
PRINT_TENSOR_UB_VARIANTS = (
    ("base", "1", "1", ("--all-dtypes",)),
    ("base", "2", "1", ("--dtype", "f32")),
    ("base", "1", "2", ("--dtype", "f32")),
    ("base", "2", "2", ("--all-dtypes",)),
    ("aligned-offset", "1", "1", ("--all-dtypes",)),
)


def _cases(device: int) -> Iterator[tuple[str, list[list[str]]]]:
    """Yield (test id, [argv, ...]) for every case in the battery.

    A case is usually one argv. It is a *sequence* only where a later invocation
    depends on an earlier one -- see the basic_mixed compile-then-cache-reuse
    case, which must stay a single test so the two halves can never be split up
    or reordered.
    """

    dev = ["--device", str(device)]

    # --- basic_mmad: the flag-sync layout x dtype matrix ---
    for m, n, k in MMAD_SHAPES:
        for la, lb in MMAD_LAYOUTS:
            for dab, dc in MMAD_TRIPLES:
                yield (
                    f"mmad-{m}x{n}x{k}-{la}{lb}-{dab}-{dc}",
                    [[
                        "basic_mmad/basic_matmul.py",
                        "--m", m, "--n", n, "--k", k,
                        "--layout-a", la, "--layout-b", lb,
                        "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                        *dev,
                    ]],
                )

    for script, label in (
        ("basic_matmul_mutex.py", "mutex"),
        ("basic_matmul_mutex_with.py", "mutex-with"),
        ("basic_matmul_auto_sync.py", "auto-sync"),
    ):
        yield (
            f"mmad-{label}",
            [[
                f"basic_mmad/{script}", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", "f16", "--dtype-b", "f16", "--dtype-c", "f32", *dev,
            ]],
        )

    for dtype in ("f16", "bf16", "f32"):
        yield (
            f"mmad-atomic-add-{dtype}",
            [[
                "basic_mmad/basic_matmul_atomic_add.py", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", dtype, "--dtype-b", dtype, "--dtype-c", "f32", *dev,
            ]],
        )

    yield ("mmad-ptr", [["basic_mmad/basic_mmad_ptr.py", *dev]])

    # --- basic_vadd: every dtype under each sync mode ---
    for mode in VADD_MODES:
        for dtype in VADD_DTYPES:
            label = mode.lstrip("-") or "flag-sync"
            yield (
                f"vadd-{label}-{dtype}",
                [[
                    "basic_vadd/basic_vadd.py", "--dtype", dtype, *dev,
                    *([mode] if mode else []),
                ]],
            )

    # --- basic_mixed ---
    # One test on purpose: the second invocation exercises cache reuse and is
    # only meaningful straight after the one that populates the cache.
    yield (
        "mixed-compile-then-cache-reuse",
        [
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
        ],
    )
    yield (
        "mixed-mutex",
        [["basic_mixed/basic_mixed_mutex.py", *dev, "--block-num", "1"]],
    )
    yield ("mixed-ub2l1", [["basic_mixed/basic_mixed_ub2l1.py", *dev]])
    yield ("mixed-store-zN", [["basic_mixed/basic_mixed_store_zN.py", *dev]])
    # m=64 is fractal-aligned (multiple of 16); m=50 exercises the zNUnAlign M
    # axis, where the dest leaf[0] is the runtime row count and the stride varies.
    for m in ("64", "50"):
        yield (
            f"mixed-store-zNUnAlign-m{m}",
            [["basic_mixed/basic_mixed_store_zNUnAlign.py", *dev, "--m", m]],
        )
    yield ("mixed-fixpipe-nz2dn", [["basic_mixed/basic_mixed_fixpipe_nz2dn.py", *dev]])

    # --- vector_ops: each of these sweeps or batches many kernels internally ---
    yield (
        "vector-masked-binary",
        [["vector_ops/masked_binary.py", "masked_binary", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-bitwise-ops",
        [["vector_ops/bitwise_ops.py", "bitwise_ops", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-binary-op",
        [[
            "vector_ops/binary_op.py", "--batch-run",
            "add", "sub", "mul", "div", "max", "min", "add_unalign", "add_brc_b32",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield ("vector-reduction-ops", [["vector_ops/reduction_ops.py", "--batch-run", *dev]])
    yield (
        "vector-load-store-scalar-after-reduction",
        [["vector_ops/load_and_store_scalar_after_reduction.py", *dev]],
    )
    yield (
        "vector-compare-mask",
        [[
            "vector_ops/compare_mask.py", "--batch-run", *COMPARE_MASK_OPS,
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-unary-ops",
        [[
            "vector_ops/unary_ops.py", "--batch-run",
            "exp", "log", "sqrt", "abs", "neg", "masked_unary", "masked_abs", "masked_neg",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-arange-op",
        [[
            "vector_ops/arange_op.py", "--batch-run", "increase", "decrease",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-interleave-op",
        [[
            "vector_ops/interleave_op.py", "--batch-run", "interleave", "deinterleave",
            "--shape", "512", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-load-dintlv-op",
        [["vector_ops/load_dintlv_op.py", "dintlv_b32", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-us-b8-op",
        [["vector_ops/load_us_b8_op.py", "us_b8", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-store-mask",
        [["vector_ops/load_store_mask.py", "load_store_mask", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-pack",
        [["vector_ops/store_pack.py", "store_pack", "--all-dtypes", *dev]],
    )
    yield (
        "vector-squeeze-op",
        [["vector_ops/squeeze_op.py", "squeeze", "--sweep", "--shapes", "64", *dev]],
    )
    yield (
        "vector-register-control-flow",
        [["vector_ops/register_control_flow.py", "register_carriers", *dev]],
    )

    # --- tensor_index ---
    yield (
        "tensor-index-scalar-control-flow",
        [["tensor_index/scalar_index_control_flow.py", *dev]],
    )
    yield (
        "tensor-index-scalar-kernel-arg",
        [["tensor_index/scalar_kernel_arg.py", *dev]],
    )

    # --- print_tensor, GM storage ---
    for blocks, calls, dtype_args in PRINT_TENSOR_VARIANTS:
        yield (
            f"print-tensor-gm-b{blocks}-c{calls}",
            [[
                "print_tensor/print_tensor.py", "--run", *dev,
                "--block-num", blocks, "--calls", calls, *dtype_args,
            ]],
        )

    # --- debug_print ---
    yield (
        "debug-print-matrix",
        [[
            "debug_print/debug_print.py", "--run", *dev,
            "--all-dtypes", "--block-num", "2", "--expect-count", "2",
        ]],
    )
    for value in ("-0.0", "nan", "inf", "-inf"):
        yield (
            f"debug-print-f16-{value}",
            [[
                "debug_print/debug_print.py", "--run", *dev,
                "--dtype", "f16", f"--value={value}",
            ]],
        )
    yield (
        "debug-print-expression",
        [["debug_print/debug_print.py", "--run", *dev, "--all-dtypes", "--expression"]],
    )
    for region in ("cube", "vector", "both"):
        yield (
            f"debug-print-mixed-{region}",
            [[
                "debug_print/debug_print_mixed.py", "--run", *dev,
                "--all-dtypes", "--print-region", region,
            ]],
        )
    yield (
        "debug-print-format",
        [["debug_print/debug_print_format.py", "--run", *dev, "--block-num", "2"]],
    )

    # --- scalar_arg_alignment ---
    yield (
        "scalar-arg-alignment",
        [["scalar_arg_alignment/scalar_arg_alignment.py", *dev]],
    )

    # --- print_tensor, UB storage ---
    for case_name, blocks, calls, dtype_args in PRINT_TENSOR_UB_VARIANTS:
        yield (
            f"print-tensor-ub-{case_name}-b{blocks}-c{calls}",
            [[
                "print_tensor/print_tensor.py", "--run", *dev, "--storage", "ub",
                "--case", case_name, "--block-num", blocks, "--calls", calls,
                *dtype_args,
            ]],
        )


def battery_cases(device: int = 0) -> list[tuple[str, list[list[str]]]]:
    """The whole battery as data. Used for the tests, and to audit coverage."""

    return list(_cases(device))


def _ids() -> list[str]:
    return [case_id for case_id, _ in battery_cases()]


@pytest.mark.parametrize("case_id", _ids())
def test_case(case_id: str, device: int) -> None:
    import example_runner

    argvs = dict(battery_cases(device))[case_id]
    for argv in argvs:
        script = example_runner.EXAMPLES_DIR / argv[0]
        assert script.is_file(), f"no such example: {argv[0]}"
        rc = example_runner.run_case([str(script), *argv[1:]])
        assert rc == 0, f"{argv[0]} {' '.join(argv[1:])} exited {rc}"
