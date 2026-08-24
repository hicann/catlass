from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import importlib.util
import math
import os
import re
import struct
import sys
import threading
import types

import pytest

tla = pytest.importorskip("catlass.tla", exc_type=ImportError)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
base_dsl_mod = pytest.importorskip("catlass.base_dsl", exc_type=ImportError)
compiler_bridge = pytest.importorskip("catlass.compiler_bridge", exc_type=ImportError)
ascend_runtime = pytest.importorskip(
    "catlass.base_dsl.runtime.ascend", exc_type=ImportError
)


def _load_debug_print_example(*, mixed: bool = False):
    fake_catlass = types.ModuleType("catlass")
    fake_catlass.kernel = lambda function: function
    if mixed:
        fake_catlass.Int32 = int
        fake_catlass.Float32 = float
    previous = sys.modules.get("catlass")
    sys.modules["catlass"] = fake_catlass
    example_dir = Path(__file__).parents[1] / "examples/end_to_end/debug_print"
    previous_debug_print = sys.modules.get("debug_print")
    try:
        if mixed:
            dependency_path = example_dir / "debug_print.py"
            dependency_spec = importlib.util.spec_from_file_location(
                "debug_print", dependency_path
            )
            assert dependency_spec and dependency_spec.loader
            dependency = importlib.util.module_from_spec(dependency_spec)
            sys.modules["debug_print"] = dependency
            dependency_spec.loader.exec_module(dependency)
        filename = "debug_print_mixed.py" if mixed else "debug_print.py"
        path = example_dir / filename
        spec = importlib.util.spec_from_file_location(
            f"{path.stem}_example", path
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if mixed:
            if previous_debug_print is None:
                del sys.modules["debug_print"]
            else:
                sys.modules["debug_print"] = previous_debug_print
        if previous is None:
            del sys.modules["catlass"]
        else:
            sys.modules["catlass"] = previous


def _load_print_tensor_example():
    fake_catlass = types.ModuleType("catlass")
    fake_catlass.kernel = lambda function: function
    fake_catlass.jit = lambda function: function
    fake_catlass.TlaExecutionError = execution.TlaExecutionError
    previous = sys.modules.get("catlass")
    sys.modules["catlass"] = fake_catlass
    try:
        path = (
            Path(__file__).parents[1]
            / "examples/end_to_end/print_tensor/print_tensor.py"
        )
        spec = importlib.util.spec_from_file_location("print_tensor_example", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules["catlass"]
        else:
            sys.modules["catlass"] = previous


def _load_basic_mixed_example():
    fake_catlass = types.ModuleType("catlass")
    fake_catlass.kernel = lambda function: function
    fake_catlass.TlaExecutionError = execution.TlaExecutionError
    previous = sys.modules.get("catlass")
    sys.modules["catlass"] = fake_catlass
    try:
        path = (
            Path(__file__).parents[1] / "examples/end_to_end/basic_mixed/basic_mixed.py"
        )
        spec = importlib.util.spec_from_file_location("basic_mixed_example", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules["catlass"]
        else:
            sys.modules["catlass"] = previous


def test_basic_mixed_output_accepts_unordered_distinct_subblocks() -> None:
    example = _load_basic_mixed_example()
    values = ", ".join(["3.0"] * 16)
    records = [
        "tla.print dtype=float32 position=UB subblock=1 "
        f"shape=[16,32] count=16 values=[{values}]",
        "tla.print dtype=float32 position=UB subblock=0 "
        f"shape=[16,32] count=16 values=[{values}]",
    ]

    assert example._verify_mixed_print_output("\n".join(records)) == records


def test_basic_mixed_output_rejects_duplicate_subblock() -> None:
    example = _load_basic_mixed_example()
    values = ", ".join(["3.0"] * 16)
    record = (
        "tla.print dtype=float32 position=UB subblock=0 "
        f"shape=[16,32] count=16 values=[{values}]"
    )

    with pytest.raises(execution.TlaExecutionError, match="duplicate subblock=0"):
        example._verify_mixed_print_output("\n".join((record, record)))


def test_print_tensor_output_verifies_and_formats_canonical_record() -> None:
    example = _load_print_tensor_example()
    stable = example._format_record(example.EXPECTED_VALUES)

    assert example._verify_public_output(stable) == (
        "tla.print dtype=float32 subblock=0 shape=[8,4] count=16 "
        "values=[0.0, -0.0, 1.0, -2.5, nan, inf, -inf, 3.25, "
        "0.0, -0.0, 1.0, -2.5, nan, inf, -inf, 3.25]"
    )


def test_print_tensor_output_formats_ub_physical_copy_shape() -> None:
    example = _load_print_tensor_example()
    values = [float(value) for value in range(16)]
    stable = example._format_record(values, shape=example.UB_SHAPE)

    assert example.UB_SHAPE == (4, 8)
    assert math.prod(example.UB_SHAPE) == 32
    assert example._verify_public_output(
        stable, values=values, shape=example.UB_SHAPE
    ) == (
        "tla.print dtype=float32 subblock=0 shape=[4,8] count=16 "
        "values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
        "9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]"
    )




@pytest.mark.parametrize(
    ("case", "kernel_name"),
    (
        ("base", "print_tensor_ub_base_kernel"),
        ("aligned-offset", "print_tensor_ub_aligned_offset_kernel"),
        ("dynamic", "print_tensor_ub_dynamic_kernel"),
    ),
)
def test_print_tensor_example_selects_ub_kernel(case, kernel_name) -> None:
    example = _load_print_tensor_example()
    args = types.SimpleNamespace(
        storage="ub", case=case, calls=1
    )

    assert example._kernel(args).fn.__name__ == kernel_name


def test_prepare_hivmc_input_selects_aic_print_tensor_helper(
    monkeypatch, tmp_path
) -> None:
    mlir_path = tmp_path / "lowered.mlir"
    mlir_path.write_text(
        "module { func.func @kernel(%workspace: i64 {tla.print_tensor.workspace}) }"
    )
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    helper_bc = tmp_path / "bc" / "Cube" / "print_tensor.aic.c310.bc"
    template_bc.write_bytes(b"bc")
    helper_bc.parent.mkdir(parents=True)
    helper_bc.write_bytes(b"bc__tla_print_tensor_abi")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    compiler_input, selected = execution._create_stamped_hivmc_input(
        mlir_path,
        execution.TlaRuntimeOptions(
            kernel_mode="aic", arch_scope="aic.c310"
        ),
    )

    assert compiler_input != mlir_path
    assert selected == f"{template_bc.resolve()},{helper_bc.resolve()}"
    assert "hivm.aic_bitcode" in compiler_input.read_text()


def test_prepare_hivmc_input_rejects_outdated_print_tensor_helper(
    monkeypatch, tmp_path
) -> None:
    mlir_path = tmp_path / "lowered.mlir"
    mlir_path.write_text(
        "module { func.func @kernel(%workspace: i64 {tla.print_tensor.workspace}) }"
    )
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    helper_bc = tmp_path / "bc" / "Vector" / "print_tensor.aiv.c310.bc"
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")
    helper_bc.parent.mkdir(parents=True)
    helper_bc.write_bytes(b"bc_tla_print_tensor_old_abi")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    with pytest.raises(execution.TlaRuntimeUnavailableError, match="ABI marker"):
        execution._create_stamped_hivmc_input(
            mlir_path,
            execution.TlaRuntimeOptions(
                kernel_mode="aiv", arch_scope="aiv.c310"
            ),
        )


@pytest.mark.parametrize(
    ("print_split", "helper_dir", "helper_name"),
    (
        ("aic", "Cube", "print_tensor.aic.c310.bc"),
        ("aiv", "Vector", "print_tensor.aiv.c310.bc"),
    ),
)
def test_prepare_hivmc_input_selects_mixed_split_print_tensor_helper(
    monkeypatch,
    tmp_path,
    print_split,
    helper_dir,
    helper_name,
) -> None:
    other_split = "aiv" if print_split == "aic" else "aic"
    mlir_path = tmp_path / "lowered.mlir"
    mlir_path.write_text(
        "module {\n"
        "func.func private @_mlir_ciface_tla_print_tensor_gm_f32()\n"
        f"func.func @kernel_mix_{other_split}(%workspace: i64 "
        "{tla.print_tensor.workspace}) { return }\n"
        f"func.func @kernel_mix_{print_split}(%workspace: i64 "
        "{tla.print_tensor.workspace}) {\n"
        "call @_mlir_ciface_tla_print_tensor_gm_f32() : () -> ()\n"
        "return\n"
        "}\n"
        "}\n"
    )
    aic_bc = tmp_path / "bc" / "meta_op.aic.c310.bc"
    aiv_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    helper_bc = tmp_path / "bc" / helper_dir / helper_name
    aic_bc.parent.mkdir(parents=True)
    aic_bc.write_bytes(b"bc")
    aiv_bc.write_bytes(b"bc")
    helper_bc.parent.mkdir(parents=True)
    helper_bc.write_bytes(b"bc__tla_print_tensor_abi")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    _, selected = execution._create_stamped_hivmc_input(
        mlir_path,
        execution.TlaRuntimeOptions(kernel_mode="mix"),
    )

    assert selected == (
        f"{aic_bc.resolve()},{aiv_bc.resolve()},{helper_bc.resolve()}"
    )


def test_debug_print_output_accepts_unordered_f32_records_from_distinct_blocks() -> (
    None
):
    example = _load_debug_print_example()

    example._verify_debug_output(
        "\n".join(
            (
                "TLA printf: core=0 block=3 v=1.250000",
                "TLA printf: core=1 block=1 v=1.250000",
                "TLA printf: core=0 block=0 v=1.250000",
                "TLA printf: core=1 block=2 v=1.250000",
            )
        ),
        dtype="f32",
        expected_value="1.250000",
        expect_count=4,
    )


def test_debug_print_output_ignores_cann_diagnostic_records() -> None:
    example = _load_debug_print_example()

    example._verify_debug_output(
        "\n".join(
            (
                "TLA printf: core=0 block=0 [WARNING]: CANN TimeStamp is invalid",
                "TLA printf: core=0 block=0 [AIV Block 0/1] ",
                "TLA printf: core=0 block=0 x=-128",
            )
        ),
        dtype="i8",
        expected_value="-128",
        expect_count=1,
    )


def test_debug_print_output_rejects_unexpected_scalar_records() -> None:
    example = _load_debug_print_example()

    with pytest.raises(RuntimeError):
        example._verify_debug_output(
            "\n".join(
                (
                    "TLA printf: core=0 block=0 x=-128",
                    "TLA printf: core=0 block=0 x=127",
                )
            ),
            dtype="i8",
            expected_value="-128",
            expect_count=1,
        )


def test_debug_print_output_rejects_duplicate_multiblock_records() -> None:
    example = _load_debug_print_example()

    with pytest.raises(RuntimeError, match="distinct blocks"):
        example._verify_debug_output(
            "\n".join("TLA printf: core=0 block=0 v=1.250000" for _ in range(4)),
            dtype="f32",
            expected_value="1.250000",
            expect_count=4,
        )


def test_debug_print_example_exposes_full_scalar_matrix() -> None:
    example = _load_debug_print_example()

    assert tuple(example.DTYPE_SPECS) == (
        "i8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f16",
        "f32",
    )
    assert example._parser().parse_args(["--all-dtypes"]).all_dtypes is True
    assert {
        token: (
            spec.parser,
            spec.value,
            spec.expected,
        )
        for token, spec in example.DTYPE_SPECS.items()
    } == {
        "i8": ("signed8", -128, "-128"),
        "i16": ("signed16", -32768, "-32768"),
        "i32": ("signed32", -37, "-37"),
        "u8": ("unsigned8", 255, "255"),
        "u16": ("unsigned16", 65535, "65535"),
        "u32": ("unsigned32", 4294967295, "4294967295"),
        "f16": ("float16", 1.25, "1.250000"),
        "f32": ("float32", 1.25, "1.250000"),
    }


def test_debug_print_expression_matrix_excludes_unsigned_arithmetic() -> None:
    example = _load_debug_print_example()
    args = example._parser().parse_args(["--all-dtypes", "--expression"])

    assert tuple(example.EXPRESSION_DTYPES) == ("i8", "i16", "i32", "f16", "f32")
    assert dict(example.EXPRESSION_SPECS) == {
        "i8": (-37, 5, "-32"),
        "i16": (-30000, 123, "-29877"),
        "i32": (-37, 5, "-32"),
        "f16": (1.25, 0.75, "2.000000"),
        "f32": (1.25, 0.75, "2.000000"),
    }
    assert [dtype for dtype, _, _ in example._selected_cases(args)] == list(
        example.EXPRESSION_DTYPES
    )


@pytest.mark.parametrize("dtype", ("u8", "u16", "u32"))
def test_debug_print_expression_rejects_unsigned_arithmetic(dtype: str) -> None:
    example = _load_debug_print_example()
    args = example._parser().parse_args(["--dtype", dtype, "--expression"])

    with pytest.raises(
        ValueError,
        match=rf"--expression does not support {dtype}; expected one of "
        r"i8, i16, i32, f16, f32",
    ):
        example._kernel(args)


@pytest.mark.parametrize(
    ("dtype", "expected_value", "line"),
    (
        ("i8", "-128", "TLA printf: core=0 block=0 x=-128"),
        ("i16", "-32768", "TLA printf: core=0 block=0 x=-32768"),
        ("i32", "-37", "TLA printf: core=0 block=0 x=-37"),
        ("u8", "255", "TLA printf: core=0 block=0 x=255"),
        ("u16", "65535", "TLA printf: core=0 block=0 x=65535"),
        ("u32", "4294967295", "TLA printf: core=0 block=0 x=4294967295"),
        ("f16", "1.250000", "TLA printf: core=0 block=0 v=1.250000"),
        ("f32", "1.250000", "TLA printf: core=0 block=0 v=1.250000"),
    ),
)
def test_debug_print_output_accepts_full_scalar_matrix(
    dtype: str, expected_value: str, line: str
) -> None:
    example = _load_debug_print_example()

    example._verify_debug_output(
        line, dtype=dtype, expected_value=expected_value, expect_count=1
    )


@pytest.mark.parametrize(
    ("print_region", "output"),
    (
        ("cube", "TLA printf: core=0 block=0 x=-37"),
        (
            "vector",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=2 block=0 v=1.250000",
                )
            ),
        ),
        (
            "both",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=2 block=0 v=1.250000",
                    "TLA printf: core=0 block=0 x=-37",
                )
            ),
        ),
    ),
)
def test_debug_print_mixed_output_accepts_requested_region(
    print_region: str, output: str
) -> None:
    example = _load_debug_print_example(mixed=True)

    example._verify_mixed_debug_output(output, print_region=print_region)


def test_debug_print_mixed_output_ignores_cann_diagnostic_records() -> None:
    example = _load_debug_print_example(mixed=True)

    example._verify_mixed_debug_output(
        "\n".join(
            (
                "TLA printf: core=0 block=0 [WARNING]: CANN TimeStamp is invalid",
                "TLA printf: core=0 block=0 [AIC Block 0/1] ",
                "TLA printf: core=0 block=0 x=-37",
            )
        ),
        print_region="cube",
    )


@pytest.mark.parametrize(
    ("print_region", "expected_kernel"),
    (
        ("cube", "debug_print_mixed_cube_kernel"),
        ("vector", "debug_print_mixed_vector_kernel"),
        ("both", "debug_print_mixed_both_kernel"),
    ),
)
def test_debug_print_mixed_selects_fixed_region_kernel(
    print_region: str, expected_kernel: str
) -> None:
    example = _load_debug_print_example(mixed=True)
    args = example._parser().parse_args(["--print-region", print_region])

    assert example._kernel(args).fn.__name__ == expected_kernel


def test_debug_print_mixed_defaults_to_both_regions() -> None:
    example = _load_debug_print_example(mixed=True)

    assert example._parser().parse_args([]).print_region == "both"


def test_debug_print_mixed_exposes_full_scalar_matrix() -> None:
    example = _load_debug_print_example(mixed=True)

    assert example._parser().parse_args(["--all-dtypes"]).all_dtypes is True
    assert tuple(example.DTYPE_SPECS) == (
        "i8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f16",
        "f32",
    )


@pytest.mark.parametrize(
    ("print_region", "dtype", "expected_value", "output"),
    (
        ("cube", "u32", "4294967295",
         "TLA printf: core=0 block=0 x=4294967295"),
        (
            "vector",
            "i8",
            "-128",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 x=-128",
                    "TLA printf: core=2 block=0 x=-128",
                )
            ),
        ),
        (
            "both",
            "f16",
            "1.250000",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=2 block=0 v=1.250000",
                    "TLA printf: core=0 block=0 v=1.250000",
                )
            ),
        ),
    ),
)
def test_debug_print_mixed_output_accepts_typed_matrix(
    print_region: str, dtype: str, expected_value: str, output: str
) -> None:
    example = _load_debug_print_example(mixed=True)

    example._verify_mixed_debug_output(
        output,
        print_region=print_region,
        dtype=dtype,
        expected_value=expected_value,
    )


@pytest.mark.parametrize(
    ("print_region", "output"),
    (
        (
            "vector",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=1 block=0 v=1.250000",
                )
            ),
        ),
        (
            "vector",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=2 block=1 v=1.250000",
                )
            ),
        ),
        ("cube", "\n".join(("TLA printf: core=0 block=0 x=-36",))),
        (
            "both",
            "\n".join(
                (
                    "TLA printf: core=1 block=0 v=1.250000",
                    "TLA printf: core=2 block=0 v=1.250000",
                )
            ),
        ),
    ),
    ids=[
        "duplicate-vector-core",
        "wrong-vector-block",
        "wrong-cube-value",
        "missing-both-cube",
    ],
)
def test_debug_print_mixed_output_rejects_invalid_native_frames(
    print_region: str, output: str
) -> None:
    example = _load_debug_print_example(mixed=True)

    with pytest.raises(RuntimeError, match="expected"):
        example._verify_mixed_debug_output(output, print_region=print_region)




@pytest.mark.parametrize(
    ("dtype", "expected_value", "line"),
    (
        ("i32", "-37", "TLA printf: core=0 block=0 x=-37"),
        ("f32", "1.250000", "TLA printf: core=0 block=0 v=1.250000"),
    ),
)
def test_debug_print_aic_output_uses_scalar_frame(
    dtype: str, expected_value: str, line: str
) -> None:
    example = _load_debug_print_example()

    example._verify_debug_output(
        line, dtype=dtype, expected_value=expected_value, expect_count=1
    )


class _FakeLowered:
    def __init__(self, text: str, module: object | None = None) -> None:
        self.module = module
        self._text = text

    def asm(self, *, generic: bool = False) -> str:
        del generic
        return self._text


def _zero_arg_kernel() -> None:
    pass


def _zero_arg_kernel_abi() -> compiler_bridge.KernelAbiLayout:
    return compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="zero_arg_kernel",
        total_size=0,
        arguments=(),
    )


@tla.kernel
def _zero_arg_tla_kernel() -> None:
    pass


def test_public_compile_dry_run_invokes_typed_bridge_and_hivmc_a5(
    monkeypatch, tmp_path
) -> None:
    tlair_mlir = "module {\n  tla.func @zero_arg_kernel() { tla.return }\n}"
    lowered_module = object()
    bridge_path = tmp_path / "_tla_type_bridge_native.so"
    hivm_compile = tmp_path / "hivmc-a5"
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    bridge_path.write_text("")
    hivm_compile.write_text("")
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")

    def fake_lower(
        self, fn, *, kind, options, generic=False, type_args=None, location=None
    ):
        del self, fn, options, type_args, location
        assert kind == "kernel"
        assert generic is False
        return _FakeLowered(tlair_mlir, module=lowered_module)

    monkeypatch.setattr(base_dsl_mod.BaseDSL, "_lower", fake_lower)
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: bridge_path)
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda: hivm_compile)
    monkeypatch.setattr(execution, "_tool_version", lambda _x: "test-version")
    monkeypatch.setattr(
        execution,
        "lower_tlair_module_to_mlir",
        lambda module, **_kwargs: compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n",
            kernel_abi=_zero_arg_kernel_abi(),
        ),
    )
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    recorded: list[tuple[str, list[str]]] = []

    def fake_run_checked(cmd, *, label, cwd, stdin_text=None):
        assert stdin_text is None
        recorded.append((label, list(cmd)))
        if label == "hivmc-a5":
            assert "hivm.aic_bitcode" not in Path(cmd[1]).read_text()
            Path(cwd, "kernel.o").write_bytes(b"obj")

    monkeypatch.setattr(execution, "_run_checked", fake_run_checked)
    monkeypatch.setenv("CATLASS_DSL_CACHE", "0")
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(tmp_path / "cache"))

    artifact = tla.compile(
        _zero_arg_tla_kernel,
        options="--npu-arch 3510",
    )

    assert artifact.compiler_bridge_path == bridge_path
    assert artifact.lowered_llvm == "module { func.func @zero_arg_kernel() }\n"
    assert artifact.expects_print_tensor is False
    assert artifact.expects_debug_fifo is False
    assert artifact.logical_mixed_handoff is None
    assert artifact._abi_packer is not None
    assert not (artifact.cache_dir / "lowered.hivmc-input.mlir").exists()
    assert recorded == [
        (
            "hivmc-a5",
            [
                str(hivm_compile),
                str(artifact.cache_dir / "lowered.mlir"),
                "--target=Ascend950PR_9589",
                "--disable-ffts",
                "--enable-hivm-compile=False",
                f"--link-aicore-bitcode={template_bc.resolve()}",
                "-o",
                str(artifact.kernel_binary_path),
            ],
        )
    ]


def test_prepare_hivmc_input_stamps_only_debug_print_mlir(
    monkeypatch, tmp_path
) -> None:
    mlir_path = tmp_path / "lowered.mlir"
    mlir_path.write_text(
        "module { func.func @kernel(%workspace: memref<?xi8> "
        "{tla.debug_print.workspace}) }"
    )
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    template_bc.write_bytes(b"bc")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    compiler_input, template_bitcode = execution._create_stamped_hivmc_input(
        mlir_path,
        execution.TlaRuntimeOptions(kernel_mode="aic", arch_scope="aic.c310"),
    )

    assert compiler_input != mlir_path
    assert template_bitcode == str(template_bc.resolve())
    assert "hivm.aic_bitcode" in compiler_input.read_text()
    assert "hivm.aic_bitcode" not in mlir_path.read_text()


def test_generated_kernel_bridge_lowers_live_module(monkeypatch, tmp_path) -> None:
    tlair_mlir = "module {\n  tla.func @zero_arg_kernel() { tla.return }\n}"
    lowered_module = object()
    hivm_compile = tmp_path / "hivmc-a5"
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    hivm_compile.write_text("")
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")

    monkeypatch.setattr(
        base_dsl_mod.BaseDSL,
        "_lower",
        lambda *_a, **_k: _FakeLowered(tlair_mlir, module=lowered_module),
    )
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: None)
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda: hivm_compile)
    monkeypatch.setattr(execution, "_tool_version", lambda _x: "test-version")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    bridge_calls: list[tuple[object, dict[str, object]]] = []

    def fake_lower_tlair_module_to_mlir(module, **kwargs):
        bridge_calls.append((module, kwargs))
        return compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n",
            kernel_abi=_zero_arg_kernel_abi(),
        )

    monkeypatch.setattr(
        execution, "lower_tlair_module_to_mlir", fake_lower_tlair_module_to_mlir
    )

    def fake_run_checked(cmd, *, label, cwd, stdin_text=None):
        del cmd, stdin_text
        assert label == "hivmc-a5"
        Path(cwd, "kernel.o").write_bytes(b"obj")

    monkeypatch.setattr(execution, "_run_checked", fake_run_checked)

    execution.compile_kernel(
        _zero_arg_kernel,
        kind="kernel",
        options={},
        runtime=execution.TlaRuntimeOptions(
            cache_enabled=False, cache_dir=tmp_path / "cache"
        ),
        type_args=None,
        decorator_location=None,
    )

    assert bridge_calls == [
        (
            lowered_module,
            {
                "mlir_print_ir_before": (),
                "mlir_print_ir_after": (),
                "mlir_print_ir_before_all": False,
                "mlir_print_ir_after_all": False,
            },
        )
    ]


def test_force_recompile_refreshes_launched_artifact_in_memory_cache(
    monkeypatch, tmp_path
) -> None:
    tlair_mlir = "module { tla.func @zero_arg_kernel() { tla.return } }"
    lowered_module = object()
    hivm_compile = tmp_path / "hivmc-a5"
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    hivm_compile.write_text("")
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")

    monkeypatch.setattr(execution, "_MEMORY_COMPILE_CACHE", {})
    monkeypatch.setattr(
        base_dsl_mod.BaseDSL,
        "_lower",
        lambda *_args, **_kwargs: _FakeLowered(
            tlair_mlir, module=lowered_module
        ),
    )
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: None)
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda: hivm_compile)
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "test-version")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])
    monkeypatch.setattr(
        execution,
        "lower_tlair_module_to_mlir",
        lambda *_args, **_kwargs: compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n",
            kernel_abi=_zero_arg_kernel_abi(),
        ),
    )

    compiled_binaries: list[bytes] = []

    def fake_run_checked(cmd, *, label, cwd, stdin_text=None):
        del cmd, stdin_text
        assert label == "hivmc-a5"
        binary = f"obj-{len(compiled_binaries) + 1}".encode()
        compiled_binaries.append(binary)
        Path(cwd, "kernel.o").write_bytes(binary)

    monkeypatch.setattr(execution, "_run_checked", fake_run_checked)

    loaded_binaries: list[bytes] = []
    launched_functions: list[int] = []
    _install_fake_launch_context(monkeypatch, device=0, stream=90)

    def fake_load_binary(**kwargs):
        loaded_binaries.append(kwargs["kernel_path"].read_bytes())
        sequence = len(loaded_binaries)
        return 100 + sequence, 200 + sequence

    monkeypatch.setattr(execution, "load_binary", fake_load_binary)
    monkeypatch.setattr(
        execution,
        "launch_kernel",
        lambda **kwargs: launched_functions.append(kwargs["function"]),
    )

    runtime = execution.TlaRuntimeOptions(
        cache_enabled=True, cache_dir=tmp_path / "cache"
    )
    first = execution.compile_kernel(
        _zero_arg_kernel,
        kind="kernel",
        options={},
        runtime=runtime,
    )
    execution.execute_kernel(
        first,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )

    recompiled = execution.compile_kernel(
        _zero_arg_kernel,
        kind="kernel",
        options={},
        runtime=replace(runtime, force_recompile=True),
    )
    cached = execution.compile_kernel(
        _zero_arg_kernel,
        kind="kernel",
        options={},
        runtime=runtime,
    )
    assert cached is recompiled
    assert cached is not first

    execution.execute_kernel(
        cached,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )

    assert compiled_binaries == [b"obj-1", b"obj-2"]
    assert loaded_binaries == [b"obj-1", b"obj-2"]
    assert launched_functions == [201, 202]


def test_compile_rejects_invalid_kernel_abi_before_hivmc(
    monkeypatch, tmp_path
) -> None:
    tlair_mlir = "module { tla.func @zero_arg_kernel() { tla.return } }"
    hivmc = tmp_path / "hivmc-a5"
    hivmc.write_text("")
    invalid_abi = replace(_zero_arg_kernel_abi(), entrypoint="wrong_kernel")

    monkeypatch.setattr(
        base_dsl_mod.BaseDSL,
        "_lower",
        lambda *_args, **_kwargs: _FakeLowered(tlair_mlir, module=object()),
    )
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: None)
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda: hivmc)
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "test-version")
    monkeypatch.setattr(
        execution,
        "lower_tlair_module_to_mlir",
        lambda *_args, **_kwargs: compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n",
            kernel_abi=invalid_abi,
        ),
    )
    monkeypatch.setattr(
        execution,
        "_run_checked",
        lambda *_args, **_kwargs: pytest.fail("hivmc must not run"),
    )

    with pytest.raises(execution.TlaKernelCompileError, match="does not match"):
        execution.compile_kernel(
            _zero_arg_kernel,
            kind="kernel",
            options={},
            runtime=execution.TlaRuntimeOptions(
                cache_enabled=False, cache_dir=tmp_path / "cache"
            ),
        )


def test_runtime_options_npu_arch_defaults_core_until_mlir() -> None:
    options = execution.runtime_options_from_kwargs({"options": "--npu-arch 3510"})
    # AIC/AIV is inferred later from lowered MLIR; Host only selects chip arch.
    assert options.arch_scope == "aiv.c310"
    assert options.kernel_mode == "aiv"


def test_runtime_options_reject_unknown_npu_arch() -> None:
    with pytest.raises(ValueError, match="Unsupported --npu-arch"):
        execution.runtime_options_from_kwargs({"options": "--npu-arch sm_100"})


def test_typed_bridge_raises_without_live_module(tmp_path) -> None:
    with pytest.raises(
        execution.TlaCompilerBridgeUnavailableError, match="live MLIR module"
    ):
        execution._run_typed_bridge_to_mlir(
            lowered_module=None, mlir_path=tmp_path / "lowered.mlir"
        )


def test_lower_tlair_module_to_mlir_uses_typed_extension(monkeypatch) -> None:
    module = object()
    calls: list[tuple[object, list[str], list[str], bool, bool]] = []

    class _FakeExtension:
        def lower_to_mlir(
            self,
            module_arg: object,
            before: list[str],
            after: list[str],
            before_all: bool,
            after_all: bool,
        ) -> dict[str, str]:
            calls.append((module_arg, before, after, before_all, after_all))
            return {
                "lowered_mlir": "module { func.func @zero_arg_kernel() }\n",
                "pass_ir_dump": "after-pass-dump",
            }

    monkeypatch.setattr(
        compiler_bridge, "_load_bridge_extension", lambda: _FakeExtension()
    )

    lowered = compiler_bridge.lower_tlair_module_to_mlir(
        module,
        mlir_print_ir_before=["tla-lower-func"],
        mlir_print_ir_after=["tla-finalize-memref"],
        mlir_print_ir_before_all=True,
    )

    assert lowered.lowered_mlir == "module { func.func @zero_arg_kernel() }\n"
    assert lowered.pass_ir_dump == "after-pass-dump"
    assert calls == [
        (
            module,
            ["tla-lower-func"],
            ["tla-finalize-memref"],
            True,
            False,
        )
    ]


def test_lower_tlair_module_to_mlir_preserves_pass_dump_on_failure(
    monkeypatch,
) -> None:
    module = object()

    class _FakeExtension:
        def lower_to_mlir(self, *_args) -> dict[str, object]:
            return {
                "success": False,
                "error": "pipeline failed",
                "lowered_mlir": "",
                "pass_ir_dump": "// ----- IR Dump After failing-pass -----\nmodule {}\n",
            }

    monkeypatch.setattr(
        compiler_bridge, "_load_bridge_extension", lambda: _FakeExtension()
    )

    with pytest.raises(
        compiler_bridge.BridgeLoweringError, match="pipeline failed"
    ) as exc_info:
        compiler_bridge.lower_tlair_module_to_mlir(module, mlir_print_ir_after_all=True)

    assert "IR Dump After failing-pass" in exc_info.value.pass_ir_dump


def test_lower_tlair_module_to_mlir_requires_typed_extension(monkeypatch) -> None:
    module = object()

    monkeypatch.setattr(
        compiler_bridge,
        "_load_bridge_extension",
        lambda: (_ for _ in ()).throw(
            compiler_bridge.BridgeUnavailableError("missing typed bridge")
        ),
    )

    with pytest.raises(
        compiler_bridge.BridgeUnavailableError, match="missing typed bridge"
    ):
        compiler_bridge.lower_tlair_module_to_mlir(module)


def test_run_tla_lowering_to_mlir_falls_back_to_tla_compile(
    monkeypatch, tmp_path
) -> None:
    lowered_path = tmp_path / "lowered.mlir"
    tla_compile = tmp_path / "TlaCompile"
    tla_compile.write_text("")

    monkeypatch.setattr(
        execution,
        "_run_typed_bridge_to_mlir",
        lambda **_kwargs: (_ for _ in ()).throw(
            execution.TlaKernelCompileError("typed bridge failed")
        ),
    )
    monkeypatch.setattr(execution, "_resolve_tla_compile", lambda: tla_compile)

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        lowered_path.write_text("module { func.func @fallback() }\n")
        return None

    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result = execution._run_tla_lowering_to_mlir(
        lowered_module=object(),
        tlair_mlir="module { tla.func @k() { tla.return } }\n",
        mlir_path=lowered_path,
        runtime=execution.TlaRuntimeOptions(),
    )

    assert result.lowered_mlir == "module { func.func @fallback() }\n"
    assert result.pass_ir_dump == ""
    assert calls == [
        (
            [
                str(tla_compile),
                str(tmp_path / "lowered.tlair.mlir"),
                "-o",
                str(lowered_path),
            ],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "env": execution._tla_compile_env(),
            },
        )
    ]


def test_tla_compile_cli_preserves_ir_dump_on_failure(monkeypatch, tmp_path) -> None:
    lowered_path = tmp_path / "lowered.mlir"
    tla_compile = tmp_path / "TlaCompile"
    tla_compile.write_text("")
    pass_ir_dump = "// ----- IR Dump After failing-pass -----\nmodule {}\n"

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        raise execution.subprocess.CalledProcessError(
            1,
            cmd,
            output="cli stdout",
            stderr=pass_ir_dump,
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    runtime = execution.TlaRuntimeOptions(print_ir=True)

    with pytest.raises(execution.TlaKernelCompileError) as exc_info:
        execution._run_tla_compile_cli_to_mlir(
            tla_compile=tla_compile,
            tlair_mlir="module { tla.func @k() { tla.return } }\n",
            mlir_path=lowered_path,
            runtime=runtime,
        )

    assert exc_info.value.pass_ir_dump == pass_ir_dump
    assert "<captured in pass IR dump>" in str(exc_info.value)
    assert calls[0][0][-2:] == [
        "--mlir-print-ir-before-all",
        "--mlir-print-ir-after-all",
    ]


def test_run_tla_lowering_to_mlir_raises_when_no_fallback_exists(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        execution,
        "_run_typed_bridge_to_mlir",
        lambda **_kwargs: (_ for _ in ()).throw(
            execution.TlaKernelCompileError("typed bridge failed")
        ),
    )
    monkeypatch.setattr(execution, "_resolve_tla_compile", lambda: None)

    with pytest.raises(execution.TlaKernelCompileError, match="typed bridge failed"):
        execution._run_tla_lowering_to_mlir(
            lowered_module=object(),
            tlair_mlir="module {}\n",
            mlir_path=tmp_path / "lowered.mlir",
            runtime=execution.TlaRuntimeOptions(),
        )


def test_runtime_options_from_lowered_mlir_updates_kernel_mode() -> None:
    runtime = execution.TlaRuntimeOptions()

    updated = execution._runtime_options_from_lowered_mlir(
        runtime,
        "module { func.func @kernel() { vector.transfer_read %arg0[%c0], %cst : memref<1xf32>, vector<1xf32> } }",
        has_logical_mixed_handoff=False,
    )

    assert updated.kernel_mode == "aiv"
    assert updated.arch_scope == "aiv.c310"


def test_build_hivmc_a5_command_links_template_bitcode_for_aic(
    monkeypatch, tmp_path
) -> None:
    compiler = tmp_path / "hivmc-a5"
    mlir_path = tmp_path / "kernel.mlir"
    kernel_path = tmp_path / "kernel.o"
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    template_bc.write_bytes(b"bc")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    command = execution._build_hivmc_a5_command(
        compiler=compiler,
        mlir_path=mlir_path,
        kernel_path=kernel_path,
        runtime=execution.TlaRuntimeOptions(
            kernel_mode="aic", arch_scope="aic.c310"
        ),
    )

    assert command == [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--disable-ffts",
        "--enable-hivm-compile=False",
        f"--link-aicore-bitcode={template_bc.resolve()}",
        "-o",
        str(kernel_path),
    ]


def test_build_hivmc_a5_command_links_template_bitcode_for_aiv(
    monkeypatch, tmp_path
) -> None:
    compiler = tmp_path / "hivmc-a5"
    mlir_path = tmp_path / "kernel.mlir"
    kernel_path = tmp_path / "kernel.o"
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    command = execution._build_hivmc_a5_command(
        compiler=compiler,
        mlir_path=mlir_path,
        kernel_path=kernel_path,
        runtime=execution.TlaRuntimeOptions(kernel_mode="aiv", arch_scope="aiv.c310"),
    )

    assert command == [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--disable-ffts",
        "--enable-hivm-compile=False",
        f"--link-aicore-bitcode={template_bc.resolve()}",
        "-o",
        str(kernel_path),
    ]


class _TypedPointer:
    dtype = "f32"

    def __init__(self, pointer: int) -> None:
        self._pointer = pointer

    def __c_pointers__(self) -> list[int]:
        return [self._pointer]


def _debug_kernel_abi(launch_args, *, entrypoint: str):
    arguments = []
    offset = 0
    for arg in launch_args:
        offset = (offset + 3) & ~3
        if isinstance(arg, execution.Numeric):
            abi_type = str(type(arg).dtype).lower()
            storage_size = (
                8 if abi_type == "index" else max(1, int(type(arg).width) // 8)
            )
            arguments.append(("scalar", abi_type, abi_type, offset, storage_size, 4))
        else:
            storage_size = 8
            arguments.append(
                ("pointer", "pointer", "!llvm.ptr", offset, storage_size, 4)
            )
        offset += storage_size
    return _kernel_abi(
        *arguments,
        total_size=(offset + 7) & ~7,
        entrypoint=entrypoint,
    )


def _debug_print_artifact(
    tmp_path, *, entrypoint: str = "debug", launch_args=(tla.Int32(7),)
):
    return execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=(
            f"module {{ func.func @{entrypoint}(%arg0: i32, "
            "%arg1: i64 {tla.debug_print.workspace}) }"
        ),
        entrypoint=entrypoint,
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_debug_kernel_abi(launch_args, entrypoint=entrypoint),
    )


def _print_tensor_artifact(
    tmp_path,
    *,
    entrypoint: str = "dump",
    shape: tuple[int, ...] = (4, 4),
    storage: str = "gm",
    static_length: int | None = None,
    mixed: bool = False,
):
    rendered_shape = ", ".join(str(extent) for extent in shape)
    length_attribute = (
        f"length = {static_length} : i64, " if static_length is not None else ""
    )
    return execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir=(
            'module { "tla.func"() ({ "tla.print_tensor"(%value, %length) '
            f"<{{{length_attribute}shape = array<i64: {rendered_shape}>}}> : "
            f"(!tla.tensor<!tla.ptr<f32, {storage}, 4>>, i64) -> () "
            '"tla.return"() : () -> () '
            "}) {function_type = () -> (), "
            f'sym_name = "{entrypoint}"}} : () -> () }}'
        ),
        lowered_llvm=(
            (
                "module {\n"
                "func.func private @_mlir_ciface_tla_print_tensor_ub_f32()\n"
                f"func.func @{entrypoint}_mix_aic(%value: i64, "
                "%workspace: i64 {tla.print_tensor.workspace}) "
                "{ return }\n"
                f"func.func @{entrypoint}_mix_aiv(%value: i64, "
                "%workspace: i64 {tla.print_tensor.workspace}) {\n"
                "call @_mlir_ciface_tla_print_tensor_ub_f32() : () -> ()\n"
                "return\n"
                "}\n"
                "}"
            )
            if mixed
            else (
                f"module {{ func.func @{entrypoint}(%workspace: i64 "
                "{tla.print_tensor.workspace}, %value: i64) }"
            )
        ),
        entrypoint=entrypoint,
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(
            ("pointer", "pointer", "!llvm.ptr", 0, 8, 4),
            total_size=8,
            entrypoint=entrypoint,
        ),
    )


def _two_print_tensor_artifact(tmp_path, *, storage: str = "gm"):
    artifact = _print_tensor_artifact(tmp_path, shape=(2, 2), storage=storage)
    first = (
        '%length0 = "arith.constant"() <{value = 4 : i64}> : () -> i64 '
        '"tla.print_tensor"(%value, %length0) '
        "<{shape = array<i64: 2, 2>}> : "
        f"(!tla.tensor<!tla.ptr<f32, {storage}, 4>>, i64) -> ()"
    )
    second = (
        '%length1 = "arith.constant"() <{value = 2 : i64}> : () -> i64 '
        '"tla.print_tensor"(%value, %length1) '
        "<{shape = array<i64: 2, 2>}> : "
        f"(!tla.tensor<!tla.ptr<f32, {storage}, 4>>, i64) -> ()"
    )
    return replace(
        artifact,
        tlair_mlir=f"module {{ func.func @dump() {{ {first} {second} }} }}",
    )


def test_print_tensor_workspace_preserves_user_argument_and_uses_abi_marker(
    tmp_path,
) -> None:
    artifact = _print_tensor_artifact(tmp_path)

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        block_num=1,
    )

    assert plan.payload == struct.pack(
        "<QQ", 0x1000, execution._PRINT_TENSOR_WORKSPACE_SENTINEL
    )
    assert plan.expects_print_tensor is True


def _install_fake_launch_context(monkeypatch, *, device: int = 7, stream: int = 99) -> None:
    from types import ModuleType

    from catlass.base_dsl.runtime import ascend_stream_adapter as stream_mod

    monkeypatch.setattr(stream_mod, "current_device", lambda: device)
    monkeypatch.setattr(stream_mod, "current_stream", lambda _device: stream)
    # Early ACL availability check in execute_kernel.
    monkeypatch.setattr(execution, "load_acl", lambda: ModuleType("fake_acl"))


def _install_print_tensor_loader(monkeypatch, output: str) -> None:
    _install_fake_launch_context(monkeypatch, device=1, stream=99)
    monkeypatch.setattr(
        execution, "load_binary", lambda **kwargs: (11, 12)
    )

    def _launch_kernel(**kwargs) -> None:
        assert kwargs["expects_print_tensor"] in (True, 2)
        os.write(1, output.encode())

    monkeypatch.setattr(execution, "launch_kernel", _launch_kernel)


def test_execute_kernel_decodes_and_formats_native_print_tensor_for_ordinary_call(
    monkeypatch, tmp_path, capfd
) -> None:
    _install_print_tensor_loader(
        monkeypatch,
        "CANN address=0xdeadbeef\n"
        "DumpTensor: call=0, block=0, data_type=float32, "
        "position=GM, shape=[2,2] dump_size=4 [0, 1.5, -2, 3]\n",
    )

    execution.execute_kernel(
        _print_tensor_artifact(tmp_path, shape=(2, 2)),
        runtime=execution.TlaRuntimeOptions(
            kernel_mode="aic", arch_scope="aic.c310"
        ),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={"block_num": 1},
    )

    assert capfd.readouterr().out == (
        "tla.print dtype=float32 shape=[2,2] count=4 values=[0.0, 1.5, -2.0, 3.0]\n"
    )


@pytest.mark.parametrize(("storage", "position"), (("gm", "GM"), ("ub", "UB")))
def test_execute_kernel_formats_combined_calls_and_blocks_in_arrival_order(
    monkeypatch, tmp_path, capfd, storage, position
) -> None:
    _install_print_tensor_loader(
        monkeypatch,
        "DumpTensor: call=1, block=1, data_type=float32, "
        f"position={position}, shape=[2,2] dump_size=2 [10, 11]\n"
        "DumpTensor: call=0, block=1, data_type=float32, "
        f"position={position}, shape=[2,2] dump_size=4 [4, 5, 6, 7]\n"
        "DumpTensor: call=1, block=0, data_type=float32, "
        f"position={position}, shape=[2,2] dump_size=2 [8, 9]\n"
        "DumpTensor: call=0, block=0, data_type=float32, "
        f"position={position}, shape=[2,2] dump_size=4 [0, 1, 2, 3]\n",
    )

    execution.execute_kernel(
        _two_print_tensor_artifact(tmp_path, storage=storage),
        runtime=execution.TlaRuntimeOptions(
            kernel_mode="aic", arch_scope="aic.c310"
        ),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={"block_num": 2},
    )

    assert capfd.readouterr().out == (
        "tla.print call=1 block=1 dtype=float32 shape=[2,2] "
        "count=2 values=[10.0, 11.0]\n"
        "tla.print call=0 block=1 dtype=float32 shape=[2,2] "
        "count=4 values=[4.0, 5.0, 6.0, 7.0]\n"
        "tla.print call=1 block=0 dtype=float32 shape=[2,2] "
        "count=2 values=[8.0, 9.0]\n"
        "tla.print call=0 block=0 dtype=float32 shape=[2,2] "
        "count=4 values=[0.0, 1.0, 2.0, 3.0]\n"
    )


def test_execute_aiv_kernel_formats_calls_blocks_and_subblocks_in_arrival_order(
    monkeypatch, tmp_path, capfd
) -> None:
    identities = (
        (0, 1, 0, "4, 5, 6, 7"),
        (1, 0, 0, "8, 9"),
        (1, 1, 0, "10, 11"),
        (0, 0, 0, "0, 1, 2, 3"),
    )
    output = "".join(
        _identified_print_tensor_record(
            call=call,
            block=block,
            subblock=subblock,
            count=2 if call else 4,
            values=values,
        )
        for call, block, subblock, values in identities
    )
    _install_print_tensor_loader(monkeypatch, output)

    execution.execute_kernel(
        _two_print_tensor_artifact(tmp_path),
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={"block_num": 2},
    )

    lines = capfd.readouterr().out.splitlines()
    assert [
        tuple(
            int(value)
            for value in re.search(
                r"call=(\d+) block=(\d+).* subblock=(\d+)", line
            ).groups()
        )
        for line in lines
    ] == [(call, block, subblock) for call, block, subblock, _ in identities]


def _identified_print_tensor_record(
    *,
    call: int = 0,
    block: int = 0,
    native_dtype: str = "float32",
    position: str = "GM",
    shape: str = "2,2",
    count: int = 4,
    values: str = "0, 1, 2, 3",
    subblock: int | None = None,
) -> str:
    rendered_subblock = f"subblock={subblock}, " if subblock is not None else ""
    return (
        f"DumpTensor: call={call}, block={block}, "
        f"{rendered_subblock}"
        f"data_type={native_dtype}, position={position}, shape=[{shape}] "
        f"dump_size={count} "
        f"[{values}]\n"
    )


def test_print_tensor_parser_retains_logical_subblock() -> None:
    record = execution._parse_native_print_tensor_records(
        _identified_print_tensor_record(subblock=1)
    )[0]

    assert record.subblock == 1


@pytest.mark.parametrize(
    "output",
    (
        "",
        (
            _identified_print_tensor_record(subblock=0)
            + _identified_print_tensor_record(subblock=0)
        ),
    ),
    ids=("no-subblock-output", "repeated-subblock-output"),
)
def test_aiv_print_tensor_record_set_allows_partial_dynamic_output(output) -> None:
    metadata = (execution._PrintTensorMetadata((2, 2), 4, "f32", "GM", call=0),)

    decoded = execution._decode_native_print_tensor_records(
        output,
        metadata=metadata,
        expected_subblocks=(0,),
    )

    assert len(decoded) == (0 if not output else 2)


@pytest.mark.parametrize("subblock", (None, 1), ids=("missing-tag", "wrong-tag"))
def test_aiv_print_tensor_record_set_rejects_invalid_subblock(subblock) -> None:
    metadata = (execution._PrintTensorMetadata((2, 2), 4, "f32", "GM", call=0),)

    with pytest.raises(
        execution.TlaExecutionError, match="unexpected call/block/subblock"
    ):
        execution._decode_native_print_tensor_records(
            _identified_print_tensor_record(subblock=subblock),
            metadata=metadata,
            expected_subblocks=(0,),
        )


def test_mixed_aiv_print_tensor_record_set_allows_one_executed_subblock() -> None:
    metadata = (execution._PrintTensorMetadata((2, 2), 4, "f32", "GM", call=0),)

    decoded = execution._decode_native_print_tensor_records(
        _identified_print_tensor_record(subblock=0),
        metadata=metadata,
        expected_subblocks=(0, 1),
    )

    assert len(decoded) == 1


@pytest.mark.parametrize(
    ("output", "match"),
    (
        (
            _identified_print_tensor_record(call=0)
            + _identified_print_tensor_record(call=7),
            r"unexpected call/block/subblock identities \[\(7, 0, None\)\]",
        ),
        (
            _identified_print_tensor_record(call=0)
            + _identified_print_tensor_record(call=1, block=3),
            r"unexpected call/block/subblock identities \[\(1, 3, None\)\]",
        ),
        (
            _identified_print_tensor_record(call=0, native_dtype="int32")
            + _identified_print_tensor_record(call=1, count=2, values="8, 9"),
            "unexpected native dtype",
        ),
        (
            _identified_print_tensor_record(call=0, position="UB")
            + _identified_print_tensor_record(call=1, count=2, values="8, 9"),
            "unexpected position",
        ),
        (
            _identified_print_tensor_record(call=0, count=3, values="0, 1, 2")
            + _identified_print_tensor_record(call=1, count=2, values="8, 9"),
            "unexpected declared count",
        ),
        (
            _identified_print_tensor_record(values="0, nope, 2, 3")
            + _identified_print_tensor_record(call=1, count=2, values="8, 9"),
            "malformed records",
        ),
        (
            "DumpTensor: bogus header\n"
            + _identified_print_tensor_record(call=0)
            + _identified_print_tensor_record(call=1, count=2, values="8, 9"),
            "malformed native record",
        ),
        (
            "DumpTensor: data_type=float32 position=GM dump_size=4 [0, 1, 2, 3]\n",
            "malformed native record",
        ),
    ),
    ids=(
        "unknown-call",
        "out-of-range-block",
        "wrong-dtype",
        "wrong-position",
        "wrong-count",
        "bad-numeric-syntax",
        "extra-header",
        "missing-identity",
    ),
)
def test_print_tensor_record_set_rejects_invalid_output_without_public_lines(
    monkeypatch, tmp_path, capfd, output, match
) -> None:
    _install_print_tensor_loader(monkeypatch, output)

    with pytest.raises(execution.TlaExecutionError, match=match):
        execution.execute_kernel(
            _two_print_tensor_artifact(tmp_path),
            runtime=execution.TlaRuntimeOptions(
                kernel_mode="aic", arch_scope="aic.c310"
            ),
            launch_args=[_TypedPointer(0x1000)],
            launch_kwargs={"block_num": 1},
        )

    assert capfd.readouterr().out == ""


@pytest.mark.parametrize(
    ("output", "match"),
    (
        (
            _identified_print_tensor_record(call=9, values="bad")
            + _identified_print_tensor_record(call=0)
            + _identified_print_tensor_record(call=0),
            "malformed records",
        ),
        (
            _identified_print_tensor_record(call=9)
            + _identified_print_tensor_record(call=8)
            + _identified_print_tensor_record(call=0)
            + _identified_print_tensor_record(call=0),
            r"unexpected call/block/subblock identities "
            r"\[\(8, 0, None\), \(9, 0, None\)\]",
        ),
    ),
    ids=("malformed-first", "unexpected-before-repeated"),
)
def test_print_tensor_record_set_has_stable_error_priority_and_sorted_identities(
    output, match
) -> None:
    metadata = (
        execution._PrintTensorMetadata((2, 2), 4, "f32", "GM", call=0),
        execution._PrintTensorMetadata((2, 2), 2, "f32", "GM", call=1),
    )

    with pytest.raises(execution.TlaExecutionError, match=match):
        execution._decode_native_print_tensor_records(
            output, metadata=metadata, block_count=1
        )


def test_print_tensor_capacity_uses_exact_aligned_native_wire_bytes() -> None:
    assert (
        execution._print_tensor_native_wire_bytes(
            execution._PrintTensorMetadata((16,), 1, "f32", "GM")
        )
        == 152
    )
    assert (
        execution._print_tensor_native_wire_bytes(
            execution._PrintTensorMetadata((16,), 16, "f32", "GM")
        )
        == 184
    )
    assert (
        execution._print_tensor_native_wire_bytes(
            execution._PrintTensorMetadata((-1,), None, "f32", "GM")
        )
        == execution._PRINT_TENSOR_FIFO_BYTES - 8
    )
    assert (
        execution._print_tensor_native_wire_bytes(
            execution._PrintTensorMetadata((16,), 16, "f16", "UB")
        )
        == 152
    )
    assert (
        execution._print_tensor_native_wire_bytes(
            execution._PrintTensorMetadata((16,), 16, "i8", "GM")
        )
        == 152
    )


def test_print_tensor_record_set_decodes_mixed_dtypes_per_call() -> None:
    metadata = (
        execution._PrintTensorMetadata((2, 2), 4, "i8", "GM", call=0),
        execution._PrintTensorMetadata((2, 2), 2, "u16", "UB", call=1),
    )
    output = _identified_print_tensor_record(
        call=1,
        native_dtype="uint16",
        position="UB",
        count=2,
        values="0, 65535",
    ) + _identified_print_tensor_record(
        call=0,
        native_dtype="int8",
        values="-128, 127, 0, -1",
    )

    decoded = execution._decode_native_print_tensor_records(output, metadata=metadata)

    assert [values for _, _, values in decoded] == [
        [0, 65535],
        [-128, 127, 0, -1],
    ]


@pytest.mark.parametrize("calls", (1, 2))
@pytest.mark.parametrize(("core_type", "max_blocks"), (("aic", 108), ("aiv", 108)))
def test_print_tensor_workspace_uses_fixed_one_mib_core_records(
    tmp_path, calls, core_type, max_blocks
) -> None:
    artifact = (
        _print_tensor_artifact(tmp_path, shape=(2, 2), static_length=4)
        if calls == 1
        else _two_print_tensor_artifact(tmp_path)
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode=core_type, arch_scope=f"{core_type}.c310"),
        launch_args=[_TypedPointer(0x1000)],
        block_num=max_blocks,
    )
    assert plan.block_num == max_blocks

    with pytest.raises(execution.TlaExecutionError, match="fixed 1 MiB"):
        execution._build_kernel_launch_plan(
            artifact=artifact,
            runtime=execution.TlaRuntimeOptions(
                kernel_mode=core_type, arch_scope=f"{core_type}.c310"
            ),
            launch_args=[_TypedPointer(0x1000)],
            block_num=max_blocks + 1,
        )


def test_mixed_aiv_print_tensor_capacity_counts_both_subblocks(
    tmp_path,
) -> None:
    artifact = _print_tensor_artifact(tmp_path, storage="ub", mixed=True)
    accepted = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[_TypedPointer(0x1000)],
        block_num=execution._PRINT_TENSOR_CORE_RECORDS // 2,
    )

    assert accepted.block_num == 54
    with pytest.raises(execution.TlaExecutionError, match="core records"):
        execution._build_kernel_launch_plan(
            artifact=artifact,
            runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
            launch_args=[_TypedPointer(0x1000)],
            block_num=55,
        )


def test_print_tensor_capacity_allows_multiple_static_calls_within_each_core_record(
    tmp_path,
) -> None:
    artifact = _two_print_tensor_artifact(tmp_path)
    artifact = replace(
        artifact,
        tlair_mlir=artifact.tlair_mlir.replace(
            "value = 4 : i64", "value = 262112 : i64"
        )
        .replace("value = 2 : i64", "value = 262112 : i64")
        .replace(
            "shape = array<i64: 2, 2>",
            "shape = array<i64: 262112>",
        ),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        block_num=1,
    )

    assert plan.block_num == 1


def test_print_tensor_capacity_allows_partial_dynamic_output(
    tmp_path,
) -> None:
    artifact = _two_print_tensor_artifact(tmp_path)
    artifact = replace(
        artifact,
        tlair_mlir=artifact.tlair_mlir.replace(
            '%length0 = "arith.constant"() <{value = 4 : i64}> : () -> i64 ',
            "",
            1,
        ),
    )
    metadata = execution._print_tensor_static_metadata_records(
        artifact.tlair_mlir, entrypoint=artifact.entrypoint
    )
    assert [record.count for record in metadata] == [None, 2]

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        block_num=1,
    )

    assert plan.block_num == 1


def test_print_tensor_metadata_requires_one_static_shape() -> None:
    with pytest.raises(execution.TlaExecutionError, match="static shape metadata"):
        execution._print_tensor_static_metadata_records(
            "module { func.func @kernel() }"
        )


def test_print_tensor_metadata_reads_generic_tlair_shape() -> None:
    mlir = (
        '"tla.print_tensor"(%value, %length) '
        "<{shape = array<i64: 2, 3>}> : "
        "(!tla.tensor<!tla.layout<!tla.shape<2,3>, !tla.stride<3,1>, "
        "!tla.shape<2,3>, RowMajor>, !tla.coord<0,0>, "
        "!tla.ptr<f32, gm, 4>>, i64) -> ()"
    )

    assert execution._print_tensor_static_metadata_records(mlir) == (
        execution._PrintTensorMetadata(
            shape=(2, 3), count=None, dtype="f32", position="GM"
        ),
    )


def test_print_tensor_metadata_reads_ub_storage() -> None:
    mlir = (
        '"tla.print_tensor"(%value, %length) '
        "<{shape = array<i64: 2, 3>}> : "
        "(!tla.tensor<!tla.layout<!tla.shape<2,3>, !tla.stride<3,1>, "
        "!tla.shape<2,3>, RowMajor>, !tla.coord<0,0>, "
        "!tla.ptr<f32, ub, 32>>, i64) -> ()"
    )

    assert execution._print_tensor_static_metadata_records(mlir) == (
        execution._PrintTensorMetadata(
            shape=(2, 3), count=None, dtype="f32", position="UB"
        ),
    )


def test_print_tensor_metadata_reads_dynamic_shape_pattern() -> None:
    mlir = (
        '"tla.print_tensor"(%value, %length) '
        "<{shape = array<i64: -1, 4>}> : "
        "(!tla.tensor<!tla.layout<!tla.shape<?,4>, !tla.stride<4,1>, "
        "!tla.shape<16,4>, RowMajor>, !tla.coord<0,0>, "
        "!tla.ptr<f32, gm, 4>>, i64) -> ()"
    )

    assert execution._print_tensor_static_metadata_records(mlir) == (
        execution._PrintTensorMetadata(
            shape=(-1, 4), count=None, dtype="f32", position="GM"
        ),
    )


def test_print_tensor_metadata_and_decode_are_scoped_to_second_entrypoint() -> None:
    first = (
        '"tla.func"() ({ '
        '"tla.print_tensor"(%value, %length0) '
        "<{length = 4 : i64, shape = array<i64: 2, 2>}> : "
        "(!tla.tensor<!tla.ptr<f32, gm, 4>>, i64) -> () "
        '"tla.print_tensor"(%value, %length1) '
        "<{length = 2 : i64, shape = array<i64: 2, 2>}> : "
        "(!tla.tensor<!tla.ptr<f32, gm, 4>>, i64) -> () "
        '}) {sym_name = "first"} : () -> ()'
    )
    second = (
        '"tla.func"() ({ '
        '"tla.print_tensor"(%value, %length) '
        "<{length = 4 : i64, shape = array<i64: 2, 2>}> : "
        "(!tla.tensor<!tla.ptr<f32, gm, 4>>, i64) -> () "
        '}) {sym_name = "second"} : () -> ()'
    )
    metadata = execution._print_tensor_static_metadata_records(
        f"module {{ {first} {second} }}", entrypoint="second"
    )

    assert metadata == (execution._PrintTensorMetadata((2, 2), 4, "f32", "GM", call=0),)
    decoded = execution._decode_native_print_tensor_records(
        _identified_print_tensor_record(call=0),
        metadata=metadata,
        block_count=1,
    )
    assert decoded[0][0].call == 0


@pytest.mark.parametrize("core_type", ("aic", "aiv"))
def test_print_tensor_launch_accepts_multiblock_block_num(tmp_path, core_type) -> None:
    plan = execution._build_kernel_launch_plan(
        artifact=_print_tensor_artifact(tmp_path),
        runtime=execution.TlaRuntimeOptions(kernel_mode=core_type, arch_scope=f"{core_type}.c310"),
        launch_args=[_TypedPointer(0x1000)],
        block_num=2,
    )

    assert plan.block_num == 2
    assert plan.expects_print_tensor is True


@pytest.mark.parametrize(
    ("block_num", "accepted"),
    (
        (65536, True),
        (65537, False),
        (0, False),
        (-1, False),
    ),
)
def test_print_tensor_launch_checks_16_bit_block_identity(block_num, accepted) -> None:
    if accepted:
        assert execution._checked_print_tensor_block_count(block_num) == block_num
    else:
        with pytest.raises(execution.TlaExecutionError, match="16-bit block"):
            execution._checked_print_tensor_block_count(block_num)


def test_mixed_print_tensor_workspace_keeps_trailing_marker(tmp_path) -> None:
    plan = execution._build_kernel_launch_plan(
        artifact=_print_tensor_artifact(tmp_path, storage="ub", mixed=True),
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[_TypedPointer(0x1000)],
        block_num=1,
    )

    assert plan.entrypoint == "dump"
    assert plan.payload == struct.pack(
        "<QQ", 0x1000, execution._PRINT_TENSOR_WORKSPACE_SENTINEL
    )
    assert plan.expects_print_tensor == 2


def test_execute_mixed_aiv_print_preserves_position_and_subblocks(
    monkeypatch, tmp_path, capfd
) -> None:
    _install_print_tensor_loader(
        monkeypatch,
        _identified_print_tensor_record(subblock=1, position="UB", values="4, 5, 6, 7")
        + _identified_print_tensor_record(
            subblock=0, position="UB", values="0, 1, 2, 3"
        ),
    )

    execution.execute_kernel(
        _print_tensor_artifact(tmp_path, shape=(2, 2), storage="ub", mixed=True),
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={"block_num": 1},
    )

    assert capfd.readouterr().out.splitlines() == [
        "tla.print dtype=float32 position=UB subblock=1 "
        "shape=[2,2] count=4 values=[4.0, 5.0, 6.0, 7.0]",
        "tla.print dtype=float32 position=UB subblock=0 "
        "shape=[2,2] count=4 values=[0.0, 1.0, 2.0, 3.0]",
    ]


def test_execute_mixed_aic_print_omits_subblock(monkeypatch, tmp_path, capfd) -> None:
    _install_print_tensor_loader(monkeypatch, _identified_print_tensor_record())
    artifact = replace(
        _print_tensor_artifact(tmp_path, shape=(2, 2), mixed=True),
        lowered_llvm=(
            "module {\n"
            "func.func private @_mlir_ciface_tla_print_tensor_gm_f32()\n"
            "func.func @dump_mix_aic(%value: i64, %workspace: i64 "
            "{tla.print_tensor.workspace}) {\n"
            "call @_mlir_ciface_tla_print_tensor_gm_f32() : () -> ()\n"
            "return\n"
            "}\n"
            "func.func @dump_mix_aiv(%value: i64, %workspace: i64 "
            "{tla.print_tensor.workspace}) { return }\n"
            "}\n"
        ),
    )

    execution.execute_kernel(
        artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={"block_num": 1},
    )

    assert capfd.readouterr().out == (
        "tla.print dtype=float32 position=GM shape=[2,2] count=4 "
        "values=[0.0, 1.0, 2.0, 3.0]\n"
    )


@pytest.mark.parametrize(
    ("launch_args", "expected_user_payload"),
    [
        (
            [_TypedPointer(0x1000), _TypedPointer(0x2000)],
            struct.pack("<QQ", 0x1000, 0x2000),
        ),
        ([tla.Int32(7), _TypedPointer(0x1000)], struct.pack("<i4xQ", 7, 0x1000)),
        ([_TypedPointer(0x1000), tla.Int32(7)], struct.pack("<Qi4x", 0x1000, 7)),
        (
            [tla.Int32(7), tla.Int32(9), _TypedPointer(0x1000)],
            struct.pack("<iiQ", 7, 9, 0x1000),
        ),
    ],
    ids=["pointer-pointer", "scalar-pointer", "pointer-scalar", "multi-scalar-pointer"],
)
def test_debug_print_workspace_preserves_normal_user_argument_slots(
    tmp_path, launch_args, expected_user_payload
) -> None:
    artifact = _debug_print_artifact(tmp_path, launch_args=launch_args)

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=launch_args,
        block_num=1,
    )

    assert plan.payload == expected_user_payload + struct.pack(
        "<Q", int.from_bytes(b"TLA_PRNT", byteorder="big")
    )
    assert plan.expects_debug_fifo is True


def test_non_print_kernel_keeps_normal_pointer_payload(tmp_path) -> None:
    launch_args = [_TypedPointer(0x1000), _TypedPointer(0x2000)]
    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module { func.func @plain() }",
        lowered_llvm="module { func.func @plain() }",
        entrypoint="plain",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_debug_kernel_abi(launch_args, entrypoint="plain"),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=launch_args,
        block_num=1,
    )

    assert plan.payload == struct.pack("<QQ", 0x1000, 0x2000)
    assert plan.expects_debug_fifo is False


def test_cache_key_uses_ir_and_debug_print_workspace_abi_revision(
    monkeypatch, tmp_path
) -> None:
    hivmc = tmp_path / "hivmc-a5"
    target = execution.TlaKernelTarget("aiv.c310", "c310", "aiv", "dav-c310-vec")
    runtime = execution.TlaRuntimeOptions()
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "test")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "test")

    plain_key = execution._cache_key(
        tlair_mlir="module { func.func @kernel() }",
        entrypoint="kernel",
        runtime=runtime,
        compiler_bridge_path=None,
        hivmc=hivmc,
        target=target,
    )
    same_plain_key = execution._cache_key(
        tlair_mlir="module { func.func @kernel() }",
        entrypoint="kernel",
        runtime=runtime,
        compiler_bridge_path=None,
        hivmc=hivmc,
        target=target,
    )
    debug_key = execution._cache_key(
        tlair_mlir="module { tla.debug_print %value : i32 }",
        entrypoint="kernel",
        runtime=runtime,
        compiler_bridge_path=None,
        hivmc=hivmc,
        target=target,
    )

    assert plain_key == same_plain_key
    assert debug_key != plain_key
    monkeypatch.setattr(
        execution,
        "_DEBUG_PRINT_WORKSPACE_ABI_REVISION",
        "debug-print-workspace-i64-v0",
    )
    assert (
        execution._cache_key(
            tlair_mlir="module { func.func @kernel() }",
            entrypoint="kernel",
            runtime=runtime,
            compiler_bridge_path=None,
            hivmc=hivmc,
            target=target,
        )
        != plain_key
    )


def test_cache_key_uses_print_tensor_workspace_abi_revision(
    monkeypatch, tmp_path
) -> None:
    hivmc = tmp_path / "hivmc-a5"
    target = execution.TlaKernelTarget("aiv.c310", "c310", "aiv", "dav-c310-vec")
    runtime = execution.TlaRuntimeOptions()
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "test")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "test")
    kwargs = {
        "tlair_mlir": (
            'module { "tla.print_tensor"(%value, %length) '
            "<{shape = array<i64: 4, 4>}> "
            ": (!tla.tensor, i64) -> () }"
        ),
        "entrypoint": "kernel",
        "runtime": runtime,
        "compiler_bridge_path": None,
        "hivmc": hivmc,
        "target": target,
    }
    key = execution._cache_key(**kwargs)

    monkeypatch.setattr(
        execution,
        "_PRINT_TENSOR_WORKSPACE_ABI_REVISION",
        "print-tensor-workspace-i64-v0",
    )

    assert execution._cache_key(**kwargs) != key


@pytest.mark.parametrize("manifest_revision", [None, "debug-print-workspace-i64-v0"])
def test_debug_print_workspace_abi_manifest_requires_current_revision(
    manifest_revision,
) -> None:
    manifest = {}
    if manifest_revision is not None:
        manifest["debug_print_workspace_abi_revision"] = manifest_revision

    assert not execution._cache_manifest_has_current_debug_print_workspace_abi(manifest)
    manifest["debug_print_workspace_abi_revision"] = (
        execution._DEBUG_PRINT_WORKSPACE_ABI_REVISION
    )
    assert execution._cache_manifest_has_current_debug_print_workspace_abi(manifest)


@pytest.mark.parametrize("manifest_revision", [None, "print-tensor-workspace-i64-v0"])
def test_print_tensor_workspace_abi_manifest_requires_current_revision(
    manifest_revision,
) -> None:
    manifest = {}
    if manifest_revision is not None:
        manifest["print_tensor_workspace_abi_revision"] = manifest_revision

    assert not execution._cache_manifest_has_current_print_tensor_workspace_abi(
        manifest
    )
    manifest["print_tensor_workspace_abi_revision"] = (
        execution._PRINT_TENSOR_WORKSPACE_ABI_REVISION
    )
    assert execution._cache_manifest_has_current_print_tensor_workspace_abi(manifest)


def _kernel_abi(
    *arguments: tuple[str, str, str, int, int, int],
    total_size: int,
    entrypoint: str = "kernel",
) -> compiler_bridge.KernelAbiLayout:
    def scalar_descriptor(
        abi_type: str,
    ) -> compiler_bridge.KernelAbiScalarDescriptor | None:
        if abi_type == "pointer":
            return None
        if abi_type == "index":
            return compiler_bridge.KernelAbiScalarDescriptor(
                compiler_bridge.KernelAbiScalarCategory.INDEX, 64, None, None
            )
        if abi_type in {"f16", "bf16", "f32"}:
            return compiler_bridge.KernelAbiScalarDescriptor(
                compiler_bridge.KernelAbiScalarCategory.FLOAT,
                int(abi_type[-2:]),
                None,
                compiler_bridge.KernelAbiFloatFormat(abi_type),
            )
        signedness = (
            compiler_bridge.KernelAbiIntegerSignedness.SIGNED
            if abi_type.startswith("si")
            else compiler_bridge.KernelAbiIntegerSignedness.UNSIGNED
            if abi_type.startswith("ui")
            else compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS
        )
        return compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            int(abi_type.lstrip("sui")),
            signedness,
            None,
        )

    return compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint=entrypoint,
        total_size=total_size,
        arguments=tuple(
            compiler_bridge.KernelAbiArgument(
                index=index,
                kind=compiler_bridge.KernelAbiArgumentKind(kind),
                scalar=scalar_descriptor(abi_type),
                mlir_type=mlir_type,
                offset=offset,
                storage_size=storage_size,
                alignment=alignment,
            )
            for index, (
                kind,
                abi_type,
                mlir_type,
                offset,
                storage_size,
                alignment,
            ) in enumerate(arguments)
        ),
    )


def test_online_cache_key_serializes_kernel_abi_version(monkeypatch, tmp_path) -> None:
    payloads: list[dict[str, object]] = []
    json_dumps = execution.json.dumps

    def capture_payload(payload, **kwargs):
        payloads.append(payload)
        return json_dumps(payload, **kwargs)

    monkeypatch.setattr(execution.json, "dumps", capture_payload)
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "version")

    runtime = execution.TlaRuntimeOptions()
    execution._cache_key(
        tlair_mlir="module {}",
        entrypoint="kernel",
        runtime=runtime,
        compiler_bridge_path=tmp_path / "bridge.so",
        hivmc=tmp_path / "hivmc-a5",
        target=execution.TlaKernelTarget(
            arch_scope="aiv.c310",
            target_arch="c310",
            core_type="aiv",
            cce_arch="dav-c310-vec",
        ),
    )

    assert len(payloads) == 1
    assert payloads[0]["cache_abi_version"] == 5


@pytest.mark.parametrize(
    ("scalar", "mlir_type", "storage_size", "expected"),
    [
        (tla.Int8(-91), "i8", 1, bytes.fromhex("a5")),
        (tla.Int16(-16657), "i16", 2, bytes.fromhex("efbe")),
        (tla.UInt16(0xBEEF), "i16", 2, bytes.fromhex("efbe")),
        (tla.Int32(-559038737), "i32", 4, bytes.fromhex("efbeadde")),
        (
            tla.Int64(-0x112233445566778),
            "i64",
            8,
            (-0x112233445566778).to_bytes(8, "little", signed=True),
        ),
        (tla.Float16(1.5), "f16", 2, bytes.fromhex("003e")),
        (tla.BFloat16(1.5), "bf16", 2, bytes.fromhex("c03f")),
        (tla.Float32(1.25), "f32", 4, bytes.fromhex("0000a03f")),
    ],
)
def test_pack_launch_args_writes_typed_scalar_bits_at_declared_width(
    scalar, mlir_type: str, storage_size: int, expected: bytes
) -> None:
    layout = _kernel_abi(
        ("scalar", mlir_type, mlir_type, 0, storage_size, 4), total_size=8
    )
    payload = execution._pack_launch_args([scalar], layout)

    assert len(payload) == 8
    assert payload == expected + bytes(8 - storage_size)


@pytest.mark.parametrize(
    ("value", "abi_type", "mlir_type", "storage_size", "expected"),
    [
        (False, "i1", "i1", 1, bytes.fromhex("00")),
        (True, "i1", "i1", 1, bytes.fromhex("01")),
        (17, "i32", "i32", 4, struct.pack("<i", 17)),
        (-17, "i32", "i32", 4, struct.pack("<i", -17)),
        (-(1 << 31), "i32", "i32", 4, struct.pack("<i", -(1 << 31))),
        ((1 << 31) - 1, "i32", "i32", 4, struct.pack("<i", (1 << 31) - 1)),
        (1.25, "f32", "f32", 4, bytes.fromhex("0000a03f")),
    ],
)
def test_pack_launch_args_writes_plain_python_scalar_bits_from_descriptor(
    value,
    abi_type: str,
    mlir_type: str,
    storage_size: int,
    expected: bytes,
) -> None:
    layout = _kernel_abi(
        ("scalar", abi_type, mlir_type, 0, storage_size, 4), total_size=8
    )

    payload = execution._pack_launch_args([value], layout)

    assert payload == expected + bytes(8 - storage_size)


@pytest.mark.parametrize("value", [-(1 << 31) - 1, 1 << 31])
def test_pack_launch_args_rejects_plain_python_int32_overflow(value: int) -> None:
    layout = _kernel_abi(("scalar", "i32", "i32", 0, 4, 4), total_size=8)

    with pytest.raises(execution.TlaUnsupportedAbiError, match="fit"):
        execution._pack_launch_args([value], layout)


@pytest.mark.parametrize(
    ("value", "abi_type", "mlir_type", "storage_size"),
    [
        (1, "index", "index", 8),
        (1, "si32", "si32", 4),
        (1, "i64", "i64", 8),
        (1.0, "f16", "f16", 2),
        (1.0, "bf16", "bf16", 2),
        (True, "index", "index", 8),
        (1, "i1", "i1", 1),
    ],
)
def test_pack_launch_args_rejects_plain_python_scalar_descriptor_mismatch(
    value, abi_type: str, mlir_type: str, storage_size: int
) -> None:
    layout = _kernel_abi(
        ("scalar", abi_type, mlir_type, 0, storage_size, 4), total_size=8
    )

    with pytest.raises(execution.TlaUnsupportedAbiError, match="does not match"):
        execution._pack_launch_args([value], layout)


@pytest.mark.parametrize("value", [False, 1, 1.0])
def test_pack_launch_args_rejects_plain_python_scalar_for_pointer(value) -> None:
    layout = _kernel_abi(("pointer", "pointer", "!llvm.ptr", 0, 8, 4), total_size=8)

    with pytest.raises(execution.TlaUnsupportedAbiError, match="pointer"):
        execution._pack_launch_args([value], layout)


def test_pack_scalar_argument_rejects_float_descriptor_without_format() -> None:
    descriptor = compiler_bridge.KernelAbiScalarDescriptor(
        compiler_bridge.KernelAbiScalarCategory.FLOAT,
        32,
        None,
        compiler_bridge.KernelAbiFloatFormat.F32,
    )
    object.__setattr__(descriptor, "float_format", None)

    with pytest.raises(
        execution.TlaUnsupportedAbiError,
        match="float scalar f32 has no format",
    ):
        execution._pack_scalar_argument(tla.Float32(1.0), descriptor, "f32", 4)


def test_pack_launch_args_rejects_missing_layout_after_validation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(execution, "_validate_kernel_abi_layout", lambda _layout: None)

    with pytest.raises(
        execution.TlaUnsupportedAbiError,
        match="kernel ABI layout is missing",
    ):
        execution._pack_launch_args([], None)


def test_pack_launch_args_rejects_scalar_argument_without_descriptor(
    monkeypatch,
) -> None:
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=8,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=None,
                mlir_type="i32",
                offset=0,
                storage_size=4,
                alignment=4,
            ),
        ),
    )
    monkeypatch.setattr(execution, "_validate_kernel_abi_layout", lambda _layout: None)

    with pytest.raises(
        execution.TlaUnsupportedAbiError,
        match="scalar argument 0 has no scalar descriptor",
    ):
        execution._pack_launch_args([tla.Int32(1)], layout)


def test_pack_launch_args_naturally_aligns_trailing_host_pointer() -> None:
    class _Ptr:
        def __init__(self, value: int) -> None:
            self.value = value

        def __c_pointers__(self):
            return [self.value]

    layout = _kernel_abi(
        ("pointer", "pointer", "memref<8xi32>", 0, 8, 4),
        ("scalar", "i16", "i16", 8, 2, 4),
        ("pointer", "pointer", "memref<8xi32>", 12, 8, 4),
        total_size=24,
    )
    payload = execution._pack_launch_args(
        [
            _Ptr(0x1111111122222222),
            tla.Int16(-16657),
            _Ptr(0x3333333344444444),
        ],
        layout,
    )

    assert payload == (
        struct.pack("<Q", 0x1111111122222222)
        + bytes.fromhex("efbe000000000000")
        + struct.pack("<Q", 0x3333333344444444)
    )


def test_pack_launch_args_builds_dynamic_memref_fields_once_per_tensor() -> None:
    class _DynamicTensor:
        def __init__(self) -> None:
            self.build_calls = 0

        def build_memref_launch_fields(self):
            self.build_calls += 1
            return {
                "aligned": 0x1234,
                "offset": 7,
                "size_0": 31,
                "stride_0": 1,
            }

    fields = ("aligned", "offset", "size_0", "stride_0")
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=32,
        arguments=tuple(
            compiler_bridge.KernelAbiArgument(
                index=index,
                kind=compiler_bridge.KernelAbiArgumentKind.MEMREF_FIELD,
                scalar=None,
                mlir_type="memref<?xf16>",
                offset=index * 8,
                storage_size=8,
                alignment=4,
                logical_index=0,
                field=field,
            )
            for index, field in enumerate(fields)
        ),
    )
    tensor = _DynamicTensor()

    first = execution._pack_launch_args([tensor], layout)
    second = execution._pack_launch_args([tensor], layout)

    assert first == struct.pack("<QQQQ", 0x1234, 7, 31, 1)
    assert second == first
    assert tensor.build_calls == 2


def test_pack_launch_args_host_payload_can_exceed_compiler_payload() -> None:
    class _Ptr:
        def __c_pointers__(self):
            return [0x1111111122222222]

    layout = _kernel_abi(
        ("scalar", "i16", "i16", 0, 2, 4),
        ("pointer", "pointer", "!llvm.ptr", 4, 8, 4),
        ("scalar", "i16", "i16", 12, 2, 4),
        total_size=16,
    )

    payload = execution._pack_launch_args([tla.Int16(1), _Ptr(), tla.Int16(2)], layout)

    assert len(payload) == 24
    assert payload == (
        bytes.fromhex("0100000000000000")
        + struct.pack("<Q", 0x1111111122222222)
        + bytes.fromhex("0200000000000000")
    )


@pytest.mark.parametrize(
    ("args", "argument", "total_size", "message"),
    [
        (
            [],
            ("scalar", "i32", "i32", 0, 4, 4),
            8,
            "argument count",
        ),
        (
            [tla.Int32(1)],
            ("pointer", "pointer", "memref<8xi32>", 0, 8, 4),
            8,
            "pointer",
        ),
        (
            [tla.Float32(1.0)],
            ("scalar", "i32", "i32", 0, 4, 4),
            8,
            "i32",
        ),
        (
            [tla.Int32(1)],
            ("scalar", "i32", "i32", 0, 8, 4),
            8,
            "storage size",
        ),
    ],
)
def test_pack_launch_args_rejects_invalid_count_kind_and_type(
    args,
    argument: tuple[str, str, int, int, int],
    total_size: int,
    message: str,
) -> None:
    layout = _kernel_abi(argument, total_size=total_size)
    with pytest.raises(execution.TlaUnsupportedAbiError, match=message):
        execution._pack_launch_args(args, layout)


def test_pack_launch_args_rejects_missing_layout() -> None:
    with pytest.raises(execution.TlaUnsupportedAbiError, match="ABI layout"):
        execution._pack_launch_args([tla.Int32(1)], None)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda layout: replace(layout, entrypoint=""), "entrypoint"),
        (
            lambda layout: replace(
                layout,
                arguments=(replace(layout.arguments[0], alignment=8),),
            ),
            "4-byte alignment",
        ),
        (
            lambda layout: replace(
                layout,
                arguments=(replace(layout.arguments[0], offset=2),),
            ),
            "4-byte aligned",
        ),
        (
            lambda layout: replace(
                layout,
                arguments=(replace(layout.arguments[0], storage_size=0),),
            ),
            "positive",
        ),
        (
            lambda layout: replace(
                layout,
                arguments=(
                    replace(
                        layout.arguments[0],
                        kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                        scalar=None,
                        mlir_type="!llvm.ptr",
                        storage_size=4,
                    ),
                ),
            ),
            "pointer storage size",
        ),
        (
            lambda layout: replace(layout, total_size=-8),
            "invalid total size",
        ),
        (
            lambda layout: replace(layout, total_size=4),
            "rounded to 8",
        ),
        (
            lambda layout: replace(layout, total_size=16),
            "exactly sufficient",
        ),
        (
            lambda layout: replace(
                layout, total_size=execution._MAX_KERNEL_ABI_PAYLOAD_SIZE + 8
            ),
            "maximum",
        ),
    ],
)
def test_pack_launch_args_rejects_malformed_layout_before_allocation(
    mutate, message: str
) -> None:
    layout = _kernel_abi(
        ("scalar", "i32", "i32", 0, 4, 4),
        total_size=8,
    )

    with pytest.raises(execution.TlaUnsupportedAbiError, match=message):
        execution._pack_launch_args([tla.Int32(1)], mutate(layout))


def test_kernel_abi_layout_rejects_overlapping_arguments() -> None:
    layout = _kernel_abi(
        ("pointer", "pointer", "!llvm.ptr", 0, 8, 4),
        ("scalar", "i32", "i32", 4, 4, 4),
        total_size=8,
    )
    with pytest.raises(execution.TlaUnsupportedAbiError, match="non-overlapping"):
        execution._pack_launch_args([object(), tla.Int32(1)], layout)


def test_kernel_abi_layout_entrypoint_must_match_artifact() -> None:
    layout = _kernel_abi(total_size=0, entrypoint="other")
    with pytest.raises(execution.TlaUnsupportedAbiError, match="does not match"):
        execution._validate_kernel_abi_layout(layout, expected_entrypoint="kernel")


def test_compiler_produced_kernel_abi_is_validated_before_artifact() -> None:
    layout = _kernel_abi(total_size=0, entrypoint="other")

    with pytest.raises(execution.TlaKernelCompileError, match="does not match"):
        execution._prepare_compiled_abi_packer(
            layout, expected_entrypoint="kernel"
        )


def test_corrupt_manifest_layout_is_compile_error() -> None:
    layout = _kernel_abi(
        ("scalar", "i32", "i32", 0, 4, 4),
        total_size=8,
    )
    descriptor = compiler_bridge.kernel_abi_to_dict(layout)
    assert descriptor is not None
    descriptor["total_size"] = 16

    with pytest.raises(execution.TlaKernelCompileError, match="exactly sufficient"):
        execution._kernel_abi_from_manifest(
            {"entrypoint": "kernel", "kernel_abi": descriptor}
        )


def test_corrupt_manifest_logical_index_is_compile_error() -> None:
    layout = _kernel_abi(
        ("pointer", "pointer", "!llvm.ptr", 0, 8, 4),
        total_size=8,
    )
    layout = replace(
        layout,
        arguments=(replace(layout.arguments[0], logical_index=-1),),
    )

    with pytest.raises(execution.TlaKernelCompileError, match="logical_index"):
        execution._kernel_abi_from_manifest(
            {
                "entrypoint": "kernel",
                "kernel_abi": compiler_bridge.kernel_abi_to_dict(layout),
            }
        )


def test_manifest_rejects_descriptor_decoder_returning_none(
    monkeypatch,
) -> None:
    monkeypatch.setattr(execution, "kernel_abi_from_dict", lambda _value: None)

    with pytest.raises(
        execution.TlaKernelCompileError,
        match="decoded to no layout",
    ):
        execution._kernel_abi_from_manifest({"entrypoint": "kernel", "kernel_abi": {}})


def test_pack_launch_args_rejects_multi_value_host_argument() -> None:
    class _TwoPointers:
        def __c_pointers__(self):
            return [0x1111, 0x2222]

    layout = _kernel_abi(("pointer", "pointer", "memref<8xi32>", 0, 8, 4), total_size=8)
    with pytest.raises(execution.TlaUnsupportedAbiError, match="exactly one"):
        execution._pack_launch_args([_TwoPointers()], layout)


def test_pack_launch_args_rejects_pointer_storage_overflow() -> None:
    class _HugePointer:
        def __c_pointers__(self):
            return [1 << 64]

    layout = _kernel_abi(("pointer", "pointer", "memref<8xi32>", 0, 8, 4), total_size=8)
    with pytest.raises(execution.TlaUnsupportedAbiError, match="fit"):
        execution._pack_launch_args([_HugePointer()], layout)


def test_build_kernel_launch_plan_uses_logical_mixed_handoff(tmp_path) -> None:
    class _Tensor:
        def __init__(self, ptr: int, shape: tuple[int, int]) -> None:
            self._ptr = ptr
            self._shape_tuple = shape
            self.stride = (shape[1], 1)

        def data_ptr(self) -> int:
            return self._ptr

    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=(
            "module { "
            "func.func @basic_mixed_mix_aic("
            "%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, "
            "%arg2: memref<32x32xf32>, %arg3: memref<32x32xf32>"
            ') attributes {mix_mode = "mix"} '
            "func.func @basic_mixed_mix_aiv("
            "%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, "
            "%arg2: memref<32x32xf32>, %arg3: memref<32x32xf32>"
            ') attributes {mix_mode = "mix"} }'
        ),
        entrypoint="ignored",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(
            ("pointer", "pointer", "memref<32x32xf32>", 0, 8, 4),
            ("pointer", "pointer", "memref<32x32xf32>", 8, 8, 4),
            ("pointer", "pointer", "memref<32x32xf32>", 16, 8, 4),
            ("pointer", "pointer", "memref<32x32xf32>", 24, 8, 4),
            total_size=32,
            entrypoint="basic_mixed",
        ),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[
            _Tensor(0x1000, (32, 32)),
            _Tensor(0x2000, (32, 32)),
            _Tensor(0x3000, (32, 32)),
            _Tensor(0x4000, (32, 32)),
        ],
        block_num=1,
    )

    assert plan.entrypoint == "basic_mixed"
    assert plan.kernel_mode == "mix"
    assert plan.block_num == 1
    assert plan.payload == struct.pack("<QQQQ", 0x1000, 0x2000, 0x3000, 0x4000)


def test_mixed_handoff_payload_follows_split_signature_not_fixed_four_args(
    tmp_path,
) -> None:
    class _Tensor:
        def __init__(self, ptr: int, shape: tuple[int, int]) -> None:
            self._ptr = ptr
            self._shape_tuple = shape
            self.stride = (shape[1], 1)

        def data_ptr(self) -> int:
            return self._ptr

    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=(
            "module { "
            "func.func @custom_mix_aic("
            "%arg0: i32, %arg1: memref<16x64xf32>, "
            "%arg2: memref<64x48xf32>, %arg3: memref<16x48xf32>, "
            "%arg4: memref<16x48xf32>"
            ') attributes {mix_mode = "mix"} '
            "func.func @custom_mix_aiv("
            "%arg0: i32, %arg1: memref<16x64xf32>, "
            "%arg2: memref<64x48xf32>, %arg3: memref<16x48xf32>, "
            "%arg4: memref<16x48xf32>"
            ') attributes {mix_mode = "mix"} }'
        ),
        entrypoint="ignored",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(
            ("scalar", "i32", "i32", 0, 4, 4),
            ("pointer", "pointer", "memref<16x64xf32>", 4, 8, 4),
            ("pointer", "pointer", "memref<64x48xf32>", 12, 8, 4),
            ("pointer", "pointer", "memref<16x48xf32>", 20, 8, 4),
            ("pointer", "pointer", "memref<16x48xf32>", 28, 8, 4),
            total_size=40,
            entrypoint="custom",
        ),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[
            tla.Int32(7),
            _Tensor(0x1000, (16, 64)),
            _Tensor(0x2000, (64, 48)),
            _Tensor(0x3000, (16, 48)),
            _Tensor(0x4000, (16, 48)),
        ],
        block_num=210,
    )

    assert plan.entrypoint == "custom"
    assert plan.kernel_mode == "mix"
    assert plan.block_num == 210
    assert plan.payload == struct.pack(
        "<I4xQQQQ",
        7,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
    )


def test_mixed_handoff_supplies_debug_workspace_without_public_argument(
    tmp_path,
) -> None:
    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module {}",
        lowered_llvm=(
            "module { "
            "func.func @debug_mixed_mix_aic("
            "%arg0: f32, %arg1: f32, "
            "%workspace: i64 "
            "{hacc.arg_type = #hacc.arg_type<workspace>, "
            'tla.debug_print.workspace}) attributes {mix_mode = "mix"} '
            "func.func @debug_mixed_mix_aiv("
            "%arg0: f32, %arg1: f32, "
            "%workspace: i64 "
            "{hacc.arg_type = #hacc.arg_type<workspace>, "
            'tla.debug_print.workspace}) attributes {mix_mode = "mix"} }'
        ),
        entrypoint="ignored",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(
            ("scalar", "f32", "f32", 0, 4, 4),
            ("scalar", "f32", "f32", 4, 4, 4),
            total_size=8,
            entrypoint="debug_mixed",
        ),
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(kernel_mode="mix"),
        launch_args=[tla.Float32(1.0), tla.Float32(0.25)],
        block_num=1,
    )

    sentinel = int.from_bytes(b"TLA_PRNT", byteorder="big")
    assert plan.entrypoint == "debug_mixed"
    assert plan.payload == struct.pack("<ffQ", 1.0, 0.25, sentinel)
    assert plan.expects_debug_fifo is True


def test_execute_kernel_uses_typed_launch_payload(monkeypatch, tmp_path) -> None:
    launches: list[tuple[str, object]] = []

    def _load_binary(**kwargs):
        launches.append(("load", kwargs))
        return (11, 12)

    def _launch_kernel(**kwargs) -> None:
        launches.append(("flat", kwargs))

    _install_fake_launch_context(monkeypatch, device=7, stream=99)
    monkeypatch.setattr(execution, "load_binary", _load_binary)
    monkeypatch.setattr(execution, "launch_kernel", _launch_kernel)

    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir=(
            'module { "tla.func"() ({}) '
            '{function_type = (i32) -> (), sym_name = "kernel"} : () -> () }'
        ),
        lowered_llvm="module {}",
        entrypoint="kernel",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(
            ("scalar", "i32", "i32", 0, 4, 4),
            total_size=8,
        ),
    )
    runtime = execution.TlaRuntimeOptions()

    result = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[tla.Int32(123)],
        launch_kwargs={"block_num": 1},
    )

    assert result.module_handle == 11
    assert result.function_handle == 12
    assert (
        "flat",
        {
            "function": 12,
            "stream": 99,
            "block_num": 1,
            "args": struct.pack("<I4x", 123),
            "expects_debug_fifo": False,
            "expects_print_tensor": False,
        },
    ) in launches


def test_execute_kernel_conveys_debug_fifo_intent_to_loader(
    monkeypatch, tmp_path
) -> None:
    launches: list[dict[str, object]] = []

    _install_fake_launch_context(monkeypatch, device=7, stream=99)
    monkeypatch.setattr(execution, "load_binary", lambda **kwargs: (11, 12))
    monkeypatch.setattr(
        execution, "launch_kernel", lambda **kwargs: launches.append(kwargs)
    )
    artifact = _debug_print_artifact(tmp_path, entrypoint="debug")

    execution.execute_kernel(
        artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[tla.Int32(7)],
        launch_kwargs={"block_num": 1},
    )

    assert launches == [
        {
            "function": 12,
            "stream": 99,
            "block_num": 1,
            "args": struct.pack("<QQ", 7, int.from_bytes(b"TLA_PRNT", byteorder="big")),
            "expects_debug_fifo": True,
            "expects_print_tensor": False,
        }
    ]


def test_execute_kernel_uses_empty_payload_for_zero_arg(monkeypatch, tmp_path) -> None:
    launches: list[tuple[str, object]] = []

    def _load_binary(**kwargs):
        launches.append(("load", kwargs))
        return (11, 12)

    def _launch_kernel(**kwargs) -> None:
        launches.append(("flat", kwargs))

    _install_fake_launch_context(monkeypatch, device=7, stream=99)
    monkeypatch.setattr(execution, "load_binary", _load_binary)
    monkeypatch.setattr(execution, "launch_kernel", _launch_kernel)

    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir='module { "tla.func"() ({}) {function_type = () -> (), sym_name = "kernel"} : () -> () }',
        lowered_llvm="module {}",
        entrypoint="kernel",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
        kernel_abi=_kernel_abi(total_size=0),
    )
    runtime = execution.TlaRuntimeOptions()

    result = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )

    assert result.module_handle == 11
    assert result.function_handle == 12
    # Plan-level payload stays empty; ``launch_kernel`` pads to 8 bytes
    # before PyACL launch_kernel.
    assert (
        "flat",
        {
            "function": 12,
            "stream": 99,
            "block_num": 1,
            "args": b"",
            "expects_debug_fifo": False,
            "expects_print_tensor": False,
        },
    ) in launches


def test_ascend_runtime_load_binary_does_not_cache(monkeypatch, tmp_path) -> None:
    calls: list[tuple[Path, str, str]] = []
    set_device_calls: list[int] = []
    resolve_calls: list[Path] = []

    class _FakeRt:
        @staticmethod
        def set_device(device):
            set_device_calls.append(device)
            return 0

    fake_acl = types.SimpleNamespace(rt=_FakeRt())
    monkeypatch.setitem(sys.modules, "acl", fake_acl)
    original_resolve = Path.resolve

    def _resolve(path, *args, **kwargs):
        resolve_calls.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _resolve)

    def _register(*, kernel_path, fn_name, kernel_mode):
        calls.append((kernel_path, fn_name, kernel_mode))
        sequence = len(calls)
        return ascend_runtime._LoadedKernel(
            100 + sequence, 200 + sequence, kernel_path
        )

    monkeypatch.setattr(ascend_runtime, "_register_kernel_binary", _register)
    kwargs = {
        "name": "kernel aiv",
        "kernel_path": tmp_path / "kernel.o",
        "device": 0,
    }

    assert ascend_runtime.load_binary(**kwargs) == (101, 201)
    assert ascend_runtime.load_binary(**kwargs) == (102, 202)
    assert len(calls) == 2
    assert set_device_calls == [0, 0]
    assert resolve_calls == [tmp_path / "kernel.o", tmp_path / "kernel.o"]


def _runtime_cached_artifact(tmp_path, *, cache_key: str) -> execution.TlaKernelArtifact:
    return execution.TlaKernelArtifact(
        cache_key=cache_key,
        cache_dir=tmp_path,
        tlair_mlir=(
            'module { "tla.func"() ({}) '
            '{function_type = () -> (), sym_name = "kernel"} : () -> () }'
        ),
        lowered_llvm="module {}",
        entrypoint="kernel",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / f"{cache_key}.o",
        kernel_abi=_kernel_abi(total_size=0),
    )


def test_launch_does_not_repeat_static_artifact_analysis(
    monkeypatch, tmp_path
) -> None:
    artifact = _runtime_cached_artifact(tmp_path, cache_key="validated")

    def _unexpected_validation(*_args, **_kwargs):
        raise AssertionError("launch repeated static Artifact analysis")

    monkeypatch.setattr(execution, "_validate_kernel_abi_layout", _unexpected_validation)

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[],
        block_num=1,
    )

    assert plan.payload == b""


def test_artifact_runtime_handles_are_cached_per_artifact(
    monkeypatch, tmp_path
) -> None:
    loads: list[tuple[str, int]] = []
    _install_fake_launch_context(monkeypatch, device=0, stream=90)

    def _load_binary(**kwargs):
        loads.append((kwargs["kernel_path"].name, kwargs["device"]))
        sequence = len(loads)
        return 100 + sequence, 200 + sequence

    monkeypatch.setattr(execution, "load_binary", _load_binary)
    monkeypatch.setattr(execution, "launch_kernel", lambda **_kwargs: None)

    artifact = _runtime_cached_artifact(tmp_path, cache_key="first")
    runtime = execution.TlaRuntimeOptions()
    first = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )
    repeated = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )
    assert (first.module_handle, first.function_handle) == (101, 201)
    assert (repeated.module_handle, repeated.function_handle) == (101, 201)
    assert artifact._runtime_handle == (101, 201)

    other = _runtime_cached_artifact(tmp_path, cache_key="second")
    other_result = execution.execute_kernel(
        other,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={"block_num": 1},
    )
    assert (other_result.module_handle, other_result.function_handle) == (102, 202)
    assert other._runtime_handle == (102, 202)
    assert loads == [("first.o", 0), ("second.o", 0)]


def test_artifact_runtime_handle_load_is_serialized(monkeypatch, tmp_path) -> None:
    artifact = _runtime_cached_artifact(tmp_path, cache_key="concurrent")
    _install_fake_launch_context(monkeypatch, device=3, stream=99)
    entered = threading.Event()
    release = threading.Event()
    loads: list[int] = []
    failures: list[BaseException] = []

    def _load_binary(**_kwargs):
        loads.append(1)
        entered.set()
        assert release.wait(timeout=5)
        return 101, 202

    monkeypatch.setattr(execution, "load_binary", _load_binary)
    monkeypatch.setattr(execution, "launch_kernel", lambda **_kwargs: None)

    def _execute() -> None:
        try:
            execution.execute_kernel(
                artifact,
                runtime=execution.TlaRuntimeOptions(),
                launch_args=[],
                launch_kwargs={"block_num": 1},
            )
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=_execute)
    second = threading.Thread(target=_execute)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not failures
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(loads) == 1
    assert artifact._runtime_handle == (101, 202)
