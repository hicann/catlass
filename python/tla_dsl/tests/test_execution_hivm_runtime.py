from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import importlib.util
import math
import os
import struct
import sys
import types

import pytest

tla = pytest.importorskip("catlass", exc_type=ImportError)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
base_dsl_mod = pytest.importorskip("catlass.base_dsl", exc_type=ImportError)
compiler_bridge = pytest.importorskip("catlass.compiler_bridge", exc_type=ImportError)


def _load_debug_print_example(*, mixed: bool = False):
    fake_catlass = types.ModuleType("catlass")
    fake_catlass.kernel = lambda function: function
    if mixed:
        fake_catlass.Int32 = int
        fake_catlass.Float32 = float
    previous = sys.modules.get("catlass")
    sys.modules["catlass"] = fake_catlass
    try:
        filename = "debug_print_mixed.py" if mixed else "debug_print.py"
        path = Path(__file__).parents[1] / "examples/end_to_end/debug_print" / filename
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules["catlass"]
        else:
            sys.modules["catlass"] = previous


def _load_print_tensor_example():
    fake_catlass = types.ModuleType("catlass")
    fake_catlass.kernel = lambda function: function
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


def test_print_tensor_output_verifies_and_formats_canonical_record() -> None:
    example = _load_print_tensor_example()
    stable = example._format_record(example.EXPECTED_VALUES)

    assert example._verify_public_output(stable) == (
        "tla.print dtype=float32 shape=[8,4] count=16 "
        "values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
        "9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]"
    )


def test_print_tensor_output_formats_ub_physical_copy_shape() -> None:
    example = _load_print_tensor_example()
    stable = example._format_record(example.EXPECTED_VALUES, shape=example.UB_SHAPE)

    assert example.UB_SHAPE == (4, 8)
    assert math.prod(example.UB_SHAPE) == 32
    assert example.EXPECTED_VALUES == [float(value) for value in range(16)]
    assert example._verify_public_output(stable, shape=example.UB_SHAPE) == (
        "tla.print dtype=float32 shape=[4,8] count=16 "
        "values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "
        "9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]"
    )


def test_print_tensor_aic_example_selects_cube_kernel() -> None:
    example = _load_print_tensor_example()
    args = example._parser().parse_args(["--arch-scope", "aic.c310"])

    assert example._kernel(args).__name__ == "print_tensor_aic_kernel"


@pytest.mark.parametrize(
    ("case", "kernel_name"),
    (
        ("base", "print_tensor_ub_base_kernel"),
        ("aligned-offset", "print_tensor_ub_aligned_offset_kernel"),
    ),
)
def test_print_tensor_ub_example_selects_aiv_case(
    case: str, kernel_name: str
) -> None:
    example = _load_print_tensor_example()
    args = example._parser().parse_args(
        ["--storage", "ub", "--case", case, "--arch-scope", "aiv.c310"]
    )

    assert example._kernel(args).__name__ == kernel_name


def test_prepare_hivmc_input_selects_aic_print_tensor_helper(
    monkeypatch, tmp_path
) -> None:
    mlir_path = tmp_path / "lowered.mlir"
    mlir_path.write_text(
        "module { func.func @kernel(%workspace: i64 "
        "{tla.print_tensor.workspace}) }"
    )
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    helper_bc = tmp_path / "bc" / "Cube" / "print_tensor.aic.c310.bc"
    template_bc.write_bytes(b"bc")
    helper_bc.parent.mkdir(parents=True)
    helper_bc.write_bytes(b"bc")
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))
    monkeypatch.setattr(execution, "_mlir_build_dirs", lambda: [tmp_path])

    compiler_input, selected = execution._create_stamped_hivmc_input(
        mlir_path,
        execution.TlaRuntimeOptions(
            core_type="aic", kernel_mode="aic", arch_scope="aic.c310"
        ),
    )

    assert compiler_input != mlir_path
    assert selected == f"{template_bc},{helper_bc}"
    assert "hivm.aic_bitcode" in compiler_input.read_text()


@pytest.mark.parametrize(
    "output",
    (
        "",
        "DumpTensor: data_type=float32 position=GM dump_size=16 [bad]",
        "DumpTensor: data_type=float32 position=GM dump_size=16 [0.0]",
        "\n".join(
            [
                "DumpTensor: data_type=float32 position=GM dump_size=16 "
                "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"
            ]
            * 2
        ),
    ),
)
def test_print_tensor_output_rejects_missing_malformed_or_duplicate_record(
    output: str,
) -> None:
    example = _load_print_tensor_example()
    with pytest.raises(execution.TlaExecutionError, match="initialization or decoding"):
        example._verify_public_output(output)


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


def test_debug_print_output_rejects_duplicate_multiblock_records() -> None:
    example = _load_debug_print_example()

    with pytest.raises(RuntimeError, match="distinct blocks"):
        example._verify_debug_output(
            "\n".join("TLA printf: core=0 block=0 v=1.250000" for _ in range(4)),
            dtype="f32",
            expected_value="1.250000",
            expect_count=4,
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

    assert example._kernel(args).__name__ == expected_kernel


def test_debug_print_mixed_defaults_to_both_regions() -> None:
    example = _load_debug_print_example(mixed=True)

    assert example._parser().parse_args([]).print_region == "both"


@pytest.mark.parametrize(
    ("print_region", "output"),
    (
        ("vector", "\n".join(
            (
                "TLA printf: core=1 block=0 v=1.250000",
                "TLA printf: core=1 block=0 v=1.250000",
            )
        )),
        ("vector", "\n".join(
            (
                "TLA printf: core=1 block=0 v=1.250000",
                "TLA printf: core=2 block=1 v=1.250000",
            )
        )),
        ("cube", "\n".join(
            (
                "TLA printf: core=0 block=0 x=-36",
            )
        )),
        ("both", "\n".join(
            (
                "TLA printf: core=1 block=0 v=1.250000",
                "TLA printf: core=2 block=0 v=1.250000",
            )
        )),
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
    ("dtype", "expected_kernel"),
    (
        ("i32", "debug_print_aic_kernel"),
    ),
)
def test_debug_print_aic_example_selects_cube_kernel(
    dtype: str, expected_kernel: str
) -> None:
    example = _load_debug_print_example()
    args = example._parser().parse_args(
        ["--arch-scope", "aic.c310", "--dtype", dtype]
    )

    assert example._kernel(args).__name__ == expected_kernel


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
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    bridge_path.write_text("")
    hivm_compile.write_text("")
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
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda _x: hivm_compile)
    monkeypatch.setattr(execution, "_tool_version", lambda _x: "test-version")
    monkeypatch.setattr(
        execution,
        "lower_tlair_module_to_mlir",
        lambda module, **_kwargs: compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n"
        ),
    )
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))

    recorded: list[tuple[str, list[str]]] = []

    def fake_run_checked(cmd, *, label, cwd, stdin_text=None):
        assert stdin_text is None
        recorded.append((label, list(cmd)))
        if label == "hivmc-a5":
            assert "hivm.aic_bitcode" not in Path(cmd[1]).read_text()
            Path(cwd, "kernel.o").write_bytes(b"obj")

    monkeypatch.setattr(execution, "_run_checked", fake_run_checked)

    artifact = tla.compile(
        _zero_arg_tla_kernel,
        cache=False,
        cache_dir=tmp_path / "cache",
        target_arch="c310",
        core_type="aic",
        kernel_mode="aic",
        arch_scope="aic.c310",
    )

    assert artifact.compiler_bridge_path == bridge_path
    assert artifact.lowered_llvm == "module { func.func @zero_arg_kernel() }\n"
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
                f"--link-aicore-bitcode={template_bc}",
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
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))

    compiler_input, template_bitcode = (
        execution._create_stamped_hivmc_input(
            mlir_path,
            execution.TlaRuntimeOptions(core_type="aic", kernel_mode="aic"),
        )
    )

    assert compiler_input != mlir_path
    assert template_bitcode == str(template_bc)
    assert "hivm.aic_bitcode" in compiler_input.read_text()
    assert "hivm.aic_bitcode" not in mlir_path.read_text()


def test_generated_kernel_bridge_lowers_live_module(monkeypatch, tmp_path) -> None:
    tlair_mlir = "module {\n  tla.func @zero_arg_kernel() { tla.return }\n}"
    lowered_module = object()
    hivm_compile = tmp_path / "hivmc-a5"
    template_bc = tmp_path / "meta_op.aiv.c310.bc"
    hivm_compile.write_text("")
    template_bc.write_bytes(b"bc")

    monkeypatch.setattr(
        base_dsl_mod.BaseDSL,
        "_lower",
        lambda *_a, **_k: _FakeLowered(tlair_mlir, module=lowered_module),
    )
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: None)
    monkeypatch.setattr(execution, "_resolve_hivmc_a5", lambda _x: hivm_compile)
    monkeypatch.setattr(execution, "_tool_version", lambda _x: "test-version")
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))

    bridge_calls: list[tuple[object, dict[str, object]]] = []

    def fake_lower_tlair_module_to_mlir(module, **kwargs):
        bridge_calls.append((module, kwargs))
        return compiler_bridge.TlaLoweringResult(
            "module { func.func @zero_arg_kernel() }\n"
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


def test_runtime_options_ignore_removed_target_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("TLA_DSL_TARGET_ARCH", "c220")
    monkeypatch.setenv("TLA_DSL_CORE_TYPE", "aic")
    monkeypatch.setenv("TLA_DSL_ARCH_SCOPE", "aic.c220")
    options = execution.runtime_options_from_kwargs({})

    assert options.target_arch == "c310"
    assert options.core_type == "aiv"
    assert options.arch_scope == "aiv.c310"


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
        compiler_bridge.lower_tlair_module_to_mlir(
            module, mlir_print_ir_after_all=True
        )

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


def test_tla_compile_cli_preserves_ir_dump_on_failure(
    monkeypatch, tmp_path
) -> None:
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
    runtime = execution.TlaRuntimeOptions(
        mlir_print_ir_before=["tla-lower-func"],
        mlir_print_ir_after=["tla-finalize-memref"],
        mlir_print_ir_before_all=True,
        mlir_print_ir_after_all=True,
    )

    with pytest.raises(execution.TlaKernelCompileError) as exc_info:
        execution._run_tla_compile_cli_to_mlir(
            tla_compile=tla_compile,
            tlair_mlir="module { tla.func @k() { tla.return } }\n",
            mlir_path=lowered_path,
            runtime=runtime,
        )

    assert exc_info.value.pass_ir_dump == pass_ir_dump
    assert "<captured in pass IR dump>" in str(exc_info.value)
    assert calls[0][0][-4:] == [
        "--mlir-print-ir-before=tla-lower-func",
        "--mlir-print-ir-after=tla-finalize-memref",
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


def test_runtime_options_from_lowered_mlir_preserves_hivmc_args() -> None:
    runtime = execution.TlaRuntimeOptions()

    updated = execution._runtime_options_from_lowered_mlir(
        runtime,
        "module { func.func @kernel() { vector.transfer_read %arg0[%c0], %cst : memref<1xf32>, vector<1xf32> } }",
    )

    assert updated.hivmc_args == ()


def test_build_hivmc_a5_command_links_template_bitcode_for_aic(
    monkeypatch, tmp_path
) -> None:
    compiler = tmp_path / "hivmc-a5"
    mlir_path = tmp_path / "kernel.mlir"
    kernel_path = tmp_path / "kernel.o"
    template_bc = tmp_path / "meta_op.aic.c310.bc"
    template_bc.write_bytes(b"bc")
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))

    command = execution._build_hivmc_a5_command(
        compiler=compiler,
        mlir_path=mlir_path,
        kernel_path=kernel_path,
        runtime=execution.TlaRuntimeOptions(
            core_type="aic", kernel_mode="aic", hivmc_args=("--extra-flag",)
        ),
    )

    assert command == [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--disable-ffts",
        "--enable-hivm-compile=False",
        f"--link-aicore-bitcode={template_bc}",
        "-o",
        str(kernel_path),
        "--extra-flag",
    ]


def test_build_hivmc_a5_command_links_template_bitcode_for_aiv(
    monkeypatch, tmp_path
) -> None:
    compiler = tmp_path / "hivmc-a5"
    mlir_path = tmp_path / "kernel.mlir"
    kernel_path = tmp_path / "kernel.o"
    template_bc = tmp_path / "meta_op.aiv.c310.bc"
    template_bc.write_bytes(b"bc")
    monkeypatch.setenv("TLA_DSL_HIVM_TEMPLATE_BC", str(template_bc))

    command = execution._build_hivmc_a5_command(
        compiler=compiler,
        mlir_path=mlir_path,
        kernel_path=kernel_path,
        runtime=execution.TlaRuntimeOptions(core_type="aiv", kernel_mode="aiv"),
    )

    assert command == [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--disable-ffts",
        "--enable-hivm-compile=False",
        f"--link-aicore-bitcode={template_bc}",
        "-o",
        str(kernel_path),
    ]


def test_ascend_loader_forwards_native_width_scalar_payload() -> None:
    payload = struct.pack("<Qi", 0x123456789ABCDEF0, 20)
    launches: list[tuple[bytes, int, int, int]] = []

    class _FakeRuntimeWrapper:
        def tla_runtime_launch_kernel(
            self,
            _function,
            _stream,
            _grid_x,
            _grid_y,
            _grid_z,
            args,
            arg_size,
            expects_debug_fifo,
            expects_print_tensor,
        ) -> int:
            size = int(arg_size)
            launches.append(
                (
                    ctypes.string_at(args, size),
                    size,
                    int(expects_debug_fifo),
                    int(expects_print_tensor),
                )
            )
            return 0

    loader = execution._AscendLoader()
    loader._module = _FakeRuntimeWrapper()
    loader.launch_with_args(
        function=1,
        stream=2,
        grid_x=1,
        grid_y=1,
        grid_z=1,
        args=payload,
        expects_debug_fifo=False,
        expects_print_tensor=False,
    )

    assert launches == [(payload, 12, 0, 0)]


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
                8
                if abi_type == "index"
                else max(1, int(type(arg).width) // 8)
            )
            arguments.append(
                ("scalar", abi_type, abi_type, offset, storage_size, 4)
            )
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
    length: int | None = None,
    storage: str = "gm",
):
    rendered_shape = ", ".join(str(extent) for extent in shape)
    print_length = min(16, math.prod(shape)) if length is None else length
    return execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir=(
            'module { "tla.print_tensor"(%value) '
            f"<{{length = {print_length} : i64, "
            f"shape = array<i64: {rendered_shape}>}}> : "
            "(!tla.tensor<!tla.layout<!tla.shape<4,4>, !tla.stride<4,1>, "
            "!tla.shape<4,4>, row_major>, !tla.coord<0,0>, "
            f"!tla.ptr<f32, {storage}, 32>>) -> () }}"
        ),
        lowered_llvm=(
            f"module {{ func.func @{entrypoint}(%workspace: i64 "
            "{tla.print_tensor.workspace}, %value: i64) }"
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


def test_print_tensor_workspace_preserves_user_argument_and_uses_v1_marker(
    tmp_path,
) -> None:
    artifact = _print_tensor_artifact(tmp_path)

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        grid=(1, 1, 1),
    )

    assert plan.payload == struct.pack(
        "<QQ", 0x1000, execution._PRINT_TENSOR_WORKSPACE_SENTINEL
    )
    assert plan.expects_print_tensor is True


def _install_print_tensor_loader(monkeypatch, output: str) -> None:
    class _FakeLoader:
        def get_current_device(self) -> int:
            return 1

        def get_current_stream(self, device: int) -> int:
            assert device == 1
            return 99

        def load_binary(self, **kwargs):
            del kwargs
            return (11, 12)

        def launch_with_args(self, **kwargs) -> None:
            assert kwargs["expects_print_tensor"] is True
            os.write(1, output.encode())

    monkeypatch.setattr(execution, "_AscendLoader", _FakeLoader)


def test_execute_kernel_decodes_and_formats_native_print_tensor_for_ordinary_call(
    monkeypatch, tmp_path, capfd
) -> None:
    _install_print_tensor_loader(
        monkeypatch,
        "CANN address=0xdeadbeef\n"
        "DumpTensor: data_type=float32 position=GM dump_size=4 [0, 1.5, -2, 3]\n",
    )

    execution.execute_kernel(
        _print_tensor_artifact(tmp_path, shape=(2, 2)),
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000)],
        launch_kwargs={},
    )

    assert capfd.readouterr().out == (
        "tla.print dtype=float32 shape=[2,2] count=4 "
        "values=[0.0, 1.5, -2.0, 3.0]\n"
    )


@pytest.mark.parametrize(
    "output",
    (
        "",
        "DumpTensor: data_type=float32 position=GM dump_size=4 [bad]\n",
        "\n".join(
            ["DumpTensor: data_type=float32 position=GM dump_size=4 [0, 1, 2, 3]"]
            * 2
        ),
    ),
    ids=["missing", "malformed", "duplicate"],
)
def test_execute_kernel_rejects_invalid_native_print_tensor_for_ordinary_call(
    monkeypatch, tmp_path, output
) -> None:
    _install_print_tensor_loader(monkeypatch, output)

    with pytest.raises(execution.TlaExecutionError, match="initialization or decoding"):
        execution.execute_kernel(
            _print_tensor_artifact(tmp_path, shape=(2, 2)),
            runtime=execution.TlaRuntimeOptions(),
            launch_args=[_TypedPointer(0x1000)],
            launch_kwargs={},
        )


def test_print_tensor_metadata_requires_one_static_shape() -> None:
    with pytest.raises(execution.TlaExecutionError, match="static shape metadata"):
        execution._print_tensor_static_metadata("module { func.func @kernel() }")


def test_print_tensor_metadata_reads_generic_tlair_shape() -> None:
    mlir = (
        '"tla.print_tensor"(%value) '
        "<{length = 4 : i64, shape = array<i64: 2, 3>}> : "
        "(!tla.tensor<!tla.layout<!tla.shape<2,3>, !tla.stride<3,1>, "
        "!tla.shape<2,3>, row_major>, !tla.coord<0,0>, "
        "!tla.ptr<f32, gm, 4>>) -> ()"
    )

    assert execution._print_tensor_static_metadata(mlir) == ((2, 3), 4, "GM")


def test_print_tensor_metadata_reads_ub_storage() -> None:
    mlir = (
        '"tla.print_tensor"(%value) '
        "<{length = 4 : i64, shape = array<i64: 2, 3>}> : "
        "(!tla.tensor<!tla.layout<!tla.shape<2,3>, !tla.stride<3,1>, "
        "!tla.shape<2,3>, row_major>, !tla.coord<0,0>, "
        "!tla.ptr<f32, ub, 32>>) -> ()"
    )

    assert execution._print_tensor_static_metadata(mlir) == ((2, 3), 4, "UB")


def test_print_tensor_decoder_rejects_wrong_storage() -> None:
    with pytest.raises(execution.TlaExecutionError, match="expected position=UB"):
        execution._decode_native_print_tensor_output(
            "DumpTensor: data_type=float32 position=GM dump_size=4 [0, 1, 2, 3]",
            count=4,
            position="UB",
        )


def test_print_tensor_decoder_rejects_extra_l1_record() -> None:
    output = "\n".join(
        (
            "DumpTensor: data_type=float32 position=UB dump_size=4 [0, 1, 2, 3]",
            "DumpTensor: data_type=float32 position=L1 dump_size=4 [0, 1, 2, 3]",
        )
    )
    with pytest.raises(execution.TlaExecutionError, match="exactly one record"):
        execution._decode_native_print_tensor_output(
            output,
            count=4,
            position="UB",
        )


@pytest.mark.parametrize(
    ("runtime", "grid", "match"),
    (
        (execution.TlaRuntimeOptions(), (2, 1, 1), "single-block"),
        (
            execution.TlaRuntimeOptions(kernel_mode="mix"),
            (1, 1, 1),
            "mixed-core",
        ),
    ),
)
def test_print_tensor_launch_rejects_unsupported_grid_or_mixed_mode(
    tmp_path, runtime, grid, match
) -> None:
    with pytest.raises(execution.TlaExecutionError, match=match):
        execution._build_kernel_launch_plan(
            artifact=_print_tensor_artifact(tmp_path),
            runtime=runtime,
            launch_args=[_TypedPointer(0x1000)],
            grid=grid,
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
        grid=(1, 1, 1),
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
        grid=(1, 1, 1),
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
    assert execution._cache_key(
        tlair_mlir="module { func.func @kernel() }",
        entrypoint="kernel",
        runtime=runtime,
        compiler_bridge_path=None,
        hivmc=hivmc,
        target=target,
    ) != plain_key


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
            'module { "tla.print_tensor"(%value) '
            "<{length = 16 : i64, shape = array<i64: 4, 4>}> "
            ": (!tla.tensor) -> () }"
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


@pytest.mark.parametrize(
    "manifest_revision", [None, "debug-print-workspace-i64-v0"]
)
def test_debug_print_workspace_abi_manifest_requires_current_revision(
    manifest_revision,
) -> None:
    manifest = {}
    if manifest_revision is not None:
        manifest["debug_print_workspace_abi_revision"] = manifest_revision

    assert not execution._cache_manifest_has_current_debug_print_workspace_abi(
        manifest
    )
    manifest["debug_print_workspace_abi_revision"] = (
        execution._DEBUG_PRINT_WORKSPACE_ABI_REVISION
    )
    assert execution._cache_manifest_has_current_debug_print_workspace_abi(manifest)


@pytest.mark.parametrize(
    "manifest_revision", [None, "print-tensor-workspace-i64-v0"]
)
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


def test_online_cache_key_serializes_kernel_abi_version(
    monkeypatch, tmp_path
) -> None:
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
    assert payloads[0]["cache_abi_version"] == 4


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
    layout = _kernel_abi(
        ("scalar", "i32", "i32", 0, 4, 4), total_size=8
    )

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
    layout = _kernel_abi(
        ("pointer", "pointer", "!llvm.ptr", 0, 8, 4), total_size=8
    )

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

    payload = execution._pack_launch_args(
        [tla.Int16(1), _Ptr(), tla.Int16(2)], layout
    )

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
        execution._validate_kernel_abi_layout(
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


def test_manifest_rejects_descriptor_decoder_returning_none(
    monkeypatch,
) -> None:
    monkeypatch.setattr(execution, "kernel_abi_from_dict", lambda _value: None)

    with pytest.raises(
        execution.TlaKernelCompileError,
        match="decoded to no layout",
    ):
        execution._kernel_abi_from_manifest(
            {"entrypoint": "kernel", "kernel_abi": {}}
        )


def test_pack_launch_args_rejects_multi_value_host_argument() -> None:
    class _TwoPointers:
        def __c_pointers__(self):
            return [0x1111, 0x2222]

    layout = _kernel_abi(
        ("pointer", "pointer", "memref<8xi32>", 0, 8, 4), total_size=8
    )
    with pytest.raises(execution.TlaUnsupportedAbiError, match="exactly one"):
        execution._pack_launch_args([_TwoPointers()], layout)


def test_pack_launch_args_rejects_pointer_storage_overflow() -> None:
    class _HugePointer:
        def __c_pointers__(self):
            return [1 << 64]

    layout = _kernel_abi(
        ("pointer", "pointer", "memref<8xi32>", 0, 8, 4), total_size=8
    )
    with pytest.raises(execution.TlaUnsupportedAbiError, match="fit"):
        execution._pack_launch_args([_HugePointer()], layout)


def test_ascend_loader_forwards_opaque_bytes_and_exact_byte_count() -> None:
    calls: list[tuple[bytes, int]] = []

    class _FakeLaunch:
        def __call__(self, *_args):
            size = int(getattr(_args[-3], "value", _args[-3]))
            calls.append((ctypes.string_at(_args[-4], size), size))
            return 0

    class _FakeModule:
        tla_runtime_launch_kernel = _FakeLaunch()

    loader = execution._AscendLoader()
    loader._module = _FakeModule()
    payload = bytes.fromhex("112233445566")

    loader.launch_with_args(
        function=1,
        stream=2,
        grid_x=3,
        grid_y=4,
        grid_z=5,
        args=payload,
        expects_debug_fifo=False,
        expects_print_tensor=False,
    )

    assert calls == [(payload, len(payload))]


def test_runtime_wrapper_c_abi_is_byte_oriented() -> None:
    source = (
        Path(execution.__file__).resolve().parents[1]
        / "csrc"
        / "mlir"
        / "lib"
        / "Tools"
        / "RuntimeWrapper.cpp"
    ).read_text()

    assert "const uint8_t *args, size_t arg_size" in source
    assert "std::vector<uint64_t> values" in source
    assert "std::memcpy(values.data(), args, arg_size)" in source
    assert "values.assign(args" not in source
    assert "rtKernelLaunch(function, block_num, args_array," in source
    assert "arg_size, nullptr, stream)" in source


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
        grid=(1, 1, 1),
    )

    assert plan.entrypoint == "basic_mixed"
    assert plan.kernel_mode == "mix"
    assert plan.grid == (1, 1, 1)
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
        grid=(5, 6, 7),
    )

    assert plan.entrypoint == "custom"
    assert plan.kernel_mode == "mix"
    assert plan.grid == (5, 6, 7)
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
        grid=(1, 1, 1),
    )

    sentinel = int.from_bytes(b"TLA_PRNT", byteorder="big")
    assert plan.entrypoint == "debug_mixed"
    assert plan.payload == struct.pack("<ffQ", 1.0, 0.25, sentinel)
    assert plan.expects_debug_fifo is True


def test_execute_kernel_uses_typed_launch_payload(monkeypatch, tmp_path) -> None:
    launches: list[tuple[str, object]] = []

    class _FakeLoader:
        def get_current_device(self) -> int:
            return 7

        def get_current_stream(self, device: int) -> int:
            assert device == 7
            return 99

        def load_binary(self, **kwargs):
            launches.append(("load", kwargs))
            return (11, 12)

        def launch_with_args(self, **kwargs) -> None:
            launches.append(("flat", kwargs))

    monkeypatch.setattr(execution, "_AscendLoader", _FakeLoader)

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
    runtime = execution.TlaRuntimeOptions(shared=3)

    result = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[tla.Int32(123)],
        launch_kwargs={},
    )

    assert result.module_handle == 11
    assert result.function_handle == 12
    assert (
        "flat",
        {
            "function": 12,
            "stream": 99,
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "args": struct.pack("<I4x", 123),
            "expects_debug_fifo": False,
            "expects_print_tensor": False,
        },
    ) in launches


def test_execute_kernel_conveys_debug_fifo_intent_to_loader(monkeypatch, tmp_path) -> None:
    launches: list[dict[str, object]] = []

    class _FakeLoader:
        def get_current_device(self) -> int:
            return 7

        def get_current_stream(self, device: int) -> int:
            assert device == 7
            return 99

        def load_binary(self, **kwargs):
            del kwargs
            return (11, 12)

        def launch_with_args(self, **kwargs) -> None:
            launches.append(kwargs)

    monkeypatch.setattr(execution, "_AscendLoader", _FakeLoader)
    artifact = _debug_print_artifact(tmp_path, entrypoint="debug")

    execution.execute_kernel(
        artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[tla.Int32(7)],
        launch_kwargs={},
    )

    assert launches == [
        {
            "function": 12,
            "stream": 99,
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "args": struct.pack(
                "<QQ", 7, int.from_bytes(b"TLA_PRNT", byteorder="big")
            ),
            "expects_debug_fifo": True,
            "expects_print_tensor": False,
        }
    ]


def test_execute_kernel_uses_empty_payload_for_zero_arg(monkeypatch, tmp_path) -> None:
    launches: list[tuple[str, object]] = []

    class _FakeLoader:
        def get_current_device(self) -> int:
            return 7

        def get_current_stream(self, device: int) -> int:
            assert device == 7
            return 99

        def load_binary(self, **kwargs):
            launches.append(("load", kwargs))
            return (11, 12)

        def launch_with_args(self, **kwargs) -> None:
            launches.append(("flat", kwargs))

    monkeypatch.setattr(execution, "_AscendLoader", _FakeLoader)

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
    runtime = execution.TlaRuntimeOptions(shared=3)

    result = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[],
        launch_kwargs={},
    )

    assert result.module_handle == 11
    assert result.function_handle == 12
    assert (
        "flat",
        {
            "function": 12,
            "stream": 99,
            "grid_x": 1,
            "grid_y": 1,
            "grid_z": 1,
            "args": b"",
            "expects_debug_fifo": False,
            "expects_print_tensor": False,
        },
    ) in launches
