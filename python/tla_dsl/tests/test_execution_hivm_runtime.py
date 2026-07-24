from __future__ import annotations

import ctypes
from pathlib import Path
import importlib.util
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
        execution._create_stamped_debug_print_hivmc_input(
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


@pytest.mark.parametrize(
    ("args", "expected"),
    (
        ([tla.Int32(-7), tla.Float32(1.5)], struct.pack("<if", -7, 1.5)),
        ([tla.Float32(1.0), tla.Float32(0.25)], struct.pack("<ff", 1.0, 0.25)),
        (
            [tla.Int16(-7), tla.Int64(9)],
            struct.pack("<h", -7) + b"\0" * 6 + struct.pack("<q", 9),
        ),
    ),
    ids=["i32-f32", "f32-f32", "i16-i64"],
)
def test_pack_launch_args_uses_native_scalar_layout(args, expected) -> None:
    assert execution._pack_launch_args(args) == expected


def test_pack_launch_args_aligns_pointer_after_scalar() -> None:
    class _Ptr:
        def __c_pointers__(self):
            return [0x123456789ABCDEF0]

    payload = execution._pack_launch_args([tla.Int32(5), _Ptr()])

    assert payload == struct.pack("<i4xQ", 5, 0x123456789ABCDEF0)


def test_ascend_loader_forwards_native_width_scalar_payload() -> None:
    payload = struct.pack("<Qi", 0x123456789ABCDEF0, 20)
    launches: list[tuple[bytes, int, int]] = []

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
        ) -> int:
            size = int(arg_size)
            launches.append(
                (ctypes.string_at(args, size), size, int(expects_debug_fifo))
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
    )

    assert launches == [(payload, 12, 0)]


class _TypedPointer:
    dtype = "f32"

    def __init__(self, pointer: int) -> None:
        self._pointer = pointer

    def __c_pointers__(self) -> list[int]:
        return [self._pointer]


def _debug_print_artifact(tmp_path, *, entrypoint: str = "debug"):
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
    artifact = _debug_print_artifact(tmp_path)

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
    artifact = execution.TlaKernelArtifact(
        cache_key="cache",
        cache_dir=tmp_path,
        tlair_mlir="module { func.func @plain() }",
        lowered_llvm="module { func.func @plain() }",
        entrypoint="plain",
        compiler_bridge_path=None,
        hivmc_path=tmp_path / "hivmc-a5",
        kernel_binary_path=tmp_path / "kernel.o",
    )

    plan = execution._build_kernel_launch_plan(
        artifact=artifact,
        runtime=execution.TlaRuntimeOptions(),
        launch_args=[_TypedPointer(0x1000), _TypedPointer(0x2000)],
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
        "<QQQQQ",
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
    )
    runtime = execution.TlaRuntimeOptions(shared=3)

    result = execution.execute_kernel(
        artifact,
        runtime=runtime,
        launch_args=[123],
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
            "args": struct.pack("<Q", 123),
            "expects_debug_fifo": False,
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
        },
    ) in launches
