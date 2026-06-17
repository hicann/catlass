from __future__ import annotations

from pathlib import Path
import struct

import pytest

tla = pytest.importorskip("catlass", exc_type=ImportError)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
base_dsl_mod = pytest.importorskip("catlass.base_dsl", exc_type=ImportError)
compiler_bridge = pytest.importorskip("catlass.compiler_bridge", exc_type=ImportError)


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
        mlir_print_ir_before=["tla-func-to-hacc"],
        mlir_print_ir_after=["tla-lower-to-std"],
        mlir_print_ir_before_all=True,
    )

    assert lowered.lowered_mlir == "module { func.func @zero_arg_kernel() }\n"
    assert lowered.pass_ir_dump == "after-pass-dump"
    assert calls == [
        (
            module,
            ["tla-func-to-hacc"],
            ["tla-lower-to-std"],
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


def test_runtime_options_for_offline_mlir_preserves_hivmc_args() -> None:
    runtime = execution.TlaRuntimeOptions()

    updated = execution._runtime_options_for_offline_ascendnpuir_mlir(
        runtime,
        'module { func.func @kernel() { vector.transfer_read %arg0[%c0], %cst : memref<1xf32>, vector<1xf32> } }',
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
            core_type="aic",
            kernel_mode="aic",
            hivmc_args=("--extra-flag",)
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


def test_pack_launch_args_uses_uint64_slots_for_typed_scalars() -> None:
    mlir = (
        'module { "tla.func"() ({}) '
        '{function_type = (i32, f32) -> (), sym_name = "kernel"} : () -> () }'
    )

    payload = execution._pack_launch_args([tla.Int32(-7), tla.Float32(1.5)], mlir)

    assert payload == struct.pack("<QQ", 0xFFFFFFF9, 0x3FC00000)


def test_pack_launch_args_uses_uint64_slots_for_mixed_scalar_and_pointer() -> None:
    class _Ptr:
        def __c_pointers__(self):
            return [0x123456789ABCDEF0]

    mlir = (
        'module { "tla.func"() ({}) '
        '{function_type = (i32, !tla.ptr<f32, gm, 4>) -> (), sym_name = "kernel"} '
        ': () -> () }'
    )

    payload = execution._pack_launch_args([tla.Int32(5), _Ptr()], mlir)

    assert payload == struct.pack("<QQ", 5, 0x123456789ABCDEF0)


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
        },
    ) in launches


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
        },
    ) in launches
