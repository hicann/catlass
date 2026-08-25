from __future__ import annotations

import inspect
from pathlib import Path
import importlib.util

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.runtime import TlaCoreAPIError
from catlass import execution
from catlass.base_dsl.arch import get_kernel_target


EXAMPLE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "end_to_end"
    / "extern_op"
    / "extern_vecadd.py"
)
DUAL_CORE_EXAMPLE_PATH = EXAMPLE_PATH.with_name("extern_dual_core.py")
EXTERN_SOURCE_CODE = 'extern "C" void tla_user_gm_to_ub_f32() {}\n'
DUAL_CORE_SOURCE_CODE = r"""
#include <cstdint>

extern "C" {
[aicore] __attribute__((noinline)) void tla_user_dual_core(int32_t value) {
  (void)value;
}
}
"""


def _gm_to_ub(source_code: str = EXTERN_SOURCE_CODE):
    @tla.extern(
        source=source_code,
        name="tla_user_gm_to_ub_f32",
    )
    def gm_to_ub(
        gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
        ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
        ele_num: tla.Int32,
    ) -> None: ...

    return gm_to_ub


GM_TO_UB = _gm_to_ub()


@tla.extern(source=EXTERN_SOURCE_CODE, name="tla_user_other_op")
def OTHER_EXTERN(
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
) -> None: ...


@tla.extern(source=DUAL_CORE_SOURCE_CODE)
def tla_user_dual_core(value: tla.Int32) -> None: ...


@tla.kernel
def extern_load_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        GM_TO_UB(gm.ptr, ub, 256)


@tla.kernel
def two_extern_ops_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        GM_TO_UB(gm.ptr, ub, 256)
        OTHER_EXTERN(gm.ptr)


@tla.kernel
def wrong_address_space_kernel(gm: tla.Tensor) -> None:
    with tla.vector():
        GM_TO_UB(gm.ptr, gm.ptr, 256)


@tla.kernel
def tensor_argument_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        GM_TO_UB(gm, ub, 256)


@tla.kernel
def outside_vector_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    GM_TO_UB(gm.ptr, ub, 256)


@tla.kernel
def extern_cube_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_dual_core(value)


@tla.kernel
def extern_mix_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_dual_core(value)
    with tla.vector():
        tla_user_dual_core(value)


@tla.kernel
def inside_vec_func_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            GM_TO_UB(gm.ptr, ub, 256)


@tla.kernel
def wrong_scalar_type_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        GM_TO_UB(gm.ptr, ub, tla.Int64(256))


@tla.kernel
def wrong_argument_count_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        GM_TO_UB(gm.ptr, ub)


@tla.kernel
def unused_extern_kernel(gm: tla.Tensor) -> None:
    pass


def _fake_gm_tensor():
    return tla.make_fake_tensor(
        tla.Float32,
        (256,),
        (1,),
        origin_shape=(256,),
        coord=(0,),
        layout_tag=tla.arch.RowMajor,
    )


def _lower_kernel(kernel, *type_args):
    return BaseDSL()._lower(
        kernel.fn,
        kind="kernel",
        options={},
        type_args=type_args,
        location=kernel.decorator_location,
    )


def test_extern_decorator_describes_inline_source_without_compiling(
    monkeypatch,
) -> None:
    source_code = 'extern "C" void op() {}\n'
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: calls.append(args))

    @tla.extern(source=source_code)
    def op(value: tla.Int32) -> None:
        raise AssertionError("the declaration body must not execute")

    assert op.source == source_code
    assert op.symbol == "op"
    assert op.arg_types == (tla.Int32,)
    assert callable(op)
    assert calls == []


def test_pointer_subscription_returns_public_typed_pointer() -> None:
    pointer_type = tla.Pointer[tla.Float32, tla.AddressSpace.gm]

    assert isinstance(pointer_type, tla.TypedPointer)
    assert pointer_type.dtype is tla.Float32
    assert pointer_type.space is tla.AddressSpace.gm


def test_pointer_subscription_requires_dtype_and_memory_space() -> None:
    with pytest.raises(TypeError, match="expects \\(dtype, memory_space\\)"):
        tla.Pointer[tla.Float32]


def test_extern_rejects_non_string_or_empty_source(tmp_path) -> None:
    with pytest.raises(TypeError, match="source must be str"):
        tla.extern(source=tmp_path / "op.cpp")
    with pytest.raises(ValueError, match="source must not be empty"):
        tla.extern(source="  \n")


def test_extern_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="name must be a C identifier"):
        tla.extern(source=EXTERN_SOURCE_CODE, name="op-name")


def test_extern_rejects_invalid_function_signature() -> None:
    def missing_return(value: tla.Int32): ...

    with pytest.raises(TypeError, match="return annotation.*is missing"):
        tla.extern(source=EXTERN_SOURCE_CODE)(missing_return)

    def variadic(*values: tla.Int32) -> None: ...

    with pytest.raises(TypeError, match="positional and fixed"):
        tla.extern(source=EXTERN_SOURCE_CODE)(variadic)

    def missing_parameter_annotation(value) -> None: ...

    with pytest.raises(TypeError, match="missing an annotation"):
        tla.extern(source=EXTERN_SOURCE_CODE)(missing_parameter_annotation)


def test_extern_frontend_emits_call_and_tracks_dependency() -> None:
    lowered = _lower_kernel(extern_load_kernel, _fake_gm_tensor())
    mlir = lowered.asm()
    assert "tla.call_extern" in mlir
    assert "@tla_user_gm_to_ub_f32" in mlir
    assert "!tla.ptr<f32, gm" in mlir
    assert "!tla.ptr<f32, ub" in mlir
    assert lowered.extern_function is GM_TO_UB
    assert lowered.extern_core_types == frozenset({"aiv"})


def test_extern_call_location_points_to_user_call() -> None:
    source_lines, first_line = inspect.getsourcelines(extern_load_kernel.fn)
    call_line = first_line + next(
        index for index, line in enumerate(source_lines) if "GM_TO_UB(" in line
    )

    lowered = _lower_kernel(extern_load_kernel, _fake_gm_tensor())
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            enable_debug_info=True,
            assume_verified=False,
        )

    assert f'test_extern_op.py":{call_line}:' in mlir


def test_extern_rejects_second_op_in_one_kernel() -> None:
    with pytest.raises(TlaCoreAPIError, match="at most one external function"):
        two_extern_ops_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_checks_pointer_address_space() -> None:
    with pytest.raises(TlaCoreAPIError, match="expects pointer.*ub"):
        wrong_address_space_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_rejects_tensor_and_requests_explicit_pointer() -> None:
    with pytest.raises(TlaCoreAPIError, match=r"Tensor; pass tensor\.ptr explicitly"):
        tensor_argument_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_checks_call_region() -> None:
    with pytest.raises(TlaCoreAPIError, match="exactly one.*vector.*cube"):
        outside_vector_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))
    with pytest.raises(TlaCoreAPIError, match="outside tla.vec.func"):
        inside_vec_func_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_infers_aic_and_mixed_call_targets() -> None:
    cube = _lower_kernel(extern_cube_kernel, tla.Int32(0))
    mixed = _lower_kernel(extern_mix_kernel, tla.Int32(0))

    assert cube.extern_core_types == frozenset({"aic"})
    assert mixed.extern_core_types == frozenset({"aic", "aiv"})
    assert mixed.asm().count("tla_user_dual_core") == 2


def test_extern_checks_argument_count_and_scalar_type() -> None:
    with pytest.raises(TlaCoreAPIError, match="expects 3 arguments, got 2"):
        wrong_argument_count_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))
    with pytest.raises(TlaCoreAPIError, match="expects Int32, got Int64"):
        wrong_scalar_type_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_unused_extern_declaration_is_not_a_kernel_dependency() -> None:
    lowered = _lower_kernel(unused_extern_kernel, _fake_gm_tensor())

    assert lowered.extern_function is None
    assert lowered.extern_core_types == frozenset()


def test_ascendc_compile_command_reuses_bitcode_backend(tmp_path, monkeypatch) -> None:
    compiler = tmp_path / "ccec"
    compiler.write_text("")
    commands = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"BC")

    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(execution, "_run_checked", fake_run)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))

    result = execution._compile_ascendc_extern_function(
        GM_TO_UB,
        artifact_dir=tmp_path,
        targets=(get_kernel_target(target_arch="c310", core_type="aiv"),),
    )

    assert [path.name for path in result] == ["extern.aiv.c310.bc"]
    command = commands[0]
    assert "-emit-llvm" in command
    assert "--cce-aicore-arch=dav-c310-vec" in command
    source = tmp_path / "extern.cpp"
    assert str(source) in command
    assert source.read_text() == EXTERN_SOURCE_CODE


def test_ascendc_multi_target_compile_uses_one_ccec_invocation_per_core(
    tmp_path, monkeypatch
) -> None:
    compiler = tmp_path / "ccec"
    compiler.write_text("")
    commands = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"BC")

    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(execution, "_run_checked", fake_run)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    targets = execution._resolve_extern_targets(
        tla_user_dual_core,
        extern_core_types={"aic", "aiv"},
        target_arch="c310",
    )

    result = execution._compile_ascendc_extern_function(
        tla_user_dual_core,
        artifact_dir=tmp_path,
        targets=targets,
    )

    assert [path.name for path in result] == [
        "extern.aic.c310.bc",
        "extern.aiv.c310.bc",
    ]
    assert len(commands) == 2
    assert "--cce-aicore-arch=dav-c310-cube" in commands[0]
    assert "--cce-aicore-arch=dav-c310-vec" in commands[1]
    assert commands[0][commands[0].index("-o") + 1].endswith(
        "extern.aic.c310.bc"
    )
    assert commands[1][commands[1].index("-o") + 1].endswith(
        "extern.aiv.c310.bc"
    )


def _cache_key_kwargs(tmp_path, target):
    return {
        "tlair_mlir": "module { func.func @kernel() }",
        "entrypoint": "kernel",
        "runtime": execution.TlaRuntimeOptions(arch_scope="aiv.c310"),
        "compiler_bridge_path": None,
        "hivmc": tmp_path / "hivmc-a5",
        "target": target,
    }


def test_extern_compile_identity_participates_in_kernel_cache_key(
    tmp_path, monkeypatch
) -> None:
    compiler = (tmp_path / "toolchain" / "bin" / "ccec").resolve()
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"ccec")
    compiler_version = ["first"]
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(
        execution,
        "_tool_version",
        lambda path: compiler_version[0] if path == compiler else "hivmc-version",
    )
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_include_dirs",
        lambda _ascend_home: [tmp_path / "include"],
    )
    kwargs = _cache_key_kwargs(tmp_path, target)
    kwargs.update(extern_function=GM_TO_UB, extern_targets=(target,))
    first = execution._cache_key(**kwargs)

    compiler_version[0] = "second"

    assert execution._cache_key(**kwargs) != first


def test_extern_source_participates_in_kernel_cache_key(tmp_path, monkeypatch) -> None:
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "version")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_extern_compile_identity",
        lambda: {"revision": "same"},
    )
    kwargs = _cache_key_kwargs(tmp_path, target)

    first = execution._cache_key(
        **kwargs,
        extern_function=_gm_to_ub('extern "C" void op() {}\n'),
        extern_targets=(target,),
    )
    second = execution._cache_key(
        **kwargs,
        extern_function=_gm_to_ub('extern "C" void op() { /* changed */ }\n'),
        extern_targets=(target,),
    )

    assert second != first


def _load_vecadd_example():
    spec = importlib.util.spec_from_file_location("extern_vecadd_example", EXAMPLE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dual_core_example():
    spec = importlib.util.spec_from_file_location(
        "extern_dual_core_example", DUAL_CORE_EXAMPLE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extern_vecadd_frontend_uses_custom_load_and_tla_compute() -> None:
    example = _load_vecadd_example()
    tensor = _fake_gm_tensor()
    lowered = _lower_kernel(example.extern_vecadd, tensor, tensor, tensor)
    mlir = lowered.asm()
    assert mlir.count("tla.call_extern") == 2
    assert "tla.add" in mlir
    assert "tla.copy" in mlir
    assert lowered.extern_function is example.tla_user_gm_to_ub_f32


def test_extern_dual_core_example_calls_one_op_from_aic_and_aiv() -> None:
    example = _load_dual_core_example()
    tensor = tla.make_fake_tensor(
        tla.Int32,
        (example.RESULT_SIZE,),
        (1,),
        origin_shape=(example.RESULT_SIZE,),
        coord=(0,),
        layout_tag=tla.arch.RowMajor,
    )
    lowered = _lower_kernel(example.extern_dual_core, tensor)
    mlir = lowered.asm()

    assert mlir.count("tla.call_extern") == 2
    assert mlir.count("@tla_user_store_i32") == 2
    assert lowered.extern_function is example.tla_user_store_i32
    assert lowered.extern_core_types == frozenset({"aic", "aiv"})
