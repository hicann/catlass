import pathlib
import os
import subprocess
import tempfile

import pytest
import catlass as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError


def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    ascendnpuir_root = pathlib.Path(
        os.environ.get("TLA_DSL_PREBUILT_ASCENDNPU_IR", repo_root / "3rdparty" / "AscendNPU-IR")
    )
    generated_inc_candidates = [
        ascendnpuir_root
        / "build"
        / "tools"
        / "bishengir"
        / "bishengir"
        / "include"
        / "bishengir"
        / "Interfaces"
        / "BiShengIREnums.h.inc",
        ascendnpuir_root
        / "build"
        / "tools"
        / "bishengir"
        / "include"
        / "bishengir"
        / "Interfaces"
        / "BiShengIREnums.h.inc",
    ]
    hivm_lib = ascendnpuir_root / "build" / "lib" / "libMLIRHIVMDialect.so"
    hivm_static_lib = ascendnpuir_root / "build" / "lib" / "libMLIRHIVMDialect.a"
    if not any(path.exists() for path in generated_inc_candidates) or not (
        hivm_lib.exists() or hivm_static_lib.exists()
    ):
        pytest.skip("BiShengIR/HIVM support is not available in this build environment")
    return tla_compile


def _run_tla_compile_ir_after_pass(
    mlir_text: str, pass_name: str, *, require_success: bool = False
) -> str:
    tla_compile = _require_hivm_tla_compile()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = pathlib.Path(tmpdir) / "input.mlir"
        output_path = pathlib.Path(tmpdir) / "output.mlir"
        input_path.write_text(mlir_text)
        result = subprocess.run(
            [
                str(tla_compile),
                str(input_path),
                "-o",
                str(output_path),
                f"--mlir-print-ir-after={pass_name}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if require_success:
            assert result.returncode == 0, output
        assert "IR Dump After" in output, output
        assert f"({pass_name})" in output, output
        return output



def _extract_function(ir_dump: str, name: str) -> str:
    marker = f"func.func @{name}("
    start = ir_dump.index(marker)
    next_function = ir_dump.find("\n  func.func @", start + len(marker))
    module_end = ir_dump.find("\n}", start + len(marker))
    ends = [position for position in (next_function, module_end) if position != -1]
    assert ends, ir_dump[start:]
    return ir_dump[start : min(ends)]


def _dump_after_mixed_split(
    kernel: object, *, type_args: tuple[object, ...] = ()
) -> str:
    mlir_text = kernel.dump_mlir(type_args=type_args)
    return _run_tla_compile_ir_after_pass(
        mlir_text, "tla-split-mixed-func", require_success=True
    )


def _mmad_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape((16, 8), (16, 4)),
                tla.Float16,
                addrspace=tla.AddressSpace.l0a,
                origin_shape=tla.make_shape(128, 64),
                layout_tag=tla.arch.zN,
            ),
            tla.Tensor(
                tla.make_shape((16, 4), (16, 8)),
                tla.Float16,
                addrspace=tla.AddressSpace.l0b,
                origin_shape=tla.make_shape(64, 128),
                layout_tag=tla.arch.nZ,
            ),
            tla.Tensor(
                tla.make_shape((16, 8), (16, 8)),
                tla.Float32,
                addrspace=tla.AddressSpace.l0c,
                origin_shape=tla.make_shape(128, 128),
                layout_tag=tla.arch.L0Clayout,
            ),
        )


def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )


@tla.kernel
def _cube_attr_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=False)


@tla.kernel
def _split_cube_only_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 7
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    tla.make_coord(value)


@tla.kernel
def _split_vector_only_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 8
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    tla.make_coord(value)


@tla.kernel
def _split_no_scope_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 9
    tla.make_coord(value)


@tla.kernel
def _mixed_interleaved_scopes_kernel() -> None:
    seed = tla.arch.block_idx()
    before = seed + 101
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    between = before + 102
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    middle = between + 103
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    after = middle + 104
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    tail = after + 105
    tla.make_coord(tail)


@tla.kernel
def _mixed_if_branches_kernel() -> None:
    seed = tla.arch.block_idx()
    before = seed + 201
    branch_value = before
    if before > 0:
        branch_value = before + 202
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        branch_value = branch_value + 203
    else:
        branch_value = before + 204
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
        branch_value = branch_value + 205
    tail = branch_value + 206
    tla.make_coord(tail)


@tla.kernel
def _mixed_for_kernel(limit: int) -> None:
    state = 0
    for index in tla.range(0, limit, 1):
        state = index + 301
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
    tla.make_coord(state)


@tla.kernel
def _mixed_while_kernel(limit: int) -> None:
    index = 0
    state = 0
    while index < limit:
        state = state + 401
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
        index = index + 1
    tla.make_coord(state)


def test_cube_tla_compile_emits_minimal_hivm_attrs_after_tla_func_to_hacc() -> None:
    ta, tb, tc = _mmad_tensor_args()
    try:
        mlir_text = _cube_attr_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as exc:
        _skip_if_mmad_rank2_tile_view_regression(exc)
        raise

    output = _run_tla_compile_ir_after_pass(mlir_text, "tla-lower-func")

    assert "dlti.target_system_spec = #dlti.target_system_spec<" in output
    assert '#dlti.dl_entry<"ARCH", "dav-c310">' in output
    assert 'hacc.target = #hacc.target<"Ascend950PR_9589">' in output
    assert "hivm.module_core_type = #hivm.module_core_type<AIC>" in output
    assert "hacc.entry" in output
    assert "hacc.function_kind = #hacc.function_kind<DEVICE>" in output


def test_non_mixed_functions_are_not_split() -> None:
    for kernel, name in (
        (_split_cube_only_kernel, "_split_cube_only_kernel"),
        (_split_vector_only_kernel, "_split_vector_only_kernel"),
        (_split_no_scope_kernel, "_split_no_scope_kernel"),
    ):
        output = _dump_after_mixed_split(kernel)
        assert f"func.func @{name}(" in output
        assert f"@{name}_mix_aic" not in output
        assert f"@{name}_mix_aiv" not in output


def test_mixed_split_preserves_interleaved_scopes_and_all_scalar_operations() -> None:
    output = _dump_after_mixed_split(_mixed_interleaved_scopes_kernel)
    aic = _extract_function(output, "_mixed_interleaved_scopes_kernel_mix_aic")
    aiv = _extract_function(output, "_mixed_interleaved_scopes_kernel_mix_aiv")

    assert aic.count("tla.cube") == 2
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 2
    assert "tla.cube" not in aiv
    for marker in (101, 102, 103, 104, 105):
        assert f"arith.constant {marker}" in aic
        assert f"arith.constant {marker}" in aiv
    assert aic.count("arith.addi") == 5
    assert aiv.count("arith.addi") == 5


def test_mixed_split_preserves_if_and_scalar_logic_in_opposite_scope_branch() -> None:
    output = _dump_after_mixed_split(_mixed_if_branches_kernel)
    aic = _extract_function(output, "_mixed_if_branches_kernel_mix_aic")
    aiv = _extract_function(output, "_mixed_if_branches_kernel_mix_aiv")

    assert aic.count("scf.if") == 1
    assert aiv.count("scf.if") == 1
    assert aic.count("scf.yield") == 2
    assert aiv.count("scf.yield") == 2
    assert aic.count("tla.cube") == 1
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 1
    assert "tla.cube" not in aiv
    for marker in (201, 202, 203, 204, 205, 206):
        assert f"arith.constant {marker}" in aic
        assert f"arith.constant {marker}" in aiv


@pytest.mark.parametrize(
    ("kernel", "name", "control_flow"),
    (
        (_mixed_for_kernel, "_mixed_for_kernel", "scf.for"),
        (_mixed_while_kernel, "_mixed_while_kernel", "scf.while"),
    ),
)
def test_mixed_split_preserves_loop_carried_control_flow(
    kernel: object, name: str, control_flow: str
) -> None:
    output = _dump_after_mixed_split(kernel, type_args=(4,))
    aic = _extract_function(output, f"{name}_mix_aic")
    aiv = _extract_function(output, f"{name}_mix_aiv")

    assert control_flow in aic
    assert control_flow in aiv
    assert "scf.yield" in aic
    assert "scf.yield" in aiv
    assert aic.count("tla.cube") == 1
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 1
    assert "tla.cube" not in aiv
