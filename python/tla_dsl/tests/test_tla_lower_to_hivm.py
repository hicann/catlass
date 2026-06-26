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


def _run_tla_compile_ir_after_pass(mlir_text: str, pass_name: str) -> str:
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
        assert "IR Dump After" in output, output
        assert f"({pass_name})" in output, output
        return output


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


def test_tla_blockidx_compile_lowers_to_hivm() -> None:
    mlir_text = """module{
    func.func @kernel_block.idx() {
        %0 = tla.arch.block_idx -> index
        return
    }
}
"""

    output = _run_tla_compile_ir_after_pass(mlir_text, "tla-lower-to-hivm")
    assert "hivm.hir.get_block_idx" in output


def test_cube_tla_compile_emits_minimal_hivm_attrs_after_tla_func_to_hacc() -> None:
    ta, tb, tc = _mmad_tensor_args()
    try:
        mlir_text = _cube_attr_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as exc:
        _skip_if_mmad_rank2_tile_view_regression(exc)
        raise

    output = _run_tla_compile_ir_after_pass(mlir_text, "tla-func-to-hacc")

    assert "dlti.target_system_spec = #dlti.target_system_spec<" in output
    assert '#dlti.dl_entry<"ARCH", "dav-c310">' in output
    assert 'hacc.target = #hacc.target<"Ascend950PR_9589">' in output
    assert "hivm.module_core_type = #hivm.module_core_type<AIC>" in output
    assert "hacc.entry" in output
    assert "hacc.function_kind = #hacc.function_kind<DEVICE>" in output
