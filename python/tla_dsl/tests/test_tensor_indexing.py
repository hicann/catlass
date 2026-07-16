# tests/test_tensor_indexing.py — tensor __getitem__ / __setitem__ (scalar_load/store)

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.is_file():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def _kernel_scalar_load_1d(meta: tla.Tensor) -> None:
    _ = meta[2]


@tla.kernel
def _kernel_scalar_load_2d(meta: tla.Tensor) -> None:
    _ = meta[1, 3]


@tla.kernel
def _kernel_scalar_store_1d(out: tla.Tensor, meta: tla.Tensor) -> None:
    elem = meta[2]
    out[0] = elem


def _gm_tensor_1d(length: int, *, dtype: type = tla.Int32) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(length),
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=tla.make_shape(length),
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            layout_tag=tla.arch.RowMajor,
        )


def _gm_tensor_2d(rows: int, cols: int, *, dtype: type = tla.Float32) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(rows, cols),
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=tla.make_shape(rows, cols),
            coord=tla.make_coord(0, 0),
            stride=tla.make_stride(cols, 1),
            layout_tag=tla.arch.RowMajor,
        )


def test_tensor_scalar_load_emits_tla_scalar_load_1d() -> None:
    meta = _gm_tensor_1d(8)
    mlir = _kernel_scalar_load_1d.dump_mlir(type_args=(meta,))
    assert "tla.scalar_load" in mlir
    assert "row_major" in mlir
    assert "tla.load" not in mlir.replace("tla.scalar_load", "")


def test_scalar_load_returns_typed_scalar_ssa() -> None:
    from mlir import ir as mlir_ir

    from catlass.base_dsl.typing import ScalarSSA

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown():
        i32 = mlir_ir.IntegerType.get_signless(32)
        f32 = mlir_ir.F32Type.get()
        i32_val = mlir_ir.Operation.create(
            "arith.constant",
            results=[i32],
            attributes={"value": mlir_ir.IntegerAttr.get(i32, 7)},
        ).results[0]
        f32_val = mlir_ir.Operation.create(
            "arith.constant",
            results=[f32],
            attributes={"value": mlir_ir.FloatAttr.get(f32, 1.5)},
        ).results[0]

        ssa_i32 = ScalarSSA.from_value(i32_val, tla.Int32)
        assert isinstance(ssa_i32, ScalarSSA)
        assert ssa_i32.dtype is tla.Int32
        assert ssa_i32.element_type == "i32"
        assert ssa_i32.ir_value() is i32_val

        ssa_f32 = ScalarSSA.from_mlir_type(f32, f32_val)
        assert isinstance(ssa_f32, ScalarSSA)
        assert ssa_f32.dtype is tla.Float32
        assert ScalarSSA.from_value(f32_val).dtype is tla.Float32


def test_tensor_scalar_load_emits_tla_scalar_load_2d() -> None:
    meta = _gm_tensor_2d(4, 8)
    mlir = _kernel_scalar_load_2d.dump_mlir(type_args=(meta,))
    assert "tla.scalar_load" in mlir


def test_tensor_scalar_store_emits_tla_scalar_store_1d() -> None:
    out = _gm_tensor_1d(4)
    meta = _gm_tensor_1d(8)
    mlir = _kernel_scalar_store_1d.dump_mlir(type_args=(out, meta))
    assert "tla.scalar_store" in mlir
    assert "tla.scalar_load" in mlir


@tla.kernel
def _kernel_scalar_value_through_dynamic_if(
    out: tla.Tensor,
    meta: tla.Tensor,
    selector: int,
) -> None:
    value = meta[0]
    if selector == 0:
        value = meta[1]
    else:
        value = meta[2]
    out[0] = value


def test_scalar_value_through_dynamic_if_emits_scf_and_store() -> None:
    out = _gm_tensor_1d(1)
    meta = _gm_tensor_1d(8)
    mlir = _kernel_scalar_value_through_dynamic_if.dump_mlir(
        type_args=(out, meta, 0)
    )
    assert "tla.scalar_load" in mlir
    assert "tla.scalar_store" in mlir
    assert "scf.if" in mlir


def test_scalar_load_store_ops_not_public() -> None:
    with pytest.raises(AttributeError, match="scalar_load"):
        _ = tla.scalar_load
    with pytest.raises(AttributeError, match="scalar_store"):
        _ = tla.scalar_store


def test_tensor_indexing_rejects_underscore_coord() -> None:
    meta = _gm_tensor_2d(4, 8)

    @tla.kernel
    def k(m: tla.Tensor) -> None:
        _ = m[1, None]

    from catlass.execution_lowering import TlaLoweringError

    with pytest.raises(TlaLoweringError, match="does not support None/underscore"):
        k.dump_mlir(type_args=(meta,))


def test_tensor_indexing_rejects_rank2_with_one_index() -> None:
    meta = _gm_tensor_2d(4, 8)

    @tla.kernel
    def k(m: tla.Tensor) -> None:
        _ = m[3]

    from catlass.execution_lowering import TlaLoweringError

    with pytest.raises(TlaLoweringError, match="index rank must match"):
        k.dump_mlir(type_args=(meta,))


def test_tensor_indexing_allows_vector_region_frontend() -> None:
    meta = _gm_tensor_1d(8)

    @tla.kernel
    def k(m: tla.Tensor) -> None:
        with tla.vector():
            _ = m[0]

    mlir = k.dump_mlir(type_args=(meta,))
    assert "tla.scalar_load" in mlir

def test_scalar_load_rejects_ub_tensor() -> None:
    with runtime_mod._eager_capture():
        ub = tla.Tensor(
            tla.make_shape(8),
            tla.Int32,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(8),
            layout_tag=tla.arch.RowMajor,
        )

    @tla.kernel
    def k(x: tla.Tensor) -> None:
        _ = x[0]

    with pytest.raises(ValueError, match="doesn't support scalar_load"):
        k.dump_mlir(type_args=(ub,))


def _run_tla_compile_ir_after_pass(mlir_text: str, pass_name: str) -> str:
    tla_compile = _require_tla_compile()
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
        if result.returncode != 0 and "unregistered operation 'tla.scalar_load'" in output:
            raise AssertionError(
                "TlaCompile binary is stale and does not register tla.scalar_load. "
                "Rebuild from catlass_DSL_vector/python/tla_dsl/csrc/mlir/build: "
                "ninja TlaCompile"
            ) from None
        assert result.returncode == 0, output
        assert "IR Dump After" in output, output
        assert f"({pass_name})" in output, output
        return output


def test_tensor_indexing_lowers_to_memref_load(compiler_tlair) -> None:
    meta = _gm_tensor_1d(16)
    tlair = compiler_tlair(_kernel_scalar_load_1d, type_args=(meta,))
    # Kernel-arg scalar_load is lowered in tla-lower-scalar-access after split-mixed.
    lowered = _run_tla_compile_ir_after_pass(tlair, "tla-lower-scalar-access")
    assert "memref.load" in lowered
    assert "tla.scalar_load" not in lowered
