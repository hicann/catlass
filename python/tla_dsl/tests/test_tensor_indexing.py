# tests/test_tensor_indexing.py — tensor __getitem__ / __setitem__ (scalar_load/store)

from __future__ import annotations

import ast
import pathlib
import subprocess
import tempfile

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.base_dsl.ast_preprocessor import (
    _FrontendControlFlowTransformer,
    _FunctionAnalyzer,
    _scope_facts_for_transform,
)


def _operation_is_nested_in_scf_if(mlir: str, operation: str) -> bool:
    operation_offset = mlir.index(operation)
    stack: list[str] = []
    last_closed = ""
    for offset, token in enumerate(mlir[:operation_offset]):
        if token == "{":
            header_start = mlir.rfind("\n", 0, offset) + 1
            header = mlir[header_start:offset]
            if "scf.if" in header:
                kind = "scf.if"
            elif "else" in header and last_closed == "scf.if":
                kind = "scf.if"
            else:
                kind = "other"
            stack.append(kind)
            last_closed = ""
        elif token == "}" and stack:
            last_closed = stack.pop()
    return "scf.if" in stack


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.is_file():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def _kernel_scalar_load_1d(meta: tla.Tensor) -> None:
    _ = meta[2]


@tla.kernel
def _kernel_bool_flag_in_if(flags: tla.Tensor) -> None:
    """Host bool / tla.Bool element used as ``if`` predicate (fag_75-style)."""
    is_valid = flags[0, 1]
    block_idx = tla.arch.block_idx()
    if is_valid and block_idx == 0:
        tla.make_coord(1, 0)


@tla.kernel
def _kernel_scalar_load_in_lazy_boolean_rhs(meta: tla.Tensor) -> None:
    index = tla.arch.block_idx()
    if meta[index] > 0 or meta[index + 1] > 0:
        tla.make_coord(1, 0)
    if meta[index + 2] > 0 and meta[index + 3] > 0:
        tla.make_coord(2, 0)


@tla.kernel
def _kernel_tensor_attribute_in_lazy_boolean_rhs(meta: tla.Tensor) -> None:
    index = tla.arch.block_idx()
    if index == 0 and meta.shape[0] > 0:
        tla.make_coord(1, 0)


@tla.kernel
def _kernel_scalar_load_2d(meta: tla.Tensor) -> None:
    _ = meta[1, 3]


@tla.kernel
def _kernel_scalar_store_1d(out: tla.Tensor, meta: tla.Tensor) -> None:
    elem = meta[2]
    out[0] = elem


@tla.kernel
def _kernel_scalar_store_float_literal(out: tla.Tensor) -> None:
    out[0] = 1.1125


@tla.kernel
def _kernel_scalar_store_dynamic_if(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        out[0] = 1
    else:
        out[1] = 2


@tla.kernel
def _kernel_scalar_store_nested_python_for(
    out: tla.Tensor, selector: int
) -> None:
    if selector == 0:
        for _ in range(1):
            out[0] = 1


@tla.kernel
def _kernel_list_store_nested_python_for(selector: int) -> None:
    values = [0]
    if selector == 0:
        for _ in range(1):
            values[0] = 1


@tla.kernel
def _kernel_tensor_store_helper_name_collision(
    out: tla.Tensor, selector: int, __tladsl_tensor_store_1: int
) -> None:
    if selector == 0:
        out[0] = __tladsl_tensor_store_1


@tla.kernel
def _kernel_scalar_store_dynamic_for(out: tla.Tensor, limit: int) -> None:
    for i in tla.range(0, limit, 1):
        out[i] = i


@tla.kernel
def _kernel_scalar_store_dynamic_while(out: tla.Tensor, limit: int) -> None:
    i = 0
    while i < limit:
        out[i] = i
        i = i + 1


@tla.kernel
def _kernel_scalar_store_local_view(out: tla.Tensor, selector: int) -> None:
    view = tla.tile_view(out, tla.make_shape(4), tla.make_coord(0))
    if selector == 0:
        view[1] = 7


_assignment_evaluation_log: list[str] = []


def _logged_value() -> int:
    _assignment_evaluation_log.append("value")
    return 3


def _logged_target(target: tla.Tensor) -> tla.Tensor:
    _assignment_evaluation_log.append("target")
    return target


def _logged_index() -> int:
    _assignment_evaluation_log.append("index")
    return 0


@tla.kernel
def _kernel_scalar_store_evaluation_order(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        _logged_target(out)[_logged_index()] = _logged_value()


@tla.kernel
def _kernel_bad_augmented_tensor_store(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        out[0] += 1


@tla.kernel
def _kernel_bad_deleted_tensor_store(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        del out[0]


@tla.kernel
def _kernel_bad_sliced_tensor_store(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        out[0:1] = 1


@tla.kernel
def _kernel_bad_chained_tensor_store(out: tla.Tensor, selector: int) -> None:
    if selector == 0:
        out[0] = out[1] = 1


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


def test_bool_tensor_load_usable_in_if_and() -> None:
    """Bool GM tensor scalar_load → i1 predicate; ``if flag and ...`` is legal."""
    flags = _gm_tensor_2d(4, 8, dtype=tla.Bool)
    mlir = _kernel_bool_flag_in_if.dump_mlir(type_args=(flags,))
    assert "tla.scalar_load" in mlir
    assert "scf.if" in mlir
    assert _operation_is_nested_in_scf_if(mlir, "arith.cmpi")
    assert "!tla.ptr<i1" in mlir
    assert "-> i1" in mlir


def test_tensor_arg_scalar_loads_lower_in_lazy_boolean_rhs() -> None:
    meta = _gm_tensor_1d(8)
    mlir = _kernel_scalar_load_in_lazy_boolean_rhs.dump_mlir(type_args=(meta,))
    assert mlir.count("tla.scalar_load") == 4
    assert mlir.count("scf.if") >= 4


def test_tensor_arg_attribute_lowers_in_lazy_boolean_rhs() -> None:
    meta = _gm_tensor_1d(8)
    mlir = _kernel_tensor_attribute_in_lazy_boolean_rhs.dump_mlir(type_args=(meta,))
    assert "scf.if" in mlir
    assert "arith.cmpi" in mlir


def test_scalar_load_returns_typed_numeric() -> None:
    from mlir import ir as mlir_ir

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

        ssa_i32 = tla.Int32(i32_val)
        assert isinstance(ssa_i32, tla.Int32)
        assert isinstance(ssa_i32, tla.Numeric)
        assert type(ssa_i32).dtype == "i32"
        assert ssa_i32.ir_value() is i32_val

        ssa_f32 = tla.Float32(f32_val)
        assert isinstance(ssa_f32, tla.Float32)
        assert type(ssa_f32).dtype == "f32"


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


def test_tensor_scalar_store_supported_in_all_runtime_control_flow() -> None:
    out = _gm_tensor_1d(8)

    if_mlir = _kernel_scalar_store_dynamic_if.dump_mlir(type_args=(out, 0))
    assert "scf.if" in if_mlir
    assert if_mlir.count("tla.scalar_store") == 2
    if_header = next(line for line in if_mlir.splitlines() if "scf.if" in line)
    assert "->" not in if_header

    for_mlir = _kernel_scalar_store_dynamic_for.dump_mlir(type_args=(out, 4))
    assert "scf.for" in for_mlir
    assert for_mlir.count("tla.scalar_store") == 1

    while_mlir = _kernel_scalar_store_dynamic_while.dump_mlir(type_args=(out, 4))
    assert "scf.while" in while_mlir
    assert while_mlir.count("tla.scalar_store") == 1


def test_tensor_scalar_store_nested_in_python_loop_is_rewritten() -> None:
    out = _gm_tensor_1d(8)
    mlir = _kernel_scalar_store_nested_python_for.dump_mlir(type_args=(out, 0))
    assert "scf.if" in mlir
    assert mlir.count("tla.scalar_store") == 1


def test_nested_python_list_store_is_not_misclassified_as_tensor_store() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tensor_store.*target.*tensor"):
        _kernel_list_store_nested_python_for.dump_mlir(type_args=(0,))


def test_tensor_store_helper_does_not_collide_with_user_name() -> None:
    out = _gm_tensor_1d(8)
    mlir = _kernel_tensor_store_helper_name_collision.dump_mlir(
        type_args=(out, 0, 7)
    )
    assert mlir.count("tla.scalar_store") == 1


def test_tensor_store_helper_reserves_indirect_global_name() -> None:
    source = (
        "def kernel(out, selector):\n"
        "    if selector == 0:\n"
        "        out[0] = globals()['__tladsl_tensor_store_1']\n"
    )
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    globals_ = {"__tladsl_tensor_store_1": 7}
    scope_facts = _scope_facts_for_transform(
        source, "indirect_global_collision.py", target
    )
    transformer = _FrontendControlFlowTransformer(
        globals_,
        filename="indirect_global_collision.py",
        source_text=source,
        root_plan=_FunctionAnalyzer(
            global_symbols=globals_, scope_facts=scope_facts
        ).analyze(target),
    )

    transformer.visit(tree)

    assert transformer.tensor_store_helper_name == "__tladsl_tensor_store_2"


def test_tensor_scalar_store_accepts_local_tensor_view() -> None:
    out = _gm_tensor_1d(8)
    mlir = _kernel_scalar_store_local_view.dump_mlir(type_args=(out, 0))
    assert "scf.if" in mlir
    assert mlir.count("tla.scalar_store") == 1


def test_tensor_scalar_store_preserves_python_assignment_evaluation_order() -> None:
    out = _gm_tensor_1d(8)
    _assignment_evaluation_log.clear()
    mlir = _kernel_scalar_store_evaluation_order.dump_mlir(type_args=(out, 0))
    assert "tla.scalar_store" in mlir
    assert _assignment_evaluation_log == ["value", "target", "index"]


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (_kernel_bad_augmented_tensor_store, "augmented tensor stores"),
        (_kernel_bad_deleted_tensor_store, "does not support deletion"),
        (_kernel_bad_sliced_tensor_store, "tensor slice assignment"),
        (_kernel_bad_chained_tensor_store, "chained tensor stores"),
    ],
)
def test_tensor_scalar_store_rejects_unsupported_assignment_forms(
    kernel, message: str
) -> None:
    out = _gm_tensor_1d(8)
    with pytest.raises(SyntaxError, match=message):
        kernel.dump_mlir(type_args=(out, 0))


def test_tensor_scalar_store_diagnostic_points_to_source_assignment() -> None:
    out = _gm_tensor_1d(8)
    with pytest.raises(SyntaxError, match="tensor slice assignment") as exc_info:
        _kernel_bad_sliced_tensor_store.dump_mlir(type_args=(out, 0))

    assert exc_info.value.filename == __file__
    assert exc_info.value.lineno is not None
    assert exc_info.value.text is not None
    assert "out[0:1] = 1" in exc_info.value.text


def test_tensor_scalar_store_python_literals() -> None:
    """Bare int/float literals: emit constant+store; cast to element type; reject bad cases."""
    out_f32 = _gm_tensor_1d(4, dtype=tla.Float32)
    mlir = _kernel_scalar_store_float_literal.dump_mlir(type_args=(out_f32,))
    assert "arith.constant" in mlir
    assert "1.1125" in mlir or "1.112500" in mlir
    assert "tla.scalar_store" in mlir

    out_f16 = _gm_tensor_1d(4, dtype=tla.Float16)
    mlir_f16 = _kernel_scalar_store_float_literal.dump_mlir(type_args=(out_f16,))
    assert "f16" in mlir_f16
    assert "tla.scalar_store" in mlir_f16

    out_i8 = _gm_tensor_1d(4, dtype=tla.Int8)

    @tla.kernel
    def k_ok(o: tla.Tensor) -> None:
        o[0] = 127
        o[1] = -128

    @tla.kernel
    def k_oob(o: tla.Tensor) -> None:
        o[0] = 128

    @tla.kernel
    def k_bad_float(o: tla.Tensor) -> None:
        o[0] = 1.5

    assert "tla.scalar_store" in k_ok.dump_mlir(type_args=(out_i8,))
    with pytest.raises(Exception, match="out of range"):
        k_oob.dump_mlir(type_args=(out_i8,))
    with pytest.raises(Exception, match="expected integer scalar"):
        k_bad_float.dump_mlir(type_args=(_gm_tensor_1d(4, dtype=tla.Int32),))


def test_tensor_scalar_store_typed_numeric() -> None:
    """Typed Numeric must match tensor dtype; no silent same-kind upcast."""
    out_f32 = _gm_tensor_1d(4, dtype=tla.Float32)
    out_i32 = _gm_tensor_1d(4, dtype=tla.Int32)

    @tla.kernel
    def k_match(o: tla.Tensor) -> None:
        o[0] = tla.Float32(1.1125)

    @tla.kernel
    def k_upcast_rejected(o: tla.Tensor) -> None:
        o[0] = tla.Int16(7)

    @tla.kernel
    def k_explicit_to(o: tla.Tensor) -> None:
        o[0] = tla.Int16(7).to(tla.Int32)

    @tla.kernel
    def k_mismatch(o: tla.Tensor) -> None:
        o[0] = tla.Float32(1)

    assert "tla.scalar_store" in k_match.dump_mlir(type_args=(out_f32,))
    assert "tla.scalar_store" in k_explicit_to.dump_mlir(type_args=(out_i32,))
    with pytest.raises(Exception, match="type mismatch"):
        k_upcast_rejected.dump_mlir(type_args=(out_i32,))
    with pytest.raises(Exception, match="type mismatch"):
        k_mismatch.dump_mlir(type_args=(out_i32,))


def test_tensor_scalar_store_rejects_non_scalar_value() -> None:
    out = _gm_tensor_1d(4, dtype=tla.Float32)
    meta = _gm_tensor_1d(8, dtype=tla.Float32)

    @tla.kernel
    def k(o: tla.Tensor, m: tla.Tensor) -> None:
        o[0] = m  # tensor, not scalar

    with pytest.raises(Exception, match="expected Numeric or scalar literal"):
        k.dump_mlir(type_args=(out, meta))


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
    mlir = _kernel_scalar_value_through_dynamic_if.dump_mlir(type_args=(out, meta, 0))
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


def test_scalar_load_accepts_ub_tensor_with_canonical_op() -> None:
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
        with tla.vector():
            _ = x[0]

    mlir = k.dump_mlir(type_args=(ub,))
    assert "tla.scalar_load" in mlir
    assert "tla.load_scalar" not in mlir


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
        if (
            result.returncode != 0
            and "unregistered operation 'tla.scalar_load'" in output
        ):
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
