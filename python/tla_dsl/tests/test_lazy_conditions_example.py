from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import catlass.tla as tla
import catlass.runtime as runtime_mod


_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "end_to_end"
    / "lazy_conditions"
    / "lazy_conditions.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("lazy_conditions_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gm_tensor(length: int) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(length),
            tla.Float32,
            addrspace=tla.AddressSpace.gm,
            origin_shape=tla.make_shape(length),
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            layout_tag=tla.arch.RowMajor,
        )


def test_example_contains_the_exact_runtime_lazy_guards() -> None:
    tree = ast.parse(_EXAMPLE.read_text())
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lazy_conditions_kernel"
    )
    source = ast.unparse(kernel)
    assert "x != 0.0 and 10.0 / x > 5.0" in source
    assert "x == 0.0 or 10.0 / x > 5.0" in source
    assert "j < 1 and x != 0.0 and (10.0 / x > 5.0)" in source
    assert "j < 1 and (x == 0.0 or 10.0 / x > 5.0)" in source


def test_divisions_are_nested_under_conditional_regions() -> None:
    example = _load_example()
    values = _gm_tensor(len(example.VALUES))
    out = _gm_tensor(4 * len(example.VALUES))
    mlir = example.lazy_conditions_kernel.dump_mlir(type_args=(values, out))

    assert mlir.count("arith.divf") == 4
    assert mlir.count("scf.if") >= 8
    scopes: list[str] = []
    divisions_in_if = 0
    for line in mlir.splitlines():
        stripped = line.strip()
        prior_scope = None
        if stripped.startswith("}"):
            prior_scope = scopes.pop()
        if "arith.divf" in stripped:
            assert "scf.if" in scopes
            divisions_in_if += 1
        if stripped.endswith("{"):
            if stripped.startswith("} else"):
                assert prior_scope is not None
                scopes.append(prior_scope)
            else:
                scopes.append("scf.if" if "scf.if" in stripped else "other")
    assert divisions_in_if == 4


def test_runtime_expected_output_keeps_if_prefix_and_adds_while_results() -> None:
    example = _load_example()
    assert example.EXPECTED == (
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
    )


def test_example_uses_torch_npu_device_and_env_cache() -> None:
    source = _EXAMPLE.read_text()
    assert "torch.npu.set_device(" in source
    assert "configure_compile_cache(" not in source
    assert "_npu_host" not in source
