from __future__ import annotations

import ast
from pathlib import Path


_CATLASS_DSL_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = (
    _CATLASS_DSL_ROOT
    / "examples"
    / "end_to_end"
    / "scalar_arg_alignment"
    / "scalar_arg_alignment.py"
)
_README = _EXAMPLE.with_name("README.md")


def _example_tree() -> ast.Module:
    return ast.parse(_EXAMPLE.read_text())


def test_scalar_arg_alignment_is_a_minimal_public_launch_example() -> None:
    source = _EXAMPLE.read_text()

    assert len(source.splitlines()) < 180
    assert "pack_argument_buffer" not in source
    assert "launch_with_args" not in source
    assert "compiler_abi" not in source
    assert "json" not in source
    assert not _README.exists()


def test_kernel_has_tensor_i16_tensor_signature_and_consumes_both_inputs() -> None:
    kernel = next(
        node
        for node in _example_tree().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "scalar_arg_alignment"
    )

    assert [argument.arg for argument in kernel.args.args] == [
        "output_tensor",
        "scalar",
        "trailing_tensor",
    ]
    assert [ast.unparse(argument.annotation) for argument in kernel.args.args] == [
        "tla.Tensor",
        "tla.Int16",
        "tla.Tensor",
    ]
    source = ast.unparse(kernel)
    assert "output_tensor[0] = scalar" in source
    assert "output_tensor[1] = trailing_tensor[0]" in source
