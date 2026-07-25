from __future__ import annotations

from pathlib import Path


_TLA_DSL_ROOT = Path(__file__).resolve().parents[1]
_RUN_DSL_TEST = _TLA_DSL_ROOT.parents[1] / "tests" / "run_dsl_test.sh"


def test_run_dsl_test_executes_ub_print_tensor_base_case() -> None:
    source = _RUN_DSL_TEST.read_text()

    assert (
        'PRINT_TENSOR_REL="examples/end_to_end/print_tensor/print_tensor.py"'
        in source
    )
    assert (
        'if [[ ! -f "${TLA_DSL_DIR}/${PRINT_TENSOR_REL}" ]]; then\n'
        '    echo "error: missing ${PRINT_TENSOR_REL} under ${TLA_DSL_DIR}" >&2'
        in source
    )
    assert (
        'python "${PRINT_TENSOR_REL}" --run --device "${DEVICE_ID}" \\\n'
        "            --storage ub --case base --arch-scope aiv.c310 --block 1 \\\n"
        "            --force-recompile"
        in source
    )
    assert (
        "print_tensor (print_tensor.py: AIV UB base case with a strict "
        "16-value prefix)" in source
    )
