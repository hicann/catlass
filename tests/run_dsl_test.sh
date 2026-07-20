#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# End-to-end validation for python/tla_dsl/examples/end_to_end/basic_mmad (basic_matmul.py, basic_mmad_ptr.py),
# python/tla_dsl/examples/end_to_end/basic_vadd (basic_vadd.py),
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed.py), and
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed_ub2l1.py).
# python/tla_dsl/examples/end_to_end/vector_ops (binary_op.py, masked_binary.py,
# bitwise_ops.py, reduction_ops.py, compare_mask.py, unary_ops.py, arange_op.py,
# interleave_op.py, squeeze_op.py).
# python/tla_dsl/examples/end_to_end/tensor_index (scalar_index_control_flow.py).
#
# Toolchain paths (env overrides first; directory-layout fallbacks last):
#   CANN:             ASCEND_HOME_PATH (source set_env.sh if not already in env)
#                     → WORKSPACE_ROOT/Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh
#   AscendNPU-IR: TLA_DSL_PREBUILT_ASCENDNPU_IR
#                     → TLA_DSL_ASCENDNPU_IR_ROOT
#                     → WORKSPACE_ROOT/AscendNPU-IR
#                     → TLA_DSL_DIR/3rdparty/AscendNPU-IR
#   TLA DSL:          TLA_DSL_DIR → CATDSL_ROOT/python/tla_dsl
#
# CANN 9.1+ ships hivmc-a5 in toolkit; no separate HIVMC sibling is required.
# LLVM/MLIR come from AscendNPU-IR build/install, not from conda.
#
# Usage:
#   bash tests/run_dsl_test.sh
#   bash tests/run_dsl_test.sh --device 0
#   TLA_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR bash tests/run_dsl_test.sh --device 0

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CATDSL_ROOT="$(realpath "${SCRIPT_PATH}/..")"
WORKSPACE_ROOT="${ASCEND_CATLASS_DSL_ROOT:-$(dirname "${CATDSL_ROOT}")}"

TLA_DSL_DIR="${TLA_DSL_DIR:-${CATDSL_ROOT}/python/tla_dsl}"

_resolve_cann_set_env_sh() {
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        printf '%s\n' "${ASCEND_HOME_PATH}/set_env.sh"
        return 0
    fi
    local fallback="${WORKSPACE_ROOT}/Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh"
    if [[ -f "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    return 1
}

# Prefer env for AscendNPU-IR; fall back to monorepo sibling, then in-tree 3rdparty.
if [[ -z "${TLA_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
    if [[ -n "${TLA_DSL_ASCENDNPU_IR_ROOT:-}" ]]; then
        TLA_DSL_PREBUILT_ASCENDNPU_IR="${TLA_DSL_ASCENDNPU_IR_ROOT}"
    elif [[ -d "${WORKSPACE_ROOT}/AscendNPU-IR" ]]; then
        TLA_DSL_PREBUILT_ASCENDNPU_IR="${WORKSPACE_ROOT}/AscendNPU-IR"
    else
        TLA_DSL_PREBUILT_ASCENDNPU_IR="${TLA_DSL_DIR}/3rdparty/AscendNPU-IR"
    fi
fi
TLA_DSL_ASCENDNPU_IR_ROOT="${TLA_DSL_ASCENDNPU_IR_ROOT:-${TLA_DSL_PREBUILT_ASCENDNPU_IR}}"

CONDA_ENV="${CONDA_ENV:-ascend-catlass-dsl}"
DEVICE_ID="${DEVICE_ID:-1}"

BASIC_MMAD_REL="examples/end_to_end/basic_mmad/basic_matmul.py"
BASIC_MMAD_PTR_REL="examples/end_to_end/basic_mmad/basic_mmad_ptr.py"
BASIC_VADD_REL="examples/end_to_end/basic_vadd/basic_vadd.py"
BASIC_MIXED_REL="examples/end_to_end/basic_mixed/basic_mixed.py"
BASIC_MIXED_UB2L1_REL="examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py"
MASKED_BINARY_REL="examples/end_to_end/vector_ops/masked_binary.py"
BITWISE_OPS_REL="examples/end_to_end/vector_ops/bitwise_ops.py"
BINARY_OP_REL="examples/end_to_end/vector_ops/binary_op.py"
REDUCTION_OPS_REL="examples/end_to_end/vector_ops/reduction_ops.py"
COMPARE_MASK_REL="examples/end_to_end/vector_ops/compare_mask.py"
COMPARE_MASK_OPS=(
    vector_vector_lt vector_vector_le vector_vector_gt vector_vector_ge vector_vector_eq vector_vector_ne
    vector_scalar_gt vector_scalar_ge
    masked_vector_vector_lt cmp_masked_fused static_dynamic_lt
)
UNARY_OPS_REL="examples/end_to_end/vector_ops/unary_ops.py"
ARANGE_OP_REL="examples/end_to_end/vector_ops/arange_op.py"
INTERLEAVE_OP_REL="examples/end_to_end/vector_ops/interleave_op.py"
SQUEEZE_OP_REL="examples/end_to_end/vector_ops/squeeze_op.py"
SCALAR_INDEX_CONTROL_FLOW_REL="examples/end_to_end/tensor_index/scalar_index_control_flow.py"

_ascendnpu_ir_dev_is_prebuilt() {
    local root="$1"
    [[ -n "${root}" ]] || return 1
    [[ -f "${root}/build/install/lib/cmake/mlir/MLIRConfig.cmake" ]] || return 1
    [[ -f "${root}/build/tools/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc" ]] || return 1
    return 0
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Run end-to-end validation for:
  - basic_mmad (basic_matmul.py --run --all-layouts --all-mmad-dtypes)
  - basic_mmad_ptr (basic_mmad_ptr.py --run)
  - basic_vadd (basic_vadd.py --run --all-dtypes, plus mutex variants)
  - basic_mixed (basic_mixed.py --run, basic_mixed_ub2l1.py --run)
  - binary_op (binary_op.py <op> --run --all-dtypes for add/sub/mul/div/max/min/add_unalign/add_brc_b32)
  - masked_binary (masked_binary.py masked_binary --run --all-dtypes)
  - bitwise_ops (bitwise_ops.py bitwise_ops --run --all-dtypes)
  - reduction_ops (reduction_ops.py <op> --run for add/max/min)
  - compare_mask (compare_mask.py <op> --run --all-dtypes for each compare-mask op)
  - unary_ops (unary_ops.py <op> --run --all-dtypes for exp/log/sqrt/abs/neg/masked_unary/masked_abs/masked_neg)
  - arange_op (arange_op.py increase --run --all-dtypes)
  - interleave_op (interleave_op.py interleave/deinterleave --run --all-dtypes)
  - squeeze_op (squeeze_op.py squeeze --run --all-dtypes)
  - scalar_index_control_flow (scalar_index_control_flow.py: GM scalar read/write,
    loop/dynamic-if/constexpr-if, vec.func)
Runs basic_mmad default MNK plus m=1, n=2, k=3.
Activates conda env "${CONDA_ENV}", sources CANN set_env.sh, exports AscendNPU-IR MLIR/LLVM
env, runs ./build.sh, then runs the test.

Options:
  -h, --help              Show this help
  --device ID             NPU device id (default: ${DEVICE_ID})

Paths (auto from script location):
  WORKSPACE_ROOT=${WORKSPACE_ROOT}   (override: ASCEND_CATLASS_DSL_ROOT)
  CATDSL_ROOT=${CATDSL_ROOT}
  TLA_DSL_DIR=${TLA_DSL_DIR}
  CONDA_ENV=${CONDA_ENV}

Toolchain (env first, layout fallback last):
  ASCEND_HOME_PATH               current: ${ASCEND_HOME_PATH:-<unset>}
    resolve: ASCEND_HOME_PATH/set_env.sh
             → WORKSPACE_ROOT/Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh
    note: sourcing CANN set_env.sh sets ASCEND_HOME_PATH automatically
  TLA_DSL_PREBUILT_ASCENDNPU_IR  current: ${TLA_DSL_PREBUILT_ASCENDNPU_IR}
  TLA_DSL_ASCENDNPU_IR_ROOT      current: ${TLA_DSL_ASCENDNPU_IR_ROOT}
    resolve: TLA_DSL_PREBUILT_ASCENDNPU_IR → TLA_DSL_ASCENDNPU_IR_ROOT
             → WORKSPACE_ROOT/AscendNPU-IR
             → TLA_DSL_DIR/3rdparty/AscendNPU-IR
  MLIR_DIR                       (default: ${MLIR_DIR:-<after Dev export>})

Example:
  bash ${SCRIPT_PATH}/run_dsl_test.sh
  bash ${SCRIPT_PATH}/run_dsl_test.sh --device 0
  TLA_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR \\
    bash ${SCRIPT_PATH}/run_dsl_test.sh --device 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --device)
            shift
            DEVICE_ID="${1:?--device requires an argument}"
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

_activate_conda() {
    if [[ -n "${CONDA_EXE:-}" ]] && [[ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
    else
        echo "error: conda not found; activate ${CONDA_ENV} manually or fix PATH." >&2
        exit 1
    fi
    conda activate "${CONDA_ENV}"
}

_export_ascendnpu_ir_dev_mlir_env() {
    local root="$1"

    export MLIR_DIR="${root}/build/install/lib/cmake/mlir"
    export LLVM_DIR="${root}/build/install/lib/cmake/llvm"
    export MLIR_TBLGEN_INCLUDE_DIR="${root}/build/install/include"
    export PATH="${root}/build/install/bin:${root}/build/bin:${PATH}"
    export PYTHONPATH="${root}/build/install/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${root}/build/install/python_packages/mlir_core/mlir/_mlir_libs:${root}/build/install/lib:${root}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    echo "    MLIR_DIR=${MLIR_DIR}"
    echo "    LLVM_DIR=${LLVM_DIR}"
    echo "    MLIR_TBLGEN_INCLUDE_DIR=${MLIR_TBLGEN_INCLUDE_DIR}"
}

_export_cann_build_env() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        local toolkit_dir
        toolkit_dir="$(dirname "${_cann_set_env_sh}")"
        if [[ -d "${toolkit_dir}/latest" ]]; then
            ASCEND_HOME_PATH="$(realpath "${toolkit_dir}/latest")"
        else
            ASCEND_HOME_PATH="$(realpath "${toolkit_dir}")"
        fi
        export ASCEND_HOME_PATH
    fi
    export BISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH:-${ASCEND_HOME_PATH}/bin}"
    echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
    echo "    BISHENG_COMPILER_PATH=${BISHENG_COMPILER_PATH}"
}

_export_toolchain_env() {
    echo "==> Exporting toolchain env"
    echo "    WORKSPACE_ROOT=${WORKSPACE_ROOT}"

    if [[ ! -d "${TLA_DSL_PREBUILT_ASCENDNPU_IR}" ]]; then
        echo "error: AscendNPU-IR directory not found: ${TLA_DSL_PREBUILT_ASCENDNPU_IR}" >&2
        exit 1
    fi
    export TLA_DSL_PREBUILT_ASCENDNPU_IR
    export TLA_DSL_ASCENDNPU_IR_ROOT
    echo "    TLA_DSL_PREBUILT_ASCENDNPU_IR=${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
    echo "    TLA_DSL_ASCENDNPU_IR_ROOT=${TLA_DSL_ASCENDNPU_IR_ROOT}"

    if ! _ascendnpu_ir_dev_is_prebuilt "${TLA_DSL_PREBUILT_ASCENDNPU_IR}"; then
        echo "error: AscendNPU-IR is not built at ${TLA_DSL_PREBUILT_ASCENDNPU_IR}" >&2
        echo "       Build it first (see python/tla_dsl/README.md §2.4)." >&2
        exit 1
    fi
    _export_ascendnpu_ir_dev_mlir_env "${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
}

_prepare_tla_dsl() {
    echo "==> Using AscendNPU-IR at ${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
    if [[ -f "${CATDSL_ROOT}/.gitmodules" ]]; then
        (
            cd "${CATDSL_ROOT}"
            git submodule update --init --depth 1 3rdparty/googletest 2>/dev/null || true
        )
    fi

    echo "==> ./build.sh (under ${TLA_DSL_DIR})"
    (
        cd "${TLA_DSL_DIR}"
        ./build.sh
    )
}

# --- main ---

if [[ ! -d "${TLA_DSL_DIR}" ]]; then
    echo "error: TLA_DSL_DIR does not exist: ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/build.sh" ]]; then
    echo "error: missing build.sh under TLA_DSL_DIR=${TLA_DSL_DIR}" >&2
    exit 1
fi

echo "==> Activating conda env: ${CONDA_ENV}"
_activate_conda

if ! _cann_set_env_sh="$(_resolve_cann_set_env_sh)"; then
    echo "error: CANN set_env.sh not found." >&2
    echo "       Set ASCEND_HOME_PATH to your CANN toolkit root (with set_env.sh)," >&2
    echo "       or source CANN set_env.sh before running this script," >&2
    echo "       or place CANN at ${WORKSPACE_ROOT}/Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh" >&2
    exit 1
fi
echo "==> Sourcing CANN: ${_cann_set_env_sh}"
# shellcheck disable=SC1090
source "${_cann_set_env_sh}"

echo "==> Exporting CANN build env"
_export_cann_build_env

_export_toolchain_env

echo "==> Using TLA_DSL_DIR=${TLA_DSL_DIR}"

_prepare_tla_dsl

if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MMAD_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MMAD_PTR_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_PTR_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_VADD_REL}" ]]; then
    echo "error: missing ${BASIC_VADD_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MIXED_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MIXED_UB2L1_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_UB2L1_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${MASKED_BINARY_REL}" ]]; then
    echo "error: missing ${MASKED_BINARY_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BITWISE_OPS_REL}" ]]; then
    echo "error: missing ${BITWISE_OPS_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BINARY_OP_REL}" ]]; then
    echo "error: missing ${BINARY_OP_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${REDUCTION_OPS_REL}" ]]; then
    echo "error: missing ${REDUCTION_OPS_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${COMPARE_MASK_REL}" ]]; then
    echo "error: missing ${COMPARE_MASK_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${UNARY_OPS_REL}" ]]; then
    echo "error: missing ${UNARY_OPS_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${ARANGE_OP_REL}" ]]; then
    echo "error: missing ${ARANGE_OP_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${SQUEEZE_OP_REL}" ]]; then
    echo "error: missing ${SQUEEZE_OP_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${SCALAR_INDEX_CONTROL_FLOW_REL}" ]]; then
    echo "error: missing ${SCALAR_INDEX_CONTROL_FLOW_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi

_run_basic_mmad_case() {
    local label="$1"
    shift
    echo "==> Running basic_mmad validation [${label}]: --run --all-layouts --all-mmad-dtypes --device ${DEVICE_ID} $*"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MMAD_REL}" --run --all-layouts --all-mmad-dtypes --device "${DEVICE_ID}" "$@"
    )
}

_run_basic_mmad_case "default MNK"
_run_basic_mmad_case "m=1 n=2 k=3" --m 1 --n 2 --k 3
_run_basic_mmad_case "mutex mode" --use-mutex
_run_basic_mmad_case "mutex with mode" --use-mutex-with

_run_basic_mmad_ptr_case() {
    echo "==> Running basic_mmad_ptr validation [ptr + offset -> make_tensor]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MMAD_PTR_REL}" --run --device "${DEVICE_ID}"
    )
}

_run_basic_mmad_ptr_case

_run_basic_vadd_case() {
    local label="$1"
    shift
    echo "==> Running basic_vadd validation [${label}]: --run --all-dtypes --device ${DEVICE_ID} $*"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_VADD_REL}" --run --all-dtypes --device "${DEVICE_ID}" "$@"
    )
}

_run_basic_vadd_case "all dtypes"
_run_basic_vadd_case "mutex mode" --use-mutex
_run_basic_vadd_case "mutex with mode" --use-mutex-with

_run_basic_mixed_case() {
    echo "==> Running basic_mixed validation [fixed shape/dtypes]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_REL}" --run --device "${DEVICE_ID}"
    )
}

_run_basic_mixed_case

_run_basic_mixed_ub2l1_case() {
    echo "==> Running basic_mixed_ub2l1 validation [fixed shape/dtypes, gm->ub->l1]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_UB2L1_REL}" --run --device "${DEVICE_ID}"
    )
}

_run_basic_mixed_ub2l1_case

_run_masked_binary_case() {
    echo "==> Running masked_binary validation [all dtypes]: masked_binary --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${MASKED_BINARY_REL}" masked_binary --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_masked_binary_case

_run_bitwise_ops_case() {
    echo "==> Running bitwise_ops validation [all dtypes]: bitwise_ops --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BITWISE_OPS_REL}" bitwise_ops --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_bitwise_ops_case

_run_binary_op_case() {
    local op="$1"
    echo "==> Running binary_op validation [${op} all dtypes]: ${op} --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BINARY_OP_REL}" "${op}" --run --all-dtypes --device "${DEVICE_ID}"
    )
}

for _binary_op in add sub mul div max min add_unalign add_brc_b32; do
    _run_binary_op_case "${_binary_op}"
done

_run_reduction_ops_case() {
    local op="$1"
    echo "==> Running reduction_ops validation [${op} f32]: ${op} --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${REDUCTION_OPS_REL}" "${op}" --run --device "${DEVICE_ID}"
    )
}

for _reduce_op in add max min; do
    _run_reduction_ops_case "${_reduce_op}"
done

_run_compare_mask_case() {
    local op="$1"
    echo "==> Running compare_mask validation [${op}]: ${op} --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${COMPARE_MASK_REL}" "${op}" --run --all-dtypes --device "${DEVICE_ID}"
    )
}

for _compare_mask_op in "${COMPARE_MASK_OPS[@]}"; do
    _run_compare_mask_case "${_compare_mask_op}"
done

_run_unary_ops_case() {
    local op="$1"
    echo "==> Running unary_ops validation [${op} all dtypes]: ${op} --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${UNARY_OPS_REL}" "${op}" --run --all-dtypes --device "${DEVICE_ID}"
    )
}

for _unary_op in exp log sqrt abs neg; do
    _run_unary_ops_case "${_unary_op}"
done

_run_masked_unary_case() {
    echo "==> Running unary_ops validation [masked_unary all dtypes]: masked_unary --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${UNARY_OPS_REL}" masked_unary --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_masked_unary_case

_run_masked_abs_case() {
    echo "==> Running unary_ops validation [masked_abs all integer dtypes]: masked_abs --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${UNARY_OPS_REL}" masked_abs --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_masked_abs_case

_run_masked_neg_case() {
    echo "==> Running unary_ops validation [masked_neg all numeric dtypes]: masked_neg --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${UNARY_OPS_REL}" masked_neg --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_masked_neg_case

_run_arange_op_case() {
    echo "==> Running arange_op validation [increase all dtypes]: increase --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${ARANGE_OP_REL}" increase --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_arange_op_case

_run_interleave_op_case() {
    echo "==> Running interleave_op validation [interleave all dtypes]: interleave --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${INTERLEAVE_OP_REL}" interleave --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_interleave_op_case

_run_deinterleave_op_case() {
    echo "==> Running interleave_op validation [deinterleave all dtypes]: deinterleave --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${INTERLEAVE_OP_REL}" deinterleave --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_deinterleave_op_case

_run_squeeze_op_case() {
    echo "==> Running squeeze_op validation [squeeze all dtypes]: squeeze --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${SQUEEZE_OP_REL}" squeeze --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_squeeze_op_case

_run_scalar_index_control_flow_case() {
    echo "==> Running scalar_index_control_flow validation [GM scalar indexing]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${SCALAR_INDEX_CONTROL_FLOW_REL}" --device "${DEVICE_ID}"
    )
}

_run_scalar_index_control_flow_case

echo "==> run_dsl_test.sh finished successfully"
