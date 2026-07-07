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
# End-to-end validation for python/tla_dsl/examples/end_to_end/basic_mmad (basic_matmul.py),
# python/tla_dsl/examples/end_to_end/basic_vadd (basic_vadd.py),
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed.py), and
# python/tla_dsl/examples/end_to_end/vector_ops (binary_op.py, masked_binary.py,
# reduction_ops.py, unary_ops.py, arange_op.py).
#
# Fixed toolchain paths relative to WORKSPACE_ROOT (= parent of catlass repo):
#   CANN:             Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh
#   AscendNPU-IR-Dev: AscendNPU-IR-Dev/
#   TLA DSL:          catlass/python/tla_dsl/
#
# CANN 9.1+ ships hivmc-a5 in toolkit; no separate HIVMC sibling is required.
# LLVM/MLIR come from AscendNPU-IR-Dev build/install, not from conda.
#
# Usage:
#   bash catlass_DSL/tests/run_dsl_test.sh
#   bash catlass_DSL/tests/run_dsl_test.sh --skip-prepare --device 0

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CATDSL_ROOT="$(realpath "${SCRIPT_PATH}/..")"
WORKSPACE_ROOT="${ASCEND_CATLASS_DSL_ROOT:-$(dirname "${CATDSL_ROOT}")}"

TLA_DSL_DIR="${TLA_DSL_DIR:-${CATDSL_ROOT}/python/tla_dsl}"
CANN_SET_ENV_SH="${WORKSPACE_ROOT}/Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh"
TLA_DSL_PREBUILT_ASCENDNPU_IR="${WORKSPACE_ROOT}/AscendNPU-IR-Dev"
TLA_DSL_ASCENDNPU_IR_ROOT="${WORKSPACE_ROOT}/AscendNPU-IR-Dev"

CONDA_ENV="${CONDA_ENV:-ascend-catlass-dsl}"
DEVICE_ID="${DEVICE_ID:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

BASIC_MMAD_REL="examples/end_to_end/basic_mmad/basic_matmul.py"
BASIC_VADD_REL="examples/end_to_end/basic_vadd/basic_vadd.py"
BASIC_MIXED_REL="examples/end_to_end/basic_mixed/basic_mixed.py"
MASKED_BINARY_REL="examples/end_to_end/vector_ops/masked_binary.py"
BINARY_OP_REL="examples/end_to_end/vector_ops/binary_op.py"
REDUCTION_OPS_REL="examples/end_to_end/vector_ops/reduction_ops.py"
UNARY_OPS_REL="examples/end_to_end/vector_ops/unary_ops.py"
ARANGE_OP_REL="examples/end_to_end/vector_ops/arange_op.py"

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
  - basic_vadd (basic_vadd.py --run --all-dtypes, plus mutex variants)
  - basic_mixed (basic_mixed.py --run)
  - binary_op (binary_op.py <op> --run --all-dtypes for add/sub/mul/div/max/min)
  - masked_binary (masked_binary.py masked_binary --run --all-dtypes)
  - reduction_ops (reduction_ops.py <op> --run for add/max/min)
  - unary_ops (unary_ops.py <op> --run --all-dtypes for exp/log/sqrt/abs/neg/masked_unary/masked_abs/masked_neg)
  - arange_op (arange_op.py increase --run --all-dtypes)
Runs basic_mmad default MNK plus m=1, n=2, k=3.
Activates conda env "${CONDA_ENV}", sources CANN set_env.sh, exports AscendNPU-IR-Dev MLIR/LLVM
env, then builds (optional) and runs the test.

Options:
  -h, --help              Show this help
  --device ID             NPU device id (default: ${DEVICE_ID})
  --skip-prepare          Skip git submodule update and ./build.sh

Paths (auto from script location):
  WORKSPACE_ROOT=${WORKSPACE_ROOT}   (override: ASCEND_CATLASS_DSL_ROOT)
  CATDSL_ROOT=${CATDSL_ROOT}
  TLA_DSL_DIR=${TLA_DSL_DIR}
  CONDA_ENV=${CONDA_ENV}

Toolchain (fixed, relative to WORKSPACE_ROOT=${WORKSPACE_ROOT}):
  CANN_SET_ENV_SH                Ascend/9.1.0-beta.1/ascend-toolkit/set_env.sh
  TLA_DSL_PREBUILT_ASCENDNPU_IR  AscendNPU-IR-Dev
  TLA_DSL_ASCENDNPU_IR_ROOT      AscendNPU-IR-Dev
  ASCEND_HOME_PATH               (default: ${ASCEND_HOME_PATH:-<after CANN source>})
  MLIR_DIR                       (default: ${MLIR_DIR:-<after Dev export>})

Example:
  bash ${SCRIPT_PATH}/run_dsl_test.sh
  SKIP_PREPARE=1 bash ${SCRIPT_PATH}/run_dsl_test.sh --device 0
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
        --skip-prepare)
            SKIP_PREPARE=1
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
        toolkit_dir="$(dirname "${CANN_SET_ENV_SH}")"
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
        echo "error: AscendNPU-IR-Dev directory not found: ${TLA_DSL_PREBUILT_ASCENDNPU_IR}" >&2
        exit 1
    fi
    export TLA_DSL_PREBUILT_ASCENDNPU_IR
    export TLA_DSL_ASCENDNPU_IR_ROOT
    echo "    TLA_DSL_PREBUILT_ASCENDNPU_IR=${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
    echo "    TLA_DSL_ASCENDNPU_IR_ROOT=${TLA_DSL_ASCENDNPU_IR_ROOT}"

    if ! _ascendnpu_ir_dev_is_prebuilt "${TLA_DSL_PREBUILT_ASCENDNPU_IR}"; then
        echo "error: AscendNPU-IR-Dev is not built at ${TLA_DSL_PREBUILT_ASCENDNPU_IR}" >&2
        echo "       Build it first (see python/tla_dsl/README.md §2.4)." >&2
        exit 1
    fi
    _export_ascendnpu_ir_dev_mlir_env "${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
}

_prepare_tla_dsl() {
    echo "==> Using AscendNPU-IR-Dev at ${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
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

if [[ ! -f "${CANN_SET_ENV_SH}" ]]; then
    echo "error: CANN set_env.sh not found: ${CANN_SET_ENV_SH}" >&2
    exit 1
fi
echo "==> Sourcing CANN: ${CANN_SET_ENV_SH}"
# shellcheck disable=SC1090
source "${CANN_SET_ENV_SH}"

echo "==> Exporting CANN build env"
_export_cann_build_env

_export_toolchain_env

echo "==> Using TLA_DSL_DIR=${TLA_DSL_DIR}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
    _prepare_tla_dsl
else
    echo "==> SKIP_PREPARE=1: skipping submodule update and build.sh"
fi

if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MMAD_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_REL} under ${TLA_DSL_DIR}" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${MASKED_BINARY_REL}" ]]; then
    echo "error: missing ${MASKED_BINARY_REL} under ${TLA_DSL_DIR}" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${UNARY_OPS_REL}" ]]; then
    echo "error: missing ${UNARY_OPS_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${ARANGE_OP_REL}" ]]; then
    echo "error: missing ${ARANGE_OP_REL} under ${TLA_DSL_DIR}" >&2
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

_run_masked_binary_case() {
    echo "==> Running masked_binary validation [all dtypes]: masked_binary --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${MASKED_BINARY_REL}" masked_binary --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_masked_binary_case

_run_binary_op_case() {
    local op="$1"
    echo "==> Running binary_op validation [${op} all dtypes]: ${op} --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BINARY_OP_REL}" "${op}" --run --all-dtypes --device "${DEVICE_ID}"
    )
}

for _binary_op in add sub mul div max min; do
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

echo "==> run_dsl_test.sh finished successfully"
