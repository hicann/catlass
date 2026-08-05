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
# End-to-end validation for python/tla_dsl/examples/end_to_end/basic_mmad (basic_matmul*.py, basic_mmad_ptr.py),
# python/tla_dsl/examples/end_to_end/basic_vadd (basic_vadd.py),
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed.py, including mutex mode), and
# python/tla_dsl/examples/end_to_end/basic_mixed (basic_mixed_ub2l1.py, basic_mixed_store_zN.py,
# basic_mixed_store_zNUnAlign.py, basic_mixed_fixpipe_nz2dn.py).
# python/tla_dsl/examples/end_to_end/vector_ops (binary_op.py, masked_binary.py,
# bitwise_ops.py, reduction_ops.py, compare_mask.py, unary_ops.py, arange_op.py,
# interleave_op.py, load_dintlv_op.py, load_store_mask.py, squeeze_op.py,
# register_control_flow.py, load_and_store_scalar_after_reduction.py, load_us_b8_op.py).
# python/tla_dsl/examples/end_to_end/tensor_index (scalar_index_control_flow.py,
# scalar_kernel_arg.py).
# python/tla_dsl/examples/end_to_end/debug_print (debug_print.py, debug_print_mixed.py,
# debug_print_format.py).
# python/tla_dsl/examples/end_to_end/scalar_arg_alignment (scalar_arg_alignment.py).
# python/tla_dsl/examples/end_to_end/print_tensor (print_tensor.py: all eight
# supported GM/UB dtypes plus multi-block and multi-call cases).
#
# Toolchain paths (env overrides first; directory-layout fallbacks last):
#   CANN:             ASCEND_HOME_PATH (source set_env.sh if not already in env)
#                     → WORKSPACE_ROOT/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh
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
    local fallback="${WORKSPACE_ROOT}/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh"
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
COMPILE_JOBS="${TLA_DSL_COMPILE_JOBS:-4}"

BASIC_MMAD_REL="examples/end_to_end/basic_mmad/basic_matmul.py"
BASIC_MMAD_AUTO_SYNC_REL="examples/end_to_end/basic_mmad/basic_matmul_auto_sync.py"
BASIC_MMAD_PTR_REL="examples/end_to_end/basic_mmad/basic_mmad_ptr.py"
BASIC_VADD_REL="examples/end_to_end/basic_vadd/basic_vadd.py"
BASIC_MIXED_REL="examples/end_to_end/basic_mixed/basic_mixed.py"
BASIC_MIXED_UB2L1_REL="examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py"
BASIC_MIXED_STORE_ZN_REL="examples/end_to_end/basic_mixed/basic_mixed_store_zN.py"
BASIC_MIXED_STORE_ZNUNALIGN_REL="examples/end_to_end/basic_mixed/basic_mixed_store_zNUnAlign.py"
BASIC_MIXED_FIXPIPE_NZ2DN_REL="examples/end_to_end/basic_mixed/basic_mixed_fixpipe_nz2dn.py"
MASKED_BINARY_REL="examples/end_to_end/vector_ops/masked_binary.py"
BITWISE_OPS_REL="examples/end_to_end/vector_ops/bitwise_ops.py"
BINARY_OP_REL="examples/end_to_end/vector_ops/binary_op.py"
REDUCTION_OPS_REL="examples/end_to_end/vector_ops/reduction_ops.py"
LOAD_STORE_SCALAR_AFTER_REDUCTION_REL="examples/end_to_end/vector_ops/load_and_store_scalar_after_reduction.py"
COMPARE_MASK_REL="examples/end_to_end/vector_ops/compare_mask.py"
COMPARE_MASK_OPS=(
    vector_vector_lt vector_vector_le vector_vector_gt vector_vector_ge vector_vector_eq vector_vector_ne
    vector_scalar_gt vector_scalar_ge
    masked_vector_vector_lt cmp_masked_fused static_dynamic_lt
)
UNARY_OPS_REL="examples/end_to_end/vector_ops/unary_ops.py"
ARANGE_OP_REL="examples/end_to_end/vector_ops/arange_op.py"
INTERLEAVE_OP_REL="examples/end_to_end/vector_ops/interleave_op.py"
LOAD_DINTLV_OP_REL="examples/end_to_end/vector_ops/load_dintlv_op.py"
LOAD_US_B8_OP_REL="examples/end_to_end/vector_ops/load_us_b8_op.py"
LOAD_STORE_MASK_REL="examples/end_to_end/vector_ops/load_store_mask.py"
STORE_PACK_REL="examples/end_to_end/vector_ops/store_pack.py"
SQUEEZE_OP_REL="examples/end_to_end/vector_ops/squeeze_op.py"
REGISTER_CONTROL_FLOW_REL="examples/end_to_end/vector_ops/register_control_flow.py"
SCALAR_INDEX_CONTROL_FLOW_REL="examples/end_to_end/tensor_index/scalar_index_control_flow.py"
SCALAR_KERNEL_ARG_REL="examples/end_to_end/tensor_index/scalar_kernel_arg.py"
DEBUG_PRINT_REL="examples/end_to_end/debug_print/debug_print.py"
DEBUG_PRINT_MIXED_REL="examples/end_to_end/debug_print/debug_print_mixed.py"
DEBUG_PRINT_FORMAT_REL="examples/end_to_end/debug_print/debug_print_format.py"
SCALAR_ARG_ALIGNMENT_REL="examples/end_to_end/scalar_arg_alignment/scalar_arg_alignment.py"
PRINT_TENSOR_REL="examples/end_to_end/print_tensor/print_tensor.py"

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
  - basic_mmad (full flag-sync mnk/layout/dtype matrix; representative manual and automatic mutex cases;
    atomic-add coverage for each supported input dtype)
  - basic_mmad_ptr (basic_mmad_ptr.py)
  - basic_vadd (basic_vadd.py with per-dtype CLI invocations, plus mutex variants)
  - basic_mixed (basic_mixed.py --run dynamic GM mnk list, including --use-mutex; basic_mixed_ub2l1.py --run,
    basic_mixed_store_zN.py --run, basic_mixed_store_zNUnAlign.py --run for m=64/m=50, basic_mixed_fixpipe_nz2dn.py --run)
  - binary_op (binary_op.py <op> --run --all-dtypes for add/sub/mul/div/max/min/add_unalign/add_brc_b32)
  - masked_binary (masked_binary.py masked_binary --run --all-dtypes)
  - bitwise_ops (bitwise_ops.py bitwise_ops --run --all-dtypes)
  - reduction_ops (reduction_ops.py <op> --run for add/max/min)
  - load_and_store_scalar_after_reduction (UB scalar load/store in tla.vector and outlined tla.vec.func)
  - compare_mask (compare_mask.py <op> --run --all-dtypes for each compare-mask op)
  - unary_ops (unary_ops.py <op> --run --all-dtypes for exp/log/sqrt/abs/neg/masked_unary/masked_abs/masked_neg)
  - arange_op (arange_op.py [increase/decrease] --run --all-dtypes)
  - interleave_op (interleave_op.py interleave/deinterleave --run --all-dtypes)
  - load_dintlv_op (load_dintlv_op.py dintlv_b32 --run --all-dtypes; f32 only)
  - load_us_b8_op (load_us_b8_op.py us_b8 --sweep --shapes 512; i8 only:
    DIST_US_B8 2x up-sample load of b8 elements)
  - load_store_mask (load_store_mask.py load_store_mask --run --all-dtypes:
    MaskSSA load/store round-trip via MaskLoadParams/MaskStoreParams for
    b8/b16/b32 UB carriers; companion vector fixed to f32)
  - store_pack (store_pack.py store_pack --run --all-dtypes; i32/i16 only)
  - squeeze_op (squeeze_op.py squeeze --run --all-dtypes)
  - register_control_flow (register_control_flow.py register_carriers --run:
    mixed VectorSSA/MaskSSA scf.for carriers and masked store)
  - scalar_index_control_flow (scalar_index_control_flow.py: GM scalar read/write,
    loop/dynamic-if/constexpr-if, vec.func, AST Numeric / index-vs-Int32 compare)
  - scalar_kernel_arg (scalar_kernel_arg.py: host Numeric kernel args used in
    same-type scalar arithmetic)
  - debug_print (all direct scalar dtypes, supported computed values, and two-block prints on AIV and AIC)
  - debug_print_mixed (all scalar dtypes in cube-only, vector-only, and combined regions)
  - debug_print_format (formatted multicall and multiblock prints on AIV and AIC)
  - scalar_arg_alignment (scalar_arg_alignment.py: tensor-i16-tensor host ABI)
  - print_tensor (print_tensor.py: all supported GM/UB dtypes with AIV/AIC
    multi-block and multi-call coverage)
Runs the basic_mmad flag-sync matrix with irregular CLI shapes (333×444×555
and 1×2×3); representative mutex and atomic-add cases use 333×444×555.
Example defaults remain regular (256×512×1024).
Activates conda env "${CONDA_ENV}", sources CANN set_env.sh, exports AscendNPU-IR MLIR/LLVM
env, runs ./build.sh, then runs the test.

Options:
  -h, --help              Show this help
  --device ID             NPU device id (default: ${DEVICE_ID})
  --compile-jobs N        Host-only compiler processes for vector dtype batches
                          (default: ${COMPILE_JOBS}; env: TLA_DSL_COMPILE_JOBS)
                          Compatible same-dtype op batches use up to four NPU
                          blocks in one fused kernel.

Paths (auto from script location):
  WORKSPACE_ROOT=${WORKSPACE_ROOT}   (override: ASCEND_CATLASS_DSL_ROOT)
  CATDSL_ROOT=${CATDSL_ROOT}
  TLA_DSL_DIR=${TLA_DSL_DIR}
  CONDA_ENV=${CONDA_ENV}

Toolchain (env first, layout fallback last):
  ASCEND_HOME_PATH               current: ${ASCEND_HOME_PATH:-<unset>}
    resolve: ASCEND_HOME_PATH/set_env.sh
             → WORKSPACE_ROOT/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh
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
        --compile-jobs)
            shift
            COMPILE_JOBS="${1:?--compile-jobs requires an argument}"
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! "${COMPILE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --compile-jobs must be a positive integer; got ${COMPILE_JOBS}" >&2
    exit 2
fi

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
    echo "       or place CANN at ${WORKSPACE_ROOT}/Ascend/9.1.0-beta.3/ascend-toolkit/set_env.sh" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MMAD_AUTO_SYNC_REL}" ]]; then
    echo "error: missing ${BASIC_MMAD_AUTO_SYNC_REL} under ${TLA_DSL_DIR}" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MIXED_STORE_ZN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_STORE_ZN_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MIXED_STORE_ZNUNALIGN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_STORE_ZNUNALIGN_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${BASIC_MIXED_FIXPIPE_NZ2DN_REL}" ]]; then
    echo "error: missing ${BASIC_MIXED_FIXPIPE_NZ2DN_REL} under ${TLA_DSL_DIR}" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${LOAD_STORE_SCALAR_AFTER_REDUCTION_REL}" ]]; then
    echo "error: missing ${LOAD_STORE_SCALAR_AFTER_REDUCTION_REL} under ${TLA_DSL_DIR}" >&2
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
if [[ ! -f "${TLA_DSL_DIR}/${REGISTER_CONTROL_FLOW_REL}" ]]; then
    echo "error: missing ${REGISTER_CONTROL_FLOW_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${SCALAR_INDEX_CONTROL_FLOW_REL}" ]]; then
    echo "error: missing ${SCALAR_INDEX_CONTROL_FLOW_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${DEBUG_PRINT_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${DEBUG_PRINT_MIXED_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_MIXED_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${DEBUG_PRINT_FORMAT_REL}" ]]; then
    echo "error: missing ${DEBUG_PRINT_FORMAT_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${SCALAR_ARG_ALIGNMENT_REL}" ]]; then
    echo "error: missing ${SCALAR_ARG_ALIGNMENT_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TLA_DSL_DIR}/${PRINT_TENSOR_REL}" ]]; then
    echo "error: missing ${PRINT_TENSOR_REL} under ${TLA_DSL_DIR}" >&2
    exit 1
fi

_run_basic_mmad_case() {
    local label="$1"
    local script="$2"
    shift 2
    echo "==> Running basic_mmad validation [${label}]: ${script} $* --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${script}" "$@" --device "${DEVICE_ID}" --force-recompile
    )
}

_run_basic_mmad_flag_sync_matrix() {
    # One process per (mnk, layout, dtype); script itself does not sweep.
    local shapes=(
        "--m 333 --n 444 --k 555"
        "--m 1 --n 2 --k 3"
    )
    local layouts=(
        "--layout-a row --layout-b row"
        "--layout-a row --layout-b col"
        "--layout-a col --layout-b row"
        "--layout-a col --layout-b col"
    )
    local triples=(
        "--dtype-a f16 --dtype-b f16 --dtype-c f32"
        "--dtype-a f16 --dtype-b f16 --dtype-c f16"
        "--dtype-a bf16 --dtype-b bf16 --dtype-c f32"
        "--dtype-a bf16 --dtype-b bf16 --dtype-c bf16"
        "--dtype-a f32 --dtype-b f32 --dtype-c f32"
    )
    local shape_args layout_args dtype_args
    for shape_args in "${shapes[@]}"; do
        for layout_args in "${layouts[@]}"; do
            for dtype_args in "${triples[@]}"; do
                # These specifications are fixed, internal argument lists.
                # shellcheck disable=SC2086
                _run_basic_mmad_case "flag sync" "${BASIC_MMAD_REL}" \
                    ${shape_args} ${layout_args} ${dtype_args}
            done
        done
    done
}

_run_basic_mmad_atomic_add_cases() {
    local dtype
    for dtype in f16 bf16 f32; do
        _run_basic_mmad_case "atomic add, ${dtype} inputs" \
            "examples/end_to_end/basic_mmad/basic_matmul_atomic_add.py" \
            --m 333 --n 444 --k 555 \
            --layout-a row --layout-b row \
            --dtype-a "${dtype}" --dtype-b "${dtype}" --dtype-c f32
    done
}

_run_basic_mmad_flag_sync_matrix
_run_basic_mmad_case "mutex mode" \
    "examples/end_to_end/basic_mmad/basic_matmul_mutex.py" \
    --m 333 --n 444 --k 555 \
    --layout-a row --layout-b row \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
_run_basic_mmad_case "mutex with mode" \
    "examples/end_to_end/basic_mmad/basic_matmul_mutex_with.py" \
    --m 333 --n 444 --k 555 \
    --layout-a row --layout-b row \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
_run_basic_mmad_case "automatic mutex synchronization" \
    "${BASIC_MMAD_AUTO_SYNC_REL}" \
    --m 333 --n 444 --k 555 \
    --layout-a row --layout-b row \
    --dtype-a f16 --dtype-b f16 --dtype-c f32
_run_basic_mmad_atomic_add_cases

_run_basic_mmad_ptr_case() {
    echo "==> Running basic_mmad_ptr validation [ptr + offset -> make_tensor]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MMAD_PTR_REL}" --device "${DEVICE_ID}" --force-recompile
    )
}

_run_basic_mmad_ptr_case

_run_basic_vadd_case() {
    local label="$1"
    shift
    # One process per dtype; script itself does not sweep.
    local dtypes=(i8 i16 i32 f16 f32)
    local dtype
    for dtype in "${dtypes[@]}"; do
        echo "==> Running basic_vadd validation [${label}]: --dtype ${dtype} --device ${DEVICE_ID} $*"
        (
            cd "${TLA_DSL_DIR}"
            python "${BASIC_VADD_REL}" --dtype "${dtype}" --device "${DEVICE_ID}" \
                --force-recompile "$@"
        )
    done
}

_run_basic_vadd_case "flag sync"
_run_basic_vadd_case "mutex mode" --use-mutex
_run_basic_vadd_case "mutex with mode" --use-mutex-with
_run_basic_vadd_case "enable atomic add" --use-atomic-add

_run_basic_mixed_case() {
    local cache_mode="$1"
    shift
    echo "==> Running basic_mixed validation [dynamic GM mnk list, tensor print, ${cache_mode}]: --run --device ${DEVICE_ID} --block-dim 1 $*"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_REL}" --run --device "${DEVICE_ID}" --block-dim 1 "$@"
    )
}

_run_basic_mixed_case "forced compilation" --force-recompile
_run_basic_mixed_case "cache reuse"
_run_basic_mixed_case "mutex mode, forced compilation" --use-mutex --force-recompile

_run_basic_mixed_ub2l1_case() {
    echo "==> Running basic_mixed_ub2l1 validation [fixed shape/dtypes, gm->ub->l1]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_UB2L1_REL}" --run --device "${DEVICE_ID}" --force-recompile
    )
}

_run_basic_mixed_ub2l1_case

_run_basic_mixed_store_zN_case() {
    echo "==> Running basic_mixed_store_zN validation [gm->ub(row->zN)->l1]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_STORE_ZN_REL}" --run --device "${DEVICE_ID}" --force-recompile
    )
}

_run_basic_mixed_store_zN_case

_run_basic_mixed_store_zNUnAlign_case() {
    local label="$1"
    shift
    echo "==> Running basic_mixed_store_zNUnAlign validation [${label}]: --run --device ${DEVICE_ID} $*"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_STORE_ZNUNALIGN_REL}" --run --device "${DEVICE_ID}" "$@"
    )
}

# m=64 is fractal-aligned (multiple of 16); m=50 exercises the zNUnAlign M axis
# where the dest leaf[0] is the runtime row count and stride is runtime-varying.
_run_basic_mixed_store_zNUnAlign_case "m=64 (fractal-aligned)" --m 64
_run_basic_mixed_store_zNUnAlign_case "m=50 (non-aligned)" --m 50

_run_basic_mixed_fixpipe_nz2dn_case() {
    echo "==> Running basic_mixed_fixpipe_nz2dn validation [fixed shape/dtypes]: --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BASIC_MIXED_FIXPIPE_NZ2DN_REL}" --run --device "${DEVICE_ID}" --force-recompile
    )
}

_run_basic_mixed_fixpipe_nz2dn_case

_run_masked_binary_case() {
    echo "==> Running masked_binary validation [all dtypes]: masked_binary --sweep --shapes 400 --device ${DEVICE_ID} --force-recompile"
    (
        cd "${TLA_DSL_DIR}"
        python "${MASKED_BINARY_REL}" masked_binary --sweep --shapes 400 \
            --device "${DEVICE_ID}" --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_masked_binary_case

_run_bitwise_ops_case() {
    echo "==> Running bitwise_ops validation [all dtypes]: bitwise_ops --sweep --shapes 400 --device ${DEVICE_ID} --force-recompile"
    (
        cd "${TLA_DSL_DIR}"
        python "${BITWISE_OPS_REL}" bitwise_ops --sweep --shapes 400 \
            --device "${DEVICE_ID}" --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_bitwise_ops_case

_run_binary_op_batch() {
    echo "==> Running binary_op validation [batched ops, all dtypes]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${BINARY_OP_REL}" --batch-run \
            add sub mul div max min add_unalign add_brc_b32 \
            --shape 400 --batch-size 4 --device "${DEVICE_ID}" \
            --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_binary_op_batch

_run_reduction_ops_batch() {
    echo "==> Running reduction_ops validation [batched add/max/min f32]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${REDUCTION_OPS_REL}" --batch-run --device "${DEVICE_ID}" \
            --force-recompile
    )
}

_run_reduction_ops_batch

_run_load_store_scalar_after_reduction_case() {
    echo "==> Running load_and_store_scalar_after_reduction validation [f32]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${LOAD_STORE_SCALAR_AFTER_REDUCTION_REL}" \
            --device "${DEVICE_ID}" --force-recompile
    )
}

_run_load_store_scalar_after_reduction_case

_run_compare_mask_batch() {
    echo "==> Running compare_mask validation [batched ops]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${COMPARE_MASK_REL}" --batch-run "${COMPARE_MASK_OPS[@]}" \
            --shape 400 --batch-size 4 --device "${DEVICE_ID}" \
            --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_compare_mask_batch

_run_unary_ops_batch() {
    echo "==> Running unary_ops validation [batched ops, all dtypes]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${UNARY_OPS_REL}" --batch-run \
            exp log sqrt abs neg masked_unary masked_abs masked_neg \
            --shape 400 --batch-size 4 --device "${DEVICE_ID}" \
            --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_unary_ops_batch

_run_arange_op_case() {
    echo "==> Running arange_op validation [batched increase/decrease, all dtypes]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${ARANGE_OP_REL}" --batch-run increase decrease \
            --shape 400 --batch-size 4 --device "${DEVICE_ID}" \
            --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_arange_op_case

_run_interleave_op_batch() {
    echo "==> Running interleave_op validation [batched interleave/deinterleave, all dtypes]: --batch-run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${INTERLEAVE_OP_REL}" --batch-run interleave deinterleave \
            --shape 512 --batch-size 4 --device "${DEVICE_ID}" \
            --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_interleave_op_batch

_run_load_dintlv_op_case() {
    echo "==> Running load_dintlv_op validation [dintlv_b32 f32]: dintlv_b32 --sweep --shapes 512 --device ${DEVICE_ID} --force-recompile"
    (
        cd "${TLA_DSL_DIR}"
        python "${LOAD_DINTLV_OP_REL}" dintlv_b32 --sweep --shapes 512 \
            --device "${DEVICE_ID}" --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_load_dintlv_op_case

_run_load_us_b8_op_case() {
    echo "==> Running load_us_b8_op validation [us_b8 i8]: us_b8 --sweep --shapes 512 --device ${DEVICE_ID} --force-recompile"
    (
        cd "${TLA_DSL_DIR}"
        python "${LOAD_US_B8_OP_REL}" us_b8 --sweep --shapes 512 \
            --device "${DEVICE_ID}" --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_load_us_b8_op_case

_run_load_store_mask_case() {
    echo "==> Running load_store_mask validation [b8/b16/b32 carriers]: load_store_mask --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${LOAD_STORE_MASK_REL}" load_store_mask --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_load_store_mask_case

_run_store_pack_case() {
    echo "==> Running store_pack validation [i32/i16 only]: store_pack --run --all-dtypes --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${STORE_PACK_REL}" store_pack --run --all-dtypes --device "${DEVICE_ID}"
    )
}

_run_store_pack_case

_run_squeeze_op_case() {
    echo "==> Running squeeze_op validation [squeeze all dtypes]: squeeze --sweep --shapes 64 --device ${DEVICE_ID} --force-recompile"
    (
        cd "${TLA_DSL_DIR}"
        python "${SQUEEZE_OP_REL}" squeeze --sweep --shapes 64 \
            --device "${DEVICE_ID}" --compile-jobs "${COMPILE_JOBS}" \
            --force-recompile
    )
}

_run_squeeze_op_case

_run_register_control_flow_case() {
    echo "==> Running register_control_flow validation [mixed VectorSSA/MaskSSA carriers]: register_carriers --run --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${REGISTER_CONTROL_FLOW_REL}" register_carriers --run \
            --device "${DEVICE_ID}" --force-recompile
    )
}

_run_register_control_flow_case

_run_scalar_index_control_flow_case() {
    echo "==> Running scalar_index_control_flow validation [GM scalar indexing + AST compare]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${SCALAR_INDEX_CONTROL_FLOW_REL}" --device "${DEVICE_ID}" --force-recompile
    )
}

_run_scalar_index_control_flow_case

_run_scalar_kernel_arg_case() {
    echo "==> Running scalar_kernel_arg validation [Numeric args arithmetic]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${SCALAR_KERNEL_ARG_REL}" --device "${DEVICE_ID}" --force-recompile
    )
}

_run_scalar_kernel_arg_case

_run_print_tensor_gm_case() {
    local arch_scope="$1"
    local calls="$2"
    local block_count="$3"
    local dtype_mode="${4:-f32}"
    local dtype_args=(--dtype f32)
    if [[ "${dtype_mode}" == "all" ]]; then
        dtype_args=(--all-dtypes)
    fi
    echo "==> Running tensor tla.print validation [${arch_scope} calls=${calls} blocks=${block_count}]"
    (
        cd "${TLA_DSL_DIR}"
        python "${PRINT_TENSOR_REL}" --run --device "${DEVICE_ID}" \
            --arch-scope "${arch_scope}" \
            --block-dim "${block_count}" --calls "${calls}" \
            "${dtype_args[@]}" --force-recompile
    )
}

for _print_tensor_arch_scope in aiv.c310 aic.c310; do
    _run_print_tensor_gm_case "${_print_tensor_arch_scope}" 1 1 all
    _run_print_tensor_gm_case "${_print_tensor_arch_scope}" 1 2
    _run_print_tensor_gm_case "${_print_tensor_arch_scope}" 2 1
    _run_print_tensor_gm_case "${_print_tensor_arch_scope}" 2 2 all
done

_run_debug_print_matrix_case() {
    local arch_scope="$1"
    echo "==> Running debug_print validation [${arch_scope} all dtypes, two blocks]"
    (
        cd "${TLA_DSL_DIR}"
        python "${DEBUG_PRINT_REL}" --run --device "${DEVICE_ID}" \
            --arch-scope "${arch_scope}" --all-dtypes --block-dim 2 \
            --expect-count 2 --force-recompile
    )
}

for _debug_arch_scope in aiv.c310 aic.c310; do
    _run_debug_print_matrix_case "${_debug_arch_scope}"
done

_run_debug_print_f16_special_case() {
    local arch_scope="$1"
    local value="$2"
    echo "==> Running f16 debug_print special value [${arch_scope} ${value}]"
    (
        cd "${TLA_DSL_DIR}"
        python "${DEBUG_PRINT_REL}" --run --device "${DEVICE_ID}" \
            --arch-scope "${arch_scope}" --dtype f16 --value="${value}" \
            --force-recompile
    )
}

for _debug_arch_scope in aiv.c310 aic.c310; do
    for _debug_f16_value in -0.0 nan inf -inf; do
        _run_debug_print_f16_special_case \
            "${_debug_arch_scope}" "${_debug_f16_value}"
    done
done

_run_debug_print_expression_matrix_case() {
    local arch_scope="$1"
    echo "==> Running computed debug_print validation [${arch_scope} i8/i16/i32/f16/f32]"
    (
        cd "${TLA_DSL_DIR}"
        python "${DEBUG_PRINT_REL}" --run --device "${DEVICE_ID}" \
            --arch-scope "${arch_scope}" --all-dtypes --expression \
            --force-recompile
    )
}

for _debug_arch_scope in aiv.c310 aic.c310; do
    _run_debug_print_expression_matrix_case "${_debug_arch_scope}"
done

_run_debug_print_mixed_case() {
    local print_region="$1"
    echo "==> Running debug_print_mixed validation [${print_region}]: --run --device ${DEVICE_ID} --print-region ${print_region}"
    (
        cd "${TLA_DSL_DIR}"
        python "${DEBUG_PRINT_MIXED_REL}" --run --device "${DEVICE_ID}" \
            --all-dtypes --print-region "${print_region}" --force-recompile
    )
}

for _debug_print_region in cube vector both; do
    _run_debug_print_mixed_case "${_debug_print_region}"
done

_run_debug_print_format_case() {
    local arch_scope="$1"
    echo "==> Running debug_print_format validation [${arch_scope}, 2 blocks]"
    (
        cd "${TLA_DSL_DIR}"
        python "${DEBUG_PRINT_FORMAT_REL}" --run --device "${DEVICE_ID}" \
            --arch-scope "${arch_scope}" --block-dim 2 --force-recompile
    )
}

for _debug_arch_scope in aiv.c310 aic.c310; do
    _run_debug_print_format_case "${_debug_arch_scope}"
done

_run_scalar_arg_alignment_case() {
    echo "==> Running scalar_arg_alignment validation [tensor + i16 + tensor]: --device ${DEVICE_ID}"
    (
        cd "${TLA_DSL_DIR}"
        python "${SCALAR_ARG_ALIGNMENT_REL}" --device "${DEVICE_ID}" --force-recompile
    )
}

_run_scalar_arg_alignment_case

_run_print_tensor_ub_case() {
    local case_name="$1"
    local calls="$2"
    local block_count="$3"
    local dtype_mode="${4:-f32}"
    local dtype_args=(--dtype f32)
    if [[ "${dtype_mode}" == "all" ]]; then
        dtype_args=(--all-dtypes)
    fi
    echo "==> Running print_tensor validation [AIV UB ${case_name} calls=${calls} blocks=${block_count}]"
    (
        cd "${TLA_DSL_DIR}"
        python "${PRINT_TENSOR_REL}" --run --device "${DEVICE_ID}" \
            --storage ub --case "${case_name}" --arch-scope aiv.c310 \
            --block-dim "${block_count}" --calls "${calls}" \
            "${dtype_args[@]}" --force-recompile
    )
}

_run_print_tensor_ub_case "base" 1 1 all
_run_print_tensor_ub_case "base" 1 2
_run_print_tensor_ub_case "base" 2 1
_run_print_tensor_ub_case "base" 2 2 all
_run_print_tensor_ub_case "aligned-offset" 1 1 all

echo "==> run_dsl_test.sh finished successfully"
