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
# End-to-end validation for python/tla_dsl/examples/end_to_end/basic_mmad (basic_matmul.py).
#
# Workspace layout (sibling dirs under ascend-catlass-dsl root):
#   Ascend/9.0.0-beta.2/ascend-toolkit/set_env.sh
#   AscendNPU-IR/          (prebuilt bishengir + build/)
#   HIVMC/hivmc-a5
#   catlass_DSL/python/tla_dsl/
#
# Usage:
#   bash catlass_DSL/tests/run_dsl_test.sh
#   bash catlass_DSL/tests/run_dsl_test.sh --skip-prepare --device 0

set -euo pipefail

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CATDSL_ROOT="$(realpath "${SCRIPT_PATH}/..")"
WORKSPACE_ROOT="${ASCEND_CATLASS_DSL_ROOT:-$(dirname "${CATDSL_ROOT}")}"

TLA_DSL_DIR="${TLA_DSL_DIR:-${CATDSL_ROOT}/python/tla_dsl}"

CONDA_ENV="${CONDA_ENV:-ascend-catlass-dsl}"
DEVICE_ID="${DEVICE_ID:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

BASIC_MMAD_REL="examples/end_to_end/basic_mmad/basic_matmul.py"

_resolve_default_paths() {
    local ascendnpu_ir="" cann_set_env="" hivmc_a5=""

    if [[ -d "${WORKSPACE_ROOT}/AscendNPU-IR" ]]; then
        ascendnpu_ir="${WORKSPACE_ROOT}/AscendNPU-IR"
    elif [[ -d "${CATDSL_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR" ]]; then
        ascendnpu_ir="${CATDSL_ROOT}/python/tla_dsl/3rdparty/AscendNPU-IR"
    fi

    if [[ -f "${WORKSPACE_ROOT}/Ascend/9.0.0-beta.2/ascend-toolkit/set_env.sh" ]]; then
        cann_set_env="${WORKSPACE_ROOT}/Ascend/9.0.0-beta.2/ascend-toolkit/set_env.sh"
    elif [[ -f "${WORKSPACE_ROOT}/Ascend/ascend-toolkit/set_env.sh" ]]; then
        cann_set_env="${WORKSPACE_ROOT}/Ascend/ascend-toolkit/set_env.sh"
    elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
        cann_set_env="/usr/local/Ascend/ascend-toolkit/set_env.sh"
    fi

    if [[ -x "${WORKSPACE_ROOT}/HIVMC/hivmc-a5" || -f "${WORKSPACE_ROOT}/HIVMC/hivmc-a5" ]]; then
        hivmc_a5="${WORKSPACE_ROOT}/HIVMC/hivmc-a5"
    elif [[ -n "${ascendnpu_ir}" && -x "${ascendnpu_ir}/build/bin/hivmc-a5" ]]; then
        hivmc_a5="${ascendnpu_ir}/build/bin/hivmc-a5"
    fi

    TLA_DSL_PREBUILT_ASCENDNPU_IR="${TLA_DSL_PREBUILT_ASCENDNPU_IR:-${ascendnpu_ir}}"
    TLA_DSL_ASCENDNPU_IR_ROOT="${TLA_DSL_ASCENDNPU_IR_ROOT:-${ascendnpu_ir}}"
    CANN_SET_ENV_SH="${CANN_SET_ENV_SH:-${cann_set_env}}"
    TLA_DSL_HIVMC_A5="${TLA_DSL_HIVMC_A5:-${hivmc_a5}}"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Run basic_mmad end-to-end validation (basic_matmul.py --run --all-layouts --all-mmad-dtypes).
Runs default MNK plus m=1, n=2, k=3.
Activates conda env "${CONDA_ENV}", sources CANN set_env.sh, then builds (optional) and runs the test.

Options:
  -h, --help              Show this help
  --device ID             NPU device id (default: ${DEVICE_ID})
  --skip-prepare          Skip git submodule update and ./build.sh

Paths (auto from script location):
  WORKSPACE_ROOT=${WORKSPACE_ROOT}   (override: ASCEND_CATLASS_DSL_ROOT)
  CATDSL_ROOT=${CATDSL_ROOT}
  TLA_DSL_DIR=${TLA_DSL_DIR}
  CONDA_ENV=${CONDA_ENV}

Toolchain (auto-detected when present):
  TLA_DSL_PREBUILT_ASCENDNPU_IR  (default: ${TLA_DSL_PREBUILT_ASCENDNPU_IR:-<not found>})
  TLA_DSL_ASCENDNPU_IR_ROOT      (default: ${TLA_DSL_ASCENDNPU_IR_ROOT:-<not found>})
  TLA_DSL_HIVMC_A5               (default: ${TLA_DSL_HIVMC_A5:-<not found>})
  CANN_SET_ENV_SH                (default: ${CANN_SET_ENV_SH:-<not found>})

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

_resolve_default_paths

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

_export_toolchain_env() {
    echo "==> Exporting toolchain env"
    echo "    WORKSPACE_ROOT=${WORKSPACE_ROOT}"
    if [[ -n "${TLA_DSL_PREBUILT_ASCENDNPU_IR}" ]]; then
        echo "    TLA_DSL_PREBUILT_ASCENDNPU_IR=${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
        export TLA_DSL_PREBUILT_ASCENDNPU_IR
    else
        echo "warning: TLA_DSL_PREBUILT_ASCENDNPU_IR not set (AscendNPU-IR not found)" >&2
    fi
    if [[ -n "${TLA_DSL_ASCENDNPU_IR_ROOT}" ]]; then
        echo "    TLA_DSL_ASCENDNPU_IR_ROOT=${TLA_DSL_ASCENDNPU_IR_ROOT}"
        export TLA_DSL_ASCENDNPU_IR_ROOT
    fi
    if [[ -n "${TLA_DSL_HIVMC_A5:-}" ]]; then
        echo "    TLA_DSL_HIVMC_A5=${TLA_DSL_HIVMC_A5}"
        export TLA_DSL_HIVMC_A5
    else
        echo "warning: TLA_DSL_HIVMC_A5 not set (HIVMC/hivmc-a5 not found)" >&2
    fi
}

_prepare_tla_dsl() {
    if [[ -d "${TLA_DSL_PREBUILT_ASCENDNPU_IR}/build" ]]; then
        echo "==> Using prebuilt AscendNPU-IR at ${TLA_DSL_PREBUILT_ASCENDNPU_IR}; skipping AscendNPU-IR submodule"
        if [[ -f "${CATDSL_ROOT}/.gitmodules" ]]; then
            (
                cd "${CATDSL_ROOT}"
                git submodule update --init --depth 1 3rdparty/googletest 2>/dev/null || true
            )
        fi
    else
        echo "==> git submodule update --init --depth 1 (from ${CATDSL_ROOT})"
        (
            cd "${CATDSL_ROOT}"
            git submodule update --init --depth 1
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

_export_toolchain_env

if [[ -z "${CANN_SET_ENV_SH}" || ! -f "${CANN_SET_ENV_SH}" ]]; then
    echo "error: CANN set_env.sh not found (set CANN_SET_ENV_SH)" >&2
    exit 1
fi
echo "==> Sourcing CANN: ${CANN_SET_ENV_SH}"
# shellcheck disable=SC1090
source "${CANN_SET_ENV_SH}"

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

echo "==> run_dsl_test.sh finished successfully"
