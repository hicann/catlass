#!/bin/bash
# This script applies CATLASS's patches to the AscendNPU-IR source tree.
#
# The AscendNPU-IR tree is searched in the same order build.sh uses:
#   1. $CATLASS_DSL_ASCENDNPU_IR_ROOT
#   2. python/tla_dsl/3rdparty/AscendNPU-IR (the submodule)
#   3. $CATLASS_DSL_PREBUILT_ASCENDNPU_IR
#
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DSL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
readonly PATCH_FOLDER="${SCRIPT_DIR}/AscendNPU-IR"

SUBMODULE_ROOT="${DSL_ROOT}/3rdparty/AscendNPU-IR"
SOURCE_CANDIDATES=(
  "${CATLASS_DSL_ASCENDNPU_IR_ROOT:-}"
  "${SUBMODULE_ROOT}"
  "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}"
)

NPU_IR_ROOT=""
for candidate in "${SOURCE_CANDIDATES[@]}"; do
  if [[ -n "${candidate}" && -f "${candidate}/build-tools/build.sh" ]]; then
    NPU_IR_ROOT="${candidate}"
    break
  fi
done
if [[ -z "${NPU_IR_ROOT}" ]]; then
  echo "Error: unable to locate an AscendNPU-IR source tree. Checked:" >&2
  echo "  \$CATLASS_DSL_ASCENDNPU_IR_ROOT" >&2
  echo "  ${SUBMODULE_ROOT}" >&2
  echo "  \$CATLASS_DSL_PREBUILT_ASCENDNPU_IR" >&2
  exit 1
fi

echo "Applying patches to ${NPU_IR_ROOT}"
cd "${NPU_IR_ROOT}"
for patch_file in "${PATCH_FOLDER}"/*.patch; do
  echo "  $(basename "${patch_file}")"
  patch -p1 < "${patch_file}"
done
echo "Finished applying patches"
