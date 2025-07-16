#!/bin/bash

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -o errexit
set -o nounset
set -o pipefail

function get_npu_model(){
    if command -v npu-smi &> /dev/null; then
        echo "Ascend$(npu-smi info -t board -i 0 -c 0 | grep \"Chip Name\" | awk '{print $NF}')"
        return 1
    else
        echo "Ascend910B1"
        return 0
    fi
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CMAKE_SOURCE_DIR=$(realpath "$SCRIPT_DIR/..")
BUILD_DIR="$CMAKE_SOURCE_DIR/build"
OUTPUT_DIR="$CMAKE_SOURCE_DIR/output"

echo -e "  ____    _  _____ _        _    ____ ____  "
echo -e " / ___|  / \|_   _| |      / \  / ___/ ___| "
echo -e "| |     / _ \ | | | |     / _ \ \___ \___ \ "
echo -e "| |___ / ___ \| | | |___ / ___ \ ___) |__) |"
echo -e " \____/_/   \_\_| |_____/_/   \_\____/____/ "

function show_help() {
    echo -e "${GREEN}Usage:${NC} $0 [options] <target>"
    echo -e "\n${BLUE}Options:${NC}"
    echo "  --clean         Clean build directories"
    echo "  --debug         Build in debug mode"
    echo "  --msdebug       Enable msdebug support"
    echo "  --enable_msprof Enable msprofiling"
    echo "  -D<option>      Additional CMake options"
    echo -e "\n${BLUE}Targets:${NC}"
    echo "  catlass_examples  Build Catlass examples"
    echo "  python_extension  Build Python extension"
    echo "  torch_library     Build Torch library"
    echo "  <other>           Other CMake targets"
}

TARGET=""
CMAKE_BUILD_TYPE="Release"
declare -a CMAKE_OPTIONS=()
CLEAN=false
POST_BUILD_HINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --clean)
            CLEAN=true
            ;;
        --debug)
            CMAKE_BUILD_TYPE="Debug"
            echo -e "${YELLOW}Debug mode enabled${NC}"
            ;;
        --msdebug)
            CMAKE_OPTIONS+=("-DENABLE_MSDEBUG=True")
            ;;
        --simulator)
            CMAKE_OPTIONS+=("-DENABLE_SIMULATOR=True")
            if NPU_MODEL=$(get_npu_model); then
                echo -e "${YELLOW}No npu-smi detected, using default model for simulator: ${NPU_MODEL}${NC}"
            fi
            CMAKE_OPTIONS+=("-DSIMULATOR_NPU_MODEL=${NPU_MODEL}")
            POST_BUILD_HINT="Please run \nexport LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${NPU_MODEL}/lib:\$LD_LIBRARY_PATH\nin your terminal before execute examples."            ;;
        --enable_msprof)
            CMAKE_OPTIONS+=("-DENABLE_MSPROF=True")
            ;;
        -D*)
            CMAKE_OPTIONS+=("$1")
            ;;
        *)
            if [[ -z "$TARGET" ]]; then
                TARGET="$1"
            else
                echo -e "${RED}Error: Multiple targets specified${NC}" >&2
                show_help
                exit 1
            fi
            ;;
    esac
    shift
done

if [[ "$CLEAN" == true ]]; then
    echo -e "${BLUE}Cleaning build directories...${NC}"
    rm -rf "$BUILD_DIR" "$OUTPUT_DIR"
    echo -e "${GREEN}Clean complete${NC}"
fi

if [[ -z "$TARGET" ]]; then
    echo -e "${RED}Error: No target specified${NC}" >&2
    show_help
    exit 1
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

function build_python_extension() {
    echo -e "${BLUE}Building Python extension...${NC}"
    cd "$CMAKE_SOURCE_DIR/examples/python_extension" || exit 1
    rm -rf build dist ./*.egg-info
    python3 setup.py bdist_wheel --dist-dir "$OUTPUT_DIR/python_extension"
    echo -e "${GREEN}Python extension built successfully${NC}"
}

function build_torch_library() {
    echo -e "${BLUE}Building Torch library...${NC}"
    cmake -B build \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR/python_extension" \
        -DCATLASS_INCLUDE_DIR="$CMAKE_SOURCE_DIR/include" \
        -DPython3_EXECUTABLE="$(which python3)" \
        -DBUILD_TORCH_LIB=True
    cmake --build build --target catlass_torch -j
    cmake --install build --component catlass_torch
    echo -e "${GREEN}Torch library built successfully${NC}"
}

# 执行构建
case "$TARGET" in
    python_extension)
        build_python_extension
        ;;
    torch_library)
        build_torch_library
        ;;
    *)
        echo -e "${BLUE}Building target: $TARGET...${NC}"
        cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
            "${CMAKE_OPTIONS[@]}"
        cmake --build "$BUILD_DIR" --target "$TARGET" -j
        cmake --install "$BUILD_DIR" --component "$TARGET"
        echo -e "${GREEN}Target '$TARGET' built successfully${NC}"
        ;;
esac

echo -e "$POST_BUILD_HINT"