#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --kernel-list <kernels>     Comma-separated list of kernels to build"
    echo "                              (e.g., basic_matmul,batched_matmul)"
    echo "  --arch-list <archs>         Comma-separated list of architectures"
    echo "                              (e.g., 2201,3510)"
    echo "  --build-type <type>         Build type (Release|Debug), default: Release"
    echo "  --skip-wheel                Skip building wheel package (editable install)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --arch-list 2201"
    echo "  $0 --kernel-list basic_matmul --arch-list 2201,3510"
}

KERNEL_LIST=""
ARCH_LIST=""
BUILD_TYPE="Release"
SKIP_WHEEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernel-list)
            KERNEL_LIST="$2"
            shift 2
            ;;
        --arch-list)
            ARCH_LIST="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --skip-wheel)
            SKIP_WHEEL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Building torch-catlass with scikit-build-core..."
echo "============================================"
[ -n "$KERNEL_LIST" ] && echo "Kernels: $KERNEL_LIST"
[ -n "$ARCH_LIST" ] && echo "Architectures: $ARCH_LIST"
echo "Build type: $BUILD_TYPE"
echo ""

CONFIG_SETTINGS=()
if [ -n "$KERNEL_LIST" ]; then
    CONFIG_SETTINGS+=(--config-settings="cmake.define.CATLASS_KERNEL_LIST=${KERNEL_LIST}")
fi
if [ -n "$ARCH_LIST" ]; then
    CONFIG_SETTINGS+=(--config-settings="cmake.define.CATLASS_ARCH_LIST=${ARCH_LIST}")
fi

if [ "$SKIP_WHEEL" = true ]; then
    .venv/bin/pip install -e . -v --no-build-isolation "${CONFIG_SETTINGS[@]}"
else
    mkdir -p dist
    .venv/bin/pip wheel . -v --no-deps --no-build-isolation -w dist/ "${CONFIG_SETTINGS[@]}"
fi

echo "============================================"
echo "Build completed successfully!"
echo "============================================"
[ "$SKIP_WHEEL" = false ] && echo "Wheel package in: $SCRIPT_DIR/dist/"
