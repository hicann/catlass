#!/bin/bash

set -e

BUILD_DIR="build"
INSTALL_DIR="$(pwd)/torch_catlass/lib"

echo "Building catlass_torch..."

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
)

if [ -n "$CATLASS_KERNEL_LIST" ]; then
    CMAKE_ARGS+=(-DCATLASS_KERNEL_LIST="$CATLASS_KERNEL_LIST")
    echo "Building kernels: $CATLASS_KERNEL_LIST"
fi

if [ -n "$CATLASS_ARCH_LIST" ]; then
    CMAKE_ARGS+=(-DCATLASS_ARCH_LIST="$CATLASS_ARCH_LIST")
    echo "Building architectures: $CATLASS_ARCH_LIST"
fi

cmake .. "${CMAKE_ARGS[@]}"

make -j$(nproc)
make install

echo "Build completed successfully!"
echo "Libraries installed to: $INSTALL_DIR"
