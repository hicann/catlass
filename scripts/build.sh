#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath $0))
CMAKE_SOURCE_PATH=$(realpath $SCRIPT_PATH/..)

CMAKE_BUILD_PATH=$CMAKE_SOURCE_PATH/build

TARGET=$1

mkdir -p $CMAKE_BUILD_PATH
cmake --no-warn-unused-cli -S$CMAKE_SOURCE_PATH -B$CMAKE_BUILD_PATH
cmake --build $CMAKE_BUILD_PATH --target $TARGET
