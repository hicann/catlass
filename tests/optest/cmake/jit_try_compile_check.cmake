# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# jit_try_compile_check.cmake
#
# Verify that a JIT template (.cpp) compiles with bisheng for a given set of
# NPU architectures. Uses the static wrapper cmake/jit_template_wrapper.asc
# (checked into git, never generated) + -DTEMPLATE_PATH to specify which
# template to include.
#
# Uses try_compile source-file variant: CMake picks bisheng based on .asc
# extension. No sub-project CMakeLists.txt needed.
#
# Usage:
#   include(jit_try_compile_check)
#   jit_try_compile_check(
#       NAME          basic_matmul
#       TEMPLATE      path/to/basic_matmul_impl.cpp
#       NPU_ARCH_LIST 2201 3510
#       INCLUDE_DIRS  dir1 dir2 ...)
# ---------------------------------------------------------------------------

function(jit_try_compile_check)
    cmake_parse_arguments(_TC "" "NAME;TEMPLATE" "NPU_ARCH_LIST;INCLUDE_DIRS" ${ARGN})

    if(NOT _TC_NAME OR NOT _TC_TEMPLATE OR NOT _TC_NPU_ARCH_LIST)
        message(FATAL_ERROR "jit_try_compile_check: NAME, TEMPLATE, NPU_ARCH_LIST required")
    endif()

    # Locate the static wrapper (checked into git, relative to this cmake module)
    get_filename_component(_TC_MODULE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" DIRECTORY)
    set(_TC_WRAPPER "${_TC_MODULE_DIR}/cmake/jit_template_wrapper.asc")
    if(NOT EXISTS "${_TC_WRAPPER}")
        message(FATAL_ERROR "jit_try_compile_check: wrapper not found at ${_TC_WRAPPER}")
    endif()

    # Resolve template path relative to current source dir
    if(NOT IS_ABSOLUTE "${_TC_TEMPLATE}")
        set(_TC_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/${_TC_TEMPLATE}")
    endif()

    # Extract template filename (used as -DTEMPLATE_PATH=name)
    get_filename_component(_TC_TEMPLATE_DIR "${_TC_TEMPLATE}" DIRECTORY)
    get_filename_component(_TC_TEMPLATE_NAME "${_TC_TEMPLATE}" NAME)

    foreach(_ARCH ${_TC_NPU_ARCH_LIST})
        # Build directory for this try_compile — one per name+arch
        set(_TC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/jit_try_compile/${_TC_NAME}/asc-${_ARCH}")

        # Build compile definitions: -D flags + include dirs + arch
        set(_TC_DEFS
            "-DTEMPLATE_PATH=${_TC_TEMPLATE_NAME}"
            "-DCATLASS_ARCH=${_ARCH}"
            "-I${_TC_TEMPLATE_DIR}"
            "--npu-arch=dav-${_ARCH}"
            -Wno-macro-redefined
        )
        foreach(_DIR ${_TC_INCLUDE_DIRS})
            list(APPEND _TC_DEFS "-I${_DIR}")
        endforeach()

        # Source-file variant: CMake picks ASC compiler from .asc extension
        try_compile(_TC_RESULT
            "${_TC_BINARY_DIR}"
            "${_TC_WRAPPER}"
            CMAKE_FLAGS
            "-DCMAKE_ASC_COMPILER=${CMAKE_ASC_COMPILER}"
            COMPILE_DEFINITIONS
            ${_TC_DEFS}
            OUTPUT_VARIABLE _TC_OUTPUT
        )

        if(_TC_RESULT)
            message(STATUS "JIT template '${_TC_TEMPLATE_NAME}' can be compiled [dav-${_ARCH}]")
        else()
            message("${_TC_OUTPUT}")
            message(FATAL_ERROR "JIT template '${_TC_TEMPLATE_NAME}' can not be compiled [dav-${_ARCH}]")
        endif()
    endforeach()
endfunction()
