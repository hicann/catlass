# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# jit_verify_template.cmake
#
# Build-time verification that JIT templates compile with bisheng for each
# target NPU architecture. Creates OBJECT libraries per template+arch so
# the build tool (Ninja/Make) naturally parallelizes all checks via -j.
#
# Usage:
#   include(jit_verify_template)
#   jit_verify_template(
#       NAME          basic_matmul
#       TEMPLATE      path/to/basic_matmul_impl.cpp
#       NPU_ARCH_LIST 2201 3510
#       INCLUDE_DIRS  dir1 dir2 ...)
# ---------------------------------------------------------------------------

function(jit_verify_template)
    cmake_parse_arguments(_TV "" "NAME;TEMPLATE" "NPU_ARCH_LIST;INCLUDE_DIRS" ${ARGN})

    if(NOT _TV_NAME OR NOT _TV_TEMPLATE OR NOT _TV_NPU_ARCH_LIST)
        message(FATAL_ERROR "jit_verify_template: NAME, TEMPLATE, NPU_ARCH_LIST required")
    endif()

    if(NOT IS_ABSOLUTE "${_TV_TEMPLATE}")
        set(_TV_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/${_TV_TEMPLATE}")
    endif()

    if(NOT EXISTS "${_TV_TEMPLATE}")
        message(FATAL_ERROR "jit_verify_template: template not found at ${_TV_TEMPLATE}")
    endif()

    foreach(_ARCH ${_TV_NPU_ARCH_LIST})
        set(_TGT jit_verify_${_TV_NAME}_${_ARCH})
        add_library(${_TGT} OBJECT EXCLUDE_FROM_ALL ${_TV_TEMPLATE})
        set_source_files_properties(${_TV_TEMPLATE} PROPERTIES LANGUAGE ASC)
        target_compile_options(${_TGT} PRIVATE
            -DCATLASS_ARCH=${_ARCH}
            --npu-arch=dav-${_ARCH}
            -Wno-macro-redefined
        )
        target_include_directories(${_TGT} PRIVATE ${_TV_INCLUDE_DIRS})
        set_property(GLOBAL APPEND PROPERTY _JIT_VERIFY_TARGETS ${_TGT})
    endforeach()
endfunction()
