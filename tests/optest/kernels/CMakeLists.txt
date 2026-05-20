# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_ALL_ARCHS "")
include_directories(${CATLASS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/jit_try_compile_check.cmake)

set(_JIT_COMPILER_TARGET catlass_kernel_jit_compiler)
add_library(${_JIT_COMPILER_TARGET} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_compiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_logger.cpp)
target_include_directories(${_JIT_COMPILER_TARGET} PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${CATLASS_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}
)
target_link_directories(${_JIT_COMPILER_TARGET} PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${_JIT_COMPILER_TARGET} PRIVATE
    catlass_kernel_utils
    ascendc_runtime
    runtime
    profapi
    mmpa
    ascend_dump
    c_sec
    error_manager
    ascendcl
    tiling_api
    nnopbase
    platform
    dl
)
install(TARGETS ${_JIT_COMPILER_TARGET}
    DESTINATION lib/jit)

set(_JIT_KERNEL_ENTRIES "")
set(_JIT_KERNEL_TEMPLATES "")

# compile options for msSanitizer
# 支持两种方式启用：
#   1. cmake -DENABLE_MSSANITIZER=ON
#   2. export MS_SANITIZE_MEMORY=1  (从环境变量自动检测)
if(NOT DEFINED ENABLE_MSSANITIZER)
    if(DEFINED ENV{MS_SANITIZE_MEMORY} AND "$ENV{MS_SANITIZE_MEMORY}" STREQUAL "1")
        set(ENABLE_MSSANITIZER ON)
    endif()
endif()
if(DEFINED ENABLE_MSSANITIZER AND ENABLE_MSSANITIZER)
    message(STATUS "msSanitizer ENABLED (via cmake arg or MS_SANITIZE_MEMORY env)")
endif()

macro(add_kernel)
    set(_K_OPTIONS "")
    set(_K_ONE_VALUE "NAME;KERNEL_TYPE;TEMPLATE")
    set(_K_MULTI_VALUE "NPU_ARCH_LIST")
    cmake_parse_arguments(_K "${_K_OPTIONS}" "${_K_ONE_VALUE}" "${_K_MULTI_VALUE}" ${ARGN})

    # 参数校验
    if(NOT DEFINED _K_NAME)
        message(FATAL_ERROR "add_kernel: NAME is required")
    endif()
    if(NOT DEFINED _K_NPU_ARCH_LIST)
        message(FATAL_ERROR "add_kernel: NPU_ARCH_LIST is required")
    endif()
    if(NOT DEFINED _K_KERNEL_TYPE)
        message(FATAL_ERROR "add_kernel: KERNEL_TYPE is required (jit or prebuilt)")
    endif()

    if("${_K_KERNEL_TYPE}" STREQUAL "jit")
        # -------------------------------------------------------
        # JIT 模式 — 收集源文件到统一库
        # -------------------------------------------------------
        set(_KERNEL_ENTRY "${_K_UNPARSED_ARGUMENTS}")
        set(_KERNEL_TEMPLATE "${_K_TEMPLATE}")

        # 收集 entry 源文件到列表，构建时统一编译
        list(APPEND _JIT_KERNEL_ENTRIES "${_KERNEL_ENTRY}")
        set(_JIT_KERNEL_ENTRIES ${_JIT_KERNEL_ENTRIES} PARENT_SCOPE)

        if(_KERNEL_TEMPLATE)
            list(APPEND _JIT_KERNEL_TEMPLATES "${_KERNEL_TEMPLATE}")
            set(_JIT_KERNEL_TEMPLATES ${_JIT_KERNEL_TEMPLATES} PARENT_SCOPE)

            set(_JIT_INCLUDE_DIRS
                "${CATLASS_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/.."
                "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_BINARY_DIR}"
            )

            jit_try_compile_check(
                NAME "${_K_NAME}"
                TEMPLATE "${_KERNEL_TEMPLATE}"
                NPU_ARCH_LIST ${_K_NPU_ARCH_LIST}
                INCLUDE_DIRS ${_JIT_INCLUDE_DIRS}
            )
        endif()

    elseif("${_K_KERNEL_TYPE}" STREQUAL "prebuilt")
        foreach(_ARCH ${_K_NPU_ARCH_LIST})
            list(FIND CATLASS_ARCH_LIST ${_ARCH} _ARCH_INDEX)
            if(_ARCH_INDEX EQUAL -1)
                continue()
            endif()

            list(APPEND CATLASS_ALL_ARCHS ${_ARCH})

            set(_INSTALL_TARGETS "")
            foreach(_MS_SUFFIX "" "_ms")
                set(_LIB_NAME "catlass_kernel_${_ARCH}_${_K_NAME}${_MS_SUFFIX}")
                list(APPEND _INSTALL_TARGETS ${_LIB_NAME})
                add_library(${_LIB_NAME} SHARED ${_K_UNPARSED_ARGUMENTS})

                target_compile_definitions(${_LIB_NAME} PRIVATE CATLASS_ARCH=${_ARCH})

                # 通用 ASC 编译选项
                target_compile_options(${_LIB_NAME} PRIVATE
                    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${_ARCH}>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
                )

                if(_ARCH STREQUAL "2201")
                    target_compile_options(${_LIB_NAME} PRIVATE
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-record-overflow=true>"
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
                    )
                endif()

                # _ms 版本附加 sanitizer 标志
                if(_MS_SUFFIX STREQUAL "_ms")
                    target_compile_options(${_LIB_NAME} PRIVATE
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-g --cce-enable-sanitizer>"
                    )
                    target_link_options(${_LIB_NAME} PRIVATE
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:--cce-enable-sanitizer --npu-arch=dav-${_ARCH}>"
                    )
                    message(STATUS "  [ms] ${_LIB_NAME} will be built with msSanitizer")
                endif()

                target_link_options(${_LIB_NAME} PRIVATE
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>
                )

                target_include_directories(${_LIB_NAME} PRIVATE
                    ${ASCEND_HOME_PATH}/include
                    ${CATLASS_DIR}/include
                    ${CMAKE_BINARY_DIR}
                )

                target_link_directories(${_LIB_NAME} PRIVATE
                    ${ASCEND_HOME_PATH}/lib64
                )

                target_link_libraries(${_LIB_NAME} PRIVATE
                    ascendc_runtime
                    runtime
                    profapi
                    mmpa
                    ascend_dump
                    c_sec
                    error_manager
                    ascendcl
                    tiling_api
                    nnopbase
                    platform
                    dl
                )

                if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
                    target_link_libraries(${_LIB_NAME} PRIVATE ascendalog)
                endif()
                if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
                    target_link_libraries(${_LIB_NAME} PRIVATE unified_dlog)
                endif()
            endforeach()

            install(TARGETS ${_INSTALL_TARGETS}
                DESTINATION lib/${_ARCH}
            )
            unset(_INSTALL_TARGETS)
        endforeach()

    else()
        message(FATAL_ERROR
            "add_kernel: unknown KERNEL_TYPE '${_K_KERNEL_TYPE}'. "
            "Expected 'jit' or 'prebuilt'.")
    endif()
endmacro()

add_subdirectory(00_basic_matmul)

if(_JIT_KERNEL_ENTRIES)
    set(_JIT_TARGET catlass_kernel_jit)
    add_library(${_JIT_TARGET} SHARED ${_JIT_KERNEL_ENTRIES})

    target_include_directories(${_JIT_TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${CATLASS_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_BINARY_DIR}
    )
    target_link_directories(${_JIT_TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/lib64
    )
    target_link_libraries(${_JIT_TARGET} PRIVATE
        ${_JIT_COMPILER_TARGET}
        catlass_kernel_utils
        dl
    )

    install(TARGETS ${_JIT_TARGET}
        DESTINATION lib/jit)
endif()

# 安装 JIT 模板
foreach(_TEMPLATE ${_JIT_KERNEL_TEMPLATES})
    install(FILES ${_TEMPLATE}
        DESTINATION jit/templates)
endforeach()
