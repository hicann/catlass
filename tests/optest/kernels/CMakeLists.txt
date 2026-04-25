# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_ALL_ARCHS "")
include_directories(${CATLASS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

macro(add_kernel NAME ARCH)
    if(DEFINED CATLASS_KERNEL_LIST AND NOT "${CATLASS_KERNEL_LIST}" STREQUAL "")
        list(FIND CATLASS_KERNEL_LIST ${NAME} _INDEX)
        if(_INDEX EQUAL -1)
            return()
        endif()
    endif()

    if("${ARCH}" MATCHES ";")
        set(_ARCH_LIST ${ARCH})
    else()
        set(_ARCH_LIST "${ARCH}")
    endif()

    foreach(_ARCH ${_ARCH_LIST})
        list(FIND CATLASS_ARCH_LIST ${_ARCH} _ARCH_INDEX)
        if(_ARCH_INDEX EQUAL -1)
            continue()
        endif()

        list(APPEND CATLASS_ALL_ARCHS ${_ARCH})

        set(_LIB_NAME "catlass_kernel_${_ARCH}_${NAME}")
        add_library(${_LIB_NAME} SHARED ${ARGN})

        target_compile_definitions(${_LIB_NAME} PRIVATE CATLASS_ARCH=${_ARCH})

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

        target_link_options(${_LIB_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>
        )

        target_include_directories(${_LIB_NAME} PRIVATE
            ${ASCEND_HOME_PATH}/include
            ${CATLASS_DIR}/include
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

        install(TARGETS ${_LIB_NAME}
            DESTINATION lib/${_ARCH}
        )
    endforeach()
endmacro()

add_subdirectory(00_basic_matmul)
