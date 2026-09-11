# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_X_ATTENTION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/x_attention.cpp
)
set_source_files_properties(${_X_ATTENTION_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(NAME x_attention
    NPU_ARCH_LIST 2201
    KERNEL_TYPE prebuilt
    ${_X_ATTENTION_SOURCES}
)

foreach(_MS_SUFFIX "" "_ms")
    set(_LIB_NAME "catlass_kernel_2201_x_attention${_MS_SUFFIX}")
    if(TARGET ${_LIB_NAME})
        target_include_directories(${_LIB_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        )
        target_compile_options(${_LIB_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:ASC>:--cce-auto-sync=on>
        )
    endif()
endforeach()
