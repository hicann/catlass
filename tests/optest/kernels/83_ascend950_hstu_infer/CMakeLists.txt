# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_HSTU_950_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_hstu_infer.cpp
)
set_source_files_properties(${_HSTU_950_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(
    NAME ascend950_hstu_infer
    KERNEL_TYPE prebuilt
    NPU_ARCH_LIST 3510
    SKIP_MS_BUILD
    ${_HSTU_950_SOURCES}
)

foreach(_SRC ${_HSTU_950_SOURCES})
    get_filename_component(_SRC_NAME ${_SRC} NAME)
    foreach(_ARCH 3510)
        foreach(_MS_SUFFIX "" "_ms")
            set(_LIB_NAME "catlass_kernel_${_ARCH}_ascend950_hstu_infer${_MS_SUFFIX}")
            if(TARGET ${_LIB_NAME})
                target_compile_definitions(${_LIB_NAME} PRIVATE L2_CACHE_HINT)
                target_include_directories(${_LIB_NAME} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
                    ${CMAKE_CURRENT_SOURCE_DIR}/launcher
                )
            endif()
        endforeach()
    endforeach()
endforeach()
