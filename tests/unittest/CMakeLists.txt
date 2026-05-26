# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(GOOGLETEST_INCLUDE_DIR ${CATLASS_3RDPARTY_DIR}/googletest/googletest/include)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${CATLASS_INCLUDE_DIR}
)

link_directories(${ASCEND_HOME_PATH}/lib64)
add_executable(catlass_unittest catlass_unittest.cpp)

function(testcase MODE ARCH)
    set(SOURCES ${ARGN})

    foreach(SRC ${SOURCES})
        get_filename_component(FILE_NAME ${SRC} NAME_WE)

        if(MODE STREQUAL NPU)
            add_library(${FILE_NAME}_npu OBJECT ${SRC})
            set_source_files_properties(${SRC} PROPERTIES LANGUAGE ASC)
            target_compile_options(${FILE_NAME}_npu PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${ARCH}>)
            target_sources(catlass_unittest PRIVATE $<TARGET_OBJECTS:${FILE_NAME}_npu>)
        endif()

        if(MODE STREQUAL STUB)
            add_library(${FILE_NAME}_stub OBJECT ${SRC})
            set_target_properties(${FILE_NAME}_stub PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
            target_include_directories(${FILE_NAME}_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stub)
            target_compile_definitions(${FILE_NAME}_stub PRIVATE CATLASS_ARCH=${ARCH} __NPU_ARCH__=${ARCH})
            target_sources(catlass_unittest PRIVATE $<TARGET_OBJECTS:${FILE_NAME}_stub>)
        endif()
    endforeach()
endfunction()


file(GLOB_RECURSE CATLASS_UNITTEST_SRCS_TILE_STUB ${CMAKE_CURRENT_SOURCE_DIR}/catlass/gemm/tile/*.cpp)
testcase(STUB 2201 ${CATLASS_UNITTEST_SRCS_TILE_STUB})

target_link_libraries(catlass_unittest PRIVATE
    gtest gtest_main m pthread
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    tiling_api nnopbase platform dl
)

if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(catlass_unittest PRIVATE ascendalog)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(catlass_unittest PRIVATE unified_dlog)
endif()
