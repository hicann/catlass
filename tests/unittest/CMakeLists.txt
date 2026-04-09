# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(GOOGLETEST_INCLUDE_DIR ${CATLASS_3RDPARTY_DIR}/googletest/googletest/include)

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${CATLASS_ARCH}>)
add_executable(catlass_unittest catlass_unittest.cpp)

# target_compile_options(catlass_unittest PRIVATE -fprofile-instr-generate -fcoverage-mapping)
file(GLOB_RECURSE CATLASS_UNITTEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(CATLASS_UNITTEST_SRC ${CATLASS_UNITTEST_SRCS})
    set_source_files_properties(${CATLASS_UNITTEST_SRC} PROPERTIES LANGUAGE ASC)
endforeach()

target_sources(catlass_unittest PRIVATE ${CATLASS_UNITTEST_SRCS})

target_include_directories(catlass_unittest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${CATLASS_INCLUDE_DIR}
)

target_link_libraries(catlass_unittest PRIVATE gtest gtest_main m)
target_link_directories(catlass_unittest PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(catlass_unittest PRIVATE
    # AscendC necessary libraries
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    # other necessary libraries
    tiling_api nnopbase platform
    # system libraries
    dl
)
# CANN 9.0.0.beta2 incompatible change
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(catlass_unittest PRIVATE ascendalog)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(catlass_unittest PRIVATE unified_dlog)
endif()
