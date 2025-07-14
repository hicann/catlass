# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Find dependent libs
# -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
set(RT_LIB_DIR ${ASCEND_HOME_PATH}/runtime/lib64)
find_library(RT_LIB1 NAMES ascendcl PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(RT_LIB2 NAMES runtime PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(RT_LIB3 NAMES tiling_api PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(RT_LIB4 NAMES platform PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(RT_LIB5 NAMES nnopbase PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(RT_LIB6 NAMES c_sec PATHS ${RT_LIB_DIR} NO_DEFAULT_PATH)
find_library(ENV_LIB1 NAMES dl)
find_library(ENV_LIB2 NAMES m)
if(NOT RT_LIB1 OR NOT RT_LIB2)
    message(FATAL_ERROR "One or more required libraries not found!")
endif()

add_subdirectory(impl)

add_executable(QuickStart main.cpp)
add_dependencies(QuickStart ascend_lib_target)
target_include_directories(QuickStart
  PUBLIC
  ${ASCEND_HOME_PATH}/compiler/tikcpp
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/runtime
  ${ASCEND_HOME_PATH}/include/experiment/msprof
  ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(QuickStart
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/impl/libascend_device.a
    ${RT_LIB1}
    ${RT_LIB2}
    ${RT_LIB3}
    ${RT_LIB4}
    ${RT_LIB5}
    ${RT_LIB6}
    ${ENV_LIB1}
    ${ENV_LIB2}
)
