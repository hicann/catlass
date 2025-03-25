# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(python_extension)
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c" "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}")
message(${CMAKE_PREFIX_PATH})
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_SITE_PACKAGES}/torch_npu/include
    ${SHARED_LIB_DIR}
    ${ASCEND_HOME_PATH}/include)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${PYTHON_SITE_PACKAGES}/torch_npu/lib
    ${SHARED_LIB_DIR})

link_libraries(ascendct_kernel "${TORCH_LIBRARIES}" torch_npu tiling_api)

pybind11_add_module(torch_ascendct SHARED AscendCTKernelWrapper.cpp PybindBindings.cpp)
add_library(ascendct_torch SHARED AscendCTKernelWrapper.cpp TorchBindings.cpp)

install(TARGETS torch_ascendct DESTINATION .)
install(TARGETS ascendct_torch DESTINATION .)
install(FILES set_env.sh DESTINATION .)
