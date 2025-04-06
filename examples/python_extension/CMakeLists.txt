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
message("Using python: ${Python3_EXECUTABLE}")
find_package(Python3 COMPONENTS Interpreter Development)
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
message(${CMAKE_PREFIX_PATH})
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(torch_act PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${SHARED_LIB_DIR}
    ${ASCEND_HOME_PATH}/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib
    ${SHARED_LIB_DIR})

link_libraries(act_kernel "${TORCH_LIBRARIES}" torch_npu tiling_api)

add_library(act_kernel_wrapper OBJECT act_kernel_wrapper.cpp)
target_compile_options(act_kernel_wrapper PRIVATE -fPIC)

pybind11_add_module(torch_act SHARED pybind_bindings.cpp $<TARGET_OBJECTS:act_kernel_wrapper>)
add_library(act_torch SHARED torch_bindings.cpp $<TARGET_OBJECTS:act_kernel_wrapper>)

install(TARGETS torch_act DESTINATION .)
install(TARGETS act_torch DESTINATION .)
install(FILES set_env.sh DESTINATION .)
