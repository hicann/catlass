# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(catlass_torch VERSION 1.0.0 LANGUAGES CXX ASC)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

if(NOT Python_EXECUTABLE AND NOT SKBUILD)
    set(_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    if(EXISTS ${_VENV_PYTHON})
        set(Python_EXECUTABLE ${_VENV_PYTHON})
    endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")

find_package(Torch REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ASCEND_HOME_PATH}/include
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${Python3_SITELIB}/torch_npu/lib
)

set(TORCH_NPU_DIR "${Python3_SITELIB}/torch_npu")

if(NOT DEFINED CATLASS_DIR)
    set(CATLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
set(CATLASS_INCLUDE_DIR ${CATLASS_DIR}/include)

if(NOT DEFINED CATLASS_ARCH_LIST)
    set(CATLASS_ARCH_LIST "2201;3510")
endif()

add_subdirectory(kernels)
add_subdirectory(src)
