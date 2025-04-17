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

include(ExternalProject)

set(ACT_SHARED_LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shared_lib)
set(ACT_SHARED_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/shared_lib)

ExternalProject_Add(
    act_shared_lib
    SOURCE_DIR ${ACT_SHARED_LIB_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ACT_SHARED_LIB_SRC_DIR}/build.sh --shared_lib_src_dir=${ACT_SHARED_LIB_SRC_DIR} --output_path=${CMAKE_CURRENT_BINARY_DIR}/shared_lib --act_src_dir=${CMAKE_CURRENT_SOURCE_DIR}/../../
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE FALSE
    BUILD_BYPRODUCTS ${ACT_SHARED_LIB_DIR}/libact_kernel.so ${ACT_SHARED_LIB_DIR}/libact_kernel.a
)
if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()
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

include_directories(_C PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ASCEND_HOME_PATH}/include
    ${ACT_SHARED_LIB_SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib
    ${ACT_SHARED_LIB_DIR})

link_libraries(libact_kernel.a "${TORCH_LIBRARIES}" torch_npu tiling_api)

add_library(act_kernel_wrapper OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper/act_kernel_wrapper.cpp)
target_compile_options(act_kernel_wrapper PRIVATE -fPIC)


if(BUILD_PYBIND)
    set(PYBIND_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/pybind_bindings.cpp)
    pybind11_add_module(_C 
    SHARED ${PYBIND_BINDINGS_SRC} $<TARGET_OBJECTS:act_kernel_wrapper>)
    add_dependencies(_C act_shared_lib)
    install(TARGETS _C DESTINATION .)
endif()
if(BUILD_TORCH)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/torch_bindings.cpp)
    add_library(act_torch 
    SHARED ${TORCH_BINDINGS_SRC} $<TARGET_OBJECTS:act_kernel_wrapper>)
    add_dependencies(act_torch act_shared_lib)
    install(TARGETS act_torch DESTINATION .)
endif()
