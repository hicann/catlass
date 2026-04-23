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

execute_process(
    COMMAND which python 
    OUTPUT_VARIABLE Python_EXECUTABLE 
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    ERROR_QUIET
)

if(NOT Python_EXECUTABLE)
    execute_process(
        COMMAND which python3 
        OUTPUT_VARIABLE Python_EXECUTABLE 
        OUTPUT_STRIP_TRAILING_WHITESPACE 
        ERROR_QUIET
    )
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'), end='')"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR 
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    ERROR_QUIET
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import os; os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'; import torch; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_DIR 
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    ERROR_QUIET
)

if(CMAKE_PREFIX_PATH AND NOT CMAKE_PREFIX_PATH STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
else()
    set(CMAKE_PREFIX_PATH ${TORCH_DIR})
endif()

find_package(Torch REQUIRED)

set(CATLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(NOT DEFINED CATLASS_ARCH_LIST)
    set(CATLASS_ARCH_LIST "2201;3510")
endif()

add_subdirectory(kernels)
add_subdirectory(src)
