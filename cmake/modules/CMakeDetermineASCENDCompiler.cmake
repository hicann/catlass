if(DEFINED ENV{ASCEND_HOME_PATH})
    set(_CMAKE_ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR
        "no, installation path found, should passing -DASCEND_HOME_PATH=<PATH_TO_ASCEND_INSTALLATION> in cmake"
    )
    set(_CMAKE_ASCEND_HOME_PATH)
endif()

message(STATUS "ASCEND_HOME_PATH:" "  $ENV{ASCEND_HOME_PATH}")

find_program(CMAKE_ASCEND_COMPILER NAMES "bisheng" PATHS "$ENV{PATH}" DOC "Ascend Compiler")

mark_as_advanced(CMAKE_ASCEND_COMPILER)

set(CMAKE_ASCEND_COMPILER_ID "Clang")
set(CMAKE_ASCEND_COMPILER_FRONTEND_VARIANT "Clang")

message(STATUS "CMAKE_ASCEND_COMPILER: " ${CMAKE_ASCEND_COMPILER})
message(STATUS "Ascend Compiler Information:")
execute_process(
    COMMAND ${CMAKE_ASCEND_COMPILER} --version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_ASCEND_SOURCE_FILE_EXTENSIONS cce)
set(CMAKE_ASCEND_COMPILER_ENV_VAR "Ascend")
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})

set(_CMAKE_ASCEND_HOST_IMPLICIT_LINK_DIRECTORIES
    ${_CMAKE_ASCEND_HOME_PATH}/lib64
)

set(_CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES
    stdc++
)

if(DEFINED ASCEND_ENABLE_SIMULATOR AND ASCEND_ENABLE_SIMULATOR)
    if(NOT DEFINED SIMULATOR_NPU_MODEL)
        message(WARNING "Simulator mode is enabled but SIMULATOR_NPU_MODEL is not defined. Try get model from LD_LIBRARY_PATH.")
        set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
        string(REGEX MATCH "simulator/([^:/]*)" SUBDIR "${LD_LIBRARY_PATH}")
        if(SUBDIR)
            set(SIMULATOR_NPU_MODEL "${CMAKE_MATCH_1}")
            message(STATUS "Matched SIMULATOR_NPU_MODEL: ${SIMULATOR_NPU_MODEL}")
        else()
            message(FATAL_ERROR "No SIMULATOR_NPU_MODEL matched!")
        endif()
    endif()

    list(APPEND _CMAKE_ASCEND_HOST_IMPLICIT_LINK_DIRECTORIES
        ${_CMAKE_ASCEND_HOME_PATH}/tools/simulator/${SIMULATOR_NPU_MODEL}/lib
        ${_CMAKE_ASCEND_HOME_PATH}/acllib/lib64/stub)
    list(APPEND _CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES
        runtime_camodel)
else()
    list(APPEND _CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES
        runtime)
endif()

if(DEFINED ASCEND_ENABLE_MSPROF AND ASCEND_ENABLE_MSPROF)
    list(APPEND _CMAKE_ASCEND_HOST_IMPLICIT_LINK_LIBRARIES profapi)
endif()

set(_CMAKE_ASCEND_HOST_IMPLICIT_INCLUDE_DIRECTORIES
    ${_CMAKE_ASCEND_HOME_PATH}/include
    ${_CMAKE_ASCEND_HOME_PATH}/include/experiment/runtime
    ${_CMAKE_ASCEND_HOME_PATH}/include/experiment/msprof
)

if(NOT DEFINED ASCEND_ENABLE_ASCENDC OR ASCEND_ENABLE_ASCENDC)
    list(APPEND _CMAKE_ASCEND_HOST_IMPLICIT_INCLUDE_DIRECTORIES
        ${_CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp
        ${_CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
        ${_CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
        ${_CMAKE_ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    )
endif()

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCENDCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCENDCompiler.cmake
    @ONLY
)
