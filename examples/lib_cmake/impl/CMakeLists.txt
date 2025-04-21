include(${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/aux_functions.cmake)

# Set sources
set(LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Set compile options for used in bisheng
set(ASCEND_BISHENG_FLAGS -O2 -fPIC -std=c++17 -xcce --cce-aicore-arch=dav-c220 -DL2_CACHE_HINT
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -w
)

# Set include directories
set_bisheng_include_dirs(ascend_device
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ACT_EXAMPLES_COMMON_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
# message(STATUS "set_bisheng_include_dirs: " ${BISHENG_INCLUDE_DIRS})

# Set linked libraries
set_bisheng_link_libs(ascend_device
    ${RT_LIB1}
    ${RT_LIB2}
    ${RT_LIB3}
    ${RT_LIB4}
    ${RT_LIB5}
    ${RT_LIB6}
    ${ENV_LIB1}
    ${ENV_LIB2}
)

bisheng_add_library(ascend_device DYNAMIC ${LIB_SOURCES})
bisheng_add_library(ascend_device STATIC ${LIB_SOURCES})

add_custom_target(
    ascend_lib_target
    DEPENDS libascend_device.so libascend_device.a
)

