add_subdirectory(impl)

include_directories(${ASCEND_HOME_PATH}/include/experiment/runtime (${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/include ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include))
link_directories(${ASCEND_HOME_PATH}/lib64)
add_executable(31_dynamic_tiling_matmul dynamic_tiling_matmul_half.cpp)