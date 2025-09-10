add_subdirectory(impl)

include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_source_files_properties(dynamic_tiling_matmul_half.cpp PROPERTIES LANGUAGE ASCEND)
add_executable(31_dynamic_tiling_matmul dynamic_tiling_matmul_half.cpp)