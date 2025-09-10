# if aicore-arch is different, maybe crash, to find the solution
add_compile_definitions(TILING_KEY_VAR)
add_compile_options(--cce-aicore-arch=dav-c220)
file(GLOB CATLASS_SHARED_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/impl/wrapper/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(generalization_kernel SHARED ${CATLASS_SHARED_LIB_SRC})
set_target_properties(generalization_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_target(generalization_lib)
add_dependencies(generalization_lib generalization_kernel)

install(TARGETS generalization_kernel
    DESTINATION bin
    COMPONENT generalization_lib)

include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_BINARY_DIR})
link_libraries(generalization_kernel)

set_source_files_properties(dynamic_tiling_matmul_half.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(31_dynamic_tiling_matmul dav-c220 dynamic_tiling_matmul_half.cpp)
add_dependencies(31_dynamic_tiling_matmul generalization_kernel)
