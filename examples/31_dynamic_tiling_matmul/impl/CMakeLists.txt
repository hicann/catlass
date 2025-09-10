# if aicore-arch is different, maybe crash, to find the solution
add_compile_definitions(TILING_KEY_VAR)
add_compile_options(--cce-aicore-arch=dav-c220)
file(GLOB CATLASS_SHARED_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(generalization_kernel SHARED ${CATLASS_SHARED_LIB_SRC})
set_target_properties(generalization_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_custom_target(generalization_lib)
add_dependencies(generalization_lib generalization_kernel)

install(DIRECTORY include DESTINATION generalization_lib
    COMPONENT generalization_lib)