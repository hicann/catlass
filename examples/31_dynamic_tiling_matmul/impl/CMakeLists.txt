# if aicore-arch is different, maybe crash, to find the solution
add_compile_definitions(TILING_KEY_VAR)
add_compile_options(--cce-aicore-arch=dav-c220)
file(GLOB CATLASS_SHARED_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/warpper/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(catlass_kernel SHARED ${CATLASS_SHARED_LIB_SRC})
set_target_properties(catlass_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_custom_target(shared_lib)
add_dependencies(shared_lib catlass_kernel)

install(DIRECTORY include DESTINATION shared_lib
    COMPONENT shared_lib)