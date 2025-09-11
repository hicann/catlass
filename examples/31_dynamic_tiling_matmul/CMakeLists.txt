# if aicore-arch is different, maybe crash, to find the solution
option(GENERATE_MATMUL_WRAPPER_CODE "Generate wrapper code for generalization matmul" OFF)

if(GENERATE_MATMUL_WRAPPER_CODE)
    execute_process(
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
    )
endif()

add_custom_target(auto_gen_code)
add_custom_command(
    TARGET auto_gen_code
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/code_gen.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/code_gen.py
    COMMENT "Generate wrapper code, launch_map.h ...done"
)

add_compile_definitions(TILING_KEY_VAR)
add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=dav-c220>)
file(GLOB CATLASS_SHARED_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/wrapper/*.cpp)
set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

add_library(generalization_kernel STATIC ${CATLASS_SHARED_LIB_SRC})
set_target_properties(generalization_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

link_libraries(generalization_kernel runtime)

add_executable(31_dynamic_tiling_matmul dynamic_tiling_matmul.cpp)
add_dependencies(catlass_examples 31_dynamic_tiling_matmul)
add_dependencies(31_dynamic_tiling_matmul generalization_kernel)
add_dependencies(generalization_kernel auto_gen_code)

install(TARGETS generalization_kernel DESTINATION shared_lib/lib COMPONENT 31_dynamic_tiling_matmul)
install(TARGETS 31_dynamic_tiling_matmul DESTINATION bin COMPONENT 31_dynamic_tiling_matmul)
install(TARGETS 31_dynamic_tiling_matmul DESTINATION bin COMPONENT catlass_examples)