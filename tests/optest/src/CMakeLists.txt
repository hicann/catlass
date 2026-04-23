set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(catlass_torch SHARED)

target_include_directories(catlass_torch PRIVATE
    ${COMMON_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${CATLASS_DIR}/include
    ${PYTHON_INCLUDE_DIR}
)

target_link_directories(catlass_torch PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${TORCH_NPU_DIR}/lib
)

target_link_libraries(catlass_torch PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(SRC_FILES)
    target_sources(catlass_torch PRIVATE ${SRC_FILES})
endif()

install(TARGETS catlass_torch DESTINATION ${PROJECT_SOURCE_DIR}/torch_catlass/lib)
