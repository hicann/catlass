# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.



set(ACOT_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(CCEC_COMPILER_OPTIONS
    # -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec # 只能启动AIV核心
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${ACOT_EXAMPLES_COMMON_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/include
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined
    -lruntime -lstdc++ -lascendcl -ltiling_api -lregister -lplatform -lascendalog -ldl
    -lascendc_runtime -lerror_manager -lprofapi -lge_common_base -lmmpa -lascend_dump -lc_sec
    -lruntime -lstdc++ -lascendcl
    # -mllvm -cce-aicore-stack-size=0x8000
    # -mllvm -cce-aicore-function-stack-size=0x8000
    # -mllvm -cce-aicore-record-overflow=true
    # -mllvm -cce-aicore-addr-transform
    # -mllvm -cce-aicore-dcci-insert-for-scalar=false
    # -I${ASCEND_HOME_PATH}/compiler/tikcpp
    # -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    # -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    # -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    # -I${ASCEND_HOME_PATH}/include
    # -I${ASCEND_HOME_PATH}/include/experiment/runtime
    # -I${ASCEND_HOME_PATH}/include/experiment/msprof
    # -I${ACOT_EXAMPLES_COMMON_SOURCE_DIR}
    # -I${CMAKE_SOURCE_DIR}/include
    # -L${ASCEND_HOME_PATH}/lib64
    # -lruntime -lstdc++ -lascendcl -lm
)

file(GLOB_RECURSE ACOT_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE ACOT_EXAMPLES_COMMON_INCLUDE_FILES ${ACOT_EXAMPLES_COMMON_SOURCE_DIR}/*.hpp)

add_custom_target(acot_examples)

function(acot_example_add_executable NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_CCEC_COMPILER} ${CCEC_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN} ${ACOT_INCLUDE_FILES} ${ACOT_EXAMPLES_COMMON_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling executable kernel: ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(acot_examples ${NAME})
endfunction()

foreach(EXAMPLE
    00_basic_matmul
    01_grouped_matmul
    02_optimized_matmul
    03_gemm # 添加示例代码目录
)
add_subdirectory(${EXAMPLE})
endforeach()
