# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(ASCENDCT_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(CCEC_COMPILER_OPTIONS_BASE
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${ASCENDCT_EXAMPLES_COMMON_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/include
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined -Wno-ignored-attributes
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

set(CCEC_COMPILER_OPTIONS_05_GEMV_AIV
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${ASCENDCT_EXAMPLES_COMMON_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/include
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined -Wno-ignored-attributes
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

file(GLOB_RECURSE ASCENDCT_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE ASCENDCT_EXAMPLES_COMMON_INCLUDE_FILES ${ASCENDCT_EXAMPLES_COMMON_SOURCE_DIR}/*.hpp)

add_custom_target(AscendCT_examples)

function(AscendCT_example_add_executable NAME)
    if (${NAME} STREQUAL "19_gemv_aiv")
        set(CCEC_COMPILER_OPTIONS ${CCEC_COMPILER_OPTIONS_05_GEMV_AIV})
    else()
        set(CCEC_COMPILER_OPTIONS ${CCEC_COMPILER_OPTIONS_BASE})
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_CCEC_COMPILER} ${CCEC_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN} ${ASCENDCT_INCLUDE_FILES} ${ASCENDCT_EXAMPLES_COMMON_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling executable kernel: ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(AscendCT_examples ${NAME})
endfunction()

foreach(EXAMPLE
    00_basic_matmul
    01_batched_matmul
    02_grouped_matmul_m
    03_matmul_add
    04_padding_matmul
    05_fa
    06_optimized_matmul
    07_grouped_matmul_m_per_token_dequant
    08_grouped_matmul
    09_splitk_matmul
    10_grouped_matmul_m_per_token_dequant_bf16
    11_grouped_matmul_k_per_token_dequant_bf16
    12_quant_matmul
    13_basic_matmul_tla
    14_optimized_matmul_tla
    15_mla
    16_grouped_matmul_k
    17_gemm
    18_groupgemm
    19_gemv_aiv
    20_gemv_aic
    21_quant_gemm
)
    add_subdirectory(${EXAMPLE})
endforeach()