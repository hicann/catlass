# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CATLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(catlass_examples)

macro(catlass_example_add_executable NAME AICORE_ARCH)
    set(EXAMPLE_DESTINATION bin)

    if(DEFINED ENABLE_SIMULATOR AND ENABLE_SIMULATOR)
        set(EXAMPLE_DESTINATION bin_sim)
    endif()

    add_executable(${NAME} ${ARGN})
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=${AICORE_ARCH}>)
    add_dependencies(catlass_examples ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT catlass_examples)
endmacro()

add_compile_options(
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-ignored-attributes>

    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-function-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-record-overflow=true>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-addr-transform>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
)

add_compile_definitions(L2_CACHE_HINT)
include_directories(
    ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CATLASS_INCLUDE_DIR}
)
link_directories(${ASCEND_HOME_PATH}/lib64)
link_libraries(dl tiling_api platform c_sec nnopbase ascendcl)

if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ASCENDC_DUMP=1)
    add_compile_definitions(ASCENDC_DEBUG)
    link_libraries(ascend_dump)
endif()

foreach(EXAMPLE
    00_basic_matmul
    01_batched_matmul
    02_grouped_matmul_slice_m
    03_matmul_add
    04_padding_matmul
    05_grouped_matmul_slice_k
    06_optimized_matmul
    07_grouped_matmul_slice_m_per_token_dequant_moe
    08_grouped_matmul
    09_splitk_matmul
    10_grouped_matmul_slice_m_per_token_dequant
    11_grouped_matmul_slice_k_per_token_dequant
    12_quant_matmul
    13_basic_matmul_tla
    14_optimized_matmul_tla
    15_gemm
    16_group_gemm
    17_gemv_aiv
    18_gemv_aic
    19_mla
    20_matmul_bias
    21_basic_matmul_preload_zN
    22_padding_splitk_matmul
)
    add_subdirectory(${EXAMPLE})
endforeach()

add_subdirectory(shared_lib)
add_subdirectory(python_extension)
