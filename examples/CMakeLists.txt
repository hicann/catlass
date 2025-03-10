# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.



set(ACTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
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
    -I${ACTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/include
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec
    #111
    # -L${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib 
    # -lnpu_drv -lnpu_drv_camodel -lstars -lffts -lruntime_camodel
    # -lpem_davinci -lmodel_top -lmcu_wrapper -lmcu_loop 

)

file(GLOB_RECURSE ACTLASS_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE ACTLASS_EXAMPLES_COMMON_INCLUDE_FILES ${ACTLASS_EXAMPLES_COMMON_SOURCE_DIR}/*.hpp)

add_custom_target(actlass_examples)

function(actlass_example_add_executable NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_CCEC_COMPILER} ${CCEC_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN} ${ACTLASS_INCLUDE_FILES} ${ACTLASS_EXAMPLES_COMMON_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling executable kernel: ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(actlass_examples ${NAME})
endfunction()

foreach(EXAMPLE
    04_gemv
)
    add_subdirectory(${EXAMPLE})
endforeach()
