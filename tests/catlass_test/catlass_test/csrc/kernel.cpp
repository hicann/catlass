
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define STRINGIFY(x) #x
#define INCLUDE_FILE(x) STRINGIFY(x)

#if defined(KERNEL_TEMPLATE_FILE)
#include INCLUDE_FILE(KERNEL_TEMPLATE_FILE)
#else
#include <cstdint>
template <typename ElementA> inline int32_t Kernel(uint8_t *deviceA) {
    // do nothing
}
#define KERNEL_TEMPLATE_NAME Kernel
#define COMPILE_PARAM int16_t
#define RUNTIME_PARAM uint8_t *deviceA
#define RUNTIME_PARAM_CALL deviceA
#endif
extern "C" int32_t run(RUNTIME_PARAM) { KERNEL_TEMPLATE_NAME<COMPILE_PARAM>(RUNTIME_PARAM_CALL); }