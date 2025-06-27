/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_MACROS_HPP
#define CATLASS_DETAIL_MACROS_HPP

#if defined(__CCE_IS_AICORE__)
#define CATLASS_HOST_DEVICE __forceinline__ [host, aicore]
#else
#pragma message("Included CATLASS headers in pure host code")
#define CATLASS_HOST_DEVICE
#endif

#define CATLASS_DEVICE __forceinline__ __aicore__
#define CATLASS_GLOBAL __global__ __aicore__

#endif  // CATLASS_DETAIL_MACROS_HPP