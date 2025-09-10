/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_DETAIL_MACROS_HPP
#define CATLASS_DETAIL_MACROS_HPP

#define CATLASS_DEVICE __forceinline__[aicore]
#if defined(__CCE__)
#define CATLASS_HOST_DEVICE __forceinline__[host, aicore]
#else
#define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__[aicore]
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#define CATLASS_GLOBAL_AIC CATLASS_GLOBAL __attribute__((aic))
#define CATLASS_GLOBAL_AIV CATLASS_GLOBAL __attribute__((aiv))
#define CATLASS_GLOBAL_MIX_C1V0 [[bisheng::core_ratio(1, 0)]] CATLASS_GLOBAL
#define CATLASS_GLOBAL_MIX_C1V1 [[bisheng::core_ratio(1, 1)]] CATLASS_GLOBAL
#define CATLASS_GLOBAL_MIX_C1V2 [[bisheng::core_ratio(1, 2)]] CATLASS_GLOBAL
#else
#define CATLASS_GLOBAL_AIC CATLASS_GLOBAL
#define CATLASS_GLOBAL_AIV CATLASS_GLOBAL
#define CATLASS_GLOBAL_MIX_C1V0 CATLASS_GLOBAL
#define CATLASS_GLOBAL_MIX_C1V1 CATLASS_GLOBAL
#define CATLASS_GLOBAL_MIX_C1V2 CATLASS_GLOBAL
#endif

#endif // CATLASS_DETAIL_MACROS_HPP