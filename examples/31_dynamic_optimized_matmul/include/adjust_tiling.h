/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADJUST_TILING_H
#define ADJUST_TILING_H

#include "adjust_tiling_b16.h"
#include "platform_info.h"

template <class DType>
void AdjustTiling(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t layoutTagA = tilingParams.layoutTagA;
    uint32_t layoutTagB = tilingParams.layoutTagB;

    AdjustTilingB16[layoutTagA][layoutTagB](tilingParams, platformInfo);
}

#endif  // ADJUST_TILING_H