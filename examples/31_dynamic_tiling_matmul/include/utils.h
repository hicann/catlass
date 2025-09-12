/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "tiling_params.h"
#include "platform_info.h"
#include "catlass/catlass.hpp"

void BalanceWorkload(uint32_t m, uint32_t n, uint32_t m1, uint32_t n1, uint32_t threshold, PlatformInfo& platformInfo)
{
    uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
    while (m1 > threshold && (CeilDivHost(m, m1 - 16) * CeilDivHost(n, n1) <= maxBlocks)) {
        m1 -= 16;
    }
    if (m < m1) {
        m1 = RoundUpHost(m, 16);
    }
    if (n < n1) {
        n1 = RoundUpHost(n, 16);
    }
}

void SetTile(TilingParams &TilingParams, uint32_t m1, uint32_t n1, uint32_t k1)
{
    // To save space, tiling parameters (m1, n1, k1) are stored as uint16_t.j
    TilingParams.m1 = static_cast<uint16_t>(m1);
    TilingParams.n1 = static_cast<uint16_t>(n1);
    TilingParams.k1 = static_cast<uint16_t>(k1);
}

bool IsExStrideLimit(uint32_t rows, uint32_t cols, uint32_t layoutTag)
{
    if (static_cast<LayoutTag>(layoutTag) == LayoutTag::TagColumnMajor) {
        return rows >= 65536;
    } else {
        return cols >= 65536;
    }
}

template <class DType>
bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1, PlatformInfo& platformInfo)
{
    bool judgeL1 = (m1 * k1 * 2 * sizeof(DType) + k1 * n1 * 2 * sizeof(DType) <= platformInfo.l1Size);
    bool judgeL0C = (m1 * n1 * 4 <= platformInfo.l0CSize) ? true : false;
    return judgeL1 && judgeL0C;
}

template <class DType>
uint32_t GetMaxK1(uint32_t m1, uint32_t n1, PlatformInfo& platformInfo)
{
    std::vector<uint32_t> k1List = {1024, 512, 256, 128};
    uint32_t k1 = 512 / sizeof(DType);
    for (const auto &k1t : k1List) {
        if (JudgeSpace<DType>(m1, n1, k1t, platformInfo)) {
            k1 = k1t;
            break;
        }
    }
    return k1;
}

#endif  // UTILS_H