/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rfa_kernel_utils.h
 * \brief
 */

#ifndef CATLASS_EXAMPLES_RFA_KERNEL_UTILS_H
#define CATLASS_EXAMPLES_RFA_KERNEL_UTILS_H

#include "catlass/catlass.hpp"

using namespace Catlass;

struct RfaKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    // 稀疏块索引
    GM_ADDR selectIdx;         // [T, headNum, maxKvBlockNum]
    GM_ADDR selectNumIdx;      // [T, headNum]
    // 输出
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workspace;
    GM_ADDR tiling;

    __aicore__ inline
    RfaKernelParamsArch35() {}

    __aicore__ inline
    RfaKernelParamsArch35(
        GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR selectIdx_, GM_ADDR selectNumIdx_,
        GM_ADDR o_, GM_ADDR lse_, GM_ADDR workspace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
        actualKvseqlen(actualKvseqlen_), selectIdx(selectIdx_), selectNumIdx(selectNumIdx_),
        o(o_), lse(lse_), workspace(workspace_), tiling(tiling_) {}
};

__aicore__ inline
uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX + qBaseTile - 1) / qBaseTile;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile - 1) / qBaseTile;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

#endif // CATLASS_EXAMPLES_RFA_KERNEL_UTILS_H
