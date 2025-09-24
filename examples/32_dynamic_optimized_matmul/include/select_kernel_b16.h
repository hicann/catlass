/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_KERNEL_HALF_H
#define SELECT_KERNEL_HALF_H

#include <limits>
#include "platform_info.h"

enum class PaddingTag : uint8_t { PADDING_NONE = 0, PADDING_ND = 1, PADDING_BLOCK_ND = 2, PADDING_NZ = 3};

double GetBandwidth(uint32_t nValue, uint32_t dValue, uint32_t srcDValue) {
    double a6 = 0.000000000000020146121020;
    double a5 = -0.000000000012456944162142;
    double a4 = -0.000000006738536427145036;
    double a3 = 0.000007301215580838747961;
    double a2 = -0.002146456956750821074703;
    double a1 = 0.312849910814454512664184;
    double a0 = 0.1;
    double unalignBand = a6 * pow(static_cast<double>(dValue), 6) + a5 * pow(static_cast<double>(dValue), 5)
        + a4 * pow(static_cast<double>(dValue), 4) + a3 * pow(static_cast<double>(dValue), 3)
        + a2 * pow(static_cast<double>(dValue), 2) + a1 * static_cast<double>(dValue) + a0;
    
    if (dValue == srcDValue && dValue <= 128) {
        if (dValue % 16 == 0) {
            unalignBand = 60;
        }
    }
    if (srcDValue >= 65536) {
        unalignBand = 1;
    }

    if (srcDValue % 256 == 0) {
        unalignBand = static_cast<double>(100) / 30 * unalignBand;
    } else if (srcDValue % 128 == 0) {
        unalignBand = static_cast<double>(80) / 30 * unalignBand;
    } else if (srcDValue % 64 == 0) {
        unalignBand = static_cast<double>(50) / 30 * unalignBand;
    } else if (srcDValue % 16 == 0) {
        unalignBand = static_cast<double>(40) / 30 * unalignBand;
    }

    unalignBand = std::min(unalignBand, 80.0);

    if (dValue % 256 == 0) {
        if (nValue < 16) {
            double b2 = -0.003332381309698882569659;
            double b1 = 0.113578920178116271610946;
            double b0 = 0.016102868630357251855667;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * nValue + b0);
        }
    } else if (dValue % 32 == 0) {
        if (nValue < 32) {
            double b2 = -0.000298086120946179481978;
            double b1 = 0.045309519479127147167929;
            double b0 = 0.035130178145161221336945;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * nValue + b0);
        }

    } else {
        if (nValue < 64) {
            double b3 = 0.000001809180573350345869;
            double b2 = -0.000469676727179688081274;
            double b1 = 0.038963259596073690493867;
            double b0 = 0.003942641759904389614499;
            unalignBand = unalignBand * (b3 * pow(nValue, 3) + b2 * pow(nValue, 2) + b1 * nValue + b0);
        }
    }
    return unalignBand;
}

void GetPaddingTag(TilingParams& tilingParams, PlatformInfo& platformInfo) {
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = tilingParams.m1;
    uint32_t n1 = tilingParams.n1;
    uint32_t k1 = tilingParams.k1;
    uint32_t splitkFactor = tilingParams.splitkFactor;

    uint64_t outterAxisA = m;
    uint64_t innerAxisA = k;
    uint32_t nValueA = std::min(m, m1);
    uint32_t dValueA = std::min(k, k1);
    if (static_cast<LayoutTag>(tilingParams.layoutTagA) == LayoutTag::TagColumnMajor) {
        outterAxisA = k;
        innerAxisA = m;
        nValueA = std::min(k, k1);
        dValueA = std::min(m, m1);
    }

    uint64_t outterAxisB = k;
    uint64_t innerAxisB = n;
    uint32_t nValueB = std::min(k, k1);
    uint32_t dValueB = std::min(n, n1);
    if (static_cast<LayoutTag>(tilingParams.layoutTagB) == LayoutTag::TagColumnMajor) {
        outterAxisB = n;
        innerAxisB = k;
        nValueB = std::min(n, n1);
        dValueB = std::min(k, k1);
    }

    double aBandwidthAiv = 30; // single core GB/s
    size_t matrixASize = static_cast<size_t>(m) * k * 2;
    if (matrixASize > 192 * 1024 * 1024) { // L2 cache size
        aBandwidthAiv = 10;
    }
    double aBandwidthBeforePaddingAic = GetBandwidth(nValueA, dValueA, innerAxisA);
    
    uint32_t tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1) * splitkFactor;
    uint32_t blockDimAic = tasksAic > platformInfo.coreNum ? platformInfo.coreNum : tasksAic;
    if (CeilDiv(m, m1) < blockDimAic / 2 && k <= k1 && CeilDiv(m, m1) <= 2) {
        aBandwidthBeforePaddingAic = aBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(m, m1) * 1.5);
    }
    double aBandwidthAfterPaddingAic = 80;

    double bBandwidthAiv = 30; // single core GB/s
    size_t matrixBSize = static_cast<size_t>(k) * n * 2;
    if (matrixBSize > 192 * 1024 * 1024) { // L2 cache size
        bBandwidthAiv = 10;
    }
    double bBandwidthBeforePaddingAic = GetBandwidth(nValueB, dValueB, innerAxisB); 
    if (CeilDiv(n, n1) < blockDimAic / 2 && k <= k1 && CeilDiv(n, n1) <= 2) {
        bBandwidthBeforePaddingAic = bBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(n, n1) * 1.5);
    }
    double bBandwidthAfterPaddingAic = 80;

    uint32_t actualM = std::min(m, m1);
    uint32_t actualN = std::min(n, n1);
    uint32_t roundMax = CeilDiv(CeilDiv(m, m1) * CeilDiv(n, n1) * splitkFactor, platformInfo.coreNum);
    size_t aMaxDataSizeAic = static_cast<size_t>(roundMax) * actualM * CeilDiv(k, splitkFactor) * 2; // Byte
    size_t bMaxDataSizeAic = static_cast<size_t>(roundMax) * actualN * CeilDiv(k, splitkFactor) * 2; // Byte

    // padding simulator
    size_t aMaxDataSizeAiv{0};
    uint32_t tasksAivA{0};
    {
        uint32_t taskRows = 16;
        uint32_t taskCols = 48 * 1024 / 2 / taskRows;
        if (innerAxisA < taskCols) {
            taskCols = innerAxisA;
        }
        if (outterAxisA < taskRows) {
            taskRows = outterAxisA;
        }
        taskCols = RoundUp(innerAxisA / CeilDiv(innerAxisA, taskCols), 16);
        uint32_t taskAivA = CeilDiv(outterAxisA, taskRows) * CeilDiv(innerAxisA, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivA, platformInfo.coreNum * 2);
        aMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 2;
    }

    size_t bMaxDataSizeAiv{0};
    uint32_t tasksAivB{0};
    {
        uint32_t taskRows = 16;
        uint32_t taskCols = 48 * 1024 / 2 / taskRows;
        if (innerAxisB < taskCols) {
            taskCols = innerAxisB;
        }
        if (outterAxisB < taskRows) {
            taskRows = outterAxisB;
        }
        taskCols = RoundUp(innerAxisB / CeilDiv(innerAxisB, taskCols), 16);
        uint32_t tasksAivB = CeilDiv(outterAxisB, taskRows) * CeilDiv(innerAxisB, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivB, platformInfo.coreNum * 2);
        bMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 2;
    }

    uint32_t tasksAiv = std::max(tasksAivA, tasksAivB);
    uint32_t blockDimAiv = CeilDiv(tasksAiv, 2) > platformInfo.coreNum ? platformInfo.coreNum : CeilDiv(tasksAiv, 2);
    uint32_t blockDim = blockDimAic;
    if (innerAxisA > 192 && innerAxisB > 192) {
        blockDim = std::max(blockDimAic, blockDimAiv);
    }

    double headCost = 1 + 7 * static_cast<double>(blockDim) / platformInfo.coreNum; // us
    double t00 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000;
    double t01 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost;
    double t10 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000
        + static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000 + headCost;
    double t11 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000
        + static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000
        + static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost + 2;
    
    double minCost = std::numeric_limits<double>::max();
    PaddingTag paddingTagA = PaddingTag::PADDING_NONE;
    PaddingTag paddingTagB = PaddingTag::PADDING_NONE;
    if (minCost > t00) {
        minCost = t00;
    }
    if (minCost > t01) {
        minCost = t01;
        paddingTagA = PaddingTag::PADDING_NONE;
        paddingTagB = PaddingTag::PADDING_NZ;
    }
    if (minCost > t10) {
        minCost = t10;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NONE;
    }
    if (minCost > t11) {
        minCost = t11;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    if ((innerAxisA < 8 || (innerAxisA < 32 && (innerAxisA % 16 != 0))) && outterAxisA > 512) {
        paddingTagA = PaddingTag::PADDING_NZ;
    }
    if ((innerAxisB < 8 || (innerAxisB < 32 && (innerAxisB % 16 != 0))) && outterAxisB > 512) {
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    PaddingTag paddingTagC = PaddingTag::PADDING_NONE;
    if (static_cast<size_t>(m) * n > 2048 * 2048 && n > 256 && (n % 128 != 0)) {
        size_t totalDataSize = static_cast<size_t>(m) * k * CeilDiv(n, n1) * 2
            + static_cast<size_t>(k) * n * CeilDiv(m, m1) * 2 + static_cast<size_t>(m) * n * 2;
        if (totalDataSize < 96 * 1024 * 1024) { // half of L2 cache size
            paddingTagC = PaddingTag::PADDING_ND;
        }
    }

    tilingParams.paddingTagA = static_cast<uint8_t>(paddingTagA);
    tilingParams.paddingTagB = static_cast<uint8_t>(paddingTagB); 
    tilingParams.paddingTagC = static_cast<uint8_t>(paddingTagC);
    tilingParams.blockDim = blockDim;
}

bool CommonMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 0;
    // kernelSerial, layoutTagA, layoutTagB, layoutTagC, paddingTagA, paddingTagB, paddingTagC, dtype(defalut 0).
    params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0);
    return true;
}

bool SmallMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 1;
    GetPaddingTag(params, platformInfo);
    if (static_cast<PaddingTag>(params.paddingTagA) == PaddingTag::PADDING_NONE
        && static_cast<PaddingTag>(params.paddingTagB) == PaddingTag::PADDING_NONE
        && static_cast<PaddingTag>(params.paddingTagC) == PaddingTag::PADDING_NONE) {

        uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
        if (taskBlocks <= platformInfo.coreNum && params.k <= params.k1) {
            params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0);
            return true;
        }
    }
    return false;
}

bool PaddingMatmulB16Handler(TilingParams &params, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 2;
    if (params.paddingTagA || params.paddingTagB || params.paddingTagC) {
        params.tilingKey.SetTilingKey(kernelSerial, 
            params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, params.paddingTagC); 
        return true;
    }
    return false;
}

bool PaddingMultiCoreSplitkMatmulB16Handler(TilingParams& params, PlatformInfo& platformInfo)
{
    uint32_t m = params.m;
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t m1t = 128, n1t = 256, k1t = 256;
    LayoutTag layoutTagA = static_cast<LayoutTag>(params.layoutTagA);
    LayoutTag layoutTagB = static_cast<LayoutTag>(params.layoutTagB);
    bool cond1 = (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagColumnMajor);
    bool cond2 = (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagRowMajor) && (m > n); 
    if (cond1 || cond2) {
        m1t = 256;
        n1t = 128;
    }
    uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1t);
    uint32_t maxSplitkFactor = 2;
    if (k > 1024) {
        maxSplitkFactor = 4;
    }
    if (k > 2048) {
        maxSplitkFactor = 8;
    }
    if (k > 4096) {
        maxSplitkFactor = 16;
    }
    if (k >= 12288) {
        maxSplitkFactor = platformInfo.coreNum;
    }
    if ((blocks <= platformInfo.coreNum / 2 && k > 5120) || (blocks <= 2 && k > 1024)) {
        params.m1 = m1t;
        params.n1 = n1t;
        params.k1 = k1t;
        params.splitkFactor = std::min(platformInfo.coreNum / blocks, maxSplitkFactor);
        GetPaddingTag(params, platformInfo);
        uint8_t kernelSerial = 3;
        params.tilingKey.SetTilingKey(kernelSerial, 
            params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, 0); 
        return true;
    }
    return false;
}

bool PaddingStreamkMatmulB16Handler(TilingParams& params, PlatformInfo& platformInfo)
{
    uint32_t m1 = params.m1;
    uint32_t n1 = params.n1;
    uint32_t k1 = params.k1;
    // Streamk ensures workload balancing by partitioning k, the L1 tile block can use the size with the best bandwidth.
    // The size setting of l1 tile does not need to consider workload balancing.
    uint32_t m1t = 128, n1t = 256, k1t = 256;
    LayoutTag layoutTagA = static_cast<LayoutTag>(params.layoutTagA);
    LayoutTag layoutTagB = static_cast<LayoutTag>(params.layoutTagB);
    bool cond1 = (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagColumnMajor);
    bool cond2 = (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagRowMajor) && (m > n); 
    if (cond1 || cond2) {
        m1t = 256;
        n1t = 128;
    }
    uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1t);
    uint32_t skBlocks = blocks % platformInfo.coreNum;
    if (blocks > platformInfo.coreNum && blocks < 8 * platformInfo.coreNum && skBlocks > 0 
        && skBlocks < 0.8 * platformInfo.coreNum && params.k > 3072) {
            params.m1 = m1t;
            params.n1 = n1t;
            params.k1 = k1t;
            GetPaddingTag(params);
            uint32_t kernelSerial = 4;
            params.tilingKey.SetTilingKey(kernelSerial, 
                params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, 0); 
        return true;
    }
    return false;
}

void SelectKernelB16(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    using HandlerPtr = bool (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
    HandlerPtr handlers[] = {
        SmallMatmulB16Handler,
        PaddingMultiCoreSplitkMatmulB16Handler,
        PaddingStreamkMatmulB16Handler,
        PaddingMatmulB16Handler,
        CommonMatmulB16Handler
    };

    for (auto handler : handlers) {
        if (handler(tilingParams, platformInfo)) {
            break;
        }
    }
}

#endif  // SELECT_KERNEL_HALF_H