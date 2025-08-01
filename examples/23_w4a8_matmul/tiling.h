/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.




#pragma once

#include <iostream>
#include <cstdint>
#include "common.h"

enum class Dtype : uint32_t {
    INT8 = 0,
    FP16,
    BF16,
    FP32,
    INT32,
};

struct TilingParams {
    uint32_t b{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t transA{0};
    uint32_t transB{0};
    uint32_t mTile{0};
    uint32_t kTile{0};
    uint32_t nTile{0};
    uint32_t paddingModeA{0};
    uint32_t paddingModeB{0};
    uint32_t splitKSlices{1};
    uint32_t swizzleDir{0};
    uint32_t swizzleCnt{0};
    uint32_t shuffleKType{0};
};

struct MatmulInfo {
    uint32_t b{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    bool transA{false};
    bool transB{false};
    bool enBias{false};
    bool enScale{false};
    bool enResidual{false};
    CubeFormat layoutA{CubeFormat::ND};
    CubeFormat layoutB{CubeFormat::ND};
    CubeFormat layoutC{CubeFormat::ND};
    Dtype dtypeA{Dtype::FP16};
    Dtype dtypeB{Dtype::FP16};
    Dtype dtypeC{Dtype::FP16};
};

void balanceWorkload(uint32_t m, uint32_t k, uint32_t n, uint32_t& m0, uint32_t k0, uint32_t& n0) {
    uint32_t m0t = 384, n0t = 0, bestm0 = 0, bestn0 = 0;
    uint64_t minData = UINT64_MAX;
    while (m0t > 0) {
        n0t = 128 * 256 / m0t / 16 * 16;
        while (n0t > 0) {
            if (n0t <= 384) {
                uint64_t data = (uint64_t)m * (uint64_t)k * ((n + n0t - 1) / n0t) + (uint64_t)k * (uint64_t)n * ((m + m0t - 1) / m0t);
                if (data <= minData) {
                    minData = data;
                    bestm0 = m0t;
                    bestn0 = n0t;
                }
            }
            not -= 16;
        }
        m0t -= 16;
    }
    uint32_t tilesRound = ((m + bestm0 - 1) / bestm0) * ((n + bestn0 - 1) / bestn0);
    tilesRound = (tilesRound + 19) / 20 * 20;
    while (((m + bestm0 - 1) / bestm0) * ((n + (bestn0 - 16) - 1) / (bestn0 - 16)) <= tilesRound && (bestn0 - 16) > 0) {
        bestn0 -= 16;
    }
    m0 = bestm0;
    n0 = bestn0;
}

void GetTiling(MatmulInfo& mmInfo, TilingParams& tilingParams) {
    tilingParams.swizzleCnt = 1;
    tilingParams.swizzleDir = 0;
    tilingParams.mTile = 128;
    tilingParams.nTile = 256;
    tilingParams.kTile = 256;
    tilingParams.b = mmInfo.b;
    tilingParams.m = mmInfo.m;
    tilingParams.k = mmInfo.k;
    tilingParams.n = mmInfo.n;
    tilingParams.transA = mmInfo.transA;
    tilingParams.transB = mmInfo.transB;
}

void operator<< (std::ostream& os, const TilingParams& t) {
    std::cout << "b: " << t.b << ",m: " << t.m << ",k: " << t.k << ",n: " << t.n << td::endl;
    std::cout << "m0: " << t.mTile << ",k0: " << t.kTile << ",n0: " << t.nTile << std::endl;
    std::cout << "padding mode A: " << t.paddingModeA << ",padding mode B: " << t.paddingModeB << std::endl;
    std::cout << "swizzle cnt: " << t.swizzleCnt << "swizzle dir: " << t.swizzleDir << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<< std::endl;
}