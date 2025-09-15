/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H
#define CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H

#include <iostream>
#include <iomanip>

#include "adjust_tiling_b16.h"
#include "select_kernel_half.h"
#include "launch_map.h"

template <class DType>
void AdjustTiling(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    uint32_t layoutTagA = tilingParams.layoutTagA;
    uint32_t layoutTagB = tilingParams.layoutTagB;

    AdjustTilingB16[layoutTagA][layoutTagB](tilingParams, platformInfo);
}

template <class DType>
void SelectKernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    SelectKernelHalf(tilingParams, platformInfo);
}

template <class DType>
void DoTilingAndSelectKernel(TilingParams &tilingParams, PlatformInfo &platformInfo)
{
    AdjustTiling<DType>(tilingParams, platformInfo);
    SelectKernel<DType>(tilingParams, platformInfo);
}

size_t DynamicOptimizedMatmulGetWorkspace(TilingParams &tilingParams)
{
    return getWorkspaceFuncMap[tilingParams.tilingKey.value](tilingParams);
}

void ExecuteDynamicOptimizedMatmul(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dTilingParams, TilingParams &tilingParams)
{

    launchKernelFuncMap[tilingParams.tilingKey.value](stream, fftsAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
}

template <class DType>
void PrintTilingParams(TilingParams &tilingParams)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0 / sizeof(DType);
    uint32_t m0 = tilingParams.m1, n0 = tilingParams.n1, k0 = 0;
    if (m0 && n0) {
        // TODO
    }
    std::cout << std::dec << "┌─────────────────────────────────────────────┐\n"
              << "│            Tiling Parameters                │\n"
              << "├───────────────────┬─────────────────────────┤\n"
              << "│ m:           " << std::setw(30) << tilingParams.m << " │\n"
              << "│ n:           " << std::setw(30) << tilingParams.n << " │\n"
              << "│ k:           " << std::setw(30) << tilingParams.k << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ layoutTagA:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.layoutTagA) << " │\n"
              << "│ layoutTagB:  " << std::setw(30) << static_cast<uint32_t>(tilingParams.layoutTagB) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ mTile:       " << std::setw(30) << static_cast<uint32_t>(tilingParams.m1) << " │\n"
              << "│ nTile:       " << std::setw(30) << static_cast<uint32_t>(tilingParams.n1) << " │\n"
              << "│ kTile:       " << std::setw(30) << static_cast<uint32_t>(tilingParams.k1) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ mTileInL0:   " << std::setw(30) << static_cast<uint32_t>(m0) << " │\n"
              << "│ nTileInL0:   " << std::setw(30) << static_cast<uint32_t>(n0) << " │\n"
              << "│ kTileInL0:   " << std::setw(30) << static_cast<uint32_t>(k0) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ paddingTagA: " << std::setw(30) << static_cast<uint32_t>(tilingParams.paddingTagA) << " │\n"
              << "│ paddingTagB: " << std::setw(30) << static_cast<uint32_t>(tilingParams.paddingTagB) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ blockDim:    " << std::setw(30) << static_cast<uint32_t>(tilingParams.blockDim) << " │\n"
              << "├───────────────────┼─────────────────────────┤\n"
              << "│ TilingKey:   " << std::hex << std::setw(30) << tilingParams.tilingKey.value << " │\n"
              << "└───────────────────┴─────────────────────────┘" << std::endl;
    std::cout << "Kernel Func Name : " << funcNameMap[tilingParams.tilingKey.value] << std::endl;
}

#endif  // CATLASS_DYNAMIC_OPTIMIZED_MATMUL_H