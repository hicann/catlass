/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_MATMUL_KERNEL_H
#define COMMON_MATMUL_KERNEL_H

#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_common_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

enum DispatchPolicyTag {
    DEFAULT = 0,
    MATMUL_AIV_SIMPLE = 1,
    MATMUL_AIV_TARNS = 2
};

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class DispatchPolicy>
CATLASS_DEVICE void AivMatmul(Catlass::GemmCoord &problemShape, Catlass::MatrixCoord &taskShape, GM_ADDR gmA,
    LayoutA &layoutA, GM_ADDR gmB, LayoutB &layoutB, GM_ADDR gmC, LayoutC &layoutC,
    Catlass::Arch::Resource<ArchTag> &resource)
{
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using BiasType = void;
    using TileCopy = Catlass::Gemm::Tile::TileCopyAiv<ArchTag, AType, BType, CType>;
    static constexpr uint32_t COMPUTE_LENGTH = 16 * 1024;
    using TileVmuls = Catlass::Gemm::Tile::TileMuls<ArchTag, AType, COMPUTE_LENGTH>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadAiv<DispatchPolicy, AType, BType, CType, BiasType, TileCopy, TileVmuls>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::MatmulAiv<void, void, BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, DispatchPolicyTag dispatchPolicy>
CATLASS_GLOBAL __attribute__((aic)) void CommonMatmulKernel(__gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ tilingData)
{
    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    /*
    * Load tiling parameters from global memory (tilingData) to local array tilingParams
    * 
    * tilingData memory layout corresponds to tilingParams as follows:
    * -------------------------------------------------------------------------
    * | Offset | Size | Variable   | Type      | Description                   |
    * |--------|------|------------|-----------|-------------------------------|
    * | 0-3    | 4    | m          | uint32_t  | matrix M dimension            |
    * | 4-7    | 4    | n          | uint32_t  | matrix N dimension            |
    * | 8-11   | 4    | k          | uint32_t  | matrix K dimension            |
    * | 16-23  | 8    | strideA    | uint64_t  | matrix B stride               |
    * | 24-31  | 8    | strideB    | uint64_t  | matrix B stride               |
    * | 32-39  | 8    | strideC    | uint64_t  | matrix C stride               |
    * | 40-41  | 2    | m1         | uint16_t  | l1 mTile(16-bit to save space)|
    * | 42-43  | 2    | n1         | uint16_t  | l1 nTile(16-bit to save space)|
    * | 44-45  | 2    | k1         | uint16_t  | l1 kTile(16-bit to save space)|
    * -------------------------------------------------------------------------
    */
    uint8_t tilingParams[28];
    // Copy data in 64-bit chunks to tilingParams array for efficiency
    // Copy bytes 0-7: m and n
    *(uint64_t *)(tilingParams) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData));
    // Copy bytes 8-11: k
    *(uint32_t *)(tilingParams + 8) = *(reinterpret_cast<__gm__ uint32_t *>(tilingData + 8));
    // Copy bytes 32-39: strideC
    *(uint64_t *)(tilingParams + 12) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 32));
    // Copy bytes 40-47: m1, n1, k1
    *(uint64_t *)(tilingParams + 20) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 40));

    /*
    * Parse tiling parameters from local array tilingParams
    * 
    * tilingParams memory layout:
    * --------------------------------------------------------------------------------
    * | Offset | Size | Variable | Type      | Source             | Description        |
    * |--------|------|----------|-----------|--------------------|--------------------|
    * | 0-3    | 4    | m        | uint32_t  | tilingParams[0:3]  | matrix M dimension |
    * | 4-7    | 4    | n        | uint32_t  | tilingParams[4:7]  | matrix N dimension |
    * | 8-11   | 4    | k        | uint32_t  | tilingParams[8:11] | matrix K dimension |
    * | 12-19  | 8    | strideC  | int64_t   | tilingParams[12:19]| matrix C stride    |
    * | 20-21  | 2    | m1       | uint16_t  | tilingParams[20:21]| block M size       |
    * | 22-23  | 2    | n1       | uint16_t  | tilingParams[22:23]| block N size       |
    * | 24-25  | 2    | k1       | uint16_t  | tilingParams[24:25]| block K size       |
    * | 26-28  | 2    | (reserved)| -        | tilingParams[26:28]| unused             |
    * ---------------------------------------------------------------------------------
    * This requires little-endian architecture to work correctly.
    */

    // read m: tilingParams[0:3]
    uint32_t m = *(reinterpret_cast<uint32_t *>(tilingParams));
    // read n: tilingParams[4:7]
    uint32_t n = *(reinterpret_cast<uint32_t *>(tilingParams + 4));
    // read k: tilingParams[8:11]
    uint32_t k = *(reinterpret_cast<uint32_t *>(tilingParams + 8));
    // read strideC: tilingParams[12:19]
    int64_t strideC = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 12)));

    // To save space, tiling parameters (m1, n1, k1) are stored as uint16_t.
    // read m1: tilingParams[20:21]
    uint32_t m1 = *(reinterpret_cast<uint16_t *>(tilingParams + 20));
    // read n1: tilingParams[22:23]
    uint32_t n1 = *(reinterpret_cast<uint16_t *>(tilingParams + 22));

    using LayoutA = Catlass::layout::VectorLayout;
    using LayoutB = Catlass::layout::VectorLayout;
    using LayoutC = Catlass::RowMajor;

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::MatrixCoord taskTileShape(m1, n1);
    LayoutA layoutA{m};
    LayoutB layoutB{n};
    LayoutC layoutC{m, n, strideC};

    // default impl: m axis as scalar axis
    if constexpr (dispatchPolicy == DispatchPolicyTag::DEFAULT) {
        constexpr uint32_t SCALAR_BUFFER_ELE_NUM = 256;
        constexpr uint32_t STAGES = 2;
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Aiv<SCALAR_BUFFER_ELE_NUM, STAGES>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource);
    } else if constexpr (dispatchPolicy == DispatchPolicyTag::MATMUL_AIV_SIMPLE) {
        constexpr uint32_t SCALAR_BUFFER_ELE_NUM = 256;
        constexpr bool IS_TILE_M = true;
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2AivSimple<SCALAR_BUFFER_ELE_NUM, IS_TILE_M>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource);
    } else if constexpr (dispatchPolicy == DispatchPolicyTag::MATMUL_AIV_TARNS) {
        constexpr uint32_t SCALAR_BUFFER_ELE_NUM = 256;
        using DispatchPolicy = Catlass::Gemm::MmadAtlasA2AivSimple<SCALAR_BUFFER_ELE_NUM>;
        AivMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>(
            problemShape, taskTileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource);
    }
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
void LaunchCommonMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dTilingParams, TilingParams &tilingParams)
{
    CommonMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, dispatchPolicyTag>
        <<<tilingParams.blockDim, nullptr, stream>>>(dA, dB, dC, dTilingParams);
}

template <class ElementA, class ElementB, class LayoutB, class ElementC, class LayoutC>
size_t CommonMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    return 0;
}

#endif  // COMMON_MATMUL_KERNEL_H