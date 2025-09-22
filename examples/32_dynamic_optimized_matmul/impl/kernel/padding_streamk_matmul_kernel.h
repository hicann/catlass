/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PADDING_MULTI_CORE_SPLITK_MATMUL_KERNEL_H
#define PADDING_MULTI_CORE_SPLITK_MATMUL_KERNEL_H

#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_padding_streamk_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopyDynamicOptimized : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using CopyGmToL1A = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, AType>;
    using CopyGmToL1B = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB>
[[bisheng::core_ratio(1, 2)]] CATLASS_GLOBAL void PaddingStreamkMatmulKernel(
    uint64_t fftsAddr, __gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ gmWA,
    __gm__ uint8_t *__restrict__ gmWB, __gm__ uint8_t *__restrict__ gmReduceW, __gm__ uint8_t *__restrict__ tilingData)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    /*
     * Load tiling parameters from global memory (tilingData) to local array tilingParams
     *
     * tilingData memory layout corresponds to tilingParams as follows:
     * --------------------------------------------------------------------------
     * | Offset | Size | Variable    | Type      | Description                   |
     * |--------|------|-------------|-----------|-------------------------------|
     * | 0-3    | 4    | m           | uint32_t  | matrix M dimension            |
     * | 4-7    | 4    | n           | uint32_t  | matrix N dimension            |
     * | 8-11   | 4    | k           | uint32_t  | matrix K dimension            |
     * | 16-23  | 8    | strideA     | uint64_t  | matrix B stride               |
     * | 24-31  | 8    | strideB     | uint64_t  | matrix B stride               |
     * | 32-39  | 8    | strideC     | uint64_t  | matrix C stride               |
     * | 40-41  | 2    | m1          | uint16_t  | l1 mTile(16-bit to save space)|
     * | 42-43  | 2    | n1          | uint16_t  | l1 nTile(16-bit to save space)|
     * | 44-45  | 2    | k1          | uint16_t  | l1 kTile(16-bit to save space)|
     * -------------------------------------------------------------------------
     */
    uint8_t tilingParams[48];
    // Copy data in 64-bit chunks to tilingParams array for efficiency
    // Copy bytes 0-7: m and n
    *(uint64_t *)(tilingParams) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData));
    // Copy bytes 8-11: k
    *(uint32_t *)(tilingParams + 8) = *(reinterpret_cast<__gm__ uint32_t *>(tilingData + 8));
    // Copy bytes 16-23: strideA
    *(uint64_t *)(tilingParams + 12) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 16));
    // Copy bytes 24-31: strideB
    *(uint64_t *)(tilingParams + 20) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 24));
    // Copy bytes 32-39: strideC
    *(uint64_t *)(tilingParams + 28) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 32));
    // Copy bytes 40-47: m1, n1, k1
    *(uint64_t *)(tilingParams + 36) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 40));

    /*
     * Parse tiling parameters from local array tilingParams
     *
     * tilingParams memory layout:
     * --------------------------------------------------------------------------------------
     * | Offset | Size | Variable     | Type      | Source             | Description        |
     * |--------|------|--------------|-----------|--------------------|--------------------|
     * | 0-3    | 4    | m            | uint32_t  | tilingParams[0:3]  | matrix M dimension |
     * | 4-7    | 4    | n            | uint32_t  | tilingParams[4:7]  | matrix N dimension |
     * | 8-11   | 4    | k            | uint32_t  | tilingParams[8:11] | matrix K dimension |
     * | 12-19  | 8    | strideA      | int64_t   | tilingParams[12:19]| matrix A stride    |
     * | 20-27  | 8    | strideB      | int64_t   | tilingParams[20:27]| matrix B stride    |
     * | 28-35  | 8    | strideC      | int64_t   | tilingParams[28:35]| matrix C stride    |
     * | 36-37  | 2    | m1           | uint16_t  | tilingParams[36:37]| block M size       |
     * | 38-39  | 2    | n1           | uint16_t  | tilingParams[38:39]| block N size       |
     * | 40-41  | 2    | k1           | uint16_t  | tilingParams[40:41]| block K size       |
     * | 42-47  | 6    | (reserved)   |  -        | tilingParams[42:47]| unused             |
     * -------------------------------------------------------------------------------------
     * This requires little-endian architecture to work correctly.
     */

    // read m: tilingParams[0:3]
    uint32_t m = *(reinterpret_cast<uint32_t *>(tilingParams));
    // read n: tilingParams[4:7]
    uint32_t n = *(reinterpret_cast<uint32_t *>(tilingParams + 4));
    // read k: tilingParams[8:11]
    uint32_t k = *(reinterpret_cast<uint32_t *>(tilingParams + 8));
    // read strideA: tilingParams[12:19]
    int64_t strideA = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 12)));
    // read strideB: tilingParams[20:27]
    int64_t strideB = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 20)));
    // read strideC: tilingParams[28:35]
    int64_t strideC = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 28)));

    // To save space, tiling parameters (m1, n1, k1) are stored as uint16_t.
    // read m1: tilingParams[36:37]
    uint32_t m1 = *(reinterpret_cast<uint16_t *>(tilingParams + 36));
    // read n1: tilingParams[38:39]
    uint32_t n1 = *(reinterpret_cast<uint16_t *>(tilingParams + 38));
    // read k1: tilingParams[40:41]
    uint32_t k1 = *(reinterpret_cast<uint16_t *>(tilingParams + 40));

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};

    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    using PaddingA = typename PaddingBuilderA::Padding;
    using PaddingB = typename PaddingBuilderB::Padding;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicCommon<enableShuffleK, enableShuffleK>;

    using AType = Catlass::Gemm::GemmType<ElementA, typename PaddingBuilderA::LayoutAfterPadding>;
    using BType = Catlass::Gemm::GemmType<ElementB, typename PaddingBuilderB::LayoutAfterPadding>;
    using CType = Catlass::Gemm::GemmType<float, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;

    constexpr uint32_t computeLength = 32 * 1024 / sizeof(float);
    using RecudeAdd = Catlass::Gemm::Kernel::StreamkReduceAdd<ArchTag, BlockScheduler float, ElementC, computeLength>;
    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Catlass::Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 0>;
        // kernel level
        using MatmulKernel = Catlass::Gemm::Kernel::DynamicPaddingMultiCoreSplitkMatmul<
            PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RecudeAdd>;
        typename MatmulKernel::Params params{
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, gmWB, gmReduceW};
        // call a kernel
        MatmulKernel matmul;
        matmul(params, resource);
    } else {
        using BlockScheduler = typename Catlass::Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 1>;
        // kernel level
        using MatmulKernel = Catlass::Gemm::Kernel::DynamicPaddingMultiCoreSplitkMatmul<
            PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RecudeAdd>;
        typename MatmulKernel::Params params{
            problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, gmWB, gmReduceW};
        // call a kernel
        MatmulKernel matmul;
        matmul(params, resource);
    }
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB>
void LaunchPaddingStreamkMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dW, uint8_t *dTilingParams, TilingParams &tilingParams)
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    uint8_t *dWA = nullptr;
    uint8_t *dWB = nullptr;
    uint8_t *dReduceW = nullptr;
    size_t sizeWA = 0, sizeWB = 0;

    dWA = dW;
    if constexpr (paddingTagA == PaddingTag::PADDING_BLOCK_ND) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, m1, k1);
    } else if constexpr (paddingTagA == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, 512 / sizeof(ElementA));
    } else if constexpr (paddingTagA == PaddingTag::PADDING_NZ) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k);
    }

    dWB = dW + sizeWA;
    if constexpr (paddingTagB == PaddingTag::PADDING_BLOCK_ND) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, k1, n1);
    } else if constexpr (paddingTagB == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, 512 / sizeof(ElementB));
    } else if constexpr (paddingTagB == PaddingTag::PADDING_NZ) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n);
    }

    dReduceW = dW + sizeWA + sizeWB;

    PaddingStreamkMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, paddingTagA, paddingTagB
        ><<<tilingParams.blockDim, nullptr, stream>>>(fftsAddr, dA, dB, dC, dWA, dWB, dReduceW, dTilingParams);
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    PaddingTag paddingTagA, PaddingTag paddingTagB>
size_t PaddingStreamkMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = static_cast<uint32_t>(tilingParams.m1);
    uint32_t n1 = static_cast<uint32_t>(tilingParams.n1);
    uint32_t k1 = static_cast<uint32_t>(tilingParams.k1);
    size_t sizeWA = 0, sizeWB = 0, sizeReduceW;
    if constexpr (paddingTagA == PaddingTag::PADDING_BLOCK_ND) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, m1, k1);
    } else if constexpr (paddingTagA == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k, 512 / sizeof(ElementA));
    } else if constexpr (paddingTagA == PaddingTag::PADDING_NZ) {
        sizeWA = PaddingBuilderA::Padding::GetWorkspaceSize(m, k);
    }

    if constexpr (paddingTagB == PaddingTag::PADDING_BLOCK_ND) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, k1, n1);
    } else if constexpr (paddingTagB == PaddingTag::PADDING_ND) {
        // Optimal bandwidth for 512 Byte aligned reads
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n, 512 / sizeof(ElementB));
    } else if constexpr (paddingTagB == PaddingTag::PADDING_NZ) {
        sizeWB = PaddingBuilderB::Padding::GetWorkspaceSize(k, n);
    }
    sizeReduceW = static_cast<size_t>(m1) * n1 * 2 * sizeof(float);
    return sizeWA + sizeWB + sizeReduceW;
}

#endif  // PADDING_MULTI_CORE_SPLITK_MATMUL_KERNEL_H