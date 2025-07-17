/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP

#include <cmath>
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class Element_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrix {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrix(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    Layout layoutDst, Layout layoutSrc)
    {
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t paddingStride = computeLayoutDst.stride(0);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{ 0 };
        if (paddingStride > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / paddingStride;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout &ubLayout = dstLayout;

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~PaddingMatrix() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template<
    class ArchTag_,
    class ElementAccumulator_,
    class ElementOut_,
    uint32_t COMPUTE_LENGTH
>
struct ReduceAdd {
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOut = ElementOut_;

    CATLASS_DEVICE
    ReduceAdd(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            accumulatorBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementOut);
        }
    }

    CATLASS_DEVICE
    void Gm2Ub(AscendC::LocalTensor<ElementAccumulator> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementAccumulator), 0, 0, 0);
        AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams, padParams);
    }

    CATLASS_DEVICE
    void Ub2Gm(AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::LocalTensor<ElementOut> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementOut), 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint64_t elementCount, uint32_t splitkFactor)
    {
        // The vec mte processes 256 bytes of data at a time.
        constexpr uint32_t ELE_PER_VECOTR_BLOCK = 256 / sizeof(ElementAccumulator);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint64_t taskPerAiv =
            (elementCount / aivNum + ELE_PER_VECOTR_BLOCK - 1) / ELE_PER_VECOTR_BLOCK * ELE_PER_VECOTR_BLOCK;
        if (taskPerAiv == 0) taskPerAiv = ELE_PER_VECOTR_BLOCK;
        uint32_t tileLen;
        if (taskPerAiv > COMPUTE_LENGTH) {
            tileLen = COMPUTE_LENGTH;
        } else {
            tileLen = taskPerAiv;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);

        uint32_t loops = (elementCount + tileLen - 1) / tileLen;
        for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
            uint32_t actualTileLen = tileLen;
            if (loopIdx == loops - 1) {
                actualTileLen = elementCount - loopIdx * tileLen;
            }

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
            Gm2Ub(accumulatorBuffer[bufferIndex], src[loopIdx * tileLen], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

            for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; ++sliceIdx) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                Gm2Ub(inputBuffer[bufferIndex],
                    src[sliceIdx * elementCount + loopIdx * tileLen], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);

                AscendC::Add(accumulatorBuffer[bufferIndex],
                    accumulatorBuffer[bufferIndex], inputBuffer[bufferIndex], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
            if constexpr (!std::is_same_v<ElementAccumulator, ElementOut>) {
                if constexpr (std::is_same_v<ElementOut, half>) {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualTileLen);
                } else {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualTileLen);
                }
            } else {
                AscendC::DataCopy(outputBuffer[bufferIndex], accumulatorBuffer[bufferIndex], tileLen);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);

            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementAccumulator> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];
    AscendC::TEventID inputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID accumulatorEventIds[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    AscendC::TEventID outputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementAccumulator) * 2
        +  BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementOut) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ReduceAdd_
>
class PaddingSplitkMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    using BlockScheduler = BlockScheduler_;
    using ReduceAdd = ReduceAdd_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        bool aNeedPadding;
        bool bNeedPadding;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutA layoutWA;
        GM_ADDR ptrWB;
        LayoutB layoutWB;
        GM_ADDR ptrWC;
        uint32_t splitkFactor = 1;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, bool aNeedPadding_, bool bNeedPadding_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutA layoutWA_,
               GM_ADDR ptrWB_, LayoutB layoutWB_, GM_ADDR ptrWC_, uint32_t splitkFactor_)
            : problemShape(problemShape_), aNeedPadding(aNeedPadding_), bNeedPadding(bNeedPadding_),
              ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_),
              ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_),
              ptrWC(ptrWC_), splitkFactor(splitkFactor_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        uint32_t align;
        bool aNeedPadding;
        bool bNeedPadding;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum)
    {
        uint32_t maxSplitkFactor;
        if (k <= 1024) {
            // When k is less than or equal to 1024, it can be divided into at most 2 parts.
            maxSplitkFactor = 2;
        } else if (k <= 2048) {
            // When k is less than or equal to 2048, it can be divided into at most 4 parts.
            maxSplitkFactor = 4;
        } else if (k <= 4096) {
            // When k is less than or equal to 4096, it can be divided into at most 8 parts.
            maxSplitkFactor = 8;
        } else {
            // else it can be divided into at most 16 parts.
            maxSplitkFactor = 16;
        }
        uint32_t splitkFactor = 1;
        uint32_t m0 = L1TileShape::M;
        uint32_t n0 = L1TileShape::N;
        uint32_t k0 = L1TileShape::K;

        uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
        splitkFactor = std::min(aicCoreNum / baseTilesCount, maxSplitkFactor);
        // Prevent the split factor form being less than 1
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(1));
        if (baseTilesCount < aicCoreNum) {
            while (splitkFactor + 1 <= maxSplitkFactor &&
                CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >=
                CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
                splitkFactor += 1;
            }
        }
        // Ensure that splitkFactor is less than the number of base tiels in the k direction.
        splitkFactor = std::min(CeilDiv(k, k0), splitkFactor);
        // If k is very large, splitting k can lead to better cache utilization.
        // If k is greater than 8192.
        if (k > 8192) {
            // split the k direction into at least 2 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(2));
        }
        // If k is greater than 32768.
        if (k > 32768) {
            // split the k direction into at least 4 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(4));
        }
        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
    {
        // prevent division of 0
        if (align == 0) {
            return 0;
        }
        return layout::RowMajor(layout.shape(0), layout.shape(1),
            (layout.shape(1) + align - 1) / align * align);
    }

    static layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
    {
        // prevent division of 0
        if (align == 0) {
            return 0;
        }
        return layout::ColumnMajor(layout.shape(0), layout.shape(1),
            (layout.shape(0) + align - 1) / align * align);
    }

    static size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }

    static size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        GemmCoord problemShape = args.problemShape;
        LayoutA layoutA{problemShape.m(), problemShape.k()};
        LayoutB layoutB{problemShape.k(), problemShape.n()};
        size_t sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        size_t sizeWB = GetWorkspaceLen(GetWorkspaceLayout(layoutB, args.align)) * args.elementSize;
        size_t sizeWC = args.elementSize * args.problemShape.m() * args.problemShape.n() *
                        GetSplitkFactor(args.problemShape.m(),
                                        args.problemShape.n(),
                                        args.problemShape.k(),
                                        args.aicCoreNum);
        return sizeWA + sizeWB + sizeWC;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        
        uint8_t *workspaceWA = nullptr;
        uint8_t *workspaceWB = nullptr;
        size_t sizeWA = 0;
        size_t sizeWB = 0;
        
        if (args.aNeedPadding) {
            workspaceWA = workspace;
            sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        } else {
            workspaceWA = args.ptrA;
        }

        if (args.bNeedPadding) {
            workspaceWB = workspace + sizeWA;
            sizeWB = GetWorkspaceLen(GetWorkspaceLayout(layoutB, args.align)) * args.elementSize;
        } else {
            workspaceWB = args.ptrB;
        }

        uint8_t *workspaceWC = workspace + sizeWA + sizeWB;

        Params params{
            args.problemShape,
            args.aNeedPadding,
            args.bNeedPadding,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            workspaceWA,
            GetWorkspaceLayout(layoutA, args.align),
            workspaceWB,
            GetWorkspaceLayout(layoutB, args.align),
            workspaceWC,
            GetSplitkFactor(args.problemShape.m(),
                            args.problemShape.n(),
                            args.problemShape.k(),
                            args.aicCoreNum)};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    PaddingSplitkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if (params.aNeedPadding || params.bNeedPadding) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape,
            GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), params.splitkFactor);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWC);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                blockCoord, matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            uint64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            uint64_t gmOffsetC = params.layoutC.GetOffset(offsetC)
                + static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n())
                * static_cast<uint64_t>(matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutWA,
                      gmB[gmOffsetB], params.layoutWB,
                      gmC[gmOffsetC], params.layoutC,
                      actualBlockShape);
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if (params.aNeedPadding) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (params.bNeedPadding) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
        }

        if (params.aNeedPadding || params.bNeedPadding) {
            // 0x0 synchronization control between AI Core
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        // reduce add
        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmWC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut*>(params.ptrC));
        gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator*>(params.ptrWC));
        ReduceAdd reduceAdd(resource);
        reduceAdd(gmC, gmWC,
            static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n()),
            params.splitkFactor);
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 1;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP
