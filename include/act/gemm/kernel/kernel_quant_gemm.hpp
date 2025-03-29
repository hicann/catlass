/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACT_GEMM_KERNEL_GEMM_HPP
#define ACT_GEMM_KERNEL_GEMM_HPP

#include "act/act.hpp"
#include "act/arch/cross_core_sync.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"
#include "act/epilogue/tile/copy_gm_to_ub.hpp"
#include "act/epilogue/tile/copy_ub_to_gm.hpp"
#include "act/gemm/helper.hpp"

using namespace Act;

namespace Act::Gemm::kernel{
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
    using CopyGm2Ub = Act::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, Act::layout::RowMajor>>;
    using CopyUb2Gm = Act::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, Act::layout::RowMajor>>;
    using ComputeLayout = Act::layout::RowMajor;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    ACT_DEVICE
    PaddingMatrix(Arch::Resource<ArchTag> &resource){
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) { // 
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element)); 
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    ACT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout){ 
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0)); 
    }

    ACT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout){ 
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1)); 
    }

    ACT_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    Layout layoutDst, Layout layoutSrc
    ){
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
            uint32_t loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH); 
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
                bufferIndex = 1 - bufferIndex;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / paddingStride; 
            coreLoops = CeilDiv(tilesPerAiv, tilesPerLoop);
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
                bufferIndex = 1 - bufferIndex;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]); 
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    ACT_DEVICE
    ~PaddingMatrix() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template<
    class BlockGemm_,
    class BlockEpilogue_ ,
    class TileScheduler_ = void
>
class KernelQuantGemm{
public:
    using BlockGemm = BlockGemm_;
    using ArchTag = typename BlockGemm::ArchTag;
    using L1TileShape = typename BlockGemm::L1TileShape;
    using ElementA = typename BlockGemm::ElementA;
    using LayoutA = typename BlockGemm::LayoutA;
    using LayoutWA = typename BlockGemm::LayoutA;
    using ElementB = typename BlockGemm::ElementB;
    using LayoutB = typename BlockGemm::LayoutB;
    using LayoutWB = typename BlockGemm::LayoutB;
    using ElementX = typename BlockGemm::ElementX;
    using LayoutX = typename BlockGemm::LayoutX;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0XBlockNum = ArchTag::L0C_SIZE / cSize;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = BlockGemm::STAGES; 
    using TileScheduler = TileScheduler_;
    
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA); 
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    typedef struct Params{
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR gmWorkspace;
        GM_ADDR ptrWA;
        LayoutA layoutWA; 
        GM_ADDR ptrWB;
        LayoutB layoutWB;
        EpilogueParams epilogueParams;

        // Methods
        ACT_HOST_DEVICE
        Params() {}

        ACT_HOST_DEVICE
        Params(GemmCoord problemShape_, GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR gmWorkspace_, 
            GM_ADDR ptrWA_, LayoutA layoutWA_, GM_ADDR ptrWB_, LayoutB layoutWB_, 
            EpilogueParams epilogueParams_) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_), gmWorkspace(gmWorkspace_), 
                ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_), 
                epilogueParams(epilogueParams_){} 
    }Params;

    struct Arguments{
        GemmCoord problemShape;
        uint32_t align;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR gmWorkspace;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        EpilogueParams epilogueParams;
    };
    
    ACT_DEVICE
    bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
    {
        return layout1.stride(0) == layout2.stride(0);
    }
    ACT_DEVICE
    bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
    {
        return layout1.stride(1) == layout2.stride(1);
    }

    static layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
    {
        if (align == 0) {
            return layout;
        }
        return layout::RowMajor(layout.shape(0), layout.shape(1),
            RoundUp(layout.shape(1), align));
    }
    
    static layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
    {
        if (align == 0) {
            return layout;
        }
        return layout::ColumnMajor(layout.shape(0), layout.shape(1),
            RoundUp(layout.shape(0), align));
    }

    static bool CanImplement(const Arguments &args){
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace){
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutWA layoutWA = GetWorkspaceLayout(layoutA, args.align);
        LayoutWB layoutWB = GetWorkspaceLayout(layoutA, args.align);
        Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.gmWorkspace,
                    args.ptrWA, layoutWA, args.ptrWB, layoutWB, args.epilogueParams};
        return params;
    }

    ACT_DEVICE
    KernelQuantGemm(){}

    ACT_DEVICE
    ~KernelQuantGemm(){}

    template<int32_t CORE_TYPE = g_coreType>
    ACT_DEVICE
    void operator()(Params &params){}

    template<>
    ACT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        if (!IsSameStride(params.layoutWA, params.layoutA) || !IsSameStride(params.layoutWB, params.layoutB)) {
            Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }
        Arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.gmWorkspace);
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        #pragma unroll
        for(uint32_t i = 0; i < l0XBlockNum; i++){
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        uint32_t singleIdx = 0;
        LayoutX layoutX(params.problemShape.m(), params.problemShape.n());
        for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextActualShape;
            uint32_t MNextGmBlockIdx = 0; uint32_t NNextGmBlockIdx = 0;
            if(loopIdx + AscendC::GetBlockNum() < coreLoops){
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx / NLoops;
                NNextGmBlockIdx = nextLoopIdx % NLoops;
                uint32_t MNextGmActual = (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = (NNextGmBlockIdx == NLoops - 1) ? (N - NNextGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                nextActualShape = MakeCoord(MNextGmActual, NNextGmActual, K); 
            }
            GemmCoord actualShape{MGmActual, NGmActual, K};
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
            MatrixCoord gmTileAOffset{MGmBlockIdx * maxMPerBlock, 0}; 
            auto gmTileA = gmA[params.layoutWA.GetOffset(gmTileAOffset)];
            MatrixCoord gmTileBOffset{0, NGmBlockIdx * maxNPerBlock}; 
            auto gmTileB = gmB[params.layoutWB.GetOffset(gmTileBOffset)];
            MatrixCoord gmTileXOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; 
            auto gmTileX = gmX[layoutX.GetOffset(gmTileXOffset)];
            MatrixCoord gmTileNextAOffset{MNextGmBlockIdx * maxMPerBlock, 0}; 
            auto gmTileNextA = gmA[params.layoutWA.GetOffset(gmTileNextAOffset)];
            MatrixCoord gmTileNextBOffset{0, NNextGmBlockIdx * maxNPerBlock}; 
            auto gmTileNextB = gmB[params.layoutWB.GetOffset(gmTileNextBOffset)];
            blockGemm(
                gmTileA, params.layoutWA,
                gmTileB, params.layoutWB,
                gmTileX, layoutX,
                gmTileNextA, gmTileNextB,
                actualShape, nextActualShape, isFirstBlock, hasNextBlock, singleIdx
            );
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
            singleIdx = (singleIdx + 1) % l0XBlockNum;
        }
        #pragma unroll
        for(uint32_t i = 0; i < l0XBlockNum; i++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
    } 

    template<>
    ACT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        Arch::Resource<ArchTag> resource;
        if (!IsSameStride(params.layoutWA, params.layoutA)) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (!IsSameStride(params.layoutWB, params.layoutB)) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if (!IsSameStride(params.layoutWA, params.layoutA) || !IsSameStride(params.layoutWB, params.layoutB)) {
            Act::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Act::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
        GemmCoord blockShape = L1TileShape::ToCoord();
        BlockEpilogue blockEpilogue(resource, blockShape, params.epilogueParams);
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        uint32_t aivNum = AscendC::GetSubBlockNum();
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / aivNum;
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.gmWorkspace);
        for(uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            GemmCoord actualShape{MGmActual, NGmActual, K};
            LayoutX layoutX(params.problemShape.m(), params.problemShape.n());
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore); 
            MatrixCoord gmTileXOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; 
            auto offsetX = layoutX.GetOffset(gmTileXOffset);
            TensorCoord gmTileScaleOffset{NGmBlockIdx * maxNPerBlock};
            auto offsetScale = params.epilogueParams.layoutScale.GetOffset(gmTileScaleOffset);
            TensorCoord gmTilePerTokenScaleOffset{MGmBlockIdx * maxMPerBlock};
            auto offsetPerTokenScale = params.epilogueParams.layoutPerTokenScale.GetOffset(gmTilePerTokenScaleOffset);
            TensorCoord gmTileBiasOffset{NGmBlockIdx * maxNPerBlock};
            auto offsetBias = params.epilogueParams.layoutBias.GetOffset(gmTileBiasOffset);
            blockEpilogue(offsetX, offsetScale, offsetPerTokenScale, offsetBias, gmX[offsetX], layoutX, actualShape);
        }
    }
private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};
}

#endif // ACT_GEMM_KERNEL_GEMM_HPP