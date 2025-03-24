/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCENDCT_GEMV_KERNLE_GEMV_AIC_HPP
#define ASCENDCT_GEMV_KERNLE_GEMV_AIC_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/cross_core_sync.hpp"
#include "AscendCT/arch/resource.hpp"
#include "AscendCT/epilogue/tile/copy_gm_to_ub.hpp"
#include "AscendCT/epilogue/tile/copy_ub_to_gm.hpp"
#include "AscendCT/gemv_coord.hpp"
#include "AscendCT/matrix_coord.hpp"

namespace AscendCT::gemv::kernel {

// tmeplate for gemv kernle, Compute z = αAx + βy
template <
    class BlockGemv_,
    class BlockEpilogue_,
    class TileScheduler_
>
class GemvEpilogue {
public:
    using BlockGemv = BlockGemv_;
    using ArchTag = typename BlockGemv::ArchTag;
    using L1TileShape = typename BlockGemv::L1TileShape;
    using L0TileShape = typename BlockGemv::L0TileShape;

    using ElementX = typename BlockGemv::ElementX;
    using LayoutX = typename BlockGemv::LayoutX;

    using ElementA = typename BlockGemv::ElementA;
    using LayoutA = typename BlockGemv::LayoutA;
    using ElementY = typename BlockGemv::ElementY;
    using LayoutY = typename BlockGemv::LayoutY;

    using BlockEpilogue = BlockEpilogue_;
    using ElementZ = typename BlockEpilogue::ElementZ;
    using LayoutZ = typename BlockEpilogue::LayoutZ;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementAccumulator = typename gemv::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using TileScheduler = TileScheduler_;

    struct Params {
        // Data members
        GemvCoord problemShape;
        GM_ADDR ptrX;
        LayoutX layoutX;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        // Methods
        ASCENDCT_HOST_DEVICE
        Params() {}

        ASCENDCT_HOST_DEVICE
        Params(
            GemvCoord const& problemShape_,
            GM_ADDR ptrX_,
            LayoutX layoutX_,
            GM_ADDR ptrA_,
            LayoutA layoutA_,
            GM_ADDR ptrWorkspace_,
            EpilogueParams const& epilogueParams_)
            : problemShape(problemShape_), ptrX(ptrX_), layoutX(layoutX_), ptrA(ptrA_), layoutA(layoutA_), ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    //TODO: add arguments
    struct Arguments {
        GemvCoord problemShape;
        ElementY alpha;
        ElementY beta;
        size_t elementSize;
        GM_ADDR ptrX;
        GM_ADDR ptrA;
        GM_ADDR ptrZ;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return args.elementSize * args.problemShape.m() * args.problemShape.n();
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        MatmulCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        LayoutX layoutX{1, n};
        LayoutA layoutA{m, n};
        LayoutZ layoutZ{m};
        typename BlockEpilogue::Params epilogueParams{args.alpha, args.beta, args.ptrZ, layoutZ, args.ptrZ, layoutZ};

        
        Params params{problemShape, args.ptrX, layoutX, args.ptrA, layoutA, workspace, epilogueParams};
        return params;
    }

    // Methods
    ASCENDCT_DEVICE
    GemvEpilogue() {}

    template <int32_t CORE_TYPE = g_coreType>
    ASCENDCT_DEVICE 
    void operator()(Params const& params);

    template <>
    ASCENDCT_DEVICE 
    void operator()<AscendC::AIC>(Params const& params) 
    {
        TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        // arch::Resource<ArchTag> resource;
        BlockGemv blockGemv(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrWorkspace);

        layout::RowMajor layoutY(1, params.problemShape.m());

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();

        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;
        uint32_t singleIdx = 0;

        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
        static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute Block location
            uint32_t MGmBlockIdx = loopIdx;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = N;
            int64_t gmOffsetX;
            int64_t gmOffsetA;
            int64_t gmOffsetY;
            int64_t gmOffsetNextX;
            int64_t gmOffsetNextA;
            int64_t gmOffsetNextY;

            if constexpr (std::is_same<LayoutA, AscendCT::layout::RowMajor>::value) {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetX = 0;
                gmOffsetA = MGmBlockIdx * maxMPerBlock;
                gmOffsetY = MGmBlockIdx * maxMPerBlock;
            }

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            uint32_t MNextGmBlockIdx;
            GemvCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx;
                uint32_t MNextGmActual = (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = N;
                nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
            }

            if constexpr (std::is_same<LayoutA, AscendCT::layout::RowMajor>::value) {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);

                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            } else {
                gmOffsetNextX = 0;
                gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
                gmOffsetNextY = MNextGmBlockIdx * maxMPerBlock;
            }

            GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            // Compute block-scoped matrix multiply-add
            blockGemv(gmX[gmOffsetX], params.layoutX,
                      gmA[gmOffsetA], params.layoutA,
                      gmY[gmOffsetY], layoutY,
                      gmX[gmOffsetNextX],
                      gmA[gmOffsetNextA],
                      actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock,
                      singleIdx);

            arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

            singleIdx++;
        }

        #pragma unroll
        for (uint32_t i = 0; i < L0C_TILE_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
        }
    }

    template <>
    ASCENDCT_DEVICE 
    void operator()<AscendC::AIV>(Params const& params) 
    {
        TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrWorkspace);

        // layout::RowMajor layoutY(1, params.problemShape.m());
        layout::VectorLayout layoutY(params.problemShape.m());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        uint32_t maxMPerBlock = L1TileShape::M;
        // uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        // uint32_t N = params.problemShape.n();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t coreLoops = MLoops;

        // Loop through the epilogue calculations of each basic block
        // GemvCoord blockShape{L1TileShape::N, L1TileShape::M};
        layout::VectorLayout::TensorCoord blockShape{L1TileShape::M};

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            // GemvCoord blockCoord = GemvCoord(0, loopIdx);
            layout::VectorLayout::TensorCoord blockCoord{loopIdx};
            uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;
            // uint32_t NGmActual = 1;

            // GemvCoord actualBlockShape = GemvCoord(NGmActual, MGmActual);
            layout::VectorLayout::TensorCoord actualBlockShape{MGmActual};

            // Get the offset
            // MatrixCoord blockOffset = blockCoord.GetCoordMN() * blockShape.GetCoordMN();
            layout::VectorLayout::TensorCoord blockOffset = blockCoord * blockShape;

            // Get the data and layout of y under the current basic block
            auto gmBlockY = gmY[layoutY.GetOffset(blockOffset)];
            auto layoutBlockY = layoutY.GetTileLayout(actualBlockShape);

            // Synchronize cross core
            arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            // Actual calculatioin logic for performing block-scoped epilogue
            blockEpilogue(blockOffset, actualBlockShape, gmBlockY, layoutBlockY);
        }
    }

private:
    // ID used for inter-core synchronization
    static constexpr arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    arch::Resource<ArchTag> resource;

    static constexpr arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};

}  // namespace AscendCT::gemv::kernel

#endif  // ASCENDCT_GEMV_KERNLE_GEMV_EPILOGUE_HPP
