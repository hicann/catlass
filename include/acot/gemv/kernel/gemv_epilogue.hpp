/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP
#define ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/gemv_coord.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::gemv::kernel
{

    // Template for matmul add kernel. Compute D = A * B + X
    // tmeplate for gemv kernle, Compute z = αAx + βy
    template <
        class BlockGemv_,
        class BlockEpilogue_,
        class TileScheduler_>
    class GemvEpilogue
    {
    public:
        using BlockGemv = BlockGemv_;
        using ArchTag = typename BlockGemv::ArchTag;
        using L1TileShape = typename BlockGemv::L1TileShape;

        using Elementx = typename BlockGemv::Elementx;
        using Layoutx = typename BlockGemv::Layoutx;

        using ElementA = typename BlockGemv::ElementA;
        using LayoutA = typename BlockGemv::LayoutA;
        using Elementy = typename BlockGemv::Elementy;
        using Layouty = typename BlockGemv::Layouty;

        using BlockEpilogue = BlockEpilogue_;
        using Elementz = typename BlockEpilogue::ElementZ;
        using Layoutz = typename BlockEpilogue::LayoutZ;
        using EpilogueParams = typename BlockEpilogue::Params;

        using TileScheduler = TileScheduler_;

        static_assert(std::is_same_v<typename BlockEpilogue::ElementY, Elementy> &&
                          std::is_same_v<typename BlockEpilogue::LayoutY, Layouty>,
                      "The yType of Gemv and Epilogue should be consistent.");

        /// Parameters structure
        struct Params
        {
            // Data members
            GemvCoord problemShape;
            GM_ADDR ptrX;
            Layoutx layoutX;
            GM_ADDR ptrA;
            LayoutA layoutA;
            GM_ADDR ptrWorkspace;
            EpilogueParams epilogueParams;

            // Methods
            ACOT_DEVICE
            Params() {}

            ACOT_DEVICE
            Params(
                GemvCoord const &problemShape_,
                GM_ADDR ptrX_, Layoutx const &layoutX_,
                GM_ADDR ptrA_, LayoutA const &layoutA_,
                GM_ADDR ptrWorkspace_, EpilogueParams const &epilogueParams_) : problemShape(problemShape_), ptrX(ptrX_), layoutX(layoutX_), ptrA(ptrA_), layoutA(layoutA_),
                                                                                ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
        };

        // Methods
        ACOT_DEVICE
        GemvEpilogue() {}

        template <int32_t CORE_TYPE = g_coreType>
        ACOT_DEVICE void operator()(Params const &params);

        template <>
        ACOT_DEVICE void operator()<AscendC::AIC>(Params const &params)
        {
            TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            // uint32_t coreLoops = matmulTileScheduler.GetCoreLoops();

            arch::Resource<ArchTag> resource;
            BlockGemv blockGemv(resource);

            // Represent the full gm
            AscendC::GlobalTensor<Elementx> gmx;
            gmx.SetGlobalBuffer((__gm__ Elementx *)params.ptrX);
            AscendC::GlobalTensor<ElementA> gmA;

            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            AscendC::GlobalTensor<Elementy> gmy;
            gmy.SetGlobalBuffer((__gm__ Elementy *)params.ptrWorkspace);

            // layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
            layout::RowMajor layouty(1, params.problemShape.m());

            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;
            uint32_t M = params.problemShape.m();
            uint32_t N = params.problemShape.n();

            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t coreLoops = MLoops;
            uint32_t singleIdx = 0;

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum())
            {

                uint32_t MGmBlockIdx = loopIdx;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = N;
                int gmOffsetx;
                int gmOffsetA;
                int gmOffsety;

                if constexpr (std::is_same<LayoutA, acot::layout::RowMajor>::value) // 行优先情况
                {
                    gmOffsetx = 0;
                    gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);
                    gmOffsety = MGmBlockIdx * maxMPerBlock;
                }
                else // 列优先情况
                {
                    gmOffsetx = 0;
                    gmOffsetA = MGmBlockIdx * maxMPerBlock;
                    gmOffsety = MGmBlockIdx * maxMPerBlock;
                }

                GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual);

                // Compute block-scoped matrix multiply-add
                blockGemv(gmx[gmOffsetx], params.layoutX,
                          gmA[gmOffsetA], params.layoutA,
                          gmy[gmOffsety], layouty,
                          actualBlockShape, singleIdx);

                AscendC::PipeBarrier<PIPE_ALL>();
                arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore); // 这行是啥，暂时没懂
                singleIdx++;
            }
        }

        template <>
        ACOT_DEVICE void operator()<AscendC::AIV>(Params const &params)
        {
            TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            // uint32_t coreLoops = matmulTileScheduler.GetCoreLoops();

            BlockEpilogue blockEpilogue(resource, params.epilogueParams);

            // Represent the full gm
            AscendC::GlobalTensor<Elementy> gmy;
            gmy.SetGlobalBuffer((__gm__ Elementy *)params.ptrWorkspace);
            // layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
            layout::RowMajor layouty(1, params.problemShape.m());

            // Get aicore information 获取核idx，核数，子核idx
            uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); // 0-19
            uint32_t aicoreNum = AscendC::GetBlockNum();                               // 20
            uint32_t subcoreIndex = AscendC::GetSubBlockIdx();
            // 扩大空间的尝试
            // GemvShape Ubshape<1024, 1>;

            // 计算总共需要的循环次数
            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;
            uint32_t M = params.problemShape.m();
            uint32_t N = params.problemShape.n();
            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t coreLoops = MLoops;

            // Loop through the epilogue calculations of each basic block
            // GemvCoord blockShape = L1TileShape::ToCoord(); // blockshape就是l1中矩阵的大小
            GemvCoord blockShape{L1TileShape::N, L1TileShape::M};

            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum)
            {
                // Compute block location
                // GemvCoord blockCoord = matmulTileScheduler.GetBlockCoord(loopIdx);
                // MatmulCoord actualBlockShape = matmulTileScheduler.GetActualBlockShape(blockCoord);
                // GemvCoord blockCoord = GemvCoord(loopIdx, 0);
                GemvCoord blockCoord = GemvCoord(0, loopIdx);
                uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;
                uint32_t NGmActual = 1;

                // GemvCoord actualBlockShape =GemvCoord(MGmActual, NGmActual); // 这里的n是错的，已修正
                GemvCoord actualBlockShape = GemvCoord(NGmActual, MGmActual);

                // Get the data and layout of y under the current basic block
                auto gmBlocky = gmy[layouty.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
                auto layoutBlocky = layouty.GetTileLayout(actualBlockShape.GetCoordMN());

                // Synchronize cross core
                arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

                // Actual calculatioin logic for performing block-scoped epilogue
                blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlocky, layoutBlocky);
            }
        }

    private:
        // ID used for inter-core synchronization
        static constexpr arch::FlagID FLAG_AIC_FINISH_STORE = 0;
        static constexpr arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
        arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
        arch::Resource<ArchTag> resource;
    };

} // namespace acot::gemv::kernel

#endif // ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP
