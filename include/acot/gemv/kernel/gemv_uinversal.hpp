/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACOT_GEMV_KERNEL_GEMV_HPP
#define ACOT_GEMV_KERNEL_GEMV_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/coord.hpp"
#include "acot/gemv_coord.hpp"

#include "acot/matrix_coord.hpp"

namespace acot::gemv::kernel
{

    // Template for Matmul kernel. Compute y = A * x
    template <
        class BlockGemv_,
        class BlockEpilogue_,
        class TileScheduler_>
    class GemvUniversal
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
        using ElementAccumulator = typename BlockGemv::ElementAccumulator;

        using TileScheduler = TileScheduler_;

        /// Parameters structure
        struct Params
        {
            // Data members
            GemvCoord problemShape;

            GM_ADDR ptrX;
            Layoutx layoutx;
            GM_ADDR ptrA;
            LayoutA layoutA;
            GM_ADDR ptrY;
            Layouty layouty;

            // Methods
            ACOT_DEVICE
            Params() {}

            ACOT_DEVICE
            Params(GemvCoord const &problemShape_, GM_ADDR ptrX_, Layoutx layoutx_, GM_ADDR ptrA_,
                   LayoutA layoutA_, GM_ADDR ptrY_, Layouty layouty_)
                : problemShape(problemShape_), ptrX(ptrX_), layoutx(layoutx_), ptrA(ptrA_), layoutA(layoutA_),
                  ptrY(ptrY_), layouty(layouty_) {}
        };

        // Methods
        ACOT_DEVICE
        GemvUniversal() {}

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
            gmy.SetGlobalBuffer((__gm__ Elementy *)params.ptrY);

            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;
            uint32_t M = params.problemShape.m();
            uint32_t N = params.problemShape.n();

            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t coreLoops = MLoops;
            uint32_t singleIdx = 0;

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum())
            {
                // Compute block location
                // MatmulCoord blockCoord = matmulTileScheduler.GetBlockCoord(loopIdx);
                // MatmulCoord actualBlockShape = matmulTileScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                // MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                // MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                // MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                // int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                // int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                // int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

                uint32_t MGmBlockIdx = loopIdx;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = N;
                int gmOffsetx;
                int gmOffsetA;
                int gmOffsety;

                if constexpr (std::is_same_v<LayoutA, acot::layout::RowMajor>) // 行优先情况
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

                // Compute block-scoped matrix-vector multiply-add
                blockGemv(gmx[gmOffsetx], params.layoutx,
                          gmA[gmOffsetA], params.layoutA,
                          gmy[gmOffsety], params.layouty,
                          actualBlockShape, singleIdx);

                AscendC::PipeBarrier<PIPE_ALL>();
                singleIdx++;
            }
        }

        template <>
        ACOT_DEVICE void operator()<AscendC::AIV>(Params const &params) {}
    };

} // namespace acot::gemv::kernel

#endif // ACOT_MATMUL_KERNEL_MATMUL_HPP
