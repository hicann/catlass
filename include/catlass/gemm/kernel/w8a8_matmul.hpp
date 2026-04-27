/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_W8A8_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_W8A8_MATMUL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/vector.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class W8A8Matmul {
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
    using ElementScale = typename BlockMmad::ElementScale;
    using LayoutScale = typename BlockMmad::LayoutScale;
    using ElementBias = typename BlockMmad::ElementBias;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_,
            LayoutA layoutA_,
            GM_ADDR ptrB_,
            LayoutB layoutB_,
            GM_ADDR ptrC_,
            LayoutC layoutC_,
            GM_ADDR ptrScale_,
            LayoutScale layoutScale_,
            GM_ADDR ptrBias_,
            LayoutBias layoutBias_
        )
            : problemShape(problemShape_)
            , ptrA(ptrA_)
            , layoutA(layoutA_)
            , ptrB(ptrB_)
            , layoutB(layoutB_)
            , ptrC(ptrC_)
            , layoutC(layoutC_)
            , ptrScale(ptrScale_)
            , layoutScale(layoutScale_)
            , ptrBias(ptrBias_)
            , layoutBias(layoutBias_)
        {
        }
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrScale;
        GM_ADDR ptrBias;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(args.problemShape.m(), args.problemShape.n());
        LayoutScale layoutScale{args.problemShape.n()};
        LayoutBias layoutBias{args.problemShape.n()};
        Params params{args.problemShape, args.ptrA,     layoutA,     args.ptrB,    layoutB,   args.ptrC,
                      layoutC,           args.ptrScale, layoutScale, args.ptrBias, layoutBias};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    W8A8Matmul()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2002)
    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::MIX>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            layout::VectorLayout::TensorCoord offsetScale{blockCoord.n() * L1TileShape::N};
            layout::VectorLayout::TensorCoord offsetBias{blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            int64_t gmOffsetScale = params.layoutScale.GetOffset(offsetScale);
            int64_t gmOffsetBias = params.layoutBias.GetOffset(offsetBias);
            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC,
                gmScale[gmOffsetScale], params.layoutScale, gmBias[gmOffsetBias], params.layoutBias, actualBlockShape
            );
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3002)
    template <>
    CATLASS_DEVICE void operator()<AscendC::MIX>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            auto gmOffsetScale = offsetC.template GetCoordByAxis<1>();
            auto gmOffsetBias = offsetC.template GetCoordByAxis<1>();

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockIdCoord);
            }
            MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M, nextBlockIdCoord.k() * L1TileShape::K};
            MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K, nextBlockIdCoord.n() * L1TileShape::N};
            int64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC,
                gmScale[params.layoutScale.GetOffset(gmOffsetScale)], params.layoutScale,
                gmBias[params.layoutBias.GetOffset(gmOffsetBias)], params.layoutBias, gmA[gmOffsetNextA],
                gmB[gmOffsetNextB], actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
            );
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_W8A8_MATMUL_HPP