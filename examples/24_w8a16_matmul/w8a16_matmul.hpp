/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_W8A16_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_W8A16_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class W8A16Matmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

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
        GM_ADDR ptrWksp;
        half deqScalar;
        half deqZeroPoint;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWksp_, half deqScalar_, half deqZeroPoint_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWksp(ptrWksp_), deqScalar(deqScalar_), deqZeroPoint(deqZeroPoint_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        half deqScalar;
        half deqZeroPoint;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        // // Calculate workspace size, using double buffer
        size_t lenWorkspace = static_cast<size_t>(L1TileShape::N) * L1TileShape::K *
            args.aicCoreNum * BUFFER_NUM;
        size_t sizeWorkspace = lenWorkspace * args.elementSize;
        return sizeWorkspace;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};

        Params params{args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            workspace,
            args.deqScalar,
            args.deqZeroPoint};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    W8A16Matmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<int8_t> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockIdxCoord.k() * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

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
            int64_t gmOffsetNextA = params.layoutWA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutWB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutWA,
                gmB[gmOffsetB], params.layoutWB,
                gmC[gmOffsetC], params.layoutC,
                gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockIdxCoord.k() * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

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
            int64_t gmOffsetNextA = params.layoutWA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutWB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutWA,
                gmB[gmOffsetB], params.layoutWB,
                gmC[gmOffsetC], params.layoutC,
                gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    static const uint32_t BUFFER_NUM = 2;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP