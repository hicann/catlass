/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_ADD_ATOMIC_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_ADD_ATOMIC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

namespace detail {

template <class T>
CATLASS_DEVICE
void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
{
    for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
        reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
    }
}

} // namespace detail

// Template for grouped matmul add kernel. Compute grouped D = A * B + X
// The matmul results atomic add to X, and then produce D.
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class GroupedMatmulAddAtomic {
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

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;

    struct Params {
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrD;
        GM_ADDR ptrLayoutC;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrD_, GM_ADDR ptrLayoutC_)
            : problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
            ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
            ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
            ptrD(ptrD_), ptrLayoutC(ptrLayoutC_)
        {}
    };

    struct Arguments {
        uint32_t problemCount;
        GemmCoord problemShape;  // used for workspace size
        size_t elementSize;
        uint8_t *ptrProblemShape;
        uint8_t *ptrA;
        uint8_t *ptrLayoutA;
        uint8_t *ptrB;
        uint8_t *ptrLayoutB;
        uint8_t *ptrD;  // also stores X
        uint8_t *ptrLayoutC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, void *workspace)
    {
        Params params{args.problemCount,
            args.ptrProblemShape,
            args.ptrA,
            args.ptrLayoutA,
            args.ptrB,
            args.ptrLayoutB,
            args.ptrD,
            args.ptrLayoutC};
        return params;
    }

    CATLASS_DEVICE
    GroupedMatmulAddAtomic()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutCList[MAX_TENSOR_COUNT];

        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutCList, params.ptrLayoutC, params.problemCount);

        BlockScheduler matmulBlockScheduler;
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementC *)params.ptrD);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t inGroupOffsetA = 0;
        int64_t inGroupOffsetB = 0;
        int64_t inGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            GemmCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutC = layoutCList[groupIdx];

            if (problemShape.k() == 0) {
                inGroupOffsetA += problemShape.m() * problemShape.k();
                inGroupOffsetB += problemShape.k() * problemShape.n();
                inGroupOffsetC += problemShape.m() * problemShape.n();
                continue;
            }

            matmulBlockScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                blockMmad(
                    gmA[inGroupOffsetA + gmOffsetA], layoutA,
                    gmB[inGroupOffsetB + gmOffsetB], layoutB,
                    gmD[inGroupOffsetC + gmOffsetC], layoutC,
                    actualBlockShape);
            }

            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetC += problemShape.m() * problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_ADD_ATOMIC_HPP