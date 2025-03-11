/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_MATMUL_KERNEL_GROUPED_MATMUL_HPP
#define ACOT_MATMUL_KERNEL_GROUPED_MATMUL_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/coord.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::matmul::kernel {

namespace detail {

template <class T>
ACOT_DEVICE
void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
{
    for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
        reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
    }
}

}  // namespace detail

// Template for grouped matmul kernel. Compute grouped C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class TileScheduler_
>
class GroupedMatmul {
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

    using TileScheduler = TileScheduler_;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrC;
        GM_ADDR ptrLayoutC;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrC_, GM_ADDR ptrLayoutC_
        ) : problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
            ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
            ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
            ptrC(ptrC_), ptrLayoutC(ptrLayoutC_)
        {
        }
    };

    // Methods
    ACOT_DEVICE
    GroupedMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params const &params);

    /// Executes matmul
    template <>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        MatmulCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutCList[MAX_TENSOR_COUNT];

        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutCList, params.ptrLayoutC, params.problemCount);

        TileScheduler matmulTileScheduler;
        arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t inGroupOffsetA = 0;
        int64_t inGroupOffsetB = 0;
        int64_t inGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            MatmulCoord problemShape = problemShapeList[groupIdx];
            if (problemShape.k() == 0 || problemShape.m() == 0 || problemShape.n() == 0) {
                inGroupOffsetA += problemShape.m() * problemShape.k();
                inGroupOffsetB += problemShape.k() * problemShape.n();
                inGroupOffsetC += problemShape.m() * problemShape.n();
                continue;
            }
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutC = layoutCList[groupIdx];

            matmulTileScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = matmulTileScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                MatmulCoord blockCoord = matmulTileScheduler.GetBlockCoord(loopIdx);
                MatmulCoord actualBlockShape = matmulTileScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                bool isFirstBlock = (loopIdx == startLoopIdx);
                bool hasNextBlock = false;
                MatmulCoord nextBlockIdCoord;
                MatmulCoord nextActualBlockShape;
                if (loopIdx + coreNum < coreLoops) {
                    hasNextBlock = true;
                    nextBlockIdCoord = matmulTileScheduler.GetBlockCoord(loopIdx + coreNum);
                    nextActualBlockShape = matmulTileScheduler.GetActualBlockShape(nextBlockIdCoord);
                }
                MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M, nextBlockIdCoord.k() * L1TileShape::K};
                MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K, nextBlockIdCoord.n() * L1TileShape::N};
                int64_t gmOffsetNextA = layoutA.GetOffset(offsetNextA);
                int64_t gmOffsetNextB = layoutB.GetOffset(offsetNextB);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[inGroupOffsetA + gmOffsetA], layoutA,
                    gmB[inGroupOffsetB + gmOffsetB], layoutB,
                    gmC[inGroupOffsetC + gmOffsetC], layoutC,
                    gmA[inGroupOffsetA + gmOffsetNextA], gmB[inGroupOffsetB + gmOffsetNextB],
                    actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
            }

            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetC += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

    template <>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace acot::matmul::kernel

#endif // ACOT_MATMUL_KERNEL_GROUPED_MATMUL_HPP