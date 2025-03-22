/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDCT_MATMUL_KERNEL_GROUPED_MATMUL_M_HPP
#define ASCENDCT_MATMUL_KERNEL_GROUPED_MATMUL_M_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/resource.hpp"
#include "AscendCT/coord.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"

__gm__ struct OpSystemRunCfg g_opSystemRunCfg{L2_OFFSET};

namespace AscendCT::gemm::kernel {

// Template for grouped matmul kernel. Compute grouped C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMatmulM {
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

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        MatmulCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        // Methods
        ASCENDCT_HOST_DEVICE
        Params() {}

        ASCENDCT_HOST_DEVICE
        Params(
            MatmulCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_)
        {
        }
    };
    struct Arguments{
        MatmulCoord problemShape;
        uint32_t problemCount;
        uint8_t *ptrGroupList;
        uint8_t *ptrA;
        uint8_t *ptrB;
        uint8_t *ptrC;
    };
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments &args, void* workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutC layoutC{m, n};
        Params params{args.problemShape, args.problemCount, args.ptrGroupList,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC};
        return params;
    }
    // Methods
    ASCENDCT_HOST_DEVICE
    GroupedMatmulM() {}
    // Methods
    ASCENDCT_HOST_DEVICE
    ~GroupedMatmulM(){}

    template <int32_t CORE_TYPE = g_coreType>
    ASCENDCT_DEVICE
    void operator()(Params const &params);

    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            MatmulCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

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
                MatmulCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                MatmulCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                    gmB[gmOffsetB], layoutB,
                    gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                    actualBlockShape
                );
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }

    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
    }
};

} // namespace AscendCT::gemm::kernel

#endif // ASCENDCT_MATMUL_KERNEL_GROUPED_MATMUL_HPP