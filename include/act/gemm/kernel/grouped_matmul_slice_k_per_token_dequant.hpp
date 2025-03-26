/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMM_KERNEL_GROUPED_MATMUL_K_PER_TOKEN_DEQUANT_HPP
#define ACT_GEMM_KERNEL_GROUPED_MATMUL_K_PER_TOKEN_DEQUANT_HPP

#include "act/act.hpp"
#include "act/arch/cross_core_sync.hpp"
#include "act/arch/resource.hpp"
#include "act/coord.hpp"
#include "act/detail/callback.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"

namespace Act::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMatmulSliceKPerTokenDequant {
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

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;

    friend class AicFinishSync;
    friend class AivWaitSync;

    struct AicFinishSync {
        using MatmulKernel = GroupedMatmulSliceKPerTokenDequant<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        ACT_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(ptr->flagAicFinishStore);
        }

        MatmulKernel *ptr;
    };

    struct AivWaitSync {
        using MatmulKernel = GroupedMatmulSliceKPerTokenDequant<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        ACT_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(ptr->flagAicFinishStore);
        }

        MatmulKernel *ptr;
    };

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementScale *ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        // Methods
        ACT_DEVICE
        Params() {}

        ACT_DEVICE
        Params(
            GemmCoord problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrScale_, LayoutScale layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
            GM_ADDR ptrD_, LayoutD layoutD_,
            GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)), layoutScale(layoutScale_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    // Methods
    ACT_DEVICE
    GroupedMatmulSliceKPerTokenDequant() {}

    template <int32_t CORE_TYPE = g_coreType>
    ACT_DEVICE
    void operator()(Params const &params);

    template <>
    ACT_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        AicFinishSync aicFinishSync{this};

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentK = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{params.problemShape.m(), params.problemShape.n(), currentK};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB.GetTileLayout(inGroupProblemShape.GetCoordKN());
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                        gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                        actualBlockShape, MakeCallback(&aicFinishSync)
                    );
                } else {
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB[gmGroupOffsetB + gmOffsetB], layoutB,
                        gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                        actualBlockShape
                    );
                    aicFinishSync();
                }
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
    ACT_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetC = 0;
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AivWaitSync aicFinishSync{this};

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentK = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{params.problemShape.m(), params.problemShape.n(), currentK};

            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());
            LayoutScale layoutScale = params.layoutScale;
            LayoutPerTokenScale layoutPerTokenScale =
                params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
            LayoutD layoutD = params.layoutD.GetTileLayout(inGroupProblemShape.GetCoordMN());

            EpilogueParams epilogueParams{
                params.ptrScale + gmGroupOffsetScale, layoutScale,
                params.ptrPerTokenScale + gmGroupOffsetPerTokenScale, layoutPerTokenScale,
                params.ptrD + gmGroupOffsetD, layoutD
            };

            blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
            blockEpilogue.UpdateParams(epilogueParams);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            GemmCoord blockShapeMNK = L1TileShape::ToCoord();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

                int64_t gmInGroupOffsetC = layoutC.GetOffset(blockCoordMNK.GetCoordMN() * blockShapeMNK.GetCoordMN());
                auto gmBlockC = gmC[gmGroupOffsetC + gmInGroupOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                blockEpilogue(
                    blockShapeMNK, blockCoordMNK,
                    actualBlockShapeMNK, gmBlockC,
                    layoutBlockC, MakeCallback(&aicFinishSync)
                );
            }

            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            gmGroupOffsetScale += inGroupProblemShape.n();
            gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
            gmGroupOffsetD += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Act::Gemm::Kernel

#endif // ACT_GEMM_KERNEL_GROUPED_MATMUL_K_PER_TOKEN_DEQUANT_HPP