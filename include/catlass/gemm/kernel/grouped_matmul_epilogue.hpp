/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class Element_
>
struct MemFill {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;

    CATLASS_DEVICE
    MemFill(Arch::Resource<ArchTag> &resource)
    {
        ubBuffer = resource.ubBuf.template GetBufferByByte<Element>(0);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    uint32_t elementCount, Element fillValue)
    {
        const uint32_t maxBurstSize = MAX_BURST_BYTES / sizeof(Element);
        const uint32_t ubBufferSize = ubBuffer.GetSize() > maxBurstSize ? maxBurstSize : ubBuffer.GetSize();
        const uint32_t batchCount = elementCount / ubBufferSize;
        const uint32_t tailElements = elementCount % ubBufferSize;

        // duplicate fillValue to ubBuffer for datacopy later
        AscendC::Duplicate<Element>(ubBuffer, fillValue, ubBufferSize);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t currentOffset = 0;

        // fill the main block by datacopy
        if (batchCount > 0) {
            for (int index = 0; index < batchCount; ++index) {
                AscendC::DataCopyPad(dst[currentOffset], ubBuffer,
                    AscendC::DataCopyExtParams(1, static_cast<uint32_t>(ubBufferSize * sizeof(Element)), 0, 0, 0));
                currentOffset += ubBufferSize;
            }
        }
        
        // fill the tail block by datacopy
        if (tailElements != 0) {
            AscendC::DataCopyPad(dst[currentOffset], ubBuffer,
                AscendC::DataCopyExtParams(1, static_cast<uint32_t>(tailElements * sizeof(Element)), 0, 0, 0));
        }
    }

    CATLASS_DEVICE
    ~MemFill() {}

private:
    static const size_t MAX_BURST_BYTES = 255 * 32;
    AscendC::LocalTensor<Element> ubBuffer;
};
// Template for grouped matmul add kernel. Compute grouped D = A * B + X
// The matmul results are written to a workspace and the epilogue adds X to produce D.
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMatmulEpilogue {
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
    using ElementGroupList = ElementGroupList_;

    using BlockEpilogue = BlockEpilogue_;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    using BlockScheduler = BlockScheduler_;
    using MemFill0 = MemFill<ArchTag, ElementC>;

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    struct Arguments {
        uint32_t problemCount;
        GemmCoord problemShape; // used for workspace size
        uint8_t *ptrGroupList;
        uint8_t *ptrA;
        uint8_t *ptrB;
        uint8_t *ptrD; // also stores X
    };

    static bool CanImplement(const Arguments &args) { return true; }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return static_cast<size_t>(sizeof(ElementC)) *
               args.problemShape.m() * args.problemShape.n() * args.problemCount;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutD layoutD{m, n};
        Params params{
            args.problemShape, args.problemCount, args.ptrGroupList,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrD, layoutD,
            workspace};
        return params;
    }

    CATLASS_DEVICE
    GroupedMatmulEpilogue() {}

    CATLASS_DEVICE
    ~GroupedMatmulEpilogue() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t inGroupOffsetA = 0;
        int64_t inGroupOffsetB = 0;
        int64_t inGroupOffsetC = 0;
        
        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentK = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord problemShape{params.problemShape.m(), params.problemShape.n(), currentK};

            if (problemShape.k() == 0) {
                inGroupOffsetA += problemShape.m() * problemShape.k();
                inGroupOffsetB += problemShape.k() * problemShape.n();
                inGroupOffsetC += problemShape.m() * problemShape.n();
                continue;
            }

            LayoutA layoutA = params.layoutA.GetTileLayout(problemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB.GetTileLayout(problemShape.GetCoordKN());
            LayoutC layoutC = LayoutC(problemShape.m(), problemShape.n());

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
                    gmWorkspace[inGroupOffsetC + gmOffsetC], layoutC,
                    actualBlockShape);

                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
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
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        MemFill0 memFill0(resource);

        BlockScheduler matmulBlockScheduler;

        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        int64_t inGroupOffsetC = 0;
        int64_t inGroupOffsetD = 0;
        uint32_t startCoreIdx = 0;

        GemmCoord blockShape = L1TileShape::ToCoord();
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentK = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord problemShape{params.problemShape.m(), params.problemShape.n(), currentK};

            LayoutC layoutC = LayoutC(problemShape.m(), problemShape.n());
            LayoutD layoutD = params.layoutD.GetTileLayout(problemShape.GetCoordMN());

            if (problemShape.k() == 0) {
                memFill0(gmWorkspace[inGroupOffsetC], problemShape.m() * problemShape.n(), 0);
                inGroupOffsetC += problemShape.m() * problemShape.n();
                inGroupOffsetD += problemShape.m() * problemShape.n();
                continue;
            }

            typename BlockEpilogue::Params epilogueParams{
                params.ptrD + sizeof(ElementD) * inGroupOffsetD, layoutD,
                params.ptrD + sizeof(ElementD) * inGroupOffsetD, layoutD};
            BlockEpilogue blockEpilogue(resource, epilogueParams);

            matmulBlockScheduler.Update(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (aicoreIndex < startCoreIdx) {
                startLoopIdx = aicoreIndex + aicoreNum - startCoreIdx;
            } else {
                startLoopIdx = aicoreIndex - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                auto gmBlockC = gmWorkspace[inGroupOffsetC +
                    layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
                blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
            }

            inGroupOffsetC += problemShape.m() * problemShape.n();
            inGroupOffsetD += problemShape.m() * problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % aicoreNum;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP
