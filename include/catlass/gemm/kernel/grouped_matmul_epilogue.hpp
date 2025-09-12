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

namespace detail {

template <class T>
CATLASS_DEVICE void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
{
    for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
        reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
    }
}

} // namespace detail
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
    class BlockScheduler_
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

    using BlockEpilogue = BlockEpilogue_;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    using BlockScheduler = BlockScheduler_;
    using MemFill0 = MemFill<ArchTag, ElementC>;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;

    struct Params {
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrLayoutC;
        GM_ADDR ptrD;

        CATLASS_HOST_DEVICE Params() {}

        CATLASS_HOST_DEVICE Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrLayoutC_,
            GM_ADDR ptrD_)
            : problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
              ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
              ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
              ptrWorkspace(ptrWorkspace_), ptrLayoutC(ptrLayoutC_),
              ptrD(ptrD_) {}
    };

    struct Arguments {
        uint32_t problemCount;
        GemmCoord problemShape; // used for workspace size
        size_t elementSize;
        uint8_t *ptrProblemShape;
        uint8_t *ptrA;
        uint8_t *ptrLayoutA;
        uint8_t *ptrB;
        uint8_t *ptrLayoutB;
        uint8_t *ptrLayoutC;
        uint8_t *ptrD; // also stores X
    };

    static bool CanImplement(const Arguments &args) { return true; }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return static_cast<size_t>(args.elementSize) *
               args.problemShape.m() * args.problemShape.n() * args.problemCount;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.problemCount, args.ptrProblemShape,
            args.ptrA, args.ptrLayoutA,
            args.ptrB, args.ptrLayoutB,
            workspace, args.ptrLayoutC,
            args.ptrD};
        return params;
    }

    CATLASS_DEVICE GroupedMatmulEpilogue() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
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
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

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
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutC layoutCList[MAX_TENSOR_COUNT];

        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutCList, params.ptrLayoutC, params.problemCount);

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
            GemmCoord problemShape = problemShapeList[groupIdx];
            LayoutC layoutC = layoutCList[groupIdx];
            LayoutD layoutD = layoutC; // D shares layout with C

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
