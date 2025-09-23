/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <class ArchTag_, class BlockScheduler_, class ElementAccumulator_, class ElementOut_, uint32_t COMPUTE_LENGTH>
struct StreamkReduceAdd {
    using ArchTag = ArchTag_;
    using BlockScheduler = BlockScheduler_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOut = ElementOut_;
    using LocalLayout = layout::RowMajor;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<ElementAccumulator, LocalLayout>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementOut, LocalLayout>>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    constexpr static uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementOut);

    CATLASS_DEVICE
    StreamkReduceAdd(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            accumulatorBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementOut);
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementOut> const &dst, AscendC::GlobalTensor<ElementAccumulator> const &src,
        LocalLayout const &layoutDst, GemmCoord const &problemShape, GemmCoord const &l1TileShape)
    {
        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t aivNum = blockDim * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        BlockScheduler matmulBlockScheduler(problemShape, l1TileShape, blockDim);
        // The number of tail blocks using the streamk algorithm
        uint32_t streamkBlockNum = matmulBlockScheduler.GetStreamkBlockNum();
        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);

        for (uint32_t skBlockId = 0; skBlockId < streamkBlockNum; ++skBlockId) {
            // If the head part of the current block and tail part of previous block are computed by the same core,
            // this flag is set to true.
            bool isHeadCross = matmulBlockScheduler.IsCross(skBlockId);
            bool isTailCross = matmulBlockScheduler.IsCross(skBlockId + 1);
            // Get the ID of first core for the current skBlock computation.
            uint32_t startCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId);
            // Get the ID of first core for the next skBlock computation.
            uint32_t endCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId);
            if (isTailCross) {
                endCoreIdx = endCoreIdx + 1;
            }

            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(normalBlockNum + skBlockId);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t tileNum = actualBlockShape.m();
            uint32_t tileLen = actualBlockShape.n();
            uint32_t tilePerCore = CeilDiv(tileNum, aivNum);
            uint32_t loops = CeilDiv(tileNum, tilePerCore);
            for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
                uint32_t tilesActual = tilePerCore;
                if (loopIdx == loops - 1) {
                    tilesActual = tileNum - loopIdx * tilePerCore;
                }
                LocalLayout gmLayoutSrc{tilesActual, tileLen, l1TileShape.n()};
                LocalLayout ubLayoutDst{tilesActual, tileLen, RoundUp(tileLen, ELE_NUM_PER_C0)};

                int64_t srcTileOffset = loopIdx * tilePerCore * l1TileShape.n();
                int64_t skBlockOffset = startCoreIdx * 2 * l1TileShape.m() * l1TileShape.n();
                if (isHeadCross) {
                    skBlockOffset = (startCoreIdx * 2 + 1) * l1TileShape.m() * l1TileShape.n();
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
                copyGm2Ub(accumulatorBuffer[bufferIndex], src[skBlockOffset + srcTileOffset], ubLayoutDst, gmLayoutSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

                for (uint32_t coreIdx = startCoreIdx + 1; coreIdx < endCoreIdx; ++coreIdx) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                    skBlockOffset = coreIdx * 2 * l1TileShape.m() * l1TileShape.n();
                    copyGm2Ub(inputBuffer[bufferIndex], src[srcTileOffset + skBlockOffset], ubLayoutDst, gmLayoutSrc);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                    AscendC::Add(accumulatorBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex],
                        inputBuffer[bufferIndex],
                        tilesActual * RoundUp(tileLen, ELE_NUM_PER_C0));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
                if constexpr (!std::is_same_v<ElementAccumulator, ElementOut>) {
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex],
                            AscendC::RoundMode::CAST_NONE,
                            tilesActual * RoundUp(tileLen, ELE_NUM_PER_C0));
                    } else {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex],
                            AscendC::RoundMode::CAST_RINT,
                            tilesActual * RoundUp(tileLen, ELE_NUM_PER_C0));
                    }
                } else {
                    AscendC::DataCopy(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex],
                        tilesActual * RoundUp(tileLen, ELE_NUM_PER_C0));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                Catlass::MatrixCoord offsetC{
                    blockCoord.m() * l1TileShape.m() + loopIdx * tilePerCore, blockCoord.n() * l1TileShape.n()};
                int64_t dstGmOffset = layoutDst.GetOffset(offsetC);
                LocalLayout ubLayoutSrc{tilesActual, tileLen, RoundUp(tileLen, ELE_NUM_PER_C0)};
                LocalLayout gmLayoutDst{tilesActual, tileLen, layoutDst.stride(0)};
                copyUb2Gm(dst[dstGmOffset], outputBuffer[bufferIndex], gmLayoutDst, ubLayoutSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementAccumulator> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];
    AscendC::TEventID inputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID accumulatorEventIds[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    AscendC::TEventID outputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{0};
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementAccumulator) * 2 +
                          BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementOut) <=
                      ArchTag::UB_SIZE,
        "Excedding the UB space!");
};

template <class PrologueA_, class PrologueB_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_,
    class ReduceAdd_>
class DynamicPaddingStreamkMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using ReduceAdd = ReduceAdd_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutC = typename BlockMmad::LayoutC;

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, typename BlockMmad::LayoutA,
        typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, typename BlockMmad::LayoutB,
        typename LayoutHelper<PrologueB>::type>;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrReduceW;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GM_ADDR ptrA_, LayoutA &layoutA_,
            GM_ADDR ptrB_, LayoutB &layoutB_, GM_ADDR ptrC_, LayoutC &layoutC_, GM_ADDR ptrWA_, GM_ADDR ptrWB_,
            GM_ADDR ptrReduceW_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_),
              layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), ptrWB(ptrWB_), ptrReduceW(ptrReduceW_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingStreamkMatmul()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA =
                    PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB =
                    PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmReduceW;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut *>(params.ptrC));
        gmReduceW.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrReduceW));
        ReduceAdd reduceAdd(resource);
        reduceAdd(gmC, gmReduceW, params.layoutC, params.problemShape, params.l1TileShape);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape, blockDim);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementAccumulator> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrReduceW);

        BlockMmad blockMmad(params.l1TileShape, resource);

        typename BlockScheduler::StreamkBlockDec streamkBlockDec;
        typename BlockScheduler::StreamkBlockDec nextStreamkBlockDec;
        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += blockDim) {

            bool isFirstBlock = (loopIdx == blockIdx);
            if (isFirstBlock) {
                matmulBlockScheduler.GetStreamkBlockDec(loopIdx, streamkBlockDec);
            } else {
                streamkBlockDec = nextStreamkBlockDec;
            }

            bool hasNextBlock = false;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                matmulBlockScheduler.GetStreamkBlockDec(loopIdx + blockDim, nextStreamkBlockDec);
            }

            // Compute initial location in logical coordinates
            MatrixCoord coordA{streamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                streamkBlockDec.blockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordB{streamkBlockDec.blockCoord.k() * params.l1TileShape.k(),
                streamkBlockDec.blockCoord.n() * params.l1TileShape.n()};
            MatrixCoord coordC{streamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                streamkBlockDec.blockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC);
            MatrixCoord coordNextA{nextStreamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                nextStreamkBlockDec.blockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordNextB{nextStreamkBlockDec.blockCoord.k() * params.l1TileShape.k(),
                nextStreamkBlockDec.blockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            int64_t gmOffsetW = params.l1TileShape.m() * params.l1TileShape.n() * 2 * blockIdx;
            LayoutC layoutW = LayoutC{
                streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.n(), params.l1TileShape.n()};

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                layoutA,
                gmB[gmOffsetB],
                layoutB,
                gmC[gmOffsetC],
                params.layoutC,
                gmW[gmOffsetW],
                layoutW,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                streamkBlockDec.actualBlockShape,
                nextStreamkBlockDec.actualBlockShape,
                isFirstBlock,
                hasNextBlock,
                streamkBlockDec.isStreamkBlock);
            // If the current block is a cross block-meaning it consists partly of the tail k portion of
            // current block and partly of head k portion of the next block-then a second blockmmad
            // computation must be invoked to process the head k segment of the next block.
            if (streamkBlockDec.isCrossBlock) {
                MatrixCoord coordA{streamkBlockDec.streamkBlockCoord.m() * params.l1TileShape.m(),
                    streamkBlockDec.streamkBlockCoord.k() * params.l1TileShape.k()};
                MatrixCoord coordB{streamkBlockDec.streamkBlockCoord.k() * params.l1TileShape.k(),
                    streamkBlockDec.streamkBlockCoord.n() * params.l1TileShape.n()};
                int64_t gmOffsetA = layoutA.GetOffset(coordA);
                int64_t gmOffsetB = layoutB.GetOffset(coordB);
                int64_t gmOffsetW = params.l1TileShape.m() * params.l1TileShape.n() * (2 * blockIdx + 1);
                LayoutC layoutW = LayoutC{streamkBlockDec.streamkActualBlockShape.m(),
                    streamkBlockDec.streamkActualBlockShape.n(), params.l1TileShape.n()};
                blockMmad(gmA[gmOffsetA],
                    layoutA,
                    gmB[gmOffsetB],
                    layoutB,
                    gmC[gmOffsetC],
                    params.layoutC,
                    gmW[gmOffsetW],
                    layoutW,
                    gmA[gmOffsetNextA],
                    gmB[gmOffsetNextB],
                    streamkBlockDec.streamkActualBlockShape,
                    nextStreamkBlockDec.actualBlockShape,
                    true,
                    false,
                    streamkBlockDec.isStreamkBlock);
            }

        }
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP