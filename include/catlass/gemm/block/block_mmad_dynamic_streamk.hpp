/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_COMMON_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_COMMON_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_, class L1TileShape_, class L0TileShape_, class AType_,
    class BType_, class CType_, class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA2DynamicStreamk<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>, L1TileShape_, L0TileShape_, AType_,
    BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2DynamicStreamk<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGmNormalBlock = typename TileCopy_::CopyL0CToGm;
    using CopyL0CToStreamkBlock = Gemm::Tile::CopyL0CT0Gm<ArchTag, float, Gemm::GemmType<float, LayoutC>>;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(GemmCoord const &l1TileShape_, Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
        : l1TileShape(l1TileShape_)
    {
        uint32_t l1ASize = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        uint32_t l1BSize = l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);

        kPartLenMax = min(L0A_PINGPONG_BUF_SIZE / sizeof(ElementA) / l1TileShape.m() / L1AAlignHelper::ELE_NUM_PER_C0 *
                              L1AAlignHelper::ELE_NUM_PER_C0,
            L0B_PINGPONG_BUF_SIZE / sizeof(ElementB) / l1TileShape.n() / L1BAlignHelper::ELE_NUM_PER_C0 *
                L1BAlignHelper::ELE_NUM_PER_C0);

        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>) {
            kPartLenMax = RoundDown<C0_NUM_PER_FRACTAL>(kPartLenMax);
        }

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1ASize * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + l1ASize * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + l1BSize * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        AscendC::GlobalTensor<float> const &gmW, LayoutC const &layoutW,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA, AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, bool isSkBlock)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1TileShape.m(), l1TileShape.k());
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1TileShape.k(), l1TileShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kTileCount = CeilDiv(actualShape.k(), l1TileShape.k());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? l1TileShape.k() : (actualShape.k() - firstTileIdx * l1TileShape.k());

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            // Load first matrix A tile in total kernel loop from GM to L1
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileACoord{0, shuffleKIdx * l1TileShape.k()};
                MatrixCoord gmTileBCoord{shuffleKIdx * l1TileShape.k(), 0};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileACoord)];
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBCoord)];
                // Load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
                copyGmToL1A(l1ATensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // Load first matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
                copyGmToL1B(l1BTensorList[l1ListId], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ? l1TileShape.k()
                                                                 : (actualShape.k() - shuffleKIdxNext * l1TileShape.k());
                MatrixCoord gmTileACoord{0, shuffleKIdxNext * l1TileShape.k()};
                MatrixCoord gmTileBCoord{shuffleKIdxNext * l1TileShape.k(), 0};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileACoord)];
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBCoord)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                uint32_t kTileCount = CeilDiv(actualShapeNext.k(), l1TileShape.k());
                uint32_t firstTileIdx = startTileIdx % kTileCount;
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (firstTileIdx < kTileCount - 1)
                                  ? l1TileShape.k()
                                  : (actualShapeNext.k() - firstTileIdx * l1TileShape.k());
                MatrixCoord gmTileACoord{0, firstTileIdx * l1TileShape.k()};
                MatrixCoord gmTileBCoord{firstTileIdx * l1TileShape.k(), 0};
                auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileACoord)];
                auto gmTileB = gmNextBlockB[layoutB.GetOffset(gmTileBCoord)];
                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShapeNext.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv(kActual, kPartLenMax);

            uint32_t l0ABufId = 0;
            uint32_t l0BBufId = 0;

            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                uint32_t kPartActual =
                    (kPartIdx < kPartLoop - 1) ? kPartLenMax : (kActual - kPartIdx * kPartLenMax);

                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0ABufId];
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kPartActual);
                // Locate the current tile of matrix A on L1
                MatrixCoord l1ACoord{0, kPartIdx * kPartLenMax};
                auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ACoord)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                }

                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                }

                // Locate the current tile on L0B
                auto l0BTile = l0BTensorList[l0BBufId];
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nRound);
                // Locate the current tile of matrix B on L1
                MatrixCoord l1BCoord{kPartIdx * kPartLenMax, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BCoord)];

                // Wait for mmad finished
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                }

                // Load current tile from L1 to L0B
                copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                }
                // Notify to do mmad
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                // Locate the current tile on L0C
                MatrixCoord l0CCoord{static_cast<uint32_t>(0U), static_cast<uint32_t>(0U)};
                auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0CCoord)];

                // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                // Wait for loading L0B
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                // If the unit flag is enabled, the unit flag is set according to the calculation progress
                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    if ((kLoopIdx == kTileCount - 1) && (kPartIdx == kPartLoop - 1)) {
                        unitFlag = 0b11;
                    } else {
                        unitFlag = 0b10;
                    }
                }
                // Perform calculation operations
                tileMmad(l0CTile, l0ATile, l0BTile, mRound, nRound, kPartActual, initC, unitFlag);

                // Notify to move the next L0B tile
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);

                l0BBufId = (l0BBufId + 1 < STAGES) ? (l0BBufId + 1) : 0;
                
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                l0ABufId = (l0ABufId + 1 < STAGES) ? (l0ABufId + 1) : 0;
            }

            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            if (isSkBlock) {
                copyL0CToGmStreamkBlock(gmW, l0CTensor, layoutW, layoutInL0C);
            } else {
                copyL0CToGmNormalBlock(gmBlockC, l0CTensor, layoutBlock, layoutInL0C);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            if (isSkBlock) {
                copyL0CToGmStreamkBlock(gmW, l0CTensor, layoutW, layoutInL0C, 0b11);
            } else {
                copyL0CToGmNormalBlock(gmBlockC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    uint32_t l1ListId{0};
    GemmCoord l1TileShape;
    uint32_t kPartLenMax;

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_MMAD_DYNAMIC_COMMON_HPP
