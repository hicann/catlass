/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Gemm::Block {

template <
    bool ENABLE_UNIT_FLAG_,
    bool IS_DYNAMIC_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_, IS_DYNAMIC_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_, IS_DYNAMIC_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
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
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
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
    static constexpr bool IS_DYNAMIC = DispatchPolicy::IS_DYNAMIC;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    template <bool IS_DYNAMIC, class L1TileShape, class L0TileShape>
    struct StaticTileConstraintVerifier {
        static constexpr bool VALID = true;
    };

    template <class L1TileShape, class L0TileShape>
    struct StaticTileConstraintVerifier<false, L1TileShape, L0TileShape> {
        static constexpr uint32_t L1A_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
        static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);

        // Check L1TileShape
        static_assert((L1A_TILE_SIZE * STAGES + L1B_TILE_SIZE * STAGES) <= ArchTag::L1_SIZE,
            "L1TileShape exceeding the L1 space!");

        // Check L0TileShape
        static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
        static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
        static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
        static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

        static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
            "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

        static constexpr bool VALID = true;
    };

    static_assert(StaticTileConstraintVerifier<IS_DYNAMIC, L1TileShape, L0TileShape>::VALID, "Unvalid tile shape.");

    template <bool IS_DYNAMIC>
    struct ParamsBase {
        CATLASS_HOST_DEVICE
        ParamsBase() = default;

        CATLASS_HOST_DEVICE
        GemmCoord GetL1TileShape() const
        {
            return L1TileShape::ToCoord();
        }

        CATLASS_HOST_DEVICE
        GemmCoord GetL0TileShape() const
        {
            return L0TileShape::ToCoord();
        }

        CATLASS_HOST_DEVICE constexpr
        size_t GetL1ASize() const
        {
            return L1TileShape::M * L1TileShape::K * sizeof(ElementA);
        }

        CATLASS_HOST_DEVICE constexpr
        size_t GetL1BSize() const
        {
            return L1TileShape::K * L1TileShape::N * sizeof(ElementB);
        }
    };

    template <>
    struct ParamsBase<true> {
        GemmCoord l1TileShape;
        GemmCoord l0TileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() = default;

        CATLASS_HOST_DEVICE
        ParamsBase(GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_) :
            l1TileShape(l1TileShape_), l0TileShape(l0TileShape_) {}

        CATLASS_HOST_DEVICE
        GemmCoord GetL1TileShape() const
        {
            return l1TileShape;
        }

        CATLASS_HOST_DEVICE
        GemmCoord GetL0TileShape() const
        {
            return l0TileShape;
        }

        CATLASS_HOST_DEVICE
        size_t GetL1ASize() const
        {
            return l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        }

        CATLASS_HOST_DEVICE
        size_t GetL1BSize() const
        {
            return l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);
        }
    };

    using Params = ParamsBase<IS_DYNAMIC>;

    template <bool IS_DYNAMIC>
    struct ArgumentsBase {
    };

    template <>
    struct ArgumentsBase<true> {
        GemmCoord l1TileShape;
        GemmCoord l0TileShape;
    };

    using Arguments = ArgumentsBase<IS_DYNAMIC>;

    static bool CanImplement(Arguments const &args)
    {
        if constexpr (IS_DYNAMIC) {
            auto &l1TileShape = args.l1TileShape;
            size_t l1ATileSize = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
            size_t l1BTileSize = l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);
            if (l1ATileSize * STAGES + l1BTileSize * STAGES > ArchTag::L1_SIZE) {
                return false;
            }

            auto &l0TileShape = args.l0TileShape;
            size_t l0ATileSize = l0TileShape.m() * l0TileShape.k() * sizeof(ElementA);
            size_t l0BTileSize = l0TileShape.k() * l0TileShape.n() * sizeof(ElementB);
            if (l0ATileSize * STAGES > ArchTag::L0A_SIZE || l0BTileSize * STAGES > ArchTag::L0B_SIZE) {
                return false;
            }

            if (l1TileShape.m() != l0TileShape.m() || l1TileShape.n() != l0TileShape.n()) {
                return false;
            }
        }
        return true;
    }

    static Params ToUnderlyingArguments(Arguments const &args, uint8_t *workspace)
    {
        (void)workspace;
        Params params;
        if constexpr (IS_DYNAMIC) {
            params.l1TileShape = args.l1TileShape;
            params.l0TileShape = args.l0TileShape;
        }
        return params;
    }

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0, Params const &params_ = {}) :
        params(params_)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + params.GetL1ASize() * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + params.GetL1ASize() * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + params.GetL1BSize() * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
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
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        GemmCoord const &actualShape)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto l1TileShape = params.GetL1TileShape();
        auto l0TileShape = params.GetL0TileShape();

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1TileShape.m(), l1TileShape.k());
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1TileShape.k(), l1TileShape.n());
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kActual = min(actualShape.k(), l1TileShape.k());

        // load first matrix A tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
        copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
        copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t mPartLoop = CeilDiv(mRound, l0TileShape.m());
        uint32_t nPartLoop = CeilDiv(nRound, l0TileShape.n());

        // main loop
        uint32_t kTileCount = CeilDiv(actualShape.k(), l1TileShape.k());
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1) {
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    l1TileShape.k() : (actualShape.k() - kLoopIdxNext * l1TileShape.k());

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                MatrixCoord gmTileAOffset{0, kLoopIdxNext * l1TileShape.k()};
                MatrixCoord gmTileBOffset{kLoopIdxNext * l1TileShape.k(), 0};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv(kActual, l0TileShape.k());

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    l1TileShape.m() : (mRound - mPartIdx * l1TileShape.m());

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        l0TileShape.k() : (kActual - kPartIdx * l0TileShape.k());

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * l1TileShape.m(), kPartIdx * l0TileShape.k()};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            l0TileShape.n() : (nRound - nPartIdx * l0TileShape.n());

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * l0TileShape.k(), nPartIdx * l0TileShape.n()};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * l1TileShape.m(), nPartIdx * l0TileShape.n()};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    Params params;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP
