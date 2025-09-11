/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_BLOCK_BLOCK_MMAD_PINGPONG_HPP
#define CATLASS_CONV2D_BLOCK_BLOCK_MMAD_PINGPONG_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/conv2d_coord.hpp"
#include "catlass/conv2d/dispatch_policy.hpp"
#include "catlass/conv2d/helper.hpp"

namespace Catlass::Conv2d::Block {

template <
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class FmapType_,
    class FilterType_,
    class OutputType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    FmapType_,
    FilterType_,
    OutputType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_; // PostIm2colShape
    using L0TileShape = L0TileShape_;
    using ElementFmap = typename FmapType_::Element;
    using LayoutFmap = typename FmapType_::Layout;
    using ElementFilter = typename FilterType_::Element;
    using LayoutFilter = typename FilterType_::Layout;
    using ElementOutput = typename OutputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Conv2d::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;
    using LayoutFmapInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutFilterInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutOutputInL0 = layout::zN;

    using L1FmapAlignHelper = Conv2d::helper::L1AlignHelper<ElementFmap, LayoutFmap>;
    using L1FilterAlignHelper = Conv2d::helper::L1AlignHelper<ElementFilter, LayoutFilter>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    
    static constexpr uint32_t ELE_NUM_A_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint32_t ELE_NUM_B_PER_C0 = BYTE_PER_C0 / sizeof(ElementFilter);

    // Check LayoutOutput
    static_assert(std::is_same_v<LayoutOutput, layout::Output>, "LayoutOutput only support Output yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, const Conv2dConfigs &configs_, uint32_t l1BufAddrStart = 0)
        : configs(configs_), copyL1ToL0A(configs_), copyL1ToL0B(configs_)
    {
        hiBlock = (L1TileShape::Ho - 1) * configs.strideH() +
            (configs.kh() - 1) * configs.dilationH() + 1;
        wiBlock = (L1TileShape::Wo - 1) * configs.strideW() +
            (configs.kw() - 1) * configs.dilationW() + 1;
        l1A_size = // {cin1, hi, wi, C0}
            L1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        l1B_size = // {cin1, Kh, Kw, cout, C0}
            L1TileShape::Cin1 * configs.kh() * configs.kw() * L1TileShape::Cout * BYTE_PER_C0;

        // Check L1TileShape
        assert((l1A_size * STAGES + l1B_size * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1A_size * STAGES;

        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(l1AOffset + l1A_size * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1BOffset + l1B_size * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementFilter>(L0B_PINGPONG_BUF_SIZE * i);

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
    ~BlockMmad() {
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
        AscendC::GlobalTensor<ElementFmap> const &gmFmap, LayoutFmap const &layoutFmap,
        AscendC::GlobalTensor<ElementFilter> const &gmFilter, LayoutFilter const &layoutFilter,
        AscendC::GlobalTensor<ElementOutput> const &gmOutput, LayoutOutput const &layoutOutput,
        PreIm2colCoord const &actualShape, uint8_t* blockPadList)
    {
        uint8_t blockPadLeft = blockPadList[0];
        uint8_t blockPadRight = blockPadList[1];
        uint8_t blockPadTop = blockPadList[2];
        uint8_t blockPadBottom = blockPadList[3];
        int32_t wiActualOrg = actualShape.wi() + blockPadLeft + blockPadRight;
        int32_t hiActualOrg = actualShape.hi() + blockPadTop + blockPadBottom;\

        uint32_t hoActual = (hiActualOrg - 1 - 
            (configs.kh() - 1) * configs.dilationH()) / configs.strideH() + 1;
        uint32_t woActual = (wiActualOrg - 1 -
            (configs.kw() - 1) * configs.dilationW()) / configs.strideW() + 1;
        uint32_t howoActual = hoActual * woActual;

        uint32_t howoRound = RoundUp<L1FmapAlignHelper::HOWO_ALIGNED>(howoActual);
        uint32_t coutRound = RoundUp<L1FilterAlignHelper::COUT_ALIGNED>(actualShape.cout());

        auto layoutFmapInL1 = LayoutFmapInL1::template MakeLayout<ElementFmap>(
            L1TileShape::Cin1, actualShape.hi(), actualShape.wi(), ELE_NUM_A_PER_C0);
        auto layoutFilterInL1 = LayoutFilterInL1::template MakeLayout<ElementFilter>(
            L1TileShape::Cin1, configs.kh(), configs.kw(), coutRound, ELE_NUM_B_PER_C0);
        auto layoutInL0C = LayoutOutputInL0::MakeLayoutInL0C(MakeCoord(howoRound, coutRound));
    
        uint32_t cin1Actual = min(actualShape.cin1(), L1TileShape::Cin1);

        // load first Fmap tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        auto layoutTileFmap = layoutFmap.GetTileLayout(
            MakeCoord(cin1Actual, actualShape.hi(), actualShape.wi(), ELE_NUM_A_PER_C0));
        copyGmToL1A(l1ATensorList[l1ListId], gmFmap, layoutFmapInL1, layoutTileFmap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        // load first Filter tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileFilter = layoutFilter.GetTileLayout(
            MakeCoord(cin1Actual, (uint32_t)configs.kh(), (uint32_t)configs.kw(),
                    actualShape.cout(), ELE_NUM_B_PER_C0));
        copyGmToL1B(l1BTensorList[l1ListId], gmFilter, layoutFilterInL1, layoutTileFilter);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        // main loop
        uint32_t cin1TileCount = CeilDiv<L1TileShape::Cin1>(actualShape.cin1());
        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCount; cin1LoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t cin1ActualNext{0};
            // preload next tile from GM to L1
            if (cin1LoopIdx < cin1TileCount - 1) {
                uint32_t cin1LoopIdxNext = cin1LoopIdx + 1;
                cin1ActualNext = (cin1LoopIdxNext < cin1TileCount - 1) ?
                    L1TileShape::Cin1 : (actualShape.cin1() - cin1LoopIdxNext * L1TileShape::Cin1);
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                FmapCoord gmTileFmapOffset{cin1LoopIdxNext * L1TileShape::Cin1, 0, 0, 0};
                FilterCoord gmTileFilterOffset{cin1LoopIdxNext * L1TileShape::Cin1, 0, 0, 0, 0};
                auto gmTileFmap = gmFmap[layoutFmap.GetOffset(gmTileFmapOffset)];
                auto gmTileFilter = gmFilter[layoutFilter.GetOffset(gmTileFilterOffset)];

                // load next Fmap tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileFmap = layoutFmap.GetTileLayout(
                    MakeCoord(cin1ActualNext, actualShape.hi(), actualShape.wi(), ELE_NUM_A_PER_C0));
                copyGmToL1A(l1ATensor, gmTileFmap, layoutFmapInL1, layoutTileFmap);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next Filter tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileFilter = layoutFilter.GetTileLayout(
                    MakeCoord(cin1ActualNext, (uint32_t)configs.kh(), (uint32_t)configs.kw(),
                              actualShape.cout(), ELE_NUM_B_PER_C0));
                copyGmToL1B(l1BTensor, gmTileFilter, layoutFilterInL1, layoutTileFilter);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // Get the loop nums on L0
            uint32_t cin1PartLoop = CeilDiv<L0TileShape::Cin1>(cin1Actual);

            for (int cin1L0Idx = 0; cin1L0Idx < cin1PartLoop; cin1L0Idx++) {
                uint32_t cin1L0Actual = (cin1L0Idx < cin1PartLoop - 1) ?
                    L0TileShape::Cin1 : (cin1Actual - cin1L0Idx * L0TileShape::Cin1);
                    
                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0AListId];
                FmapCoord l1AOffset{cin1L0Idx * L0TileShape::Cin1, 0, 0, 0};
                auto l1ATile = l1ATensor[l1AOffset.GetOffset(l1AOffset)];

                // Locate the current tile on L0B
                auto l0BTile = l0BTensorList[l0BListId];
                FilterCoord l1BOffset{cin1L0Idx * L0TileShape::Cin1, 0, 0, 0, 0};
                auto l1BTile = l1BTensor[l1BOffset.GetOffset(l1BOffset)];

                // Wait for mmad finished
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                // Load current tile from L1 to L0A
                copyL1ToL0A(l0ATile, l1ATile, 
                            blockPadList, actualShape.hi(), actualShape.wi(), cin1L0Actual, howoRound);
                // Wait for mmad finished
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                // Load current tile from L1 to L0B
                copyL1ToL0B(l0BTile, l1BTile, cin1L0Actual, coutRound);

                // Notify to do mmad
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                // Wait for loading L0B
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = (cin1LoopIdx == 0) && (cin1L0Idx == 0);

                // If the unit flag is enabled, the unit flag is set according to the calculation progress
                uint8_t unitFlag = 0b00;
                // if constexpr (ENABLE_UNIT_FLAG) {
                //   // AscendC::printf("ENABLE_UNIT_FLAG\n");
                //   if ((cin1LoopIdx == cin1TileCount - 1) && (cin1L0Idx == cin1PartLoop - 1)) {
                //     unitFlag = 0b11;
                //   } else {
                //     unitFlag = 0b10;
                //   }
                // }
                
                // Perfrom calculation operations
                tileMmad(
                    l0CTensor, l0ATile, l0BTile,
                    howoActual, actualShape.cout(),
                    // howoRound, coutRound,
                    cin1L0Actual * configs.kh() * configs.kw() * ELE_NUM_A_PER_C0,
                    initC, unitFlag);
                AscendC::PipeBarrier<PIPE_M>();

                // Notify to move the next L0A, L0B tile
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            l1ListId = l1ListIdNext;
            cin1Actual = cin1ActualNext;
        }

        // copy block out
        uint32_t cout1Actual = coutRound / ELE_NUM_A_PER_C0;
        LayoutOutput layoutBlock = layoutOutput.GetTileLayout(
            MakeCoord(cout1Actual, hoActual, woActual, ELE_NUM_A_PER_C0));

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock, 0b11);
        }
    }

protected:
    Conv2dConfigs configs;
    uint32_t hiBlock, wiBlock;
    uint32_t l1A_size, l1B_size;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementFmap> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementFilter> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementFmap> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementFilter> l0BTensorList[STAGES];
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

} // namespace Catlass::Conv2d::Block

#endif // CATLASS_CONV2D_BLOCK_BLOCK_MMAD_PINGPONG_HPP
