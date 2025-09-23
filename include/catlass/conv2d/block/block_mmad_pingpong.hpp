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
    class FmapL1TileShape_,
    class FilterL1TileShape_,
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
    FmapL1TileShape_,
    FilterL1TileShape_,
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
    using FmapL1TileShape = FmapL1TileShape_;
    using FilterL1TileShape = FilterL1TileShape_;
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
    using LayoutFmapInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutFilterInL0 = typename CopyL1ToL0B::LayoutDst;
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

    // Check LayoutOutput
    static_assert(std::is_same_v<LayoutOutput, layout::Output>, "LayoutOutput only support Output yet!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementFmap);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static constexpr uint32_t ELE_NUM_A_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint32_t ELE_NUM_B_PER_C0 = BYTE_PER_C0 / sizeof(ElementFilter);

    // Check big and small L1TileShape::Cin1 
    static_assert(
        (FilterL1TileShape::Cin1 >= FmapL1TileShape::Cin1) &&
        (FilterL1TileShape::Cin1 % FmapL1TileShape::Cin1 == 0),
        "FilterL1TileShape::Cin1 must be a multiple of FmapL1TileShape::Cin1");

    static constexpr uint32_t K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1;

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, const Conv2dConfigs &configs_, uint32_t l1BufAddrStart = 0)
        : configs(configs_), copyL1ToL0A(configs_), copyL1ToL0B(configs_)
    {
        hiBlock = (FmapL1TileShape::Ho - 1) * configs.strideH() +
            (configs.kh() - 1) * configs.dilationH() + 1;
        wiBlock = (FmapL1TileShape::Wo - 1) * configs.strideW() +
            (configs.kw() - 1) * configs.dilationW() + 1;
        l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        l1B_size = FilterL1TileShape::Cin1 * configs.kh() * configs.kw() * 
            FilterL1TileShape::Cout * BYTE_PER_C0;

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
        Conv2d5HdCoord const &actualShape, uint8_t* blockPadList)
    {
        uint8_t blockPadLeft = blockPadList[0];
        uint8_t blockPadRight = blockPadList[1];
        uint8_t blockPadTop = blockPadList[2];
        uint8_t blockPadBottom = blockPadList[3];
        uint32_t wiActual = actualShape.w();
        uint32_t hiActual = actualShape.h();
        int32_t wiActualOrg = wiActual + blockPadLeft + blockPadRight;
        int32_t hiActualOrg = hiActual + blockPadTop + blockPadBottom;

        uint32_t hoActual = (hiActualOrg - 1 - 
            (configs.kh() - 1) * configs.dilationH()) / configs.strideH() + 1;
        uint32_t woActual = (wiActualOrg - 1 -
            (configs.kw() - 1) * configs.dilationW()) / configs.strideW() + 1;
        uint32_t howoActual = hoActual * woActual;

        uint32_t howoRound = RoundUp<L1FmapAlignHelper::HOWO_ALIGNED>(howoActual);
        uint32_t coutRound = RoundUp<L1FilterAlignHelper::COUT_ALIGNED>(actualShape.cout());

        auto layoutFmapInL1 = LayoutFmapInL1::template MakeLayout<ElementFmap>(
            (uint32_t)1, FmapL1TileShape::Cin1, hiActual, wiActual, ELE_NUM_A_PER_C0);
        auto layoutFilterInL1 = LayoutFilterInL1::template MakeLayout<ElementFilter>(
            FilterL1TileShape::Cin1, configs.kh(), configs.kw(), coutRound, ELE_NUM_B_PER_C0);
        auto layoutInL0C = LayoutOutputInL0::MakeLayoutInL0C(MakeCoord(mL1C, coutRound));
    
        uint32_t cin1Actual = min(actualShape.cin1(), FmapL1TileShape::Cin1);
        uint32_t cin1FilterActual = min(actualShape.cin1(), FilterL1TileShape::Cin1);
       
        // load first Fmap tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        auto layoutTileFmap = layoutFmap.GetTileLayout(
            MakeCoord((uint32_t)1, cin1Actual, hiActual, wiActual, ELE_NUM_A_PER_C0));
        copyGmToL1A(l1ATensorList[l1AListId], gmFmap, layoutFmapInL1, layoutTileFmap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        // load first Filter tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto layoutTileFilter = layoutFilter.GetTileLayout(
            MakeCoord(cin1FilterActual, (uint32_t)configs.kh(), (uint32_t)configs.kw(),
                      actualShape.cout(), ELE_NUM_B_PER_C0));
        copyGmToL1B(l1BTensorList[l1BListId], gmFilter, layoutFilterInL1, layoutTileFilter);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t hoL0Tile = Max(L0TileShape::M / woActual, 1);
        uint32_t mL0A = hoL0Tile * woActual;
        uint32_t mL0C = RoundUp<C0_NUM_PER_FRACTAL>(mL0A);
        uint32_t mPartLoop = CeilDiv(hoActual, hoL0Tile);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(coutRound);
        uint32_t cin1L0Tile = Max(L0TileShape::K / (configs.kh() * configs.kw() * ELE_NUM_A_PER_C0), 1);
        uint32_t mL1C = mL0C * mPartLoop;

        // main loop
        uint32_t cin1TileCnt =
            CeilDiv<FmapL1TileShape::Cin1>(actualShape.cin1());
        uint32_t cin1FilterTileCnt =
            CeilDiv<FilterL1TileShape::Cin1>(actualShape.cin1());
        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCnt; cin1LoopIdx++) {
            uint32_t  cin1FmapIdx = cin1LoopIdx % K_FMAP_PER_FILTER;

            uint32_t l1AListIdNext = (l1AListId + 1 < STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < STAGES) ? (l1BListId + 1) : 0;

            uint32_t cin1ActualNext{0};
            uint32_t cin1FilterActualNext{0};

            // preload next tile A from GM to L1
            if (cin1LoopIdx < cin1TileCnt - 1) {
                uint32_t cin1LoopIdxNext = cin1LoopIdx + 1;
                cin1ActualNext = (cin1LoopIdxNext < cin1TileCnt - 1) ?
                    FmapL1TileShape::Cin1 : (actualShape.cin1() - cin1LoopIdxNext * FmapL1TileShape::Cin1);
                
                // Get L1 tensor A for next stage
                auto l1ATensor = l1ATensorList[l1AListIdNext];
                // Get GM tile A for next stage
                FmapCoord gmTileFmapOffset{0, cin1LoopIdxNext * FmapL1TileShape::Cin1, 0, 0, 0};
                auto gmTileFmap = gmFmap[layoutFmap.GetOffset(gmTileFmapOffset)];

                // load next Fmap tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                layoutTileFmap = layoutFmap.GetTileLayout(
                    MakeCoord((uint32_t)1, cin1ActualNext, hiActual, wiActual, ELE_NUM_A_PER_C0));
                copyGmToL1A(l1ATensor, gmTileFmap, layoutFmapInL1, layoutTileFmap);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                // preload next tile B from GM to L1
                if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                    uint32_t cin1FilterIdxNext = cin1LoopIdxNext / K_FMAP_PER_FILTER;
                    cin1FilterActualNext = (cin1FilterIdxNext < cin1FilterTileCnt - 1) ?
                        FilterL1TileShape::Cin1 : (actualShape.cin1() - cin1FilterIdxNext * FilterL1TileShape::Cin1);
                
                    // Get L1 tensor B for next stage
                    auto l1BTensor = l1BTensorList[l1BListIdNext];
                    // Get GM tile B for next stage
                    FilterCoord gmTileFilterOffset{cin1FilterIdxNext * FilterL1TileShape::Cin1, 0, 0, 0, 0};
                    auto gmTileFilter = gmFilter[layoutFilter.GetOffset(gmTileFilterOffset)];

                    // load next Filter tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                    auto layoutTileFilter = layoutFilter.GetTileLayout(
                        MakeCoord(cin1FilterActualNext, (uint32_t)configs.kh(), (uint32_t)configs.kw(),
                                actualShape.cout(), ELE_NUM_B_PER_C0));
                    copyGmToL1B(l1BTensor, gmTileFilter, layoutFilterInL1, layoutTileFilter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
                }
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv(cin1Actual, cin1L0Tile);

            int32_t hiPartOffsetOrg = -blockPadTop;
            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t hoPartActual = (mPartIdx < mPartLoop - 1) ?
                    hoL0Tile : (hoActual - mPartIdx * hoL0Tile);
                uint32_t mPartActual = hoPartActual * woActual;
                uint32_t mPartRound = RoundUp<C0_NUM_PER_FRACTAL>(mPartActual);

                // compute hiPartActual from hoPartActual
                uint8_t tilePadTop = 0, tilePadBottom = 0;
                uint32_t hiPartActualOrg = (hoPartActual - 1) * configs.strideH() + 
                    (configs.kh() - 1) * configs.dilationH() + 1;
                uint32_t hiPartActual = hiPartActualOrg;
                if (hiPartOffsetOrg < 0) {
                    tilePadTop = -hiPartOffsetOrg;
                    hiPartActual -= tilePadTop;
                }
                if(hiPartOffsetOrg + hiPartActualOrg > hiActual) {
                    tilePadBottom = hiPartOffsetOrg + hiPartActualOrg - hiActual;
                    hiPartActual -= tilePadBottom;
                }
                uint32_t hiPartOffset = (hiPartOffsetOrg < 0) ? 0 : hiPartOffsetOrg;

                uint8_t tilePadList[4] = {blockPadLeft, blockPadRight, tilePadTop, tilePadBottom};

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t cin1PartActual = (kPartIdx < kPartLoop - 1) ?
                        cin1L0Tile : (cin1Actual - kPartIdx * cin1L0Tile);
                    uint32_t kPartActual = 
                        cin1PartActual * configs.kh() * configs.kw() * ELE_NUM_A_PER_C0;

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutFmapInL0 layoutFmapInL0 = LayoutFmapInL0::template MakeLayout<ElementFmap>(mPartRound, kPartActual);
                    // Locate the current tile of matrix A on L1
                    FmapCoord l1AOffset{0, kPartIdx * cin1L0Tile, hiPartOffset, 0, 0};
                    auto l1ATile = l1ATensor[layoutFmapInL1.GetOffset(l1AOffset)];

                    // Wait for mmad finished
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutFmapInL0, layoutFmapInL1,
                        tilePadList, hiPartActual);
                    
                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (coutRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutFilterInL0 layoutFilterInL0 = LayoutFilterInL0::template MakeLayout<ElementFilter>(kPartActual, nPartActual);
                        // Load current tile of matrix B on L1
                        FilterCoord l1BOffset{
                            cin1FmapIdx * FmapL1TileShape::Cin1 + kPartIdx * cin1L0Tile,
                            0, 0, nPartIdx * L0TileShape::N, 0};
                        auto l1BTile = l1BTensor[layoutFilterInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        if ((cin1FmapIdx == 0) &&
                            (mPartIdx == 0) && (kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutFilterInL0, layoutFilterInL1);
                        
                        if (((cin1FmapIdx == K_FMAP_PER_FILTER - 1) || (cin1LoopIdx == cin1TileCnt - 1)) &&
                            (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                        }

                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * mL0C, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = (cin1LoopIdx == 0) && (kPartIdx == 0);
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((cin1LoopIdx == cin1TileCnt - 1) && (kPartIdx == kPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perfrom calculation operations
                        tileMmad(
                            l0CTile, l0ATile, l0BTile,
                            mPartRound, nPartActual, kPartActual,
                            initC, unitFlag);

                        // Notify to move the next L0A, L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
                hiPartOffsetOrg += configs.strideH() * hoPartActual;
            }
            l1AListId = l1AListIdNext;
            cin1Actual = cin1ActualNext;
            if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                l1BListId = l1BListIdNext;
                cin1FilterActual = cin1FilterActualNext;
            }
        }

        // copy block out
        uint32_t cout1Actual = coutRound / ELE_NUM_A_PER_C0;
        LayoutOutput layoutBlock = layoutOutput.GetTileLayout(
            MakeCoord((uint32_t)1, cout1Actual, hoActual, woActual, ELE_NUM_A_PER_C0));

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock, mL0A);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock, mL0A, 0b11);
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
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    __aicore__ inline int32_t Max(int32_t a, int32_t b) {
        if (a > b) {
            return a;
        } else {
            return b;
        }
    }
    __aicore__ inline int32_t Min(int32_t a, int32_t b) {
        if (a < b) {
            return a;
        } else {
            return b;
        }
    }
};

} // namespace Catlass::Conv2d::Block

#endif // CATLASS_CONV2D_BLOCK_BLOCK_MMAD_PINGPONG_HPP
