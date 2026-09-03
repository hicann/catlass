/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef CATLASS_EPPILOGUE_BLOCK_EPILLOGUE_BLOCK_EPILLOGUE_HSTU_SCALE_HPP
#define CATLASS_EPPILOGUE_BLOCK_EPILLOGUE_BLOCK_EPILLOGUE_HSTU_SCALE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

namespace Catlass::Epilogue::Block {

template <class DstType_, class SrcType_, class TileShape_, class TileElemWiseEpilogue_, class TileCopy_>
class BlockEpilogue<EpilogueSilu, DstType_, SrcType_, TileShape_, TileElemWiseEpilogue_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueSilu;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementSrc = typename SrcType_::Element;
    using LayoutTagSrc = typename SrcType_::Layout;

    using ElementDst = typename DstType_::Element;
    using LayoutTagDst = typename DstType_::Layout;
    static constexpr AscendC::TPosition PositionDst = DstType_::POSITION;

    // Check the element type of C and D
    static_assert(std::is_same_v<ElementSrc, float>, "Element type of Src must be float");

    // Check the layout type of C and D
    static_assert(
        std::is_same_v<LayoutTagSrc, layout::RowMajor> &&
            (((PositionDst == AscendC::TPosition::GM) && (std::is_same_v<LayoutTagDst, layout::RowMajor>)) ||
             ((PositionDst == AscendC::TPosition::A1) && (std::is_same_v<LayoutTagDst, layout::zN>))),
        "Layout type of Src must be RowMajor, Dst must be RowMajor or zN");

    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    using TileShape = TileShape_;

    static constexpr uint32_t UB_TILE_M = tla::get<0>(TileShape{});
    static constexpr uint32_t UB_TILE_N = tla::get<1>(TileShape{});

    static constexpr uint32_t AIV_CORE_NUM = DispatchPolicy::AIV_CORE_NUM;
    static constexpr uint32_t UB_SRC_DATA_PING_SIZE = UB_TILE_M * UB_TILE_N * sizeof(ElementSrc) / AIV_CORE_NUM;
    static constexpr uint32_t UB_SRC_BUFF_PING_SIZE = RoundUp(UB_SRC_DATA_PING_SIZE, 1024);
    static constexpr uint32_t UB_SRC_BUFF_SIZE = UB_SRC_BUFF_PING_SIZE * UB_STAGES;

    static constexpr uint32_t UB_DST_DATA_PING_SIZE = UB_TILE_M * UB_TILE_N * sizeof(ElementDst) / AIV_CORE_NUM;
    static constexpr uint32_t UB_DST_BUFF_PING_SIZE = RoundUp(UB_DST_DATA_PING_SIZE, 1024);
    static constexpr uint32_t UB_DST_BUFF_SIZE = UB_DST_BUFF_PING_SIZE * UB_STAGES;

    using TileElemWiseEpilogue = TileElemWiseEpilogue_;

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, uint32_t ubSrcOffset, uint32_t ubSrcPingSize, uint32_t ubDstOffset,
        uint32_t ubDstPingSize)
    {
        for (uint32_t i = 0; i < UB_STAGES; i++) {
            ubOut[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubDstOffset + i * ubDstPingSize);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE void InitEvent(Arch::Resource<ArchTag>& resource)
    {
        uint32_t eventIdOffset = EVENT_ID0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdOffset + i);
        }
    }

    CATLASS_DEVICE void ClearEvent()
    {
        uint32_t eventIdOffset = EVENT_ID0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdOffset + i);
        }
    }

    CATLASS_DEVICE void GetSubBlockShape(MatrixCoord actualShape)
    {
        MatrixCoord subblockShape{
            CeilDiv(actualShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())), actualShape.column()};

        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(subblockShape, actualShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& tensorl1Dst, TensorSrc const& tensorUbSrc, MatrixCoord actualShape, float scale,
        uint32_t ubStageId)
    {
        using ElementUbDst = typename TensorDst::Element;
        using LayoutUbDst = typename TensorDst::Layout;
        using TensorUbDst = tla::Tensor<
            AscendC::LocalTensor<ElementUbDst>, LayoutUbDst, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
        using CopyUbToDst = typename TileCopy_::template CopyUbToDst<TensorDst, TensorUbDst>;
        CopyUbToDst copyUbToDst;

        uint32_t blockRowNum = actualShape.row();
        uint32_t rowBlockTile = CeilDiv(blockRowNum, 1);

        uint32_t subRowTile = CeilDiv(rowBlockTile, static_cast<uint32_t>(AscendC::GetSubBlockNum())) * 1;
        uint32_t actualSubRowOffset = subRowTile * AscendC::GetSubBlockIdx();
        if (blockRowNum <= actualSubRowOffset) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
            return;
        }
        uint32_t actualSubRowNum = min(subRowTile, blockRowNum - actualSubRowOffset);
        MatrixCoord subBlockShape{actualSubRowNum, actualShape.column()};

        auto tensorSubUbSrc =
            GetTile(tensorUbSrc, tla::MakeCoord(0, 0), tla::MakeShape(actualSubRowNum, actualShape.column()));

        auto layoutSubUbDst = tla::MakeLayout<ElementDst, LayoutTagDst>(actualSubRowNum, actualShape.column());
        auto tensorSubUbDst = tla::MakeTensor(ubOut[ubStageId], layoutSubUbDst, Arch::PositionUB{});

        // Perform epilogue calculation
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
        tileEpilogue(tensorSubUbDst, tensorSubUbSrc, subBlockShape, scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);

        // // Copy the data of D
        auto tensorSubL1Dst = GetTile(
            tensorl1Dst, tla::MakeCoord(actualSubRowOffset, 0), tla::MakeShape(actualSubRowNum, actualShape.column()));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
        copyUbToDst(tensorSubL1Dst, tensorSubUbDst);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& tensorl1Dst, TensorSrc const& tensorUbSrc, MatrixCoord actualShape, float scale,
        uint32_t ubStageId, MatrixCoord topLeftDotCoord)
    {
        using ElementUbDst = typename TensorDst::Element;
        using LayoutUbDst = typename TensorDst::Layout;
        using TensorUbDst = tla::Tensor<
            AscendC::LocalTensor<ElementUbDst>, LayoutUbDst, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
        using CopyUbToDst = typename TileCopy_::template CopyUbToDst<TensorDst, TensorUbDst>;
        CopyUbToDst copyUbToDst;

        uint32_t blockRowNum = actualShape.row();
        uint32_t rowBlockTile = CeilDiv(blockRowNum, 1);

        uint32_t subRowTile = CeilDiv(rowBlockTile, static_cast<uint32_t>(AscendC::GetSubBlockNum())) * 1;
        uint32_t actualSubRowOffset = subRowTile * AscendC::GetSubBlockIdx();
        if (blockRowNum <= actualSubRowOffset) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
            return;
        }
        uint32_t actualSubRowNum = min(subRowTile, blockRowNum - actualSubRowOffset);
        MatrixCoord subBlockShape{actualSubRowNum, actualShape.column()};

        MatrixCoord subTopLeftDotCoord{topLeftDotCoord.row() + actualSubRowOffset, topLeftDotCoord.column()};

        auto tensorSubUbSrc =
            GetTile(tensorUbSrc, tla::MakeCoord(0, 0), tla::MakeShape(actualSubRowNum, actualShape.column()));

        auto layoutSubUbDst = tla::MakeLayout<ElementDst, LayoutTagDst>(actualSubRowNum, actualShape.column());
        auto tensorSubUbDst = tla::MakeTensor(ubOut[ubStageId], layoutSubUbDst, Arch::PositionUB{});

        // Perform epilogue calculation
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
        tileEpilogue(tensorSubUbDst, tensorSubUbSrc, subBlockShape, scale, subTopLeftDotCoord);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);

        // // Copy the data of D
        auto tensorSubL1Dst = GetTile(
            tensorl1Dst, tla::MakeCoord(actualSubRowOffset, 0), tla::MakeShape(actualSubRowNum, actualShape.column()));

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
        copyUbToDst(tensorSubL1Dst, tensorSubUbDst);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
    }

private:
    AscendC::LocalTensor<ElementDst> ubOut[UB_STAGES];

    TileElemWiseEpilogue tileEpilogue;
};

} // namespace Catlass::Epilogue::Block

#endif
