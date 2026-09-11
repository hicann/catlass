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
#ifndef CATLASS_EPILOGUE_BLOCK_EPPILOGUE_ELEMWISE_NO_SOURCE_FROM_UB_HPP
#define CATLASS_EPILOGUE_BLOCK_EPPILOGUE_ELEMWISE_NO_SOURCE_FROM_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"

namespace Catlass::Epilogue::Block {
template <class CType_, class DType_, class TileElemWiseEpilogue_, class TileCopy_, class SrcUbShape_>
class BlockEpilogue<EpilogueElemWiseNoSourceFromUB, CType_, DType_, TileElemWiseEpilogue_, TileCopy_, SrcUbShape_> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueElemWiseNoSourceFromUB;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename CType_::Element;
    using LayoutTagSrc = typename CType_::Layout;

    using ElementDst = typename DType_::Element;
    using LayoutTagDst = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;

    // Check the element type of Src and Dst
    static_assert(std::is_same_v<ElementSrc, float>, "Element type of Src must be float");
    // Check the layout type of Src and Dst
    static_assert(
        std::is_same_v<LayoutTagSrc, layout::RowMajor> && std::is_same_v<LayoutTagDst, layout::RowMajor>,
        "Layout type of Src, Dst must be RowMajor");

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");

    // Epilogue params definition
    struct Params {};

    static constexpr uint32_t UB_TILE_ROW = tla::get<0>(SrcUbShape_{});
    static constexpr uint32_t UB_TILE_COL = tla::get<1>(SrcUbShape_{});

    // Fixpipe SPLIT_M writes one half of the MM output to each vector core. The
    // epilogue converts that float input to ElementDst in place, so every
    // epilogue stage must start at the corresponding MM-output stage boundary.
    static constexpr uint32_t SRC_UB_SIZE_PING =
        RoundUp<256>(CeilDiv<2>(UB_TILE_ROW) * UB_TILE_COL * sizeof(ElementSrc));
    static constexpr uint32_t SRC_UB_SIZE = SRC_UB_SIZE_PING * UB_STAGES;
    static constexpr uint32_t GELU_OUT_SIZE_PING =
        RoundUp<256>(CeilDiv<2>(UB_TILE_ROW) * UB_TILE_COL * sizeof(ElementDst));
    static constexpr uint32_t GELU_OUT_SIZE = GELU_OUT_SIZE_PING * UB_STAGES;

    static_assert(GELU_OUT_SIZE_PING <= SRC_UB_SIZE_PING, "In-place epilogue output must fit in its source stage");
    static_assert(SRC_UB_SIZE <= ArchTag::UB_SIZE, "Epilogue source stages exceed the UB capacity");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            // Output reuses the input buffer of the same ping-pong stage.
            ubTemp[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(i * SRC_UB_SIZE_PING);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + i);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + i);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const& gmBlockDst, LayoutTagDst const& layoutTagBlockDst,
        AscendC::LocalTensor<ElementSrc> const& ubBlockSrc, LayoutTagSrc const& layoutTagBlockSrc,
        GemmCoord const& actualBlockShapeMNK, uint32_t ubStageId)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        // Calculate the offset and the shape of the current subblock
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()};

        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape =
            MatrixCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of C
        auto layoutTagSubblockSrc = layoutTagBlockSrc.GetTileLayout(actualSubblockShape);

        // Get the layout on UB
        auto roundColumnNum = RoundUp<16>(actualSubblockShape.column());
        LayoutTagSrc layoutTagComputeOut{actualSubblockShape.row(), roundColumnNum};

        // Perform epilogue calculation
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
        tileEpilogue(ubTemp[ubStageId], ubBlockSrc, layoutTagComputeOut, layoutTagSubblockSrc, actualSubblockShape);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);

        // Get the data and layout of D
        auto gmSubblockDst = gmBlockDst[layoutTagBlockDst.GetOffset(subblockOffset)];
        auto layoutTagSubblockDst = layoutTagBlockDst.GetTileLayout(actualSubblockShape);

        // Copy the data of D
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + ubStageId);
        copyUbToGmD(gmSubblockDst, ubTemp[ubStageId], layoutTagSubblockDst, layoutTagComputeOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + ubStageId);
    }

private:
    Params params;

    AscendC::LocalTensor<ElementDst> ubTemp[UB_STAGES];

    TileElemWiseEpilogue tileEpilogue;
    CopyUbToGmD copyUbToGmD;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPPILOGUE_ELEMWISE_NO_SOURCE_FROM_UB_HPP
