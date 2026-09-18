/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Epilogue::Block {

/**
 * @brief AIV epilogue for SYRK workspace path: D = alpha * P + beta * Y.
 *
 * P is float in GM workspace (RowMajor tile). Y is ElementY, D is ElementD in GM.
 * Processes the tile in UB-sized row chunks.
 */
template <class ArchTag_, class ElementY_, class ElementD_>
class BlockEpilogueSyrkAxpby {
public:
    using ArchTag = ArchTag_;
    using ElementY = ElementY_;
    using ElementD = ElementD_;
    using ElementCompute = float;

    static constexpr uint32_t MAX_COLS = 256;
    static constexpr uint32_t COMPUTE_ROWS = 64;
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementCompute);
    static constexpr uint32_t ALIGN_COLS = RoundUp(MAX_COLS, ELE_NUM_PER_BLK);
    static constexpr uint32_t CHUNK_ELEMS = COMPUTE_ROWS * ALIGN_COLS;

    using LayoutY = detail::TagToLayout_t<ElementY, layout::RowMajor>;
    using LayoutD = detail::TagToLayout_t<ElementD, layout::RowMajor>;
    using LayoutP = detail::TagToLayout_t<ElementCompute, layout::RowMajor>;

    static constexpr uint32_t UB_BYTES_USED = CHUNK_ELEMS * sizeof(ElementCompute) + // ubP
                                              CHUNK_ELEMS * sizeof(ElementY) +       // ubY
                                              CHUNK_ELEMS * sizeof(ElementCompute) + // ubYFp32
                                              CHUNK_ELEMS * sizeof(ElementD);        // ubD
    static_assert(UB_BYTES_USED <= ArchTag::UB_SIZE, "BlockEpilogueSyrkAxpby exceeds UB");

    struct Params {
        ElementCompute alpha{1.0f};
        ElementCompute beta{0.0f};
        GM_ADDR ptrY{nullptr};
        GM_ADDR ptrD{nullptr};
        LayoutY layoutY{};
        LayoutD layoutD{};

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            ElementCompute alpha_, ElementCompute beta_, GM_ADDR ptrY_, GM_ADDR ptrD_, LayoutY layoutY_,
            LayoutD layoutD_)
            : alpha(alpha_), beta(beta_), ptrY(ptrY_), ptrD(ptrD_), layoutY(layoutY_), layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogueSyrkAxpby() = default;

    CATLASS_DEVICE
    BlockEpilogueSyrkAxpby(Arch::Resource<ArchTag>& resource, Params const& params) : params_(params)
    {
        uint32_t offset = 0;
        ubP_ = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += CHUNK_ELEMS * sizeof(ElementCompute);
        ubY_ = resource.ubBuf.template GetBufferByByte<ElementY>(offset);
        offset += CHUNK_ELEMS * sizeof(ElementY);
        ubYFp32_ = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
        offset += CHUNK_ELEMS * sizeof(ElementCompute);
        ubD_ = resource.ubBuf.template GetBufferByByte<ElementD>(offset);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~BlockEpilogueSyrkAxpby()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    /**
     * @param tensorP  RowMajor float tile covering the block to fuse (may use padded stride).
     * @param gmRow/gmCol  Destination window origin in Y/D.
     * @param rows/cols    Actual tile shape.
     */
    template <class TensorP>
    CATLASS_DEVICE void operator()(
        TensorP const& tensorP, uint32_t gmRow, uint32_t gmCol, uint32_t rows, uint32_t cols,
        int64_t batchOffsetElems = 0)
    {
        if (rows == 0 || cols == 0) {
            return;
        }

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params_.ptrY) + batchOffsetElems);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params_.ptrD) + batchOffsetElems);
        auto tensorY = tla::MakeTensor(gmY, params_.layoutY, Arch::PositionGM{});
        auto tensorD = tla::MakeTensor(gmD, params_.layoutD, Arch::PositionGM{});

        uint32_t alignCols = RoundUp(cols, ELE_NUM_PER_BLK);
        uint32_t maxRowsPerChunk = CHUNK_ELEMS / alignCols;
        if (maxRowsPerChunk == 0) {
            maxRowsPerChunk = 1;
        }

        for (uint32_t r = 0; r < rows;) {
            uint32_t chunkRows = (rows - r) < maxRowsPerChunk ? (rows - r) : maxRowsPerChunk;
            uint32_t chunkElems = chunkRows * alignCols;

            // ubY_ is also the MTE3 output buffer. Do not let the next Y load
            // overwrite it before the previous GM store has completed.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            auto tileP = GetTile(tensorP, tla::MakeCoord(r, 0), tla::MakeShape(chunkRows, cols));
            auto layoutPUb = tla::MakeLayout(
                tla::MakeShape(chunkRows, cols), tla::MakeStride(static_cast<int64_t>(alignCols), tla::Int<1>{}));
            auto tensorPUb = tla::MakeTensor(ubP_, layoutPUb, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            using CopyGmToUbP = Tile::CopyGm2UbTla<ArchTag, decltype(tileP), decltype(tensorPUb)>;
            CopyGmToUbP copyGmToUbP;
            copyGmToUbP(tensorPUb, tileP);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Muls(ubP_, ubP_, params_.alpha, chunkElems);
            AscendC::PipeBarrier<PIPE_V>();

            if (params_.beta != static_cast<ElementCompute>(0)) {
                auto tileY = GetTile(tensorY, tla::MakeCoord(gmRow + r, gmCol), tla::MakeShape(chunkRows, cols));
                auto layoutYUb = tla::MakeLayout<ElementY, layout::RowMajor>(chunkRows, alignCols);
                auto tensorYUb = tla::MakeTensor(ubY_, layoutYUb, Arch::PositionUB{});

                using CopyGmToUbY = Tile::CopyGm2UbTla<ArchTag, decltype(tileY), decltype(tensorYUb)>;
                CopyGmToUbY copyGmToUbY;
                copyGmToUbY(tensorYUb, tileY);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                AscendC::Cast(ubYFp32_, ubY_, AscendC::RoundMode::CAST_NONE, chunkElems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(ubYFp32_, ubYFp32_, params_.beta, chunkElems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(ubP_, ubP_, ubYFp32_, chunkElems);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::Cast(ubD_, ubP_, AscendC::RoundMode::CAST_RINT, chunkElems);
            // Cast is the last vector read of ubP_. Release ubP_ to MTE2 only
            // after this instruction, otherwise the next chunk can overwrite
            // it while the current cast is still in flight.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            auto tileD = GetTile(tensorD, tla::MakeCoord(gmRow + r, gmCol), tla::MakeShape(chunkRows, cols));
            auto layoutDUb = tla::MakeLayout<ElementD, layout::RowMajor>(chunkRows, alignCols);
            auto tensorDUb = tla::MakeTensor(ubD_, layoutDUb, Arch::PositionUB{});
            using CopyUbToGmD = Tile::CopyUb2GmTla<ArchTag, decltype(tensorDUb), decltype(tileD)>;
            CopyUbToGmD copyUbToGmD;
            copyUbToGmD(tileD, tensorDUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            r += chunkRows;
        }
    }

private:
    Params params_;
    AscendC::LocalTensor<ElementCompute> ubP_;
    AscendC::LocalTensor<ElementY> ubY_;
    AscendC::LocalTensor<ElementCompute> ubYFp32_;
    AscendC::LocalTensor<ElementD> ubD_;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP
