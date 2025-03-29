/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMV_BLOCK_BLOCK_SWIZZLE_HPP
#define ACT_GEMV_BLOCK_BLOCK_SWIZZLE_HPP

#include "act/act.hpp"
#include "act/detail/alignment.hpp"
#include "act/gemv_coord.hpp"
#include "act/matrix_coord.hpp"

namespace Act::Gemv::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Gemvs
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemvIdentityBlockSwizzle {
    /// Data members

    GemvCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;
    uint32_t splitKSlices = 1;  // splite k dim into virtual cores

    /// Methods

    ACT_DEVICE
    GemvIdentityBlockSwizzle() {}

    ACT_DEVICE
    GemvIdentityBlockSwizzle(GemvCoord const& problemShape_, MatrixCoord const& tileMN_, uint32_t splitKSlices_ = 1)
        : problemShape(problemShape_), tileMN(tileMN_), splitKSlices(splitKSlices_) {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    ACT_DEVICE
    GemvIdentityBlockSwizzle(GemvCoord const& problemShape_, MatrixCoord const& tileMN_, MatrixCoord const& loopsMN_, uint32_t splitKSlices_ = 1)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_), splitKSlices(splitKSlices_) {
    }

    ACT_DEVICE
    void
    Update(GemvCoord const& problemShape_, MatrixCoord const& tileMN_, uint32_t splitKSlices_ = 1) 
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        splitKSlices = splitKSlices_;

        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    ACT_DEVICE
    void Update(GemvCoord const& problemShape_, MatrixCoord const& tileMN_, MatrixCoord const& loopsMN_, uint32_t splitKSlices_ = 1) 
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
        splitKSlices = splitKSlices_;
    }

    ACT_DEVICE
    uint32_t GetCoreLoops() const 
    {
        return loopsMN.row() * loopsMN.column();  //
    }

    ACT_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) 
    {
        return taskIdx / (GetCoreLoops() * splitKSlices);
    }

    ACT_DEVICE
    GemvCoord GetBlockCoord(uint32_t taskIdx) 
    {
        uint32_t kIdx = taskIdx % (GetCoreLoops() * splitKSlices) / GetCoreLoops();
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) {  // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }
            return GemvCoord{mIdx, nIdx};
        } else if constexpr (SwizzleDirection == 1) {  // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemvCoord{mIdx, nIdx};
        }
    }

    ACT_DEVICE
    GemvCoord GetActualBlockShape(GemvCoord blockCoord) 
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ? (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ? (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        return GemvCoord{mActual, nActual};
    }
};

}  // namespace Act::Gemv::Block

#endif  // ACT_GEMV_BLOCK_BLOCK_SWIZZLE_HPP