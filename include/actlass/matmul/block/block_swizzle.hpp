/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP
#define ASCENDCT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/detail/alignment.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"

namespace AscendCT::gemm::block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Matmuls
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct MatmulIdentityBlockSwizzle {
    /// Data members

    MatmulCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    /// Methods

    ASCENDCT_DEVICE
    MatmulIdentityBlockSwizzle() {}

    ASCENDCT_DEVICE
    MatmulIdentityBlockSwizzle(MatmulCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    ASCENDCT_DEVICE
    MatmulIdentityBlockSwizzle(MatmulCoord const &problemShape_, MatrixCoord const &tileMN_,
        MatrixCoord const &loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_) {}

    ASCENDCT_DEVICE
    void Update(MatmulCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;

        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    ASCENDCT_DEVICE
    void Update(MatmulCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
    }

    ASCENDCT_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ASCENDCT_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / (GetCoreLoops());
    }

    ASCENDCT_DEVICE
    MatmulCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
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
            return MatmulCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
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
            return MatmulCoord{mIdx, nIdx, 0};
        }
    }

    ASCENDCT_DEVICE
    MatmulCoord GetActualBlockShape(MatmulCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        uint32_t kActual = problemShape.k();
        return MatmulCoord{mActual, nActual, kActual};
    }
};

/// Block swizzling function for Splitk Matmuls
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkMatmulIdentityBlockSwizzle {
    /// Data members

    MatmulCoord problemShape;
    MatmulCoord tileShape;
    MatmulCoord loopsMNK;
    uint32_t splitkFactor = 1;  // splite k dim into virtual cores

    /// Methods

    ASCENDCT_DEVICE
    SplitkMatmulIdentityBlockSwizzle() {}

    ASCENDCT_DEVICE
    SplitkMatmulIdentityBlockSwizzle(
        MatmulCoord const &problemShape_, MatmulCoord const &tileShape_, uint32_t splitkFactor_ = 1
    ) : problemShape(problemShape_), tileShape(tileShape_), splitkFactor(splitkFactor_)
    {
        loopsMNK = CeilDiv(problemShape, tileShape);
    }

    ASCENDCT_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    ASCENDCT_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        return taskIdx % GetCoreLoops() / mnLoops;
    }

    ASCENDCT_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMNK.m() * loopsMNK.n() * splitkFactor;
    }

    ASCENDCT_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    ASCENDCT_DEVICE
    MatmulCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return MatmulCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return MatmulCoord{mIdx, nIdx, kIdx};
        }
    }

    ASCENDCT_DEVICE
    MatmulCoord GetActualBlockShape(MatmulCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t splitkSliceLen;
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileShape.m()) : tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileShape.n()) : tileShape.n();
        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
            (problemShape.k() - blockCoord.k() * tileShape.k()) : splitkSliceLen;
        return MatmulCoord{mActual, nActual, kActual};
    }
};

}  // namespace AscendCT::gemm::block

#endif  // ASCENDCT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP