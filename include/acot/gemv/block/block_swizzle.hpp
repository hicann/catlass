/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP
#define ACOT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP

#include "acot/acot.hpp"
#include "acot/detail/alignment.hpp"
#include "acot/gemv_coord.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::gemv::block
{

    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Block swizzling function for Matmuls
    template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
    struct GemvIdentityBlockSwizzle
    {
        /// Data members

        GemvCoord problemShape;
        MatrixCoord tileMN;        // 矩阵和向量分块的大小
        MatrixCoord loopsMN;       // 分块后的矩阵和向量的循环次数
        uint32_t splitKSlices = 1; // splite k dim into virtual cores

        /// Methods

        ACOT_DEVICE
        GemvIdentityBlockSwizzle() {} // 默认构造函数

        ACOT_DEVICE // 传入tileMN，并通过problemShape和tileMN来算loopsMN
        GemvIdentityBlockSwizzle(GemvCoord const &problemShape_, MatrixCoord const &tileMN_, uint32_t splitKSlices_ = 1)
            : problemShape(problemShape_), tileMN(tileMN_), splitKSlices(splitKSlices_)
        {
            loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
        }

        ACOT_DEVICE // 显性传入loopsMN,即已经计算好的循环次数
        GemvIdentityBlockSwizzle(GemvCoord const &problemShape_, MatrixCoord const &tileMN_,
                                 MatrixCoord const &loopsMN_, uint32_t splitKSlices_ = 1)
            : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_), splitKSlices(splitKSlices_)
        {
        }

        ACOT_DEVICE // 更新各个参数值，并重算loopsMN，这个函数允许动态更新矩阵形状和块大小
            void
            Update(GemvCoord const &problemShape_, MatrixCoord const &tileMN_, uint32_t splitKSlices_ = 1)
        { // 重算loopsMN
            problemShape = problemShape_;
            tileMN = tileMN_;
            splitKSlices = splitKSlices_;

            loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
        }

        ACOT_DEVICE
        void Update(GemvCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_,
                    uint32_t splitKSlices_ = 1)
        { // 重传各参数和loopsMN
            problemShape = problemShape_;
            tileMN = tileMN_;
            loopsMN = loopsMN_;
            splitKSlices = splitKSlices_;
        }

        ACOT_DEVICE
        uint32_t GetCoreLoops() const // 总循环次数：行方向切分的循环数 * 列方向切分的循环数
        {
            return loopsMN.row() * loopsMN.column(); //
        }

        ACOT_DEVICE
        uint32_t GetBatchIdx(uint32_t taskIdx) // 返回当前任务的批次索引。
        {
            return taskIdx / (GetCoreLoops() * splitKSlices);
        }

        ACOT_DEVICE
        GemvCoord GetBlockCoord(uint32_t taskIdx) // 给定的任务索引 taskIdx，返回当前任务所在的块的坐标
        {
            uint32_t kIdx = taskIdx % (GetCoreLoops() * splitKSlices) / GetCoreLoops();
            uint32_t innerIdx = taskIdx % GetCoreLoops(); // 计算当前任务在矩阵和向量块中的位置
            if constexpr (SwizzleDirection == 0)
            { // Zn
                uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
                uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
                uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

                uint32_t nRow = SwizzleOffset;
                if (tileBlockIdx == tileBlockLoop - 1)
                {
                    nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
                }
                uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
                uint32_t nIdx = inTileBlockIdx / nRow;
                if (tileBlockIdx % 2 == 1)
                {
                    nIdx = loopsMN.column() - nIdx - 1;
                }
                return GemvCoord{mIdx, nIdx};
            }
            else if constexpr (SwizzleDirection == 1)
            { // Nz
                uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
                uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
                uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

                uint32_t nCol = SwizzleOffset;
                if (tileBlockIdx == tileBlockLoop - 1)
                {
                    nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
                }
                uint32_t mIdx = inTileBlockIdx / nCol;
                uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
                if (tileBlockIdx % 2 == 1)
                {
                    mIdx = loopsMN.row() - mIdx - 1;
                }
                // return GemvCoord{mIdx, nIdx, kIdx};
                return GemvCoord{mIdx, nIdx};
            }
        }

        ACOT_DEVICE
        GemvCoord GetActualBlockShape(GemvCoord blockCoord)
        {
            // uint32_t splitKLen = problemShape.k() / splitKSlices;
            uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ? (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
            uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ? (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
            // uint32_t kActual = (blockCoord.k() == (splitKSlices - 1)) ? (problemShape.k() - blockCoord.k() * splitKLen) : splitKLen;
            return GemvCoord{mActual, nActual};
        }
    };

} // namespace acot::matmul::block

#endif // ACOT_MATMUL_BLOCK_BLOCK_SWIZZLE_HPP