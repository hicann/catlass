/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_BROADCAST_ADD_HPP
#define CATLASS_EPILOGUE_TILE_TILE_BROADCAST_ADD_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

/// BroadcastAdd computes the elementwise addition of a tensor of shape (m, n) and a tensor
/// of shape (m, n) after broadcasting. 

/// @brief Computes the elementwise addition of a tensor with shape (m, n) and a tensor with
/// original shape (1, n) broadcast to (m, n).
/// @tparam ArchTag_ is the architecture tag.
/// @tparam ComputeType_ includes the element type and layout information.
/// @tparam TileShape_ is the shape (m, n).
template <
    class ArchTag_,
    class ComputeType_,
    class TileShape_
>
struct TileRowBroadcastAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    CATLASS_DEVICE
    TileRowBroadcastAdd() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        constexpr uint32_t maxRepeatTimes = 255;
        // 单个block内元素数量
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);

        // 一行block的数量
        constexpr uint32_t blkNumPerColumn = TileShape::COLUMN / eleNumPerBlk;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        // src1需要向m方向broadcast，每次迭代后dst和src0都向下移一行，而src1不变，进而等价实现broadcast
        repeatParams.dstRepStride = blkNumPerColumn;
        repeatParams.src0RepStride = blkNumPerColumn;
        repeatParams.src1RepStride = 0;

        // 行方向，计算的repeat重复次数
        constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
        // 一次repeate中计算的元素个数，一次repeat计算的数据大小/元素个数（也等于8个block*每个block的元素个数）
        constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        for (uint32_t rowOffset = 0; rowOffset < TileShape::ROW; rowOffset += rowNumPerCompute) {
            // 计算了rowNumPerCompute行后，还剩多少行
            uint32_t residueM = TileShape::ROW - rowOffset;
            // 本次repeattime是多少，取residueM和rowNumPerCompute最小值。
            uint8_t repeatTimes = static_cast<uint8_t>((residueM > rowNumPerCompute) ? rowNumPerCompute : residueM);
            for (uint32_t colOffset = 0; colOffset < TileShape::COLUMN; colOffset += colNumPerCompute) {
                // 在行方向循环，因为先前同行的数据被访问过，有利于缓存命中
                uint32_t residueN = TileShape::COLUMN - colOffset;
                // 单次迭代计算的数据个数，为单次行方向可计算数据量或行方向上剩余元素，取最小值。
                uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                AscendC::Add(
                    ubOut[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn0[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn1[colOffset],
                    mask, repeatTimes, repeatParams
                );
            }
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif
