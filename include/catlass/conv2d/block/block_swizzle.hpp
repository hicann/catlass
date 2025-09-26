/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_BLOCK_BLOCK_SWIZZLE_HPP
#define CATLASS_CONV2D_BLOCK_BLOCK_SWIZZLE_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/conv2d_coord.hpp"

namespace Catlass::Conv2d::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Conv2ds
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct Conv2dIdentityBlockSwizzle {
    /// Data members

    Conv2d5HdCoord problemShape; // {Batch, Ho, Wo, Cout, Cin1}
    HoWoCoutCoord tiles;
    HoWoCoutCoord loops;

    /// Methods

    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle(Conv2d5HdCoord const &problemShape_, HoWoCoutCoord const &tiles_)
        : problemShape(problemShape_), tiles(tiles_) {
        loops = CeilDiv(HoWoCoutCoord(problemShape.GetCoordHoWoCout()), tiles);      
    }

    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle(Conv2d5HdCoord const &problemShape_, HoWoCoutCoord const &tiles_,
        HoWoCoutCoord const &loops_)
        : problemShape(problemShape_), tiles(tiles_), loops(loops_) {}

    CATLASS_DEVICE
    void Update(Conv2d5HdCoord const &problemShape_, HoWoCoutCoord const &tiles_) {
        problemShape = problemShape_;
        tiles = tiles_;
        loops = CeilDiv(HoWoCoutCoord(problemShape.GetCoordHoWoCout()), tiles);
    }

    CATLASS_DEVICE
    void Update(Conv2d5HdCoord const &problemShape_, HoWoCoutCoord const &tiles_, HoWoCoutCoord const &loops_) {
        problemShape = problemShape_;
        tiles = tiles_;
        loops = loops_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const {
        return loops.ho() * loops.wo() * loops.cout(); 
    }

    CATLASS_DEVICE
    uint32_t GetLoops() const {
        return problemShape.batch() * this->GetCoreLoops(); 
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) {
        return taskIdx / (GetCoreLoops());
    }

    CATLASS_DEVICE
    Conv2d5HdCoord GetBlockCoord(uint32_t taskIdx) {
        uint32_t outerIdx = this->GetBatchIdx(taskIdx);
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // howoCout (Zn)
            uint32_t tileBlockLoop = CeilDiv(loops.howo(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loops.cout());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loops.cout());

            uint32_t nHoWo = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nHoWo = loops.howo() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t howoIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nHoWo;
            uint32_t coutIdx = inTileBlockIdx / nHoWo;
            if (tileBlockIdx % 2 == 1) {
                coutIdx = loops.cout() - coutIdx - 1;
            }
            uint32_t hoIdx = howoIdx / loops.wo();
            uint32_t woIdx = howoIdx % loops.wo();
            return Conv2d5HdCoord{outerIdx, hoIdx, woIdx, coutIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // coutHoWo (Nz)
            uint32_t tileBlockLoop = CeilDiv(loops.cout(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loops.howo());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loops.howo());

            uint32_t nCout = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCout = loops.cout() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t howoIdx = inTileBlockIdx / nCout;
            uint32_t coutIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCout;
            if (tileBlockIdx % 2 == 1) {
                howoIdx = loops.howo() - howoIdx - 1;
            }
            uint32_t hoIdx = howoIdx / loops.wo();
            uint32_t woIdx = howoIdx % loops.wo();
            return Conv2d5HdCoord{outerIdx, hoIdx, woIdx, coutIdx, 0};
        }
    }

    CATLASS_DEVICE
    Conv2d5HdCoord GetActualBlockShape(Conv2d5HdCoord blockCoord) {
        uint32_t hoActual = (blockCoord.h() == loops.ho() - 1) ? 
            (problemShape.h() - blockCoord.h() * tiles.ho()) : tiles.ho();
        uint32_t woActual = (blockCoord.w() == loops.wo() - 1) ? 
            (problemShape.w() - blockCoord.w() * tiles.wo()) : tiles.wo();
        uint32_t coutActual = (blockCoord.cout() == loops.cout() - 1) ? 
            (problemShape.cout() - blockCoord.cout() * tiles.cout()) : tiles.cout();
        uint32_t cin1Actual = problemShape.cin1();
        return Conv2d5HdCoord{1, hoActual, woActual, coutActual, cin1Actual};
    }
};

}  // namespace Catlass::Conv2d::Block

#endif  // CATLASS_CONV2D_BLOCK_BLOCK_SWIZZLE_HPP
