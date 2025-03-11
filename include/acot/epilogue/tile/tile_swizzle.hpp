/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_EPILOGUE_TILE_TILE_SWIZZLE_HPP
#define ACOT_EPILOGUE_TILE_TILE_SWIZZLE_HPP

#include "acot/acot.hpp"
#include "acot/detail/alignment.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::epilogue::tile {

struct EpilogueIdentityTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    ACOT_DEVICE
    EpilogueIdentityTileSwizzle() = default;

    ACOT_DEVICE
    EpilogueIdentityTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    ACOT_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ACOT_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx / loopsMN.column(), loopIdx % loopsMN.column() };
    }

    ACOT_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

struct EpilogueHorizontalTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    ACOT_DEVICE
    EpilogueHorizontalTileSwizzle() = default;

    ACOT_DEVICE
    EpilogueHorizontalTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    ACOT_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ACOT_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx % loopsMN.row(), loopIdx / loopsMN.row() };
    }

    ACOT_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

}

#endif  // ACOT_EPILOGUE_TILE_TILE_SWIZZLE_HPP
