/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_LAYOUT_VECTOR_HPP
#define ACOT_LAYOUT_VECTOR_HPP

#include "acot/acot.hpp"
#include "acot/coord.hpp"

namespace acot::layout
{

    struct VectorLayout
    {
    public:
        /// Logical rank of tensor
        static constexpr int RANK = 1;

        /// Index type used for coordinates
        using Index = uint32_t;

        /// Long index type used for offsets
        using LongIndex = int64_t;

        /// Logical coordinate，shape也是1维的
        using Shape = Coord<RANK, Index>;

        /// Stride vector， stride是1维的
        using Stride = Coord<RANK, LongIndex>;

    public:
        // Methods

        // Constructor
        ACOT_HOST_DEVICE
        VectorLayout() : stride_(MakeCoord(LongIndex(1))) {}

        // Ctor
        ACOT_HOST_DEVICE
        VectorLayout(Stride stride) : stride_(stride) {};

        /// Returns the stride of the layout
        ACOT_HOST_DEVICE
        Stride stride() const
        {
            return stride_;
        }

        /// Returns the stride of the layout 新增
        ACOT_HOST_DEVICE
        Stride &stride()
        {
            return stride_;
        }

        /// Returns the stride of the layout 新增
        ACOT_HOST_DEVICE
        typename Stride::Index stride(int idx) const
        {
            return stride_[idx];
        }

        /// Returns the stride of the layout 新增
        ACOT_HOST_DEVICE
        typename Stride::Index &stride(int idx)
        {
            return stride_[idx];
        }

        /// Returns the layout of a tile.
        ACOT_HOST_DEVICE
        VectorLayout GetTileLayout(Stride const &tileShape) const
        {
            return VectorLayout(tileShape);
        }

    private:
        /// Stride data member
        Stride stride_;
    };

} // namespace acot::layout

#endif // ACOT_LAYOUT_VECTOR_HPP