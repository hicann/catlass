/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_COORD_HPP
#define ACOT_GEMV_COORD_HPP

#include "acot/coord.hpp"

namespace acot
{

    /// Shape of a matrix multiply-add operation
    template <
        /// Rows of matrix product
        uint32_t M_ = 1,
        /// Columns of the matrix (number of elements in the input vector)
        uint32_t N_ = 1>
    struct GemvShape
    {
        static constexpr uint32_t M = M_;
        static constexpr uint32_t N = N_;

        static constexpr int64_t MN = M * N;

        static constexpr int64_t COUNT = MN;

        /// Returns a Coord object
        ACOT_HOST_DEVICE
        static Coord<2> ToCoord()
        {
            return MakeCoord(M, N);
        }
    };

    /// GemvCoord is a structure derived from Coord<2> that specifies a location within the
    /// coordinate space of a GEMV problem.
    struct GemvCoord : public Coord<2, uint32_t>
    {
        /// Integer-valued index
        using Index = uint32_t;

        /// Base type is a Coord of rank=2
        using Base = Coord<2, Index>;

        /// GEMV M dimension - rows of the output vector (y)
        static constexpr int M_INDEX = 0;

        /// GEMV N dimension - columns of the matrix (length of the input vector x)
        static constexpr int N_INDEX = 1;

        /// Default ctor
        ACOT_HOST_DEVICE
        GemvCoord() {}

        /// Constructs from Coord<2> and a batch
        ACOT_HOST_DEVICE
        GemvCoord(Coord<2, Index> const &coord) : Base(coord) {}

        /// Helper to construct from M, N coordinates
        ACOT_HOST_DEVICE
        GemvCoord(Index m, Index n) : Base(MakeCoord(m, n)) {}

        /// Returns the GEMV M coordinate (row of the result y)
        ACOT_HOST_DEVICE
        Index const &m() const
        {
            return this->At(M_INDEX);
        }

        /// Returns reference to the GEMV M coordinate
        ACOT_HOST_DEVICE
        Index &m()
        {
            return this->At(M_INDEX);
        }

        /// Returns the GEMV N coordinate (column of the matrix A or the input vector x)
        ACOT_HOST_DEVICE
        Index const &n() const
        {
            return this->At(N_INDEX);
        }

        /// Returns reference to the GEMV N coordinate
        ACOT_HOST_DEVICE
        Index &n()
        {
            return this->At(N_INDEX);
        }

        ACOT_HOST_DEVICE
        auto GetCoordMN() const
        {
            return this->GetCoordByAxis<M_INDEX, N_INDEX>();
        }
    };

} // namespace acot

#endif // ACOT_GEMV_COORD_HPP