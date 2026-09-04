/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "catlass/coord.hpp"

using Catlass::Coord;
using Catlass::MakeCoord;

TEST(CoordTest, ConstructionAndAccess)
{
    constexpr Coord<3> constant(4);
    static_assert(constant[0] == 4 && constant[1] == 4 && constant[2] == 4);

    Coord<3> coord = MakeCoord(1u, 2u, 3u);
    EXPECT_EQ(coord[0], 1u);
    EXPECT_EQ(coord.At<1>(), 2u);
    coord.At(2) = 8u;
    EXPECT_EQ(coord[2], 8u);
}

TEST(CoordTest, Arithmetic)
{
    Coord<3> lhs = MakeCoord(12u, 9u, 6u);
    Coord<3> rhs = MakeCoord(3u, 2u, 1u);

    auto sum = lhs + rhs;
    auto add_scalar = lhs + 2u;
    auto difference = lhs - rhs;
    auto subtract_scalar = lhs - 2u;
    auto product = lhs * rhs;
    auto quotient = lhs / rhs;
    auto remainder = lhs % rhs;

    EXPECT_EQ(sum, MakeCoord(15u, 11u, 7u));
    EXPECT_EQ(add_scalar, MakeCoord(14u, 11u, 8u));
    EXPECT_EQ(difference, MakeCoord(9u, 7u, 5u));
    EXPECT_EQ(subtract_scalar, MakeCoord(10u, 7u, 4u));
    EXPECT_EQ(product, MakeCoord(36u, 18u, 6u));
    EXPECT_EQ(quotient, MakeCoord(4u, 4u, 6u));
    EXPECT_EQ(remainder, MakeCoord(0u, 1u, 0u));

    lhs += rhs;
    EXPECT_EQ(lhs, MakeCoord(15u, 11u, 7u));
}

TEST(CoordTest, PredicatesAndReductions)
{
    Coord<4> coord = MakeCoord(7u, 2u, 9u, 3u);
    EXPECT_TRUE(static_cast<bool>(coord));
    EXPECT_FALSE(!coord);
    EXPECT_EQ(coord.Argmin(), 1);
    EXPECT_EQ(coord.Argmax(), 2);
    EXPECT_EQ(Coord<4>::Min(coord, MakeCoord(5u, 4u, 10u, 1u)), MakeCoord(5u, 2u, 9u, 1u));

    Coord<4> zero;
    EXPECT_FALSE(static_cast<bool>(zero));
    EXPECT_TRUE(!zero);
}

TEST(CoordTest, ProjectedAxes)
{
    Coord<4> coord = MakeCoord(10u, 20u, 30u, 40u);
    auto projected = coord.GetCoordByAxis<3, 1>();
    EXPECT_EQ(projected[0], 40u);
    EXPECT_EQ(projected[1], 20u);
}
