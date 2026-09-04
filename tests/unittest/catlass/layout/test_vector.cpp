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

#include "catlass/layout/vector.hpp"

using namespace Catlass;
using namespace Catlass::layout;

TEST(VectorLayoutTest, ConstructionOffsetAndTile)
{
    VectorLayout layout(17);
    EXPECT_EQ(layout.shape(0), 17u);
    EXPECT_EQ(layout.stride(0), 1);
    EXPECT_EQ(layout.GetOffset(MakeCoord(5u)), 5);

    auto tile = layout.GetTileLayout(MakeCoord(8u));
    EXPECT_EQ(tile.shape(0), 8u);
    EXPECT_EQ(tile.stride(0), 1);
}
