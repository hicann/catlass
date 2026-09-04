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

#include "catlass/layout/matrix.hpp"

using namespace Catlass;
using namespace Catlass::layout;

namespace {
template <class Layout>
void ExpectFractalMetadata(Layout const& layout, uint32_t orgRows, uint32_t orgCols)
{
    EXPECT_EQ(layout.orgShape(0), orgRows);
    EXPECT_EQ(layout.orgShape(1), orgCols);
    for (int i = 0; i < Layout::RANK; ++i) {
        EXPECT_EQ(layout.shape(i), layout.shape()[i]);
        EXPECT_EQ(layout.stride(i), layout.stride()[i]);
    }
}

TEST(MatrixLayoutTest, RowAndColumnMajor)
{
    RowMajor row_major(4, 7);
    EXPECT_EQ(row_major.shape(0), 4u);
    EXPECT_EQ(row_major.shape(1), 7u);
    EXPECT_EQ(row_major.stride(0), 7);
    EXPECT_EQ(row_major.stride(1), 1);
    EXPECT_EQ(row_major.GetOffset(MatrixCoord(uint32_t(2), uint32_t(3))), 17);
    EXPECT_EQ(row_major.Capacity(), 28);
    EXPECT_EQ(row_major.GetTileLayout(MatrixCoord(uint32_t(2), uint32_t(3))).stride(0), 7);

    ColumnMajor column_major(4, 7);
    EXPECT_EQ(column_major.stride(0), 1);
    EXPECT_EQ(column_major.stride(1), 4);
    EXPECT_EQ(column_major.GetOffset(MatrixCoord(uint32_t(2), uint32_t(3))), 14);
    EXPECT_EQ(column_major.Capacity(), 28);
}

TEST(MatrixLayoutTest, FractalLayouts)
{
    nZ nz(13, 17, 2, 7, 4, 5, 1, 32, 2, 64);
    zN zn(13, 17, 2, 7, 4, 5, 1, 32, 2, 64);
    zZ zz(13, 17, 2, 7, 4, 5, 1, 32, 2, 64);
    L0C l0c(13, 17, 16, 1, 16, 2, 16, 256, 1, 256);
    nN nn(13, 17, 2, 7, 4, 5, 1, 32, 2, 64);
    Weight4BitnZ weight(13, 17, 2, 7, 4, 5, 1, 32, 2, 64);

    ExpectFractalMetadata(nz, 13, 17);
    ExpectFractalMetadata(zn, 13, 17);
    ExpectFractalMetadata(zz, 13, 17);
    ExpectFractalMetadata(l0c, 13, 17);
    ExpectFractalMetadata(nn, 13, 17);
    ExpectFractalMetadata(weight, 13, 17);

    MatrixCoord coord(uint32_t(3), uint32_t(6));
    EXPECT_EQ(nz.GetOffset(coord), 101);
    EXPECT_EQ(zn.GetOffset(coord), 101);
    EXPECT_EQ(zz.GetOffset(coord), 96);
    EXPECT_EQ(l0c.GetOffset(coord), 54);
    EXPECT_EQ(nn.GetOffset(coord), 96);
    EXPECT_EQ(weight.GetOffset(coord), 101);
}

TEST(MatrixLayoutTest, PaddedLayouts)
{
    PaddingRowMajor row(7, 11, 4, 8);
    EXPECT_EQ(row.orgShape(0), 7u);
    EXPECT_EQ(row.orgShape(1), 11u);
    EXPECT_EQ(row.shape(1), 2u);
    EXPECT_EQ(row.shape(3), 2u);
    EXPECT_EQ(row.GetOffset(MatrixCoord(uint32_t(5), uint32_t(9))), 105);

    PaddingColumnMajor column(7, 11, 4, 8);
    EXPECT_EQ(column.shape(1), 2u);
    EXPECT_EQ(column.shape(3), 2u);
    EXPECT_EQ(column.GetOffset(MatrixCoord(uint32_t(5), uint32_t(9))), 101);
}

TEST(MatrixLayoutTest, SpecializedLayouts)
{
    auto fmap = NDC1HWC0::MakeLayout(2, 3, 4, 5, 6, 7);
    EXPECT_EQ(fmap.shape(0), 2u);
    EXPECT_EQ(fmap.shape(1), 6u);
    EXPECT_EQ(fmap.shape(2), 5u);
    EXPECT_EQ(fmap.shape(4), 12u);
    EXPECT_EQ(fmap.GetOffset(Conv3d6HdCoord(1, 2, 3, 4)), 4858);

    auto filter = KDC1KHKWN1N0C0::MakeLayout(6, 3, 4, 5);
    EXPECT_EQ(filter.orgShape(0), 6u);
    EXPECT_EQ(filter.orgShape(3), 5u);
    EXPECT_EQ(filter.shape(0), 5u);
    EXPECT_EQ(filter.shape(2), 4u);
    EXPECT_EQ(filter.GetOffset(Conv3dFracZ3dCoord(2, 1)), 2 * filter.stride(3) + filter.stride(1));
}
} // namespace
