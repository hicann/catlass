/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// Test for CopyGmToL1
class TileCopyGmToL1TestAscend950: public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() {
        uint32_t row = GetParam().row;
        uint32_t col = GetParam().col;
        _setShape<Element, isTrans>(row, col);
    }

    template <class Element>
    void BaseCheck(AscendCCallLog const &logTileCopy){

    }
};

// Testcase, from RowMajor -> zN, Ascend950, long stride场景
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_FALSE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_GE(_very_long_stride, STRIDE_LIMIT);
    
    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    // log check
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    AscendCCallLog logTileCopy = logs[0];

    auto logTileCopyGmTensor = logTileCopy.GetArgsAt(1).RawValue();
    auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(0).RawValue();
    ASSERT_EQ(logTileCopyGmTensor, &gmTensor);
    ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

    ASSERT_EQ(logs.size(), 1); // still one tile-copy ops
    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1); 
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _very_long_stride); // `srcDValue` can cover long stride
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyGmToL1,
    TileCopyGmToL1TestAscend950,
    ::testing::Values(
        TestMatrixShape{128U, 256U},  // aligned
        TestMatrixShape{1U, 256U},    // 1-d
        TestMatrixShape{42U, 256U},   // not aligned
        TestMatrixShape{42U, 283U}    // not aligned (on both sides)
    )
);

#endif // CATLASS_ARCH == 3510

