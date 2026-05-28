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
#include "catlass/detail/dependent_false.hpp"

#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for L1->L0A TileCopy Utilities
class TileCopyL1ToL0ATestAscend950 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
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
    void BaseCheck(AscendCCallLog const &logTileCopy, AscendC::LocalTensor<Element>& l0aTensor, AscendC::LocalTensor<Element>& l1Tensor) 
    {
        // The API name should be "LoadData"
        ASSERT_EQ(logTileCopy.name, "LoadData");

        // Check the given address
        auto logTileCopyL0ATensor = logTileCopy.GetArgsAt(0).RawValue();
        auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(1).RawValue();
        ASSERT_EQ(logTileCopyL0ATensor, &l0aTensor);
        ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

        // Check for the data type
        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

};

// zNTozN Basic test, from zN->zN, Ascend950, basic
TEST_P(TileCopyL1ToL0ATestAscend950, zNTozNTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0aTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, _rows_by_fractal);  // Unit: 16(element)
    ASSERT_EQ(loadDataArg->kStep, _cols_by_fractal);  // Unit: 32(Byte)
    ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, false);
}

// nZTozN basic test, from nZ->zN, Ascend950, transpose
TEST_P(TileCopyL1ToL0ATestAscend950, nZTozNTest)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0aTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, CeilDiv<2>(_cols_by_fractal));    // Unit: 16(element) [it is trans]
    ASSERT_EQ(loadDataArg->kStep, 2 * _rows_by_fractal);            // Unit: 32(Byte)    [it is trans]
    ASSERT_EQ(loadDataArg->srcStride, CeilDiv<2>(_cols_by_fractal)); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal);            // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, true);
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0A,
    TileCopyL1ToL0ATestAscend950,
    ::testing::Values(
        TestMatrixShape{128U, 64U},
        TestMatrixShape{1U, 128U},
        TestMatrixShape{64U, 42U},
        TestMatrixShape{123U, 8U}
    )
);

#endif // CATLASS_ARCH == 3510