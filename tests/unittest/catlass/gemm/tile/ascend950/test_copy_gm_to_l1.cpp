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
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));
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

// ColumnMajor -> nZ(B1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonZB1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->ndNum, _1);
    ASSERT_EQ(p->nValue, _col);
    ASSERT_EQ(p->dValue, _row);
    ASSERT_EQ(p->srcDValue, _row);
    ASSERT_EQ(p->dstNzC0Stride, _col_round);
    ASSERT_EQ(p->dstNzNStride, _1);
}

// RowMajor -> zN(A1), 3-param Ascend950 specialization, long-stride split path.
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNA1TestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    uint32_t veryLongStride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, veryLongStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < logs.size(); ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), i * veryLongStride * sizeof(Element));
        const auto* p = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->nValue, _1);
        ASSERT_EQ(p->dValue, _col);
        ASSERT_EQ(p->srcDValue, _0);
        ASSERT_EQ(p->dstNzNStride, _0);
    }
}

// RowMajor -> zN(B1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNB1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->ndNum, _1);
    ASSERT_EQ(p->nValue, _row);
    ASSERT_EQ(p->dValue, _col);
    ASSERT_EQ(p->srcDValue, _col);
    ASSERT_EQ(p->dstNzC0Stride, _row_round);
    ASSERT_EQ(p->dstNzNStride, _1);
}

// RowMajor -> zZ(B1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozZB1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _row / C0_NUM_PER_FRACTAL;
    uint32_t remains = _row % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _col;
    uint32_t dstMatrixStride = RoundUp<ELE_NUM_PER_C0>(_col) * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, ndNum);
        ASSERT_EQ(p->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dValue, _col);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _col);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, _1);
        ASSERT_EQ(p->nValue, remains);
        ASSERT_EQ(p->dValue, _col);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _col);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, _0);
    }
}

// RowMajor -> RowMajor(A1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, RowMajorToRowMajorA1ContiguousTest)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::RowMajor;
    using GmType = Gemm::GemmType<Element, Layout>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;

    setShape<Element>();
    Layout layoutSrc = Layout::template MakeLayout<Element>(_row, _col);
    Layout layoutDst = Layout::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    BaseCheck<Element>(logs[0]);
    ASSERT_EQ(*logs[0].GetArgsAt(2).Value<uint32_t>(), _row * _col);
}

// ColumnMajor -> nN(A1/B1), 3-param Ascend950 specializations.
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNA1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _col / C0_NUM_PER_FRACTAL;
    uint32_t remains = _col % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _row;
    uint32_t dstMatrixStride = RoundUp<ELE_NUM_PER_C0>(_row) * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, ndNum);
        ASSERT_EQ(p->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dValue, _row);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _row);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, _1);
        ASSERT_EQ(p->nValue, remains);
        ASSERT_EQ(p->dValue, _row);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _row);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, _0);
    }
}

TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNB1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _col / C0_NUM_PER_FRACTAL;
    uint32_t remains = _col % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _row;
    uint32_t dstMatrixStride = RoundUp<ELE_NUM_PER_C0>(_row) * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, ndNum);
        ASSERT_EQ(p->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dValue, _row);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _row);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* p = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(p->ndNum, _1);
        ASSERT_EQ(p->nValue, remains);
        ASSERT_EQ(p->dValue, _row);
        ASSERT_EQ(p->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(p->srcDValue, _row);
        ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstNzNStride, _1);
        ASSERT_EQ(p->dstNzMatrixStride, _0);
    }
}

// VectorLayout -> zN(A1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, VectorTozNA1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc(_col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_1, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* p = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->ndNum, _1);
    ASSERT_EQ(p->nValue, _1);
    ASSERT_EQ(p->dValue, _col);
    ASSERT_EQ(p->srcDValue, _col);
    ASSERT_EQ(p->dstNzC0Stride, C0_NUM_PER_FRACTAL);
    ASSERT_EQ(p->dstNzNStride, _1);
}

// VectorLayout(GM) -> VectorLayout(A1), 3-param Ascend950 specialization.
TEST_P(TileCopyGmToL1TestAscend950, VectorToVectorA1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::VectorLayout;
    using GmType = Gemm::GemmType<Element, Layout, AscendC::TPosition::GM>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    Layout layoutSrc(_col);
    Layout layoutDst(_col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    BaseCheck<Element>(logs[0]);
    const auto* p = logs[0].GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(p->blockCount, _1);
    ASSERT_EQ(p->blockLen, _col / ELE_NUM_PER_C0);
    ASSERT_EQ(p->srcStride, _0);
    ASSERT_EQ(p->dstStride, _0);
}

// DynamicOptimized PaddingRowMajor -> zN, direct Ascend950 coverage.
TEST_P(TileCopyGmToL1TestAscend950, DynamicOptimizedPaddingRowMajorTozNTest)
{
    using Element = half;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    LayoutSrc layoutSrc(_row, _col, C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* p = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->nValue, _row);
    ASSERT_EQ(p->dValue, _col);
    ASSERT_EQ(p->srcDValue, ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzC0Stride, _row_round);
}

// DynamicOptimized PaddingColumnMajor -> nZ, direct Ascend950 coverage.
TEST_P(TileCopyGmToL1TestAscend950, DynamicOptimizedPaddingColumnMajorTonZTest)
{
    using Element = half;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::PaddingColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc(_row, _col, ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* p = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->nValue, _col);
    ASSERT_EQ(p->dValue, _row);
    ASSERT_EQ(p->srcDValue, ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzC0Stride, _col_round);
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
