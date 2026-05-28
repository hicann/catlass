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

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for GM->L1 TileCopy Utilities
class TileCopyGmToL1TestAtlasA2 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
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

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

/// Testsuite for CopyGmToL1

// RowMajorTozNB1TestBasic (#17): RowMajor→zN(B1) 3-template parameters
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNB1TestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using GmTypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, GmTypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->ndNum, _1);
    ASSERT_EQ(p->nValue, _row);
    ASSERT_EQ(p->dValue, _col);
    ASSERT_EQ(p->srcDValue, _col);
    ASSERT_EQ(p->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
}

// Testcase, from RowMajor -> zN, AtlasA2, long-stride场景(需拆分逐行搬运)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> && std::is_same_v<LayoutDst, layout::zN>,
        "This testsuite is RowMajor->zN");
    static_assert(!std::is_same_v<Element, AscendC::int4b_t>, "It is not covered where datatype is `int4b_t`");

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_FALSE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_GE(_very_long_stride, STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();

    ASSERT_EQ(logs.size(), _row);

    for (int i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_C0);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// PaddingRowMajorTozNTestBasic (#5): PaddingRowMajor→zN
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingRowMajorTozNTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->dValue, layoutSrc.orgShape(1));
    ASSERT_EQ(p->nValue, layoutSrc.orgShape(0));
    ASSERT_EQ(p->srcDValue, layoutSrc.stride(0));
    ASSERT_EQ(p->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
}

// PaddingColumnMajorTonZTestBasic (#6): PaddingColumnMajor→nZ
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingColumnMajorTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->dValue, layoutSrc.orgShape(0));
    ASSERT_EQ(p->nValue, layoutSrc.orgShape(1));
    ASSERT_EQ(p->srcDValue, layoutSrc.stride(2));
    ASSERT_EQ(p->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
}

/// Testsuite for CopyGmToL1GMMPTD

// Testcase, from RowMajor -> zN, AtlasA2, one-row场景(调用GMMPTD)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestOneRow_GMMPTD)
{
    if (GetParam().row != 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_EQ(layoutSrc.shape(0), 1);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    ASSERT_EQ(logs.size(), 1);

    auto logTileCopyGmTensor = logTileCopy.GetArgsAt(1).RawValue();
    auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(0).RawValue();
    ASSERT_EQ(logTileCopyGmTensor, &gmTensor);
    ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

    const AscendC::DataCopyParams* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyArg->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
    ASSERT_EQ(dataCopyArg->blockLen, _1);
    ASSERT_EQ(dataCopyArg->srcStride, _0);
    ASSERT_EQ(dataCopyArg->dstStride, C0_NUM_PER_FRACTAL - _1);
}

// RowMajorTozNMultiRowGMMPTD (#18): GMMPTD multi-row Nd2Nz
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNMultiRowGMMPTD)
{
    if (GetParam().row <= 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_GT(layoutSrc.shape(0), 1);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(p->ndNum, _1);
    ASSERT_EQ(p->nValue, _row);
    ASSERT_EQ(p->dValue, _col);
    ASSERT_EQ(p->srcDValue, _col);
    ASSERT_EQ(p->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(p->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
}

/// Testsuite for CopyGmToL1DynamicOptimized

// RowMajorTozNDynamicOptimizedFewRows (#19): DynamicOptimized rows<=16
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNDynamicOptimizedFewRows)
{
    if (GetParam().row > 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_LE(layoutSrc.shape(0), 16);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(p->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
        ASSERT_EQ(p->srcStride, _0);
    }
}

/// Testsuite for `CopyGmToL1IntervalDataCopy`

// RowMajorTozNIntervalDataCopyHalf (#21): half RowMajor→zN
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(p->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
        ASSERT_EQ(p->srcStride, _0);
    }
}

// PaddingRowMajorTozNIntervalDataCopyHalf (#22): half PaddingRowMajor→zN
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingRowMajorTozNIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), layoutSrc.orgShape(0));

    for (uint32_t i = 0; i < layoutSrc.orgShape(0); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(p->blockCount, CeilDiv(layoutSrc.orgShape(1), ELE_NUM_PER_C0));
        ASSERT_EQ(p->srcStride, _0);
    }
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyGmToL1,
    TileCopyGmToL1TestAtlasA2,
    ::testing::Values(
        TestMatrixShape{128U, 256U},  // aligned
        TestMatrixShape{1U, 256U},    // 1-d
        TestMatrixShape{42U, 256U},   // not aligned
        TestMatrixShape{42U, 283U}    // not aligned (on both sides)
    )
);

#endif // CATLASS_ARCH == 2201