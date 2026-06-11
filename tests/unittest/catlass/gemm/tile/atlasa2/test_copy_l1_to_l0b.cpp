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

#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for L1->L0B TileCopy Utilities
class TileCopyL1ToL0BTest : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    // 重写SetUp方法，在每个测试用例执行前进行初始化操作
    void SetUp() override
    {
        // 调用父类的SetUp方法完成基础初始化
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() {
        uint32_t row = GetParam().row;
        uint32_t col = GetParam().col;
        _setShape<Element, isTrans>(row, col);
    }

    template <class Element, bool isTrans=false>
    void BaseCheck(AscendCCallLog const &logTileCopy){
        // 验证调用名称：应为"LodaData" 或 "LoadDataWithTranspose"
        if constexpr(isTrans){
            ASSERT_EQ(logTileCopy.name, "LoadDataWithTranspose");
        } else {
            ASSERT_EQ(logTileCopy.name, "LoadData");
        }
        ASSERT_EQ(logTileCopy.args.size(), 3);

        // 验证数据类型是否一致
        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

// Testcase, from nZ -> nZ (AtlasA2), basic场景
TEST_P(TileCopyL1ToL0BTest, nZTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    // call CopyL1ToL0B
    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true/*inner trans*/>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    // call copyL1ToL0B
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    // 日志获取
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();  // 获取所有调用日志
    AscendCCallLog logTileCopy = logs[0];

    // AtlasA2实现下包括多次搬运
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (int i=0; i<_rows_by_fractal; i++){
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        // 验证输入输出张量的数据偏移
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal);
        
        // 验证LoadData2DParams参数是否正确
        const AscendC::LoadData2DParams* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);
        ASSERT_EQ(loadDataArg->repeatTimes, _cols_by_fractal);
        ASSERT_EQ(loadDataArg->srcStride, _1);
        ASSERT_EQ(loadDataArg->sid, _0);
        ASSERT_EQ(loadDataArg->dstGap, _0);
        ASSERT_EQ(loadDataArg->ifTranspose, _0);
        ASSERT_EQ(loadDataArg->addrMode, _0);
    }
}

// Testcase, from zZ -> nZ (AtlasA2), Basic场景
TEST_P(TileCopyL1ToL0BTest, zZTonZTestBasic)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zZ;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;
    static_assert(!std::is_same_v<Element, float>, "This testcase(zZTonZTestBasic) do not support when Element is float");
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    // call CopyL1ToL0B
    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    
    // 准备执行
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    // 日志获取
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();  // 获取所有调用日志

    // AtlasA2实现下包括多次搬运
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (int i = 0; i < _rows_by_fractal; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        // 验证输入输出地址是否一致
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal); // DST
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal); // SRC

        // 验证LoadData2DParams参数是否正确
        
        const AscendC::LoadData2DParams* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);                 // 分形矩阵起始(为0)
        ASSERT_EQ(loadDataArg->repeatTimes, _cols_by_fractal);  // 搬运的分型迭代次数（每个迭代处理512B数据)
        ASSERT_EQ(loadDataArg->srcStride, _1);                  // 相邻分形间前一个分形和后一分形起始地址间隔(单位:512B)
        ASSERT_EQ(loadDataArg->dstGap, _0);                      // 相邻迭代间目的操作数前一个至后一个分形起始间隔(单位:512B) 分形间大Z方向相同
        ASSERT_EQ(loadDataArg->ifTranspose, _1);                 // 相邻迭代间目的操作数前一个至后一个分形起始间隔(单位:512B)
        ASSERT_EQ(loadDataArg->addrMode, _0);                   // 预留参数，置0
    }
}

// Testcase, from zZ -> nZ (AtlasA2), float specialization
TEST_P(TileCopyL1ToL0BTest, zZTonZTestFloat)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zZ;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;

    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element, true>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(1) * 2 * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
        ASSERT_EQ(p->startIndex, static_cast<uint16_t>(0));
        ASSERT_EQ(p->repeatTimes, static_cast<uint8_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1))));
        ASSERT_EQ(p->srcStride, static_cast<uint16_t>(1));
        ASSERT_EQ(p->dstGap, static_cast<uint16_t>(0));
        ASSERT_EQ(p->dstFracGap, static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)) - 1));
        ASSERT_EQ(p->addrMode, static_cast<uint8_t>(0));
    }
}

// zNTozNTestBasic (#5): 3-param zN(B1)→zN(B2), LoadData, no transpose
TEST_P(TileCopyL1ToL0BTest, zNTozNTestBasic)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    // loop count = layoutDst.shape(3)  (cols-by-fractal direction)
    ASSERT_EQ(logs.size(), layoutDst.shape(3));

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(3) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(3) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(p->startIndex, _0);
        ASSERT_EQ(p->repeatTimes, layoutDst.shape(1));
        ASSERT_EQ(p->srcStride, layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstGap, layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1);
        ASSERT_EQ(p->ifTranspose, _0);
        ASSERT_EQ(p->addrMode, _0);
    }
}

// zNTozNTestBasic2Param (#11): 2-param zN(A1)→nZ, LoadData, transpose
TEST_P(TileCopyL1ToL0BTest, zNTozNTestBasic2Param)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CopyL1ToL0B<ArchTag, L1Type> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(1) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(p->startIndex, _0);
        ASSERT_EQ(p->repeatTimes, CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        ASSERT_EQ(p->srcStride, layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL);
        ASSERT_EQ(p->dstGap, layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1);
        ASSERT_EQ(p->ifTranspose, _1);
        ASSERT_EQ(p->addrMode, _0);
    }
}

// nNTozNTestBasic (#6): 3-param nN(B1)→zN(B2), LoadData single call, transpose
TEST_P(TileCopyL1ToL0BTest, nNTozNTestBasic)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nN;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    // nN→zN: single LoadData call (no loop)
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
    ASSERT_EQ(p->startIndex, _0);
    ASSERT_EQ(p->repeatTimes, layoutDst.shape(1) * layoutDst.shape(3));
    ASSERT_EQ(p->srcStride, layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL);
    ASSERT_EQ(p->dstGap, layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1);
    ASSERT_EQ(p->ifTranspose, _1);
    ASSERT_EQ(p->addrMode, _0);
}

// nZTozNTestBasic2Param (#13): 2-param nZ(A1)→nZ, LoadData, no transpose (branch)
TEST_P(TileCopyL1ToL0BTest, nZTozNTestBasic2Param)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CopyL1ToL0B<ArchTag, L1Type> copyL1ToL0B;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    // nZ→nZ with same shape: single LoadData call
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
    ASSERT_EQ(p->startIndex, _0);
    ASSERT_EQ(p->repeatTimes, layoutDst.shape(1) * layoutDst.shape(3));
    ASSERT_EQ(p->srcStride, _1);
    ASSERT_EQ(p->dstGap, _0);
    ASSERT_EQ(p->ifTranspose, _0);
    ASSERT_EQ(p->addrMode, _0);
}

INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0B,
    TileCopyL1ToL0BTest,
    ::testing::Values(
        TestMatrixShape{128U, 64U},   // aligned
        TestMatrixShape{256U, 128U},  // aligned
        TestMatrixShape{64U, 64U},    // aligned
        TestMatrixShape{128U, 128U}   // aligned
    )
);

#endif // CATLASS_ARCH == 2201
