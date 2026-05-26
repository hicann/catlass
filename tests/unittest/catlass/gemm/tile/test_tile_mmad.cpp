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
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "stub/ascendc_logger.h"

using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;

// 定义TileMmadTest测试类，继承自AscendCTest测试框架基类
class TileMmadTest : public AscendCTest {
protected:
    // 重写SetUp方法，在每个测试用例执行前进行初始化操作
    void SetUp() override
    {
        // 调用父类的SetUp方法完成基础初始化
        AscendCTest::SetUp();
    }
};

// 测试用例：基础Mmad功能测试（无偏置）
TEST_F(TileMmadTest, BasicMmad)
{
    // 定义数据类型和架构标签
    using ElementMmad = float;                  // 定义计算数据类型为float
    using ArchTag = Catlass::Arch::AtlasA2;     // 指定目标架构为AtlasA2
    using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;  // 定义矩阵A的类型（行优先布局）
    using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;  // 定义矩阵B的类型（行优先布局）
    using BiasType = void;                      // 无偏置，使用void类型

    // 创建TileMmad实例，指定架构标签和矩阵类型
    TileMmad<ArchTag, AType, BType, BiasType> tileMmad;

    // 创建本地张量对象，用于存储输入输出数据
    AscendC::LocalTensor<ElementMmad> l0CTensor;  // 输出矩阵C的本地张量
    AscendC::LocalTensor<ElementMmad> l0ATensor;  // 输入矩阵A的本地张量
    AscendC::LocalTensor<ElementMmad> l0BTensor;  // 输入矩阵B的本地张量

    // 设置GEMM计算的维度参数（m=128, n=128, k=128）
    int m = 128;  // 矩阵A的行数/C的行数
    int n = 128;  // 矩阵B的列数/C的列数
    int k = 128;  // 矩阵A的列数/B的行数

    // 调用TileMmad的计算接口，执行矩阵乘法操作
    tileMmad(l0CTensor, l0ATensor, l0BTensor, m, n, k);

    // 获取AscendC调用日志记录器实例，用于验证底层调用是否符合预期
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();  // 获取所有调用日志
    AscendCCallLog logMmad = logs[0];  // 获取第一条日志（预期为Mmad调用）

    // 验证日志数量：应只有1条Mmad调用记录
    ASSERT_EQ(logs.size(), 1);
    // 验证调用名称：应为"Mmad"
    ASSERT_EQ(logMmad.name, "Mmad");
    // 验证参数数量：Mmad无偏置版本应包含4个参数（C, A, B, MmadParams）
    ASSERT_EQ(logMmad.args.size(), 4);

    // 验证输入输出张量的地址是否匹配
    auto logMmadL0CTensor = logMmad.GetArgsAt(0).RawValue();  // 获取日志中C张量地址
    auto logMmadL0ATensor = logMmad.GetArgsAt(1).RawValue();  // 获取日志中A张量地址
    auto logMmadL0BTensor = logMmad.GetArgsAt(2).RawValue();  // 获取日志中B张量地址
    ASSERT_EQ(logMmadL0CTensor, &l0CTensor);  // 验证C张量地址一致
    ASSERT_EQ(logMmadL0ATensor, &l0ATensor);  // 验证A张量地址一致
    ASSERT_EQ(logMmadL0BTensor, &l0BTensor);  // 验证B张量地址一致

    // 验证MmadParams中的维度参数是否正确
    const AscendC::MmadParams* arg3 = logMmad.GetArgsAt(3).Value<AscendC::MmadParams>();  // 获取MmadParams参数
    ASSERT_EQ(arg3->m, m);  // 验证m维度
    ASSERT_EQ(arg3->n, n);  // 验证n维度
    ASSERT_EQ(arg3->k, k);  // 验证k维度

    // 验证数据类型是否正确
    const std::type_index& T0 = logMmad.GetArgsTAt(0).Type();  // 获取第一个参数的数据类型
    ASSERT_EQ(T0, typeid(ElementMmad));  // 验证数据类型为float
}

// 测试用例：带偏置的Mmad功能测试
TEST_F(TileMmadTest, MmadWithBias)
{
    // 定义数据类型和架构标签
    using ElementMmad = float;                  // 定义计算数据类型为float
    using ArchTag = Catlass::Arch::AtlasA2;     // 指定目标架构为AtlasA2
    using AType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;  // 定义矩阵A的类型（行优先布局）
    using BType = Catlass::Gemm::GemmType<ElementMmad, Catlass::layout::RowMajor>;  // 定义矩阵B的类型（行优先布局）
    using BiasType = ElementMmad;               // 定义偏置类型为float
    using TileMmadTest = TileMmad<ArchTag, AType, BType, BiasType>;  // 定义带偏置的TileMmad类型

    // 创建带偏置的TileMmad实例
    TileMmadTest tileMmad;

    // 创建本地张量对象，用于存储输入输出数据（包含偏置张量）
    AscendC::LocalTensor<ElementMmad> l0CTensor;    // 输出矩阵C的本地张量
    AscendC::LocalTensor<ElementMmad> l0ATensor;    // 输入矩阵A的本地张量
    AscendC::LocalTensor<ElementMmad> l0BTensor;    // 输入矩阵B的本地张量
    AscendC::LocalTensor<ElementMmad> l0BiasTensor; // 偏置张量

    // 调用TileMmad的计算接口，执行带偏置的矩阵乘法操作（m=128, n=128, k=128）
    tileMmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, 128, 128, 128);

    // 获取AscendC调用日志记录器实例，验证底层调用是否符合预期
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();  // 获取所有调用日志

    // 验证日志数量：应只有1条Mmad调用记录
    ASSERT_EQ(logs.size(), 1);
    // 获取第一条日志（预期为Mmad调用）
    AscendCCallLog logMmad = logs[0];
    // 验证调用名称：应为"Mmad"
    ASSERT_EQ(logMmad.name, "Mmad");
    // 验证参数数量：Mmad带偏置版本应包含5个参数（C, A, B, Bias, MmadParams）
    ASSERT_EQ(logMmad.args.size(), 5);

    // 验证输入输出张量及偏置张量的地址是否匹配
    auto logMmadL0CTensor = logMmad.GetArgsAt(0).RawValue();    // 获取日志中C张量地址
    auto logMmadL0ATensor = logMmad.GetArgsAt(1).RawValue();    // 获取日志中A张量地址
    auto logMmadL0BTensor = logMmad.GetArgsAt(2).RawValue();    // 获取日志中B张量地址
    auto logMmadL0BiasTensor = logMmad.GetArgsAt(3).RawValue(); // 获取日志中偏置张量地址
    ASSERT_EQ(logMmadL0CTensor, &l0CTensor);  // 验证C张量地址一致
    ASSERT_EQ(logMmadL0ATensor, &l0ATensor);  // 验证A张量地址一致
    ASSERT_EQ(logMmadL0BTensor, &l0BTensor);  // 验证B张量地址一致
    ASSERT_EQ(logMmadL0BiasTensor, &l0BiasTensor); // 验证偏置张量地址一致
    
    // 验证MmadParams中的维度参数是否正确
    const AscendC::MmadParams* arg4 = logMmad.GetArgsAt(4).Value<AscendC::MmadParams>();  // 获取MmadParams参数
    ASSERT_EQ(arg4->m, 128);  // 验证m维度
    ASSERT_EQ(arg4->n, 128);  // 验证n维度
    ASSERT_EQ(arg4->k, 128);  // 验证k维度

        // 验证数据类型是否正确
    const std::type_index& T0 = logMmad.GetArgsTAt(0).Type();  // 获取第一个参数的数据类型
    ASSERT_EQ(T0, typeid(ElementMmad));  // 验证数据类型为float
}
