# 快速上手指南
## 环境准备
环境配套信息，可查看README中[软件硬件配套说明](../README.md#软件硬件配套说明)。

下载CANN开发套件包，点击[下载链接](https://www.hiascend.com/zh/developer/download/community/result?module=cann)选择对应的开发套件包`Ascend-cann-toolkit_<version>_linux-<arch>.run`。 CANN开发套件包依赖固件驱动，如需安装请查阅[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)页面。

安装CANN开发套件包。以下为root用户默认路径安装演示。
```
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
```
设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 使用ASCENDC Templates开发Matmul算子
本示例主要展示如何基于ASCENDC Templates快速搭建一个NPU上的BasicMatmul实现。示例中使用已提供的下层基础组件完成Device层和Kernel层组装，并调用算子输出结果。ASCENDC Templates分层示意图见[api文档](api.md)。
### Kernel层算子定义
Kernel层模板由Block层组件构成。这里首先定义三个Block层组件。
`<class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>`。
1. `BlockMmad_`为block层mmad计算接口，定义方式如下：
```
using DispatchPolicy = Act::Gemm::MmadAtlasA2Pingpong<true>; //流水排布使用
using L1TileShape = Act::GemmShape<128, 256, 256>; // L1基本块
using L0TileShape = Act::GemmShape<128, 256, 64>; // L0基本块
using AType = Act::Gemm::GemmType<ElementA, LayoutA>;     //封装了A矩阵的数据类型和排布信息
using BType = Act::Gemm::GemmType<ElementB, LayoutB>;     //封装了B矩阵的数据类型和排布信息
using CType = Act::Gemm::GemmType<ElementC, LayoutC>;     //封装了C矩阵的数据类型和排布信息

using BlockMmad = Act::Gemm::Block::BlockMmad<DispatchPolicy,
    L1TileShape,
    L0TileShape,
    AType,
    BType,
    CType>;
```
2. `BlockEpilogue_`为block层后处理，本文构建基础matmul，不涉及后处理，这里传入void。
```
using BlockEpilogue = void;
```
3. `BlockScheduler_`该模板类定义数据走位方式，提供计算offset的方法。此处使用定义好的GemmIdentityBlockSwizzle。参考[Swizzle策略说明](swizzle_explanation.md)文档了解更多swizzle信息。
```
using BlockScheduler = typename Act::Gemm::Block::GemmIdentityBlockSwizzle<>;
```
4. 基于上述组件即可完成BasicMatmul示例的Kernel层组装。
```
using MatmulKernel = Act::Gemm::Kernel::BasicMatmul<BlockMmad, void, TileSchedule>;
```
### Device层算子定义
基于Kernel层组装的算子，完成核函数的编写。
1. 使用ACT_GLOBAL修饰符定义Matmul函数，并传入算子的类型参数。
```
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACT_GLOBAL
void BasicMatmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC);
```
2. BasicMatmul的调用接口为`()`运算符，需要传入Params作为参数。
```
typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
```
3. 最后，实例化一个kernel，并执行该算子。
```
MatmulKernel matmul;
matmul(params);
```
### 算子调用
调用算子我们需要指定矩阵的输入输出的数据类型和数据排布信息，并使用`<<<>>>`的方式调用核函数。
```
BasicMatmul<<<BLOCK_NUM, nullptr, stream>>>(
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
```
### 算子编译
使用cmake，调用`act_example_add_executable`函数指定target名称和编译文件。如下所示，00_basic_matmul为target名称，basic_matmul.cpp为需要编译的文件。
```
# CMakeLists.txt
act_example_add_executable(
    00_basic_matmul
    basic_matmul.cpp
)
```
在项目目录下，调用`build.sh`，即可编译examples中的kernel代码。
```
# 编译examples内所有用例
bash scripts/build.sh act_examples
# 编译指定用例
bash scripts/build.sh 00_basic_matmul
```
### 算子执行
切换到可执行文件的编译目录`build/bin`下，执行算子样例程序。
```
cd build/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```
执行结果如下，表明基于ASCENDC Templates编写的Kernel已经成功执行。
```
Compare success.
```
### 代码样例
完整的基础matmul样例参照[examples/00_basic_matmul/basic_matmul.cpp](../examples/00_basic_matmul/basic_matmul.cpp)。
该示例支持A/B矩阵为rowMajor数据排布输入。

## 版权声明
Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

## 许可证
[CANN Open Software License Agreement Version 1.0](../LICENSE)