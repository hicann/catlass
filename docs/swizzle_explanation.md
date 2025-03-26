# Swizzle策略说明

Swizzle策略决定了当前AI Core以哪种顺序计算哪些基本块。调整Swizzle参数可能可以提高缓存命中率，从而减小数据读取开销，提高矩阵乘整体计算效率。

下面图中的序号表示AI Core序号，每一个方块表示C矩阵的一个基本块。箭头方向表示基本块的排列顺序，并不表示计算顺序，排列中的前20个基本块是同时开始计算的（这里假设AI Core数量为20）。

默认参数为SwizzleOffset=1，SwizzleDirection=0，如图所示：

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<>;
```

![图1](./images/swizzle10.png)

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
```
![图1](./images/swizzle30.png)

```c++
 using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
```
![图1](./images/swizzle31.png)

如果C矩阵的的大小为M x N，经验而言，当M >= N时，采用SwizzleOffset=3，SwizzleDirection=0，可以达到较好的矩阵计算性能。当M < N时，采用SwizzleOffset=3，SwizzleDirection=1，可以达到较好的矩阵计算性能。可以探索其他参数设置以达到更高的缓存命中率，从而进一步提高矩阵计算性能。

## 版权声明
Copyright (c) 2024 Huawei Technologies Co., Ltd.

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