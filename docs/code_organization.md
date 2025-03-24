# 代码组织结构
## 昇腾算子模板库代码组织结构
这篇文档描述了昇腾算子模板库的代码仓结构，主要包含的内容如下：
- include包含每层分层的代码头文件
- examples包含基于模板库的算子编程代码示例和相关组件
- docs包含昇腾算子模板库的相关介绍文档
- scripts包含模板库样例的构建脚本
## include
昇腾算子模板库是一套基于AscendC开发的算子模板库，提供昇腾硬件上Matmul算子定制化开发的极致性能。代码仓提供的能力对标硬件的不同层级展开。block层对应于NPU的单核单基块的层级，tile层对应于分片粒度的数据搬运和计算的层级，basic对应于基础API的层级。这些组件可以在相应的算子内的不同运算层级被使用。
include目录下的头文件是按照如下的文件层级进行组织的。
```
├── include
│   └── AscendCT
│       ├── AscendCT.hpp // 定义了基本的数据信息，如基本块长度等
│       ├── arch
│       │   ├── arch.hpp // 定义了架构相关的基本数据信息，如L1/L0大小，UB大小等
│       │   ├── cross_core_sync.hpp // 核间同步操作
│       │   ├── local_tensor_buffer.hpp // 定义了LocalTensorBuffer相关的操作，包括内存管理的初始化，内存申请和释放的接口
│       │   └── resource.hpp // 包含Tpipe和核内的各层级内存资源
│       ├── coord.hpp // 定义了Coord结构体，用于tiling的坐标运算
│       ├── detail
│       │   ├── alignment.hpp // 用于对齐计算的基本函数，如RoundUp，CeilDiv等
│       │   ├── dependent_false.hpp // 用于static_assert的检查函数的信息
│       │   └── macros.hpp // 辅助宏的定义
│       ├── epilogue
│       │   ├── block
│       │   │   ├── block_epilogue.hpp // block层级后处理模板定义
│       │   │   └── block_epilogue_elemwise_one_source.hpp // 带有add操作的后处理模板实现
│       │   ├── dispatch_policy.hpp // dispatch policy的定义
│       │   └── tile
│       │       ├── copy_gm_to_ub.hpp // tile层gm到ub数据搬运操作实现
│       │       ├── copy_ub_to_gm.hpp // tile层ub到gm数据搬运操作实现
│       │       ├── tile_copy.hpp  // tile层搬运接口定义
│       │       └── tile_elemwise_add.hpp // tile层add计算操作实现
│       ├── layout
│       │   ├── layout.hpp // layout头文件，主要包含matrix和vector相关layout的定义
│       │   ├── matrix.hpp // 包含矩阵运算的layout的定义
│       │   └── vector.hpp // vector相关的layout定义
│       ├── gemm
│       │   ├── block
│       │   │   ├── block_mmad_pingpong.hpp // block层的模板实现，包括doublebuffer的相应实现
│       │   │   ├── block_mmad.hpp // block层的模板定义
│       │   │   └── block_swizzle.hpp // 包含swizzle计算的相关实现
│       │   ├── dispatch_policy.hpp // policy流水的定义，包含archtag，流水阶段，UNIF_FLAG配置
│       │   ├── helper.hpp // 包含alignhelper的辅助函数的定义
│       │   ├── kernel
│       │   │   ├── batched_matmul.hpp // kernel层的batchedMatmul实现
│       │   │   ├── grouped_matmul.hpp // kernel层的groupedMatmul实现
│       │   │   ├── matmul_epilogue.hpp // kernel层的带后处理的matmul（如matmul+add）的实现
│       │   │   ├── basic_matmul.hpp // kernel层的matmul实现
│       │   │   └── padding_matmul.hpp  // kernel层的paddingMatmul实现
│       │   ├── GemmType // GemmType的定义
│       │   └── tile
│       │       ├── copy_gm_to_l1.hpp // tile层GM到L1搬运的具体实现
│       │       ├── copy_l0c_to_gm.hpp // tile层L0c到GM搬运的具体实现
│       │       ├── copy_l1_to_l0.hpp // tile层L1到L0搬运的具体实现
│       │       ├── tile_copy.hpp // tilecopy的头文件定义
│       │       └── tile_mmad.hpp // tile_mmad对基础api的封装
│       ├── gemm_coord.hpp // 矩阵运算的基础坐标运算封装
│       └── matrix_coord.hpp
├── CMakeLists.txt
├── LICENSE
└── README.md
```
## examples
examples文件夹下提供了当前基于分层组件所构建的示例，展示如何基于基础组件搭建一个matmul算子。
```
├── examples
│   ├── 00_basic_matmul
│   │   ├── CMakeLists.txt   // 编译文件
│   │   └── basic_matmul.cpp // 基础的matmul构建示例
│   ├── 01_batched_matmul
│   ├── 02_grouped_matmul
│   ├── 03_matmul_add
│   ├── 04_padding_matmul
│   ├── 05_fa
│   ├── CMakeLists.txt
│   └── common               // 包含辅助函数
```
## docs
docs文件夹下包含项目的所有文档。
```
├── docs
│   ├── images                 // 文档图片
│   ├── recommended_configuration.md // 算子性能最佳的推荐配置介绍
│   ├── api.md                 // 昇腾算子模板库api介绍
│   ├── code_organization.md   // 代码组织结构介绍
│   ├── quickstart.md          // 快速上手指南
│   └── swizzle_explanation.md // swizzle使用介绍
```
## scripts
scripts文件夹下包含样例构建脚本。
```
├── scripts
│   └── build.sh // 基础构建脚本
```
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