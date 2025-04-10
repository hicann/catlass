# 代码组织结构
## Ascend C模板库代码组织结构
这篇文档描述了Ascend C模板库的代码仓结构，主要包含的内容如下：
- include包含模块分层的相关实现
- examples包含基于模板库的算子编程代码示例
- docs包含模板库的相关介绍文档
- scripts包含模板库样例的构建脚本
## include
Ascend C模板库是一套基于Ascend C开发的算子模板库，提供昇腾硬件Gemm类算子定制化开发的极致性能。模板库的分层对应硬件的不同层级展开。block层对应于NPU的单核单基块的层级，tile层对应于分片粒度的数据搬运和计算的层级，basic对应于基础API的层级。这些组件可以在相应的算子内的不同运算层级被使用。
include目录下的头文件是按照如下的文件层级进行组织的。
```
|── include
|    |── act
|        |── arch
|            |── arch.hpp // 定义了架构相关的基本数据信息，如L1/L0大小，UB大小等
|            |── cross_core_sync.hpp // 核间同步操作
|            |── local_tensor_buffer.hpp // 定义了LocalTensorBuffer相关的操作，包括内存管理的初始化，内存申请和释放的接口
|            |── resource.hpp   // 包含Tpipe和核内的各层级内存资源
|        |── detail
|            |── alignment.hpp   // 用于对齐计算的基本函数，如RoundUp，CeilDiv等
|            |── callback.hpp    // 回调函数
|            |── dependent_false.hpp  // 用于static_assert的检查函数的信息
|            |── macros.hpp         // 辅助宏的定义
|            |── tag_to_layout.hpp   // layout构造
|        |── epilogue
|            |── block
|                |── block_epilogue.hpp  //block层后处理操作
|                |── block_epilogue_elemwise_one_source.hpp  //带有add操作的后处理模板实现
|                |── block_epilogue_fa_rescal_o.hpp      //fa的recale_o后融合操作
|                |── block_epilogue_fa_softmax.hpp   //fa的softmax后融合操作
|                |── block_epilogue_gemm.hpp           //gemm的epilogue实现
|                |── block_epilogue_gemv.hpp           //gemv的epilogue实现
|                |── block_epilogue_mla_fd_rescal_o.hpp   //mla_fd的recale_o后融合操作
|                |── block_epilogue_mla_rescal_o.hpp // mla的rescale后融合操作
|                |── block_epilogue_mla_softmax.hpp  // mla的softmax后融合操作
|                |── block_epilogue_per_token_dequant.hpp   // per token dequant后融合操作
|            |── tile
|                |── copy_gm_to_ub.hpp      // tile层gm到ub数据搬运操作实现
|                |── copy_ub_to_gm.hpp      // tile层ub到gm数据搬运操作实现
|                |── tile_broadcast_inplace_by_column.hpp   // tile层将一列广播为矩阵
|                |── tile_broadcast_inplace_by_row.hpp     //  tile层将一行广播为矩阵
|                |── tile_broadcast_mul.hpp      //tile层mul计算广播操作
|                |── tile_broadcast_one_blk.hpp // tile层单基块广播操作
|                |── tile_cast.hpp          // cast操作后融合封装
|                |── tile_copy.hpp          // tile层搬运接口定义         
|                |── tile_elemwise_add.hpp  // tile层add计算操作实现
|                |── tile_elemwise_mul.hpp  // tile层mul计算操作实现
|                |── tile_elemwise_muls.hpp  // tile层muls计算操作实现
|                |── tile_swizzle.hpp       // tile层后融合swizzle操作
|            |── dispatch_policy.hpp
|        |── gemm
|            |── block
|                |── block_mmad.hpp              // block层的模板定义
|                |── block_mmad_fa_pv.hpp        // block层pv实现
|                |── block_mmad_fa_qk.hpp        // block层qk实现
|                |── block_mmad_gemm.hpp        // block层gemm实现
|                |── block_mmad_mla_pv.hpp       // block层mla pv实现
|                |── block_mmad_mla_qk.hpp       // block层mla qk实现
|                |── block_mmad_pingpong.hpp     // / block层的模板实现，包括doublebuffer的相应实现
|                |── block_mmad_pingpong_tla.hpp // block层基于tla的doublebuffer实现
|                |── block_mmad_preload.hpp        //block层preload实现
|                |── block_mmad_preload_async.hpp  //block层preload异步加载实现
|                |── block_mmad_preload_async_with_callback.hpp  // block层async_callback实现
|                |── block_mmad_preload_tla.hpp   // block层基于tla的preload实现
|                |── block_swizzle.hpp            // block层swizzle实现
|            |── kernel
|                |── basic_matmul.hpp             // kernel层basic_matmul
|                |── basic_matmul_tla.hpp         // kernel层基于tla的basic_matmul
|                |── batched_matmul.hpp           // kernel层batched_matmul
|                |── gemm.hpp                     // kernel层gemm实现
|                |── grouped_matmul.hpp           // kernel层grouped_matmul
|                |── grouped_matmul_slice_k.hpp   // kernel层k轴切分groupMatmul
|                |── grouped_matmul_slice_k_per_token_dequant.hpp // kernfel层k轴切分groupMatmul量化实现
|                |── grouped_matmul_slice_m.hpp  // kernel层m轴切分groupedMamtul
|                |── grouped_matmul_slice_m_per_token_dequant.hpp // kernel层m轴切分groupedMamtul量化实现
|                |── grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp // kernel层m轴切分groupmatmul多阶段量化实现
|                |── matmul_epilogue.hpp        // kernel层MatmulEpilogue实现 
|                |── optimized_matmul.hpp      // kernel层optimized_matmul实现
|                |── optimized_matmul_tla.hpp // kernel层基于tla的optimized matmul实现
|                |── padding_matmul.hpp       // kernel层padding matmul实现
|                |── quant_matmul.hpp         // kernel层量化matmul实现
|                |── quant_matmul_multistage_workspace.hpp // kernel层多阶段量化matmul实现
|                |── splitk_matmul.hpp        // kernel层splitk matmul实现
|            |── tile
|                |── copy_gm_to_l1.hpp        // tile层gm到l1搬运
|                |── copy_gm_to_ub.hpp        // tile层gm到ub搬运
|                |── copy_l0c_to_gm.hpp       // tile层l0c到gm搬运
|                |── copy_l1_to_l0a.hpp       // tile层l1到l0a搬运
|                |── copy_l1_to_l0b.hpp       // tile层l1到l0b搬运
|                |── copy_ub_to_gm.hpp        // tile层ub到gm搬运
|                |── tile_copy.hpp            // tile层copy封装
|                |── tile_mmad.hpp            // tile层mmad封装
|            |── dispatch_policy.hpp          // DispatchPolicy定义
|            |── gemm_type.hpp                // GemmType的定义
|            |── helper.hpp                   // 辅助函数
|        |── gemv
|            |── block
|                |── block_gemv.hpp           //gemv的block层实现
|                |── block_aic.hpp            //gemv_aic的block层实现
|                |── block_aiv.hpp            //gemv_aiv的block层实现
|            |── kernel
|                |── kernel_gemv_aic.hpp      //gemv_aic的kernel层实现
|                |── kernel_gemv_aiv.hpp      //gemv_aiv的kernel层实现
|            |── tile
|                |── matrix_copy_gm_to_ub.hpp //copy_gm_to_ub的tile层实现
|                |── tile_copy.hpp            //gemv的tile copy实现
|                |── tile_vmad.hpp            //gemv的tile vmad实现
|                |── tile_vmuls.hpp           //gemv的tile vmuls实现
|                |── vec_copy_gm_to_ub.hpp    //gemv的tile gm_to_ub实现
|                |── vec_copy_ub_to_gm.hpp    //gemv的tile ub_to_gm实现
|            |── helper.hpp                   // 辅助函数
|        |── layout
|            |── layout.hpp                   // layout头文件，主要包含matrix和vector相关layout的定义
|            |── matrix.hpp                   //包含矩阵运算的layout的定义
|            |── vector.hpp                   //vector相关的layout定义
|        |── act.hpp                 // 定义了基本的数据信息，如基本块长度等
|        |── coord.hpp
|        |── gemm_coord.hpp                  // gemm的基础坐标运算封装
|        |── gemv_coord.hpp                  // gemv的基础坐标运算封装
|        |── matrix_coord.hpp                // 矩阵运算坐标封装
|    |── tla
|        |── numeric
|            |── integer_sequence.hpp        // integer_sequence定义
|            |── integral_constant.hpp       // integer_constant定义
|            |── math.hpp                    // math相关计算
|        |── int_tuple.hpp                   // int_tupe定义
|        |── layout.hpp                      // layout定义
|        |── tensor.hpp                      // tensor封装定义
|        |── tuple.hpp                       // tuple定义
|        |── type_traits.hpp                 // type_traits定义
```
## examples
examples文件夹下提供了当前基于分层组件所构建的示例，展示如何基于基础组件搭建一个matmul算子。
```
├── examples
    |── 00_basic_matmul                 // 基础matmul 
    |── 01_batched_matmul               // 批处理matmul
    |── 02_grouped_matmul_slice_m       // grouped matmul m切分
    |── 03_matmul_add                   // matmul-add
    |── 04_padding_matmul               // padding-matmul
    |── 05_grouped_matmul_slice_k       // grouped matmul k切分
    |── 06_optimized_matmul             // 集成性能优化点matmul
    |── 07_grouped_matmul_slice_m_per_token_dequant_moe   // group matmul m切分量化,提供某些moe场景使用
    |── 08_grouped_matmul               // group matmul
    |── 09_splitk_matmul                // splitk优化 matmul
    |── 10_grouped_matmul_slice_m_per_token_dequant // group matmul m轴切分量化
    |── 11_grouped_matmul_slice_k_per_token_dequant // group matmul k轴切分量化
    |── 12_quant_matmul                // 量化matmul
    |── 13_basic_matmul_tla            // 基于tla的basic matmul
    |── 15_gemm                        // gemm模板样例实现
    |── 16_group_gemm                  // group_gemm模板样例实现
    |── 17_gemv_aiv                    // gemv_aiv模板样例实现
    |── 18_gemv_aic                    // gemv_aic模板样例实现
    |── common                         // 辅助函数
    |── python_extension               // python接入示例
    |── shared_lib                     // 静态编译接入示例
    |── CMakeLists.txt                 // CMake文件
```
## docs
docs文件夹下包含项目的所有文档。
```
├── docs
    |—— images
        |—— api_level.png         // 接口层级示例
        |—— swizzle10.png         // swizzle图示
        |—— swizzle30.png         // swizzle图示
        |—— swizzle31.png         // swizzle图示
    |—— tla
        |—— 01_layout.md          // tla-layout介绍
        |—— 02_tensor.md          // tla-tensor介绍
    |—— api.md                    // api接口介绍
    |—— code_organization.md      // 文件组织介绍
    |—— quickstart.md             // 搭建指南
    |—— swizzle_explanation.md         // swizzle解释
```
## scripts
scripts文件夹下包含样例构建脚本。
```
├── scripts
    |── build.sh           // 基础构建脚本
```
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