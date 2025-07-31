# 更新日志


## Catlass [1.1.0](https://gitee.com/Ascend/catlass/releases/tag/v1.1.0) (2025-07-31)


- 新增Example示例
  - **新增** 20_matmul_bias
  - **新增** 21_basic_matmul_preload_zN（科大讯飞联创贡献）
  - **新增** 22_padding_splik_matmul（科大讯飞联创贡献）
  - **新增** python_extension
  - **新增** shared_lib
### 新增特性
  - **新增** [matmul_bias Kernel层](include/catlass/gemm/kernel/matmul_bias.hpp)
  - **优化** `OptimizedMatmul`不padding时不启动AIV
  - **优化** 所有Kernel添加`PIPE_ALL`，防止整网影响
  - **新增** 支持float类型矩阵乘
### 资料与工具
- **新增** [tutorials快速上手示例](docs/tutorials.md)
- CATLASS工程适配下列调测工具，并提供基础使用文档
  - [ascendc_dump](docs/tools/ascendc_dump.md)
  - [print](docs/tools/print.md)
  - [msprof](docs/tools/performance_tools.md#上板性能采集)
  - [msprof simulator](docs/tools/performance_tools.md#性能流水仿真)
  - [profiling](docs/tools/performance_tools.md#msprof使用示例)
- 将毕昇编译器适配至CMake工程，整改CMake编译脚本为标准的CMake函数调用
### BugFix
- 修复block_mmad预加载nextBlock时的引用错误
- 隔离Kernel侧`AscendC`的`inline`定义，避免异构编程时无法使用部分标准库
- 修改l2offset设置的重定义问题
### 测试
- 增加头文件自包含测试
- 其他
  - **优化** 使用非毕昇编译器时，将CATLASS_GLOBAL宏的定义清空，使得部分CATLASS结构体可以在纯Host代码使用，提升Tiling代码开发效率
  - **整改** 整改CMake工程，支持bisheng编译器；解决安全编译问题





## [1.0.0](https://gitee.com/Ascend/catlass/releases/tag/v1.0.0) (2025/05/23)

- `Device` `Kernel`-`Block`-`Tile`-`Basic`四层分层编程框架
- 提供`matmul`/`grouped_matmul`/`mla`等共20个算子示例
- 提供shared_lib动静态库接入工程，pybind/torchscripts接入工程
- 基础用例测试

## ©️ 版权声明

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.  
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").  
Please refer to the License for details. You may not use this file except in compliance with the License.  

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.  
See LICENSE in the root of the software repository for the full text of the License.

## 📜 许可证

[CANN Open Software License Agreement Version 1.0](LICENSE)
