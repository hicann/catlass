# CATLASS

## 📌 简介

CATLASS(**CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**)，中文名为昇腾算子模板库，是一个聚焦于提供高性能矩阵乘类算子基础模板的代码库。  

通过抽象分层的方式将矩阵类算子代码模板化。算子计算逻辑可以进行白盒化组装，让算子代码可复用，可替换，可局部修改。针对昇腾硬件特点进行设计，可以支持复杂场景流水排布，如FA等。在上层代码逻辑共享的同时，可以支持底层硬件差异特化。

本代码仓为CATLASS联创代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型算子的高性能实现代码样例。

## 🧩 模板分层设计

![api_level](docs/images/api_level.png)

分层详细介绍和各层级api，见[api](docs/api.md)文档。

## 📁 目录结构说明

```bash
catlass
├── cmake       # cmake工程文件
├── docs        # 文档
├── examples    # kernel使用样例
├── include     # 模板头文件
├── scripts     # 编译脚本
└── tests       # 测试用例
```

## 💻 软硬件配套说明

- 硬件平台：
  - **CPU**: `aarch64`/`x86_64`
  - **NPU**: `Atlas A2 训练系列产品`/`Atlas 800I A2 推理产品`/`A200I A2 Box 异构组件``
    - `Atlas 800T A2 训练服务器`
    - `Atlas 900 A2 PoD 集群基础单元`
    - `Atlas 200T A2 Box16 异构子框`
    - `Atlas 800I A2 推理服务器`
    - `A200I A2 Box 异构组件`

- 软件版本：
  - `gcc >= 7.5`（已测试`7.5`，`8.3`，`9.3`，`11.4`，建议使用9.3以上版本。）
  - `cmake >= 3.15`
  - `python >= 3.10`

- CANN版本：

| CANN包类别 | 版本要求                    | 获取方式                                                                                                             |
| ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 社区版     | 8.2.RC1.alpha002 及之后版本 | [社区CANN包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002) |
| 商用版     | 8.1.RC1及之后版本           | 请咨询对应Support/SupportE获取                                                                                       |

- 对于某些调测工具，可能需要较以上版本更加新的CANN版本，可参考[调测工具文档](#toolbox)。

## 🚀 快速上手

以`00_basic_matmul`算子样例为例，快速上手CATLASS算子开发：

1. 使能CANN环境变量

```bash
# root用户安装（默认路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 编译算子样例

```bash
bash scripts/build.sh 00_basic_matmul
```

3. 执行算子样例
切换到可执行文件的编译目录`output/bin`下，执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```

出现`Compare success.`打屏，说明算子运行成功，精度比较通过。

## 📚 文档介绍

### 📖 API文档

- [api](./docs/api.md) - CATLASS通用矩阵乘法Gemm API的描述。
- [dispatch_policies](./docs/dispatch_policies.md) - BlockMmad一个重要模板参数`DispatchPolicy`的描述。
- [quickstart](./docs/quickstart.md) - 模板库的快速开始。
- [swizzle_explanation](./docs/swizzle_explanation.md) - AI Core计算基本块的顺序之Swizzle策略的描述。

### 🧰 调测工具文档 <span id="toolbox"></span>

我们已经在CATLASS示例工程中适配了大多数CANN提供的调测工具，开发算子时，可基于CATLASS示例工程进行初步开发调优，无需关注具体的工具适配操作，待算子基础功能、性能达到预期，再迁移到其他工程中。

#### 🚗 功能调试

- [msDebug](./docs/tools/msdebug.md) - 类gdb/lldb的调试工具msDebug
  - ⚠️ **注意** 这个功能依赖于[8.2.RC1.alpha003](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003)版本的社区版或`8.2.RC1`之后的商用版。
- [printf](./docs/tools/print.md) - 在算子device代码进行打印调试
  - ⚠️ **注意** 这个功能将在未来的`CANN 8.3`开始支持。

#### ✈️ 性能调优

- [msProf&Profiling](./docs/tools/performance_tools.md) - 性能调优工具`msProf`和`Profiling`
  - [单算子性能分析：msProf](./docs/tools/performance_tools.md#用msProf进行单算子性能分析)
  - [整网性能分析：Profiling](./docs/tools/performance_tools.md#用Profiling进行整网性能分析)

## 👥 合作贡献者

[华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

## 🔒 安全声明

[CATLASS仓库 安全声明](./SECURITYNOTE.md)

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
