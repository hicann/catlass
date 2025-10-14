# CATLASS

## 🔥 Latest News

- [2025/09] CATLASS模板库正式开源

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
├── cmake          # cmake工程文件
├── docs           # 文档
├── examples       # kernel算子样例
├── include        # 模板头文件
├── scripts        # 编译脚本
|   └── build.sh   # 算子样例编译脚本
├── tests          # 测试用例
└── tools          # 相关工具
```

## 💻 软硬件配套说明

- 硬件平台：
  - **CPU**: `aarch64`/`x86_64`
  - **NPU**: `Atlas A2 训练系列产品`/`Atlas 800I A2 推理产品`/`A200I A2 Box 异构组件`
    - `Atlas 800T A2 训练服务器`
    - `Atlas 900 A2 PoD 集群基础单元`
    - `Atlas 200T A2 Box16 异构子框`
    - `Atlas 800I A2 推理服务器`
    - `A200I A2 Box 异构组件`

- 软件版本：
  - `gcc >= 7.5, < 13`（已测试`7.5`，`8.3`，`9.3`，`11.4`，建议使用9.3以上版本。）
  - `cmake >= 3.22`
  - `python >= 3.10`

- CANN版本：
  - 社区版`CANN`包（[8.2.RC1.alpha002](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002)及之后版本）


- 对于某些调测工具，可能需要较以上版本更加新的CANN版本，可参考[调测工具文档](#toolbox)。

## ⚡️ 快速上手

以[`00_basic_matmul`](examples/00_basic_matmul)算子样例为例，快速上手CATLASS算子开发：

1. 使能CANN环境变量
关于CANN环境准备请参考官网[安装说明](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

```bash
# root用户安装（默认路径）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 编译算子样例
在主目录下，执行下述指令。
```bash
bash scripts/build.sh 00_basic_matmul
```

3. 执行算子样例
切换到可执行文件的编译目录`output/bin`下，运行算子样例程序如下。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
./00_basic_matmul 256 512 1024 0
```

出现`Compare success.`打屏，说明算子运行成功，精度比较通过。

## 📚 文档介绍

### 📖 基础文档

按照由浅入深的次序，对模板库的相关内容展开介绍。

- [quickstart](./docs/quickstart.md) - 快速上手实践模板库，以基础的Matmul算子开发为实践背景认识使用模板库。
- [catlass_optimize_guidance](./docs/catlass_optimize_guidance.md) - 模板库的进阶教程，介绍模板库下的基础调优方式，如何通过Tiling调参、应用不同的Dispatch策略的方式，快速获得性能提升。
- [api](./docs/api.md) - 介绍CATLASS模板库的通用矩阵乘法Gemm API。
- [swizzle_explanation](./docs/swizzle_explanation.md) - 对模板库中Swizzle策略的基本介绍，这影响了AI Core上计算基本块间的顺序。
- [dispatch_policies](./docs/dispatch_policies.md) - 对模板库在`Block`层面上`BlockMmad`中的一个重要模板参数`DispatchPolicy`的介绍。

### 🧰 调测工具文档

我们已经在CATLASS示例工程中适配了大多数CANN提供的调测工具，开发算子时，可基于CATLASS示例工程进行初步开发调优，无需关注具体的工具适配操作，待算子基础功能、性能达到预期，再迁移到其他工程中。

#### 🚗 功能调试

- [msDebug](./docs/tools/msdebug.md) - 类gdb/lldb的调试工具msDebug
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本为[8.2.RC1.alpha003](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003)。
- [printf](./docs/tools/print.md) - 在算子device代码进行打印调试
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本在CANN 8.3后（如[8.3.RC1.alpha001](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha001)）。

#### ✈️ 性能调优

- [msProf&Profiling](./docs/tools/performance_tools.md) - 性能调优工具`msProf`和`Profiling`
  - [单算子性能分析：msProf](./docs/tools/performance_tools.md#用msProf进行单算子性能分析)
  - [整网性能分析：Profiling](./docs/tools/performance_tools.md#用Profiling进行整网性能分析)
- [msTuner_CATLASS](./tools/tuner/README.md) - Tiling自动寻优工具

## 👥 合作贡献者

### [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### 科大讯飞 研究院工程组

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITYNOTE.md)
- [许可证](LICENSE)