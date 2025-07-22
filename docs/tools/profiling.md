# 在CATLASS样例工程使用Profiling

Profiling工具提供了AI任务运行性能数据、昇腾AI处理器系统数据等性能数据的采集和解析能力。

其中，msprof采集通用命令是性能数据采集的基础，用于提供性能数据采集时的基本信息，包括参数说明、AI任务文件、数据存放路径、自定义环境变量等。

- 提示：CANN提供如下两套性能采集工具
  - [msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/optool/atlasopdev_16_0082.html)是单算子性能分析工具，对应的指令为`msprof op`或`msopprof`
  - [Profiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/devaids/Profiling/atlasprofiling_16_0010.html)是整网性能分析工具，对应的指令为`msprof`

- 本文档介绍的是**后者**。

# 使用示例

下面以对`00_basic_matmul`为例，进行Profiling的使用说明。

1. 基于[快速上手](../README.md#快速上手)，增加编译选项`--enable_profiling`， 使能`profiling api`编译算子样例。

```bash
bash scripts/build.sh --enable_profiling 00_basic_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，用`msprof`执行算子样例程序。

```bash
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID（可选）
msprof ./00_basic_matmul 256 512 1024 0
```

## op_summary结构

```bash
├──aicore_time  # 算子在AIcore中的计算时间
├──memory_bound # 内存带宽
├──mte2_ratio   # MTE2搬运单元的时间比例
├──mte2_time    # MTE2搬运单元的时间
├──mte3_ratio   # MTE3搬运单元的时间比例
├──mte3_time    # MTE3搬运单元的时间
├──scalar_ratio # Scalar计算单元的时间比例
├──scalar_time  # Scalar计算单元的时间
├──vec_ratio    # Vector计算单元的时间比例
└──vec_time     # Vector计算单元的时间
```
