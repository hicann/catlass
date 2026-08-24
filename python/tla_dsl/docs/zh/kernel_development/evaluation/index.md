---
nav_order: 20
---

# 调试与优化

本文档介绍 CATLASS DSL 算子开发中的通用调试调优手段。

## 调试思路

建议遵循“先功能、后性能”的开发流程：

1. **功能验证**：用较小的 shape 先跑通编译与执行，通过打印输出确认数据流与控制流正确。
2. **性能采集**：功能正确后，用性能分析工具采集算子耗时、流水利用与内存搬运等指标。
3. **瓶颈定位**：结合指标与 kernel 结构定位瓶颈，逐项优化后复测。

## 功能调试

- [打印调试](print.md)：使用 `tla.print` 在 kernel 内部打印标量、格式化字符串与 Tensor，适合定位运行时数据流与控制流问题。
- 该方案无需额外工具，编译运行即可查看输出，是功能调试的首选手段。

## 性能调优

- [性能分析工具](performance_tools.md)：基于 [msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/Profiling/atlasprofiling_16_0010.html)（单算子）与 [Profiling](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlasag_25_0056.html)（整网）采集并分析算子性能
  - [单算子性能分析：msProf](performance_tools.md#用msprof进行单算子性能分析)
  - [整网性能分析：Profiling](performance_tools.md#用profiling进行整网性能分析)
