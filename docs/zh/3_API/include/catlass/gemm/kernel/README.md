# Gemm/Kernel 类模板概述

## API 清单

- [basic_matmul](./basic_matmul.md)：Common 模板基础矩阵乘。
- [batched_matmul_softmax_tla](./batched_matmul_softmax_tla.md)：批量矩阵乘
  与行 Softmax 融合模板。
- [basic_matmul_tla_visitor](./basic_matmul_tla_visitor.md)：EVG 的 GM
  workspace kernel 入口。
- [basic_matmul_tla_ub_visitor](./basic_matmul_tla_ub_visitor.md)：EVG 的 UB
  workspace kernel 入口。

## API 拆解

命名空间为`Catlass::Gemm::Kernel`，类模板包含如下核心成员：

- `Params`：kernel 通过 `<<< >>>` 调用核函数时的入参。
- `Arguments`：device 封装后 kernel 使用的入参。
- `bool CanImplement`：`Arguments` 校验接口。
- `size_t GetWorkspaceSize`：基于 `Arguments` 计算所需 workspace 大小。
- `Params ToUnderlyingArguments`：将 `Arguments` 转换为核函数入参
  `Params`。
- `void operator()<AscendC::AIC>`：输入 `Params`，完成 AIC 上的 MMAD
  计算。
- `void operator()<AscendC::AIV>`：输入 `Params`，完成 AIV 上的计算，
  如前处理、尾处理等。
