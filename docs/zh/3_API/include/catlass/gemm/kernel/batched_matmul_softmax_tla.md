# BatchedMatmulSoftmaxTla

> 代码路径：`include/catlass/gemm/kernel/batched_matmul_softmax_tla.hpp`
>

## 功能说明

`BatchedMatmulSoftmaxTla` 是批量矩阵乘与逐行 Softmax 的融合 MIX kernel 模板。

执行方式为：

1. AIC 调用 BatchedMatmul TLA kernel 计算 FP16 logits，并写入输出地址。
2. AIC 完成后设置跨核同步标志。
3. AIV 等待同步标志后，对输出地址中的 logits 原地执行逐行 Softmax。

该模板不需要额外 logits workspace，适用于 `S = row_softmax(A * B)` 的
FP16 融合场景。

## 模板参数

```cpp
template <
    class MatmulKernel_,
    class LayoutTagA_,
    class LayoutTagB_,
    class LayoutTagC_>
class BatchedMatmulSoftmaxTla;
```

- `MatmulKernel_`：AIC 侧计算 FP16 logits 的 BatchedMatmul kernel。
- `LayoutTagA_`、`LayoutTagB_`、`LayoutTagC_`：A、B、C 的布局标签。

## Arguments 关键字段

```cpp
struct Arguments {
    uint32_t batch;
    GemmCoord problemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;
};
```

其中：

- `problemShape` 表示单个 batch 内的 `M/N/K`。
- `ptrA`、`ptrB`、`ptrC` 分别为 A、B、C 的 GM 起始地址。

## Params 关键字段

```cpp
struct Params {
    uint32_t batch;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;
};
```

`ToUnderlyingArguments()` 将 `Arguments` 转换为上述 `Params`，供 device
kernel 调用。

## Host Helper

- `SelectBatchedMatmulSoftmaxBlockNum()`：按输出 tile 数选择 16、24 或
  全部 AIC core。

Softmax 的行 tile 由 epilogue 在 device 侧按运行时的 `N` 与剩余 UB 推导，
因此 `Arguments` 无需额外传入 tiling 配置。

## 适用条件

- A 为 FP16 RowMajor，B 为 FP16 ColumnMajor，C 为 FP16 RowMajor。
- `batch`、`M`、`N`、`K` 均大于 0。
- A、B、C 地址均非空。

完整实例见 `experimental/matmul/87_ascend950_batched_matmul_softmax/`。
