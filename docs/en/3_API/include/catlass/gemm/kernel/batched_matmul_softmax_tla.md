# BatchedMatmulSoftmaxTla
>
> Code path: `include/catlass/gemm/kernel/batched_matmul_softmax_tla.hpp`
>

## Description

`BatchedMatmulSoftmaxTla` is a fused MIX kernel template for batched matrix
multiplication followed by row-wise Softmax.

The execution workflow is as follows:

1. AIC invokes a BatchedMatmul TLA kernel to compute FP16 logits and writes
   them to the output address.
2. AIC sets a cross-core synchronization flag after the matmul stage is complete.
3. AIV waits for the synchronization flag and applies row-wise Softmax in
   place on the output logits.

This template does not require an extra logits workspace and is intended for
FP16 `S = row_softmax(A * B)` fusion scenarios.

## Template Parameters

```cpp
template <
    class MatmulKernel_,
    class LayoutTagA_,
    class LayoutTagB_,
    class LayoutTagC_>
class BatchedMatmulSoftmaxTla;
```

- `MatmulKernel_`: the BatchedMatmul kernel that computes the FP16 logits on AIC.
- `LayoutTagA_`, `LayoutTagB_`, and `LayoutTagC_`: Layout tags of A, B, and C.

## Key Fields of Arguments

```cpp
struct Arguments {
    uint32_t batch;
    GemmCoord problemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrC;
};
```

Where:

- `problemShape` describes the `M/N/K` dimensions of each batch.
- `ptrA`, `ptrB`, and `ptrC` are the GM start addresses of A, B, and C.

## Key Fields of Params

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

`ToUnderlyingArguments()` converts `Arguments` to `Params` for device kernel invocation.

## Host Helpers

- `SelectBatchedMatmulSoftmaxBlockNum()`: Selects 16, 24, or all AIC cores
  from output-tile occupancy.

The Softmax row tile is derived on device by the epilogue from the runtime `N`
and the remaining UB, so `Arguments` carries no extra tiling configuration.

## Usage Conditions

- A is FP16 RowMajor, B is FP16 ColumnMajor, and C is FP16 RowMajor.
- `batch`, `M`, `N`, and `K` are greater than 0.
- A, B, and C addresses are not null.

See `experimental/matmul/87_ascend950_batched_matmul_softmax/` for a complete instantiation.
