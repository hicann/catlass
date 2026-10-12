# Conv Operator Development Guide

This document targets Conv operator development scenarios. Using Basic Conv2D as the entry point, it explains the complete workflow from problem definition and template parameter selection to Kernel assembly and Host invocation, helping developers quickly get started with the development and debugging of Conv2D operators.

## Conv Computation Principles

**Problem Definition**

For a 2D convolution, the shape of the input (Fmap) is $(N, C_{\text{in}}, H_i, W_i)$, the shape of the weight (Filter) is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of the output (Output) is $(N, C_{\text{out}}, H_o, W_o)$. The computation formula for each position in the output is:

$$
\text{out}(n, c_{\text{out}}, h_o, w_o) = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} \text{fmap}(n, c_{\text{in}}, h_i+i, w_i+j) \cdot \text{filter}(c_{\text{out}}, c_{\text{in}}, i, j)
$$

The output spatial dimensions are jointly determined by the input, padding, dilation, and stride:

$$
\begin{aligned}
H_o &= \lfloor (H_i + \text{padTop} + \text{padBottom} - \text{dilationH} \times (K_h - 1) - 1) / \text{strideH} \rfloor + 1 \\
W_o &= \lfloor (W_i + \text{padLeft} + \text{padRight} - \text{dilationW} \times (K_w - 1) - 1) / \text{strideW} \rfloor + 1
\end{aligned}
$$

Conv2D can essentially be transformed into a matrix multiplication (Im2col + GEMM). In Catlass, the sliding-window computation is implemented directly through the tile-level BlockConv2d component, avoiding explicit Im2col rearrangement.

**Data Layout**

Conv2D involves three data layouts, all of which are 5-dimensional Tensors:

| Tensor | Layout         | Dimension Meaning            | Default C0 |
| ------ | -------------- | ---------------------------- | ---------- |
| Fmap   | `NC1HWC0`      | `[Batch, Cin1, Hi, Wi, C0]`  | 16         |
| Filter | `CI1KHKWCOCI0` | `[Cin1, Kh, Kw, Cout, C0]`   | 16         |
| Output | `NC1HWC0`      | `[Batch, Cout1, Ho, Wo, C0]` | 16         |

Here `C0 = 16` is the hardware alignment granularity (`BYTE_PER_C0 / sizeof(fp16)`), `Cin1 = CeilDiv(Cin, C0)`, and `Cout1 = CeilDiv(Cout, C0)`.

The corresponding Layout type definitions in Catlass are located in `include/catlass/layout/tensor.hpp`:

- `layout::NC1HWC0` — used by Fmap and Output, with strides `[Cin1*Hi*Wi*C0, Hi*Wi*C0, Wi*C0, C0, 1]`
- `layout::CI1KHKWCOCI0` — used by Filter, with strides `[Kh*Kw*Cout*C0, Kw*Cout*C0, Cout*C0, C0, 1]`

## Full Code Walkthrough

**1. Parameter Definition and Command-Line Parsing**

The `Options` struct defines and parses the convolution parameters:

```cpp
struct Options {
    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};               // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};                // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {2, 2};                   // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                 // {dilationH, dilationW}
    int32_t deviceId{0};
    Catlass::Conv2dParams problemParams{};
    // Parse() parses each parameter from argv and then calls:
    // problemParams = Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);
};
```

`Conv2dParams` (`include/catlass/conv_coord.hpp`) automatically computes the following derived quantities from the input shape:

| Field             | Meaning                     | Computation                                      |
| ----------------- | --------------------------- | ------------------------------------------------ |
| `cin1()`          | Input C1 dimension          | `CeilDiv(cin, C0)`                               |
| `cout()`          | Output channels             | User-specified                                   |
| `cout1()`         | Output C1 dimension         | `CeilDiv(cout, C0)`                              |
| `coutRound()`     | Aligned output channels     | `(cout + C0 - 1) / C0 * C0`                      |
| `ho()`/`wo()`     | Output spatial dimensions   | Computed by the pad/stride/dilation formula      |
| `postIm2colShape` | Task space                  | `{Batch, Ho, Wo, Cout, Cin1}`                    |

**2. Environment Initialization and Memory Allocation**

```cpp
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(options.deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// Extract each dimension from Conv2dParams
uint32_t c0 = options.problemParams.C0;        // 16
uint32_t batch = options.problemParams.batch();
uint32_t hi = options.problemParams.hi();
uint32_t wi = options.problemParams.wi();
uint32_t cin1 = options.problemParams.cin1();
uint32_t ho = options.problemParams.ho();
uint32_t wo = options.problemParams.wo();
uint32_t cout1 = options.problemParams.cout1();
uint32_t cout = options.problemParams.cout();
uint32_t coutRound = options.problemParams.coutRound();
uint32_t kh = options.problemParams.kh();
uint32_t kw = options.problemParams.kw();

// Compute the size required by each Tensor (unit: number of elements)
size_t lenFmap = batch * cin1 * hi * wi * c0;
size_t lenFilter = cin1 * kh * kw * cout * c0;
size_t lenOutput = batch * ho * wo * coutRound;
```

Note that `cout` in `lenFilter` is the original (unaligned) channel count, because the last dimension of the Filter Layout is C0=16 and the physical size is `cin1 * kh * kw * cout * c0`.

**3. Host Data Preparation**

```cpp
using LayoutFmap = layout::NC1HWC0;
using LayoutFilter = layout::CI1KHKWCOCI0;
using LayoutOutput = layout::NC1HWC0;

LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

std::vector<fp16_t> hostFmap(lenFmap);
std::vector<fp16_t> hostFilter(lenFilter);
golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);
```

The Layout objects are used to compute the offsets within each Tensor. The Host data is filled with random numbers and then copied to the Device.

**4. Template Parameter Selection**

This is the core part of Conv2D operator development. A set of template parameters must be selected based on the target shape and hardware constraints.

- **4.1 Architecture Tag**

```cpp
using ArchTag = Arch::AtlasA2;
```

Specifies the target Ascend chip architecture. Catlass currently mainly supports `Arch::AtlasA2`.

- **4.2 Dispatch Policy — Pipeline Policy**

```cpp
constexpr uint32_t L1A_STAGES = 2;
constexpr uint32_t L1B_STAGES = 2;
constexpr uint32_t L0A_STAGES = 2;
constexpr uint32_t L0B_STAGES = 2;
constexpr uint32_t L0C_STAGES = 1;
constexpr bool ENABLE_UNIT_FLAG = false;

using DispatchPolicy =
    Conv::ConvAtlasA2Pingpong<L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
```

`ConvAtlasA2Pingpong` controls the number of Double Buffer stages at each level of storage (L1/L0A/L0B/L0C):

| Parameter          | Default | Description                                    |
| ------------------ | ------- | ---------------------------------------------- |
| `L1A_STAGES`       | 2       | Multi-buffer depth for Fmap from GM to L1      |
| `L1B_STAGES`       | 2       | Multi-buffer depth for Filter from GM to L1    |
| `L0A_STAGES`       | 2       | Multi-buffer depth for Fmap from L1 to L0A     |
| `L0B_STAGES`       | 2       | Multi-buffer depth for Filter from L1 to L0B   |
| `L0C_STAGES`       | 1       | L0C output buffer depth                        |
| `ENABLE_UNIT_FLAG` | false   | Whether to enable the unit flag                |

Internally, `BlockConv2d` caps the stage count: `L1A_STAGES = min(DispatchPolicy::L1A_STAGES, MAX_STAGES)`, `MAX_STAGES = 2`.

- **4.3 Tile Shape — Block Shapes**

```cpp
using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<8, 12, 8>;   // (hoBlock, woBlock, cin1BlockSmall)
using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<96, 8>;  // (coutBlock, cin1BlockBig)
using L0TileShape = Catlass::Conv2dL0Shape<16, 96, 16>;         // (mL0, nL0, kL0)
```

These three Tile Shapes determine the blocking granularity of the convolution computation at each level of storage:

| Tile Shape                      | Meaning                     | Role                                                  |
| ------------------------------- | --------------------------- | ----------------------------------------------------- |
| `FmapL1TileShape<Ho, Wo, Cin1>` | Basic block of Fmap loaded to L1 | Loads a Fmap tile of `[Ho, Wo, Cin1, C0]` each time |
| `FilterL1TileShape<Cout, Cin1>` | Basic block of Filter loaded to L1 | Loads a Filter tile of `[Cin1, Kh, Kw, Cout, C0]` each time |
| `L0TileShape<M, N, K>`          | Mmad tile on L0A/L0B        | Computation granularity of each Matrix Multiply       |

Note the relationship between `FmapL1TileShape::Cin1` and `FilterL1TileShape::Cin1`: `K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1`, which means that for each Filter block loaded, the Fmap tile must be looped multiple times along the K direction.

- **4.4 Data Type and Layout Combination**

```cpp
using FmapType = Gemm::GemmType<half, LayoutFmap>;
using FilterType = Gemm::GemmType<half, LayoutFilter>;
using OutputType = Gemm::GemmType<half, LayoutOutput>;
```

The element type and Layout are combined into a GemmType, which is passed to BlockConv2d.

- **4.5 BlockConv and Kernel Assembly**

```cpp
using BlockConv2d = Conv::Block::BlockConv2d<
    DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape,
    FmapType, FilterType, OutputType>;
using BlockEpilogue = void;

using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

using Conv2dKernel = Conv::Kernel::BasicConv2d<BlockConv2d, BlockEpilogue, BlockScheduler>;
using Conv2dAdapter = Conv::Device::DeviceConv<Conv2dKernel>;
```

| Component        | Role                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `BlockConv2d`    | Implements a single `Fmap_tile * Filter_tile -> Output_tile` matrix multiplication        |
| `BlockEpilogue`  | Post-processing (currently `void`, no post-processing)                                     |
| `BlockScheduler` | Task partitioning strategy, determining which output tile each AI Core processes           |
| `BasicConv2d`    | Kernel-level wrapper, including loop scheduling and boundary handling                      |
| `DeviceConv`     | Host-side adapter, wrapping CanImplement, GetWorkspaceSize, Initialize, and execution      |

**5. Task Scheduling and Boundary Handling**

`Conv2dIdentityBlockSwizzle<3,0>` defines the task partitioning method. The loop logic inside the Kernel (`basic_conv2d.hpp`):

```cpp
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < loops; loopIdx += AscendC::GetBlockNum()) {
    // 1. Get the logical coordinate of the current block
    Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
    // 2. Get the actual block shape (handle boundaries)
    Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);
    // 3. Compute the valid range of Fmap along H/W (including boundary padding handling)
    // 4. Compute the GM offsets of each tensor
    // 5. Call blockConv2d for execution
    blockConv2d(gmFmap[...], layoutFmap, gmFilter[...], layoutFilter,
                gmOutput[...], layoutOutput, actualConv2dBlockShape, blockPadList);
}
```

Boundary handling logic: the corresponding `hiStart / hiEnd` and `wiStart / wiEnd` are computed from `hoStart / woStart`, stride, dilation, and padding. When the start position exceeds the boundary, the padding amount is recorded and the range is clipped.

**6. Execution and Precision Verification**

```cpp
Conv2dKernel::Arguments arguments{options.problemParams, deviceFmap, deviceFilter, deviceOutput};
Conv2dAdapter conv2d_op;
conv2d_op.CanImplement(arguments);
size_t sizeWorkspace = conv2d_op.GetWorkspaceSize(arguments);
// Allocate workspace (if any)
conv2d_op.Initialize(arguments, deviceWorkspace);
conv2d_op(stream, aicCoreNum);
ACL_CHECK(aclrtSynchronizeStream(stream));

// Copy the result back to Host and compare with golden
std::vector<fp16_t> hostOutput(lenOutput);
ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
std::vector<float> hostGolden(lenOutput);
golden::ComputeConv2d(options.problemParams, hostFmap, layoutFmap, hostFilter, layoutFilter,
                      hostGolden, layoutOutput);
std::vector<uint64_t> errorIndices = golden::CompareData(hostOutput, hostGolden, cin1 * kh * kw * c0);
```

-`CanImplement`: Checks whether the current shape is within the range supported by the Kernel (BasicConv2d always returns true)
\-`GetWorkspaceSize`: Returns the required workspace size (BasicConv2d returns 0)
\-`golden::ComputeConv2d`: CPU reference implementation, used for precision comparison

**7. Resource Release**

```cpp
ACL_CHECK(aclrtFree(deviceFmap));
ACL_CHECK(aclrtFree(deviceFilter));
ACL_CHECK(aclrtFree(deviceOutput));
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(options.deviceId));
ACL_CHECK(aclFinalize());
```

## Kernel Execution Workflow

The Kernel implementation of `BasicConv2d` is located in `include/catlass/conv/kernel/basic_conv2d.hpp`, and its core flow is as follows:

```text
[GM] Fmap ──GM→L1──→ [L1] Fmap Tile ──L1→L0A──→ [L0A] zN format
[GM] Filter ──GM→L1──→ [L1] Filter Tile ──L1→L0B──→ [L0B] nZ format
                                                     ↓
                                              [Cube] Mmad (L0A × L0B → L0C)
                                                     ↓
[GM] Output ←──L0C→GM── [L0C] Accumulator
```

Workflow description:

1. **GM → L1**: Fmap and Filter are moved from Global Memory to L1 tile by tile; format conversion is completed during the move (in-line NZ)
2. **L1 → L0A/L0B**: Re-movement from L1 to L0A/L0B, further adjusted to the format required by the Cube computation
3. **Cube Mmad**: L0A × L0B matrix multiplication, with the result accumulated into L0C
4. **L0C → GM**: Write the L0C result back to Global Memory
5. **K loop**: Loop along the `Cin1` dimension, accumulating the results of multiple K blocks

## Summary

This document provides a complete introduction to the development workflow of the Basic Conv2D operator in CATLASS.
