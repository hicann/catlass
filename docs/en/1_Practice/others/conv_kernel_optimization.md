# Conv Operator Performance Optimization

## Before We Start

This document targets performance tuning scenarios for Conv2D/Conv3D operators. Using `examples/33_basic_conv2d` as the baseline, it introduces a practical path from task block balance, pipeline depth, and Tile Shape selection to boundary handling optimization.

The current default configuration of `33_basic_conv2d`:

| Parameter         | Value                                       |
| ----------------- | ------------------------------------------- |
| DispatchPolicy    | `ConvAtlasA2Pingpong<2, 2, 2, 2, 1, false>` |
| FmapL1TileShape   | `<8, 12, 8>` Ho, Wo, Cin1                   |
| FilterL1TileShape | `<96, 8>` Cout, Cin1                        |
| L0TileShape       | `<16, 96, 16>` M, N, K                      |
| BlockScheduler    | `Conv2dIdentityBlockSwizzle<3, 0>`          |

The task space dimensions are `{Batch, Ho, Wo, Cout, Cin1}`, and the number of basic blocks is:

```text
totalLoops = batch * ceilDiv(Ho, 8) * ceilDiv(Wo, 12) * ceilDiv(Cout, 96)
```

## Overall Tuning Flow

Extreme performance tuning of Conv operators is recommended to proceed in the following order:

1. First, use the basic template to get correctness working, and record `Batch/Hi/Wi/Cin/Cout/Kh/Kw`, pads/strides/dilations, data types, and the number of available AIC cores.
2. Use `msprof op` to obtain on-device performance data, and determine whether the main bottleneck is insufficient Cube utilization, insufficient MTE2 read bandwidth, insufficient MTE3/Fixpipe write-out, or insufficient Vector post-processing.
3. Calculate the number of task blocks and judge how well it matches the number of AIC cores.
4. Perform single-point optimization for the current bottleneck (task block → Tile Shape → Multi Buffer → boundary handling), avoiding stacking multiple features at once.
5. Use the simulated pipeline diagram to check whether there are long bubbles or mutual waiting among MTE2, MTE3, Cube, and Vector.
6. When multiple candidate configurations have similar performance, prefer the solution with lower startup overhead, smaller workspace, and lower maintenance cost.

## Case Collection

### Case 1: Task Block Load Balancing

**Scenario Characteristics**

The tasks of Conv2D are split by `Conv2dIdentityBlockSwizzle` over the `{Batch, Ho, Wo, Cout}` space. The number of basic task blocks is jointly determined by FmapL1TileShape and FilterL1TileShape:

```cpp
uint32_t hoTiles = ceilDiv(Ho, FmapL1TileShape::Ho);
uint32_t woTiles = ceilDiv(Wo, FmapL1TileShape::Wo);
uint32_t coTiles = ceilDiv(Cout, FilterL1TileShape::Cout);
uint32_t totalLoops = batch * hoTiles * woTiles * coTiles;
```

When `totalLoops` is far less than the number of AIC cores, a large number of cores sit idle; when `totalLoops` is far greater than the number of cores and is not evenly divisible, load imbalance occurs in the tail round.

**Optimization Method**

**Prioritize adjusting the tile shape so that the total number of task blocks approaches an integer multiple of the number of AIC cores.** Taking the typical 48 cores of Atlas A2 as an example:

```text
Expectation: totalLoops ≈ N * 48  N >= 2
```

Adjustment directions:

| Phenomenon                                     | Adjustment                                                       | Side Effect                                                    |
| ---------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `totalLoops < 48`                              | Reduce `FmapL1TileShape::Ho/Wo` or `FilterL1TileShape::Cout`     | Increases the number of loop iterations, possibly amplifying MTE2 traffic |
| `totalLoops` far greater than 48 and tail round imbalanced | Try adjusting `SwizzleOffset` and `SwizzleDirection` | Does not affect traffic, only rearranges task assignment       |

**Adjust Swizzle parameters to achieve tail round balance.** The two directions of `Conv2dIdentityBlockSwizzle<offset, direction>`:

- `direction = 0` (default): traverse in the order `Ho → Wo → Cout`. Suitable for scenarios where `hoTiles * woTiles` is large.
- `direction = 1`: traverse in the order `Cout → Ho → Wo`. Suitable for scenarios where `coTiles` is large.

`SwizzleOffset` controls the Swizzle block size. By changing the task assignment order, it makes the tail round work distribute more evenly across cores.

**Empirical Judgement**

- When `totalLoops` is in the range `[48, 96)`, the tail round load imbalance problem is most prominent.

- If `totalLoops` is exactly an integer multiple of 48, Swizzle adjustments usually yield no benefit.
- Small-batch scenarios (batch=1) are more prone to insufficient task blocks, so reducing the tile shape should be considered first.
- Swizzle adjustment does not change traffic, only rearranges tasks, making it a low-risk starting point for optimization.

### Case 2: Pipeline Depth and Multi-Buffer Tuning

**Scenario Characteristics**

`ConvAtlasA2Pingpong` controls the Double/Triple Buffer depth of each level of storage through `L1A/L1B/L0A/L0B/L0C_STAGES`. The current default configuration is `L1A=2, L1B=2, L0A=2, L0B=2, L0C=1, UnitFlag=false`.

`L0C_STAGES=1` means that when L0C writes out to GM, Cube must wait for the write-out to complete before starting the next round of computation. When Cube computes relatively fast while write-out is relatively slow in the K loop (Cin1 dimension), a bubble is produced at the end of each K loop iteration.

**Optimization Method**

**Enable the L0C pipeline with `L0C_STAGES=2, ENABLE_UNIT_FLAG=true`.**

```cpp
constexpr uint32_t L0C_STAGES = 2;        // change from 1 to 2
constexpr bool ENABLE_UNIT_FLAG = true;    // change from false to true
using DispatchPolicy =
    Conv::ConvAtlasA2Pingpong<2, 2, 2, 2, L0C_STAGES, ENABLE_UNIT_FLAG>;
```

Principle: the number of L0C output buffers increases from 1 to 2. While Cube is computing, the L0C result of the current tile can be written out to GM in parallel, achieving overlap between Cube computation and L0C write-out.

**Adjust the L1A/L1B stage count.** If the L1 space permits, increase `L1A_STAGES` or `L1B_STAGES` from 2 to 3. Triple Buffer allows MTE2 transfers to overlap more fully with Cube computation:

```cpp
// Triple Buffer configuration example
constexpr uint32_t L1A_STAGES = 3;
constexpr uint32_t L1B_STAGES = 3;
```

A rough estimate of the L1 space budget:

| Buffer     | Size per stage                              | Total size for 3 stages  |
| ---------- | ------------------------------------------- | ------------------------ |
| L1A Fmap   | `Ho × Wo × Cin1 × C0 × sizeof(fp16)`        | 3x a single stage        |
| L1B Filter | `Cin1 × Kh × Kw × Cout × C0 × sizeof(fp16)` | 3x a single stage        |
| Total      | —                                           | Must be < 512KB (L1 capacity) |

**Empirical Judgement**

- When Profiling shows a high MTE2 proportion and low Cube utilization, prioritize increasing the stage count.
- The benefit of the L0C pipeline `L0C_STAGES=2` is more obvious in scenarios with longer K loops (larger Cin1).
- When increasing L1 stages from 2 to 3, confirm that L1 can hold all buffers. If the limit is exceeded, consider reducing the tile shape of Fmap or Filter to lower the size of a single stage.
- The `conv_bias` example uses `L1A=1, L1B=1, L0C=1, UnitFlag=true`: it has no double buffering but does have the unit flag. This shows that non-pipelined scenarios do not need L1 double buffering, but the L0C unit flag still helps write-out.

### Case 3: Tile Shape Selection and Tuning

**Scenario Characteristics**

The three Tile Shapes of Conv2D each control the blocking granularity at a different storage level:

| Tile Shape                      | Controlled Granularity                    | Impact                                         |
| ------------------------------- | ----------------------------------------- | ---------------------------------------------- |
| `FmapL1TileShape<Ho, Wo, Cin1>` | Tile size of Fmap loaded into L1 each time | Number of Ho×Wo loop iterations, L1 usage     |
| `FilterL1TileShape<Cout, Cin1>` | Tile size of Filter loaded into L1 each time | Number of Cout loop iterations, L1 usage     |
| `L0TileShape<M, N, K>`          | Computation granularity of a single Cube Mmad | Cube utilization                          |

The three are related through `K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1`. Each time a Filter tile is loaded, Fmap tiles must be looped `K_FMAP_PER_FILTER` times in the K direction.

**Optimization Method**

**Increase Ho/Wo of Fmap L1TileShape to reduce outer loop iterations.** When MTE2 bound, increasing Ho/Wo can reduce the amount of repeated Fmap data transfer from GM to L1:

```text
Fmap repeated read count ≈ batch × coTiles × cin1Tiles × (reduction ratio of hoTiles × woTiles)
```

The default `<8, 12>` has Ho×Wo=96. You can try increasing it to `<16, 16>` (Ho×Wo=256), reducing the number of iterations by about 2.7x; or `<32, 12>` (Ho×Wo=384).

**Increase Cout of FilterL1TileShape.** Increasing Cout can reduce the number of iterations in the Cout direction and lower repeated Filter reads:

```text
Filter repeated read count ≈ batch × hoTiles × woTiles × cin1Tiles × (reduction ratio of coTiles)
```

The default `<96>` yields only 1 tile when Cout=80. If Cout is large (e.g. 512), you can gradually increase it from 96 to 128, 192, up to the L1 space limit.

**Increase M/N/K of L0TileShape.** L0TileShape affects the compute utilization of a single Cube Mmad. If M/N is too small, Cube will not be fully loaded; if too large, it may exceed the L0A/L0B capacity.

| L0Shape         | Mmad computation count per call | L0A size   | Typical Scenario              |
| --------------- | ------------------------------- | ---------- | ----------------------------- |
| `<16, 96, 16>`  | 24K                             | 16×16 fp16 | Current default               |
| `<32, 96, 32>`  | 96K                             | 32×32 fp16 | Cube bound, ample space       |
| `<16, 128, 16>` | 32K                             | 16×16 fp16 | Cube bound, larger N axis     |

**Empirical Judgement**

- The three need to be adjusted in coordination: after increasing the Fmap/Filter tile, L1 usage rises. If multiple stages cannot be supported, you can keep stage=2 first, and trade off between reducing the tile shape and increasing stages.
- The M/N of L0TileShape should be aligned with the corresponding dimensions of L1TileShape as much as possible, to avoid an overly small tail block. For example, when `FmapL1TileShape::Ho=16`, the M of L0 had better divide 16 evenly (such as 16 or 32).
- When Cube bound, prioritize increasing L0TileShape; when MTE2 bound, prioritize increasing L1TileShape.
- You can use CATLASS's `msTuner` to automatically search for the optimal combination; it is recommended to keep the search space within 5000.

### Case 4: Padding and Boundary Handling Optimization

**Scenario Characteristics**

Boundary handling for Conv2D occurs in the computation of each output tile. The basic_conv2d kernel performs the following operations in each loop iteration:

```cpp
// Compute the valid hi range (including padding correction)
int32_t hiStart = hoStart * strideH - padTop;
int32_t hiEnd = hiStart + (actualH - 1) * strideH + (kh - 1) * dilationH;
if (hiStart < 0) { blockPadTop = 0 - hiStart; hiStart = 0; }
if (hiEnd > hi - 1) { blockPadBottom = hiEnd - (hi - 1); hiEnd = hi - 1; }

// Compute the valid wi range (including padding correction)
int32_t wiStart = woStart * strideW - padLeft;
// ...similar boundary checks...
```

These boundary checks and clipping are executed on every tile. When the number of batches is very large, the Scalar overhead cannot be ignored.

**Optimization Method**

**Pre-pad the input data.** On the Host side or in the AIV pre-processing stage, pre-pad the Fmap so that the Device side no longer needs runtime boundary checks. Taking `padTop=padBottom=padLeft=padRight=P` as an example:

```cpp
// Host-side pre-padding: add P rows/columns of 0 around the Fmap
size_t paddedHi = hi + 2 * P;
size_t paddedWi = wi + 2 * P;
// Recompute the Layout and offsets so that tile start positions begin directly at (0,0)
```

After pre-padding, the boundary checks inside the Kernel are simplified to:

```text
hiStart = hoStart * strideH
hiEnd = hiStart + (actualH - 1) * strideH + (kh - 1) * dilationH
(no boundary check needed, since the ho/wo range guarantees no out-of-bounds access)
```

**Eliminate tail block logic for regular shapes.** When `Ho % FmapL1TileShape::Ho == 0` and `Wo % FmapL1TileShape::Wo == 0`, all tiles are full blocks, and `actualBlockShape` is always equal to `FmapL1TileShape`. The `GetActualBlockShape` call can be skipped through a compile-time branch:

```cpp
if constexpr (isFullTile) {
    // Always use FmapL1TileShape, no runtime computation needed
} else {
    // Compute actualBlockShape at runtime
}
```

**Merge Padding into Filter/Fmap writes.** If the `Kh/Kw` of the Filter is fixed and the padding is known, the padding can be written directly into the Fmap data during the gen_data stage, so that the Fmap Layout on the Device side can skip the padding offset calculation.

**Empirical Judgement**

- Pre-padding is suitable for deployment scenarios where the padding values are fixed and the shape does not change frequently.
- If the padding values change with the input (such as dynamic shapes), pre-padding requires re-preparing the data each time, and the benefit may be offset by Host-side overhead.
- Regular shapes (Ho/Wo divisible by the tile) give the greatest benefit, because the runtime computation of `actualBlockShape` and the tail block branch can be completely eliminated.
- When the Scalar proportion in Profiling is higher than 10%, boundary handling optimization is worth investing in.

## Summary

Conv operator performance optimization can refer to this document.
