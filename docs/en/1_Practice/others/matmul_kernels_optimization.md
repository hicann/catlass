# Matmul Operator Performance Optimization

## Foreword

This document targets Matmul operator performance tuning scenarios and provides a practical path from performance data to example selection, Tiling adjustment, pipeline optimization, bandwidth optimization, and special template switching.

## Overall Tuning Workflow

For ultimate Matmul performance tuning, it is recommended to proceed in the following order:

1. First get correctness working with a basic template, and record `M/N/K`, data types, A/B/C layouts, strides, available AIC/AIV core counts, and workspace constraints.
2. Use `msprof op` to obtain on-device performance data, and determine whether the main bottleneck is insufficient Cube utilization, insufficient MTE2 read bandwidth, insufficient MTE3/Fixpipe write-out, insufficient Vector post-processing, or excessive Scalar header overhead.
3. Compute the number of task blocks:

   ```cpp
   mTiles = CeilDiv(M, m1);
   nTiles = CeilDiv(N, n1);
   taskBlocks = mTiles * nTiles;
   ```

   If `taskBlocks` does not match the AIC core count, adjust `m1/n1` first or switch to a different core-splitting template.

4. Perform single-point optimization on the current bottleneck, avoiding stacking multiple features at once, which would make the source of gains hard to determine.
5. Use the simulation pipeline diagram to check whether MTE2, MTE3, Cube, Vector, and Scalar have long bubbles or are waiting on each other.
6. When multiple candidate templates have similar performance, prefer the solution with lower startup overhead, smaller workspace, and lower maintenance cost.

## Case Collection

### Case 1: From Basic Template to General High-Performance Template

**Scenario characteristics**

Basic Matmul is commonly seen in scenarios where M/N/K are relatively regular, the number of C matrix task blocks is sufficient, and there is no complex post-processing. In this case, you should first choose [00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp) or [43_ascend950_basic_matmul](../../../../examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp) to establish a baseline.

For `43_ascend950_basic_matmul`, the README already lists the key template parameters supported by `MmadPingpong` and `MmadPreloadAsyncWithCallback`, including `l1AStages/l1BStages/l0AStages/l0BStages/l0CStages`, `enableL1Resident`, `enableShuffleK`, and so on. During ultimate performance tuning, these parameters should be treated as switches for pipeline and caching strategies rather than fixed configurations.

**Tuning steps**

1. **First adjust the number of task blocks.**

   For the Common template, the basic number of task blocks is `CeilDiv(M, m1) * CeilDiv(N, n1)`. When the number of task blocks is less than the AIC core count, prefer reducing `m1` or `n1`; when the number of task blocks is far greater than the AIC core count and the per-round load is uneven, prefer trying Swizzle or StreamK.

2. **Then adjust the basic block shape.**

   Within the L1/L0A/L0B/L0C capacity constraints, the larger `m1/n1` is, the fewer times A/B are re-read; however, if `m1/n1` is too large, the number of task blocks decreases, causing uneven load. The read volume can be estimated to first order using the following formula:

   ```text
   readBytes ~= sizeof(input) * M * N * K * (1 / m1 + 1 / n1)
   ```

   Therefore, MTE2-bound scenarios usually want to increase `m1/n1`; when Cube utilization is insufficient, you usually want to decrease `m1/n1` or introduce K-axis core splitting.

3. **Enable Multi Buffer.**

   Multi Buffer is a basic optimization whose goal is to overlap data movement and computation on L1/L0A/L0B/L0C. The vast majority of blockMmad components carry this capability by default. When tuning parameters, you need to confirm that the buffer stages at each level do not exceed the corresponding on-chip storage space.

4. **Enable Preload.**

   When the simulation pipeline diagram shows bubbles in MTE2 at the K loop or C block switch points, use strategies such as `MmadAtlasA2Preload` and `MmadPreloadAsyncWithCallback` to issue the next round of GM-to-L1 data movement in advance. Refer to [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp), [21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp), and [43_ascend950_basic_matmul](../../../../examples/43_ascend950_basic_matmul/README.md).

5. **Enable ShuffleK as needed.**

   When multiple AICs start from the same K block at the same time and GM same-address access conflicts are obvious, enable staggered reading along the K direction. On A2/A3, it is more worth trying when K is large and neither A nor B is fully resident; the specific gain depends on shape, layout, and hardware generation.

**Empirical judgment**

- If the shape is very small, MIX or general optimization templates such as `06_optimized_matmul` may become slower due to AIV startup and Scalar logic; in this case, prefer `31_small_matmul` or a pure Cube template.
- If B is already in zN format, `21_basic_matmul_preload_zN` is usually lighter than templates with Padding pre-processing.
- If L1 residency is enabled, the resident A/B tiles should no longer enable multi-buffer as ordinary tiles, otherwise L1 space will be wasted.

### Case 2: MTE2 Read Bandwidth Optimization

**Scenario characteristics**

MTE2 bound often comes from three types of problems:

- The A/B matrices are repeatedly read by multiple C blocks.
- RowMajor/ColumnMajor inputs have strides that are not 512-Byte aligned, making inline format-conversion data movement inefficient.
- Multiple cores access the same GM address at the same time, causing read conflicts.

**Optimization 1: Padding pre-processing**

When unaligned strides cause a significant drop in GM-to-L1 data movement bandwidth, AIV can be used to perform Padding or format rearrangement on A/B, making subsequent AIC reads more regular. There are already three types of Padding approaches in the repository:

| Approach           | Characteristics                                  | Applicability tendency                                                         |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------ |
| `PADDING_ND`       | Only pads the stride, simple to implement        | Stride unaligned but conversion requirements are light                         |
| `PADDING_BLOCK_ND` | Rearranges at L1 tile granularity                | Stride is very large or accesses within a tile need to be regular              |
| `PADDING_NZ`       | Converts to a zN/nZ layout close to L1 reads     | Inline conversion loss is large, or generalized performance is the priority    |

Refer to [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp) and [102_dynamic_optimized_matmul/doc/CommonMatmul_en.md](../../../../examples/102_dynamic_optimized_matmul/docs/en/CommonMatmul_en.md). When making an actual choice, it is recommended to use a model to estimate:

```text
Time without Padding ~= AIC unaligned read volume / unaligned effective bandwidth
Time with Padding ~= AIV Padding read+write time + AIC aligned read time + MIX startup overhead
```

Only when the AIC data movement time saved by Padding is greater than the AIV pre-processing and startup overhead is Padding a net gain. For small shapes or scenarios where the matrix is read only once, be cautious about enabling Padding.

**Optimization 2: Large-block reads and basic block selection**

In MTE2-bound scenarios, increasing `m1/n1` can reduce the number of times A/B are re-read; increasing `k1` or using a more efficient layout can reduce the number of data movement instructions in the K loop. When tuning parameters, check at the same time:

- Whether L1 can still hold A/B multi-buffers.
- Whether L0A/L0B/L0C satisfy the corresponding tile and stage requirements.
- Whether the number of task blocks can still saturate the AICs.
- Whether the tail block introduces obvious load imbalance.

**Optimization 3: Special small-axis data movement**

When M is very small or K is very small, ordinary ND2NZ inline conversion may not be the most efficient data movement method. `102_dynamic_optimized_matmul/doc/CommonMatmul.md` lists special paths such as row-by-row `DataCopy`, K=16 contiguous data movement, and switching to an equivalent RowMajor vector read when M=1. Such optimizations are suitable to be written into the select logic of a generalized template, rather than being hard-coded for a single shape.

### Case 3: Load Balancing and K-Axis Core Splitting

**MultiCoreSplitK**

When the C matrix is small, `taskBlocks < aicCoreNum`, and K is large enough, splitting blocks only along the M/N directions cannot saturate the AICs. In this case, you can use [09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp) or [22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp) to split the K direction among multiple AICs as well.

The gain of MultiCoreSplitK comes from more AICs participating in the computation, at the cost of:

- Workspace is needed to store partial sums of different K segments.
- AIV is needed to perform ReduceAdd.
- The larger the C matrix, the higher the workspace and ReduceAdd overhead.
- If K is not large enough, the parallel gain brought by splitting K may not outweigh the accumulation overhead.

Therefore, it is more suitable for scenarios with "small C, large K".

**StreamK**

MultiCoreSplitK solves the problem of too few task blocks, but it cannot fully solve the uneven load in the tail round. For example, when `taskBlocks` is not divisible by the AIC core count, some AICs will be idle in the last round. [37_streamk_matmul](../../../../examples/37_streamk_matmul/streamk_matmul.cpp) splits only the K-direction tasks of the tail round, distributing the tail-round work more evenly among all AICs.

The two key points of StreamK:

- **Split only the tail round.** The non-tail rounds are already full-core, so K is not split further, avoiding meaningless workspace and ReduceAdd.
- **Compute the tail round in advance.** Move the partial sums of the tail round forward to the second-to-last round, so that the AIV ReduceAdd can run in parallel with subsequent Cube computation, reducing the Vector tail.

To determine whether to use StreamK, you can first compute:

```cpp
fullRounds = taskBlocks / aicCoreNum;
tailBlocks = taskBlocks % aicCoreNum;
```

If `tailBlocks == 0`, StreamK is usually unnecessary; if `tailBlocks` is small and the K direction can be split, StreamK is more likely to bring gains.

**Small template**

For small shapes, the performance bottleneck is often not Cube or MTE2, but Scalar header overhead, scheduling overhead, and general loop logic. The idea of the [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp) example is to reduce runtime logic such as BlockScheduler, loops, and offset computation. The applicability judgment can refer to:

- `taskBlocks < aicCoreNum`.
- Each AIC processes at most one or very few C blocks.
- `K <= k1`, the K-direction loop is short.
- The total time is very small, and Scalar accounts for an obvious proportion in Profiling.

### Case 4: L1 Residency to Reduce Repeated Reads

When a certain matrix tile is repeatedly used by multiple task blocks, L1 residency can be considered. You can refer to the following two examples for optimization:

- [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp): Full loading of the A matrix under the Common template.
- [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp): The single-core split-K template naturally considers L1 tile reuse.

L1 residency is suitable for scenarios with high read repetition, where the resident matrix can fit into L1, and where the remaining L1 space after residency can still support data movement for the other matrix. Common trade-offs:

- Residency can reduce GM read volume, but it compresses multi-buffer space.
- The resident matrix should no longer be multi-buffered as an ordinary tile.
- It needs to be paired with a dedicated swizzle or task scheduling so that the resident data has opportunities for reuse between adjacent tasks.

In `43_ascend950_basic_matmul`, the usage conditions of `enableL1Resident` are already given: when `mTiles = 1` and `nTiles > CoreNum`, or `nTiles = 1` and `mTiles > CoreNum`, and `K < 2 * k1`, enabling L1 residency can be considered; if `l0CStages = 2` is enabled at the same time, pay attention to the L0C space and `enableUnitFlag` constraints.

### Case 5: MTE3/Fixpipe Unaligned Write-Out Optimization

When M/N values are large and K is small, and Cube computation is relatively light, C write-out may become the main bottleneck. If the N axis or stride of C is unaligned, the efficiency of Fixpipe writing directly to Global Memory decreases.

The [46_ascend950_matmul_fixpipe_opti](../../../../examples/46_ascend950_matmul_fixpipe_opti/46_ascend950_matmul_fixpipe_opti.md) example demonstrates the optimization idea for unaligned write-out on Ascend 950: splitting the L0C result into the UB of the corresponding Vector, and then having AIV write it back to GM using a data movement method more suitable for unaligned scenarios. This solution is applicable to:

- C matrix write-out is unaligned.
- K is small and MN is large, so write-out accounts for a high proportion.
- The Vector-side transfer overhead can be covered by the write-out bandwidth gain.

If M/N are already aligned, this solution may bring no gain, and may even slightly degrade due to the extra path. Therefore, it should be used as a targeted optimization for Fixpipe bound, rather than as a default replacement for ordinary write-out.

## Summary

For Matmul operators, you can refer to this document for optimization
