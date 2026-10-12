# CV Fused Operator Performance Tuning Case Collection

## Foreword

This document summarizes performance analysis and tuning methods for CV fused operators in CATLASS. Here, CV fusion means using both AIC and AIV within the same kernel, placing stages such as matrix computation, post-processing, reduction, Softmax, or data reorganization into a single pipeline.

This document mainly covers three examples:

| Example        | Path                                          | Fusion form                                                  |
| -------------- | --------------------------------------------- | ------------------------------------------------------------ |
| quant_matmul   | `examples/12_quant_matmul`                    | AIC executes int8 matmul, AIV executes per-token/per-channel dequantization and write-back |
| streamk_matmul | `examples/37_streamk_matmul`                  | AIC executes normal blocks and tail-round Stream-K blocks, AIV reduces tail-round partial sums |
| FAI            | `examples/49_ascend950_flash_attention_infer` | AIC executes QK/PV, AIV executes Online Softmax/RescaleO     |

When tuning the performance of a CV fused operator, you cannot look only at the compute utilization of a single stage. Its tuning benefit comes from stage overlap. The final latency is usually determined jointly by the longest pipeline, pipeline bubbles, inter-core tail effects, GM workspace transfers, and synchronization overhead.

## General Analysis Methods

### Obtaining performance data

It is recommended to first run the example as a single operator and collect Profiling data, to confirm the proportions of AIC, AIV, MTE2, MTE3, and FixPipe. For how to use performance tools in CATLASS, refer to [msProf&Profiling](../evaluation/performance_tools.md).

When collecting data, it is recommended to prepare two sets of results:

| Data                                        | Purpose                                                        |
| ------------------------------------------- | -------------------------------------------------------------- |
| Total fused kernel latency                  | Determine the overall benefit and the final effect of different parameter combinations |
| AIC/AIV/MTE/FixPipe pipeline proportions    | Determine whether Cube/Vector or transfer/write-back is the bottleneck |
| Per-core task amount or per-core latency    | Determine whether the inter-core load is balanced              |
| Workspace size and alignment information    | Determine whether intermediate results between CV stages introduce an extra GM bottleneck |

If you need to observe finer pipeline parallelism, you can combine this with [performance pipeline simulation](../evaluation/performance_tools.md#performance-pipeline-simulation), focusing on whether AIC and AIV are waiting on each other for a long time.

### Building a fusion benefit model

The following simplified model can be used to judge whether a CV fusion optimization is worth pursuing:

```text
serial_latency = AIC_stage_latency + AIV_stage_latency + transfer_writeback_latency
fused_latency ~= max(AIC_pipeline_latency, AIV_pipeline_latency, transfer_pipeline_latency) + head_tail_flush_cost + sync_cost
fusion_gain = (serial_latency - fused_latency) / serial_latency
```

Ideally, the AIC and AIV latencies are close, and intermediate data does not need to be repeatedly moved in and out of GM, so the fusion benefit is more pronounced. If one side is far slower than the other, the other side will wait, and the fusion benefit converges to the latency of the bottleneck stage. If the tiling is too small, there are too many stages, or workspace access is inefficient, the fused performance may even degrade due to scheduling, synchronization, and transfer issues.

### Designing the optimization plan

Tuning a CV fused operator is usually advanced in the following order:

1. Adjust the TileShape or basic block size so that the per-block computation granularity is large enough while satisfying L1/L0/UB resource limits.
2. Adjust the block scheduler and swizzle so that the number of basic blocks better matches the number of AICs, achieving load balance.
3. Adjust the CV pipeline depth or workspace stages so that AIC production and AIV consumption overlap as much as possible.
4. Check the intermediate result path, preferring on-chip caches or aligned workspaces to avoid inefficient GM transit.
5. Based on Profiling, locate MTE2, MTE3, and FixPipe issues, and optimize for input layout, stride, workspace alignment, and the write-back path.

## Case Collection

### Case 1: quant_matmul dequantization fusion

**Example structure**

`examples/12_quant_matmul/quant_matmul.cpp` implements a quantized matrix multiply with int8 inputs, int32 accumulation, and half output:

```text
C_int32 = A_int8 * B_int8
D_half = C_int32 * scale[n] * perTokenScale[m]
```

The key template configuration in the example is as follows:

```cpp
using L1TileShape = GemmShape<128, 256, 512>;
using L0TileShape = GemmShape<128, 256, 128>;
constexpr uint32_t workspaceStages = 2;
constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
using EpilogueTileShape = MatrixShape<32, 256>;
using MatmulKernel =
    Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;
```

The fusion boundary of this kernel is in `include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp`:

| Stage    | Execution unit | Work                                                                            |
| -------- | -------------- | ------------------------------------------------------------------------------- |
| Matmul   | AIC            | Reads A/B, executes int8 BlockMmad, FixPipe writes the int32 intermediate result to workspace |
| Epilogue | AIV            | Reads int32 from workspace, loads `scale` and `perTokenScale`, completes the broadcast multiply, type conversion, and D write-back |

AIC and AIV form a producer-consumer pipeline on workspace slots. After AIC finishes writing the current stage, it sets `flagAicFinishStoreList[stageId]`; after AIV finishes consuming, it sets `flagAivFinishComputeList[stageId]`; AIC waits for AIV to finish before reusing the same stage.

**Performance bottleneck diagnosis**

If Profiling shows a high AIC proportion, look at the matmul side first:

| Symptom                        | Possible cause                                 | Tuning direction                              |
| ------------------------------ | ---------------------------------------------- | --------------------------------------------- |
| AIC runs for a long time, AIV is sparse | `L1TileShape` is too small or there are too many basic blocks | Increase the M/N/K tiles and reduce the number of loop iterations |
| High MTE2 proportion           | A/B input layout or stride is unfriendly       | Check whether A RowMajor and B ColumnMajor match the business inputs |
| High FixPipe proportion        | Writing int32 intermediate results to workspace is under heavy pressure | Check workspace address alignment and stage layout |

If the AIV proportion is high, look at the epilogue side first:

| Symptom                          | Possible cause                   | Tuning direction                                   |
| -------------------------------- | -------------------------------- | -------------------------------------------------- |
| AIC frequently waits for AIV to release the stage | Dequantization UB tile is too small or Vector computation is heavy | Adjust `EpilogueTileShape`, `ubStages`, or reduce scale transfers |
| High proportion of scale/per-token scale transfers | Repeated scale access, M/N tiles do not match | Adjust M/N tiles so that scale reuse and output write-back are more continuous |
| Abnormal tail-block AIV latency  | M/N misalignment causes many UB and GM tail-block transfers | Prefer a TileShape that reduces the tail-block ratio |

**Tuning highlights**

**1. Balance the AIC and AIV stages**

`L1TileShape<128, 256, 512>` means each AIC basic block outputs `128 x 256` int32 intermediate results. On the AIV side, `EpilogueTileShape<32, 256>` splits one AIC tile into 4 epilogue tiles along the M direction. If AIV becomes the bottleneck, you cannot simply increase the matmul tile; a larger AIC tile increases the data that AIV must consume in a single stage, which may make AIC wait more frequently for stage release.

Candidate search can be done as follows:

```text
candidate 1: fix L1TileShape::M/N, adjust EpilogueTileShape::ROW
candidate 2: fix EpilogueTileShape, adjust L1TileShape::M/N to improve inter-core balance
candidate 3: adjust L1TileShape::K and L0TileShape::K together to improve Cube efficiency
```

Each candidate group requires confirming the workspace size:

```text
workspace_size = L1TileShape::M * L1TileShape::N * aicCoreNum * workspaceStages * sizeof(uint32_t)
```

When the workspace is too large, GM access and cache pressure may offset the fusion benefit.

**2. Adjust workspaceStages**

`workspaceStages = 2` is a typical double buffer. If the pipeline diagram shows that AIC often waits for AIV after finishing a stage write, you can try increasing the number of stages to give AIC more lookahead. However, more stages is not always better; it increases the workspace capacity and the number of synchronization flags linearly, and may also amplify the pressure of reading and writing intermediate results in GM.

It is recommended to increase the number of stages only when the following conditions are all satisfied:

| Condition                              | Description                                  |
| -------------------------------------- | -------------------------------------------- |
| AIC single-block latency is shorter than AIV single-block latency | AIC has a chance to cover AIV by computing ahead |
| Workspace bandwidth is not the bottleneck | Otherwise adding stages only increases GM pressure |
| There are enough basic blocks          | When there are too few, the head/tail flush overhead is high and the benefit of adding stages is limited |

**3. Adjust BlockScheduler and Swizzle**

The example selects `GemmIdentityBlockSwizzle<3, 0>` when `m > n`, and `<3, 1>` otherwise. During tuning, first estimate the number of basic blocks using the current TileShape:

```text
block_num = CeilDiv(M, L1TileShape::M) * CeilDiv(N, L1TileShape::N)
tail_blocks = block_num % aicCoreNum
```

If `tail_blocks` is small but non-zero, there will be a tail-round straggler effect; if `block_num < aicCoreNum`, AIC utilization is insufficient. In this case, first adjust `L1TileShape::M/N` so that the number of basic blocks approaches an integer multiple of the AIC count, then fine-tune the swizzle direction and offset to improve A/B access locality.

### Case 2: streamk_matmul tail-round early computation

**Example structure**

`examples/37_streamk_matmul` implements Stream-K matmul. Normal blocks are still computed directly by AIC and written back to C. In the tail round, the basic blocks that are insufficient to fill all AICs are split along the K direction, producing more work units. AIC writes partial sums to workspace, and AIV then reduces them to obtain the result C.

The key configuration of the example is as follows:

```cpp
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Streamk<enableUnitFlag, enableShuffleK>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using BlockScheduler = typename Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 0>;
constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
using ReduceAdd =
    Catlass::Gemm::Kernel::StreamkReduceAdd<ArchTag, BlockScheduler, ElementAccumulator, ElementC, computeLength>;
```

For the detailed principles of Stream-K, refer to [StreamkMatmul documentation](../../../../examples/102_dynamic_optimized_matmul/docs/en/StreamkMatmul_en.md). From the CV fusion perspective, it has two key points:

| Optimization             | CV fusion implication                                  |
| ------------------------ | ------------------------------------------------------ |
| Tail-round K splitting   | AIC splits the tail round into more K slices, reducing tail-round AIC idleness |
| Early tail-round computation | AIC produces tail-round partial sums ahead of time, allowing AIV reduction to overlap with subsequent Cube computation |

**Tail-round early computation**

The tail-round shape of a normal matmul is usually:

```text
round 0..r-1: AIC computes complete basic blocks
round r:      a few AICs compute the tail-round basic blocks, the remaining AICs stay idle
```

After Stream-K tail-round splitting, if AIV reduction is performed only after all normal blocks are complete, the Vector reduction falls at the end of the kernel and becomes an extra tail. The scheduling in the example advances the tail-round Stream-K blocks to around the second-to-last round, so that AIV can start reading workspace and reducing partial sums earlier. The abstract pipeline is as follows:

```text
time slice t:     AIC computes the advanced Stream-K tail-round partial sums -> workspace
time slice t+1:   AIV reduces the tail-round partial sums
time slice t+1:   AIC keeps computing normal complete blocks
```

This approach essentially uses the Cube computation time of subsequent normal blocks to hide the Vector reduction time.

**Performance bottleneck diagnosis**

Stream-K is suitable for shapes with unbalanced tail-round loads. The empirical judgment is as follows:

```text
B = CeilDiv(M, m1) * CeilDiv(N, n1) # m1/n1 are the L1TileShape::M/N values used to split the task blocks
a good gain is likely when B / C > 1 and B % C <= C * 0.8
```

Here `C` is the number of AICs. You can judge whether Stream-K should be used as follows:

| Symptom          | Judgment                                       |
| ---------------- | ---------------------------------------------- |
| `B % C == 0`     | No tail round; Stream-K reduction overhead usually brings no benefit |
| `B < C`          | Too few M/N basic blocks; consider Split-K or smaller M/N tiles first |
| `B / C == 1`     | Only one round of normal blocks; the room for early tail-round hiding is limited |
| Very few tail-round blocks | Stream-K is more likely to benefit             |
| K is very short  | After K splitting there are too few slices; the AIV reduction overhead may exceed the benefit |

**Tuning highlights**

**1. Tune the M/N basic blocks first, then decide whether to use Stream-K**

Stream-K addresses tail-round imbalance, not all load imbalance problems. If an unreasonable `L1TileShape::M/N` choice causes the number of basic blocks to be far smaller than the number of AICs, adjust the TileShape first. Only when there are enough full rounds and the tail-round block count is insufficient does Stream-K more readily show a benefit.

It is recommended to compute first:

```text
m_blocks = CeilDiv(M, L1TileShape::M)
n_blocks = CeilDiv(N, L1TileShape::N)
mn_blocks = m_blocks * n_blocks
normal_rounds = mn_blocks / aicCoreNum
tail_blocks = mn_blocks % aicCoreNum
k_slices_per_block = CeilDiv(K, L1TileShape::K)
streamk_work = tail_blocks * k_slices_per_block
```

The closer `streamk_work` is to or the more it exceeds the AIC count, the more easily the tail round can be evenly distributed; if `k_slices_per_block` is too small, the tail-round splitting granularity is insufficient.

**2. Control the UB pressure of AIV reduction**

`StreamkReduceAdd` uses UB to read multiple partial sums and perform Add/Cast/write-back. In the example, `computeLength = 192 * 1024 / sizeof(ElementAccumulator)`, and the UB space constraint must be satisfied. During reduction, the number of rows processed by each AIV is jointly determined by `COMPUTE_LENGTH`, the number of AIVs participating in the reduction, and the tile column width.

If AIV reduction becomes the bottleneck, check:

| Parameter        | Impact                                          |
| ---------------- | ----------------------------------------------- |
| `L1TileShape::N` | Determines the partial sum length per row, affecting the number of rows UB can process at once |
| `tail_blocks`    | Determines the number of Stream-K blocks that need reduction |
| `splitkSliceNum` | Determines how many partial sums each output block must accumulate |
| `computeLength`  | Determines the upper limit of data transferred and reduced by AIV each time |

The tuning direction is usually to avoid an excessively large `L1TileShape::N`, which would let AIV process only very few rows at a time, while also avoiding tiles that are too small, which would reduce the efficiency of AIC normal blocks.

**3. Check the workspace size and alignment**

The Stream-K workspace size is:

```text
max(2MB, L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator) * aicCoreNum * 2)
```

Each AIC has two workspace slices, used to handle the two partial sum segments before and after crossing blocks. If Profiling shows a high MTE3 or MTE2 proportion, first check whether the workspace start address, the stride of each workspace slice, and the stride of the output C matrix meet the requirements for efficient transfers. Alignment issues directly affect AIC writing partial sums and AIV reading partial sums.

**4. Verify whether the tail-round early computation actually covers the reduction**

When observing the pipeline diagram, focus on three things:

| Goal        | Expected phenomenon                        |
| ----------- | ------------------------------------------ |
| Early AIC tail round | Stream-K blocks are not all concentrated at the end of the kernel |
| Overlapped AIV reduction | AIV Add/Cast/MTE3 overlaps with subsequent AIC BlockMmad |
| Short final flush | Only a small amount of AIV reduction and write-back remains at the end of the kernel |

If AIV reduction is still concentrated at the end, it usually means there are insufficient normal blocks available for early computation, the tail-round splitting granularity is insufficient, or the synchronization point is too late.

### Case 3: FAI multi-stage CV pipeline

The detailed performance optimization analysis of the `examples/49_ascend950_flash_attention_infer` example is in [FA operator performance optimization](./FA_kernel_optimization.md). This document only summarizes the key points.

The FAI computation chain is:

```text
QK(AIC) -> Online Softmax(AIV) -> PV(AIC) -> RescaleO(AIV)
```

This is a typical multi-stage CV fusion. Compared with quant_matmul and streamk_matmul, FAI has stronger data dependencies: Softmax depends on the QK result, PV depends on the Softmax result, and RescaleO depends on the PV result and the online softmax state. Therefore, the tuning focus is not simply increasing concurrency, but making adjacent `kvSeq` blocks form a stable pipeline.

The key optimization directions in the FAI example include:

| Direction              | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| TileShape adjustment   | Increasing `qSeqBase/kvSeqBase` can reduce the number of loops and synchronizations, but increases UB/L1/L0 usage |
| CV pipeline            | Reduce AIC/AIV mutual waiting through cross-block overlap of QK, Softmax, PV, and RescaleO |
| On-chip transit        | AIV moves the Softmax result P from UB to the AIC-side L1, avoiding `UB -> GM -> L1` transit |
| Multi-core load balance | Under causal or sparse masks, split by the number of valid blocks to avoid stragglers caused by evenly splitting by q rows |
| Paged Attention        | Check the block table, KV cache page continuity, and alignment to avoid MTE2 scattered reads |
| FixPipe and workspace alignment | Ensure that the output and workspace segment start addresses are aligned to requirements such as 512B |

For more detailed parameter explanations, resource budgets, and tuning paths, refer to [FA operator performance optimization](./FA_kernel_optimization.md).

## Fused Operator Tuning Checklist

### Tiles and resources

| Check item                | Description                                |
| ------------------------- | ------------------------------------------ |
| Whether L1/L0/UB exceed limits | Calculate the static buffer usage before enlarging tiles |
| Number of M/N basic blocks | Avoid `block_num < aicCoreNum` or an excessively small tail round as much as possible |
| K-direction splitting granularity | Stream-K and Split-K need enough K slices to balance |
| epilogue tile             | In quant_matmul, the AIV tile must be balanced against the AIC output tile |

### CV pipeline

| Check item           | Description                                    |
| -------------------- | ---------------------------------------------- |
| Whether AIC waits for AIV | May be due to AIV computation, transfers, or insufficient stages |
| Whether AIV waits for AIC | May be due to Cube, MTE2, or FixPipe bottlenecks |
| Whether the number of stages is appropriate | Too few stages provide insufficient coverage; too many increase workspace and synchronization pressure |
| Head/tail flush proportion | With small shapes or few loops, the fused pipeline benefit is limited |

### Transfers and alignment

| Check item   | Description                                   |
| ------------ | --------------------------------------------- |
| Input layout | The physical layout of A/B/Q/K/V must match the kernel's TLA/Layout definitions |
| GM workspace | Keep the start address and segment offsets aligned to 512B as much as possible |
| stride       | The contiguous-dimension stride should satisfy the requirements of efficient DataCopy/FixPipe |
| Intermediate result path | Prefer on-chip transit; when GM storage is necessary, reduce repeated reads and writes |

### Load balancing

| Check item          | Description                         |
| ------------------- | ----------------------------------- |
| Number of basic blocks per core | For Matmul, look at the M/N basic block distribution first |
| Number of valid blocks per core | For FAI causal/mask scenarios, estimate weights by valid blocks |
| Tail-round size     | For Stream-K, focus on `B % C` and the number of K slices |
| swizzle             | Adjust the swizzle direction and offset after tiles are determined |

## Summary

The core of performance optimization for CV fused operators is "balancing" and "coverage": make the single-stage latencies of AIC and AIV as close as possible, use stages, workspace, or on-chip caches to overlap production and consumption, and at the same time avoid the bloat caused by tiling, synchronization, and GM transit.

The three examples have different emphases:

| Example        | Most important tuning question                               |
| -------------- | ------------------------------------------------------------ |
| quant_matmul   | Whether AIC int8 matmul and AIV dequantization are balanced, and whether the workspace stage is appropriate |
| streamk_matmul | Whether the tail round is sufficiently unbalanced, and whether early computation covers AIV reduction |
| FAI            | Whether QK/Softmax/PV/RescaleO form a stable pipeline, and whether multi-core valid blocks are balanced |
