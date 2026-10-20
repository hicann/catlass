# Flash Attention Infer Operator Performance Optimization

## Before We Start

This document mainly explains how to perform performance analysis and tuning of the Flash Attention Infer (FAI) sample in `examples/49_ascend950_flash_attention_infer`, starting from the operator structure and source implementation. This document analyzes FAI sample tuning from the following perspectives:

- Where the main performance bottlenecks of the FAI sample may come from.
- Which parameters are compile-time template parameters, and which parameters come from tiling or runtime inputs.
- Which on-chip resources and execution paths are affected when adjusting TileShape, the CV pipeline, the core partitioning strategy, the input layout, or the Paged Attention parameters.

## Sample Introduction

This sample targets the Flash Attention inference forward operator `FlashAttentionInfer`. The computation flow is:

```text
S = Q * K^T * scale + mask
P = softmax(S)
O = P * V
```

In the source implementation, this computation is split into four main stages:

| Stage          | Execution unit | Main work                                                | Data flow   |
| -------------- | -------------- | -------------------------------------------------------- | ----------- |
| QK             | AIC            | Compute `Q * K^T`                                        | Q/K -> S    |
| Online Softmax | AIV            | Handle mask, max, sum, exp and produce P                 | S -> P      |
| PV             | AIC            | Compute `P * V`                                          | P/V -> Otmp |
| RescaleO       | AIV            | Normalize and update O based on the online softmax state | Otmp -> O   |

FAI computation involves multiple CV operations. The goal of optimizing the FAI operator is to make the chain `QK -> Online Softmax -> PV -> RescaleO` form as stable a pipeline as possible between AIC/AIV, and to avoid long tails across cores as much as possible.

Several key implementation points of the current sample are as follows:

```cpp
// examples/49_ascend950_flash_attention_infer/fai_tiling.h
constexpr int32_t BLOCK_BASE_SIZE = 128;

// examples/49_ascend950_flash_attention_infer/fai_kernel.h
using L1TileShape = tla::Shape<_128, _128, _128>;
using L0TileShape = L1TileShape;

// examples/49_ascend950_flash_attention_infer/fai_kernel_utils.h
constexpr uint32_t KERNEL_TASK_NUM = 3;
```

The three dimensions of `L1TileShape` correspond to `qSeqBase`, `kvSeqBase`, and `embedBase` respectively. In the current sample, `L1TileShape::K` is required to match the head dimension, and by default it applies to the `headDim = 128` scenario.

Note in particular that `BLOCK_BASE_SIZE` and `L1TileShape` are not two completely independent parameters. `BLOCK_BASE_SIZE` is used in the tiling stage to count q/kv basic blocks and multi-core weights, while `L1TileShape` is used for the tile shape actually executed by the kernel. Modifying only one of them may cause the tiling weight model and the kernel's actual loop granularity to become inconsistent.

## Optimization Approaches

### Adjusting the TileShape Parameter Combination

TileShape determines the basic granularity of each QK, Softmax, PV, and RescaleO processing step. A larger `qSeqBase` or `kvSeqBase` usually reduces the number of outer loops, the number of synchronizations, and the number of small-block transfers, but it increases UB, L1, and L0 resource usage, and may also introduce more invalid computation in mask or tail-block scenarios.

Before adjusting, at least the following static resource items need to be checked:

```cpp
MM1_RESULT_SIZE = qSeqBase / CV_RATIO * kvSeqBase * sizeof(ElementS);
MM2_RESULT_SIZE = qSeqBase / CV_RATIO * Align64(embedBase) * sizeof(ElementOTmp);
SHARE_UB_SIZE = CeilDiv(qSeqBase, NUM2) * sizeof(ElementS);
MM2_LEFT_SIZE = qSeqBase * kvSeqBase * sizeof(ElementP);
```

These resource items mainly affect:

| Resource    | Main occupancy                                                      |
| ----------- | ------------------------------------------------------------------- |
| UB          | `bmm1TensorList[2]`, `bmm2TensorList[2]`, `sumUb`, `expUb`, `maxUb` |
| L1          | `mm2AL1TensorList[KERNEL_TASK_NUM]`                                 |
| L0A/L0B/L0C | BlockMmad tiles of QK/PV                                            |
| Sync events | Pipeline synchronization among C1/V1, V1/C2, C2/V2                  |

Common adjustment directions are as follows:

| Symptom                                              | Adjustment direction                                         | Risk                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------- |
| Many q/kv outer loops, frequent synchronization      | Increase `qSeqBase` or `kvSeqBase`                           | Higher UB/L1/L0 usage                                             |
| AIV Softmax or RescaleO starts frequently            | Prefer increasing `qSeqBase`                                 | q tail-block waste may increase                                   |
| Too many small QK/PV matmuls                         | Prefer increasing `kvSeqBase`                                | More invalid kv blocks under causal or sparse masks               |
| Obvious causal mask tail-block waste                 | Increase `kvSeqBase` cautiously, reduce the kv tile if needed | Loop and synchronization counts increase                          |
| `headDim != 128`                                     | Modify `L1TileShape::K` accordingly and check static buffers | BlockMmad and intermediate buffer constraints may not be satisfied |

If you only enlarge TileShape without computing a resource budget, optimization easily degenerates into random trial and error. A more reasonable approach is to first list candidate `qSeqBase/kvSeqBase/embedBase` values according to the target shape, confirm that UB/L1/L0 resources can accommodate them, and then perform performance verification.

### Optimizing AIC/AIV Cooperative Pipelining

In FAI, AIC handles QK/PV matrix multiplication, and AIV handles Online Softmax and RescaleO. Although there are stage dependencies within the same `kvSeq` block, adjacent `kvSeq` blocks can form a pipeline overlap. The ideal steady-state form is:

```text
task t:     AIC QK
task t+1:   AIV OnlineSoftmax
task t+2:   AIC PV
task t+3:   AIV RescaleO
```

The key pipeline-related objects in the source code include:

| Object                               | Role                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `taskId`                             | Pipeline task index                                      |
| `runInfo[4]`                         | Stores the execution state of different pipeline stages  |
| `bmm1TensorList[2]`                  | Double buffer of the QK output UB                        |
| `bmm2TensorList[2]`                  | Double buffer of the PV output UB                        |
| `mm2AL1TensorList[KERNEL_TASK_NUM]`  | L1 cache for the Softmax result P                        |
| `sumUb/maxUb/expUb[KERNEL_TASK_NUM]` | Online Softmax state cache                               |

The synchronization paths are as follows:

| Sync flag         | Direction  | Meaning                                     |
| ----------------- | ---------- | ------------------------------------------- |
| `SYNC_C1_V1_FLAG` | AIC -> AIV | The QK result is ready; AIV can run Softmax |
| `SYNC_V1_C2_FLAG` | AIV -> AIC | P has been moved to L1; AIC can run PV      |
| `SYNC_C2_V2_FLAG` | AIC -> AIV | The PV result is ready; AIV can update O    |

The key data path in the sample is that AIV moves the Softmax result P directly from UB to the AIC-side L1:

```cpp
using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
```

This path avoids the `UB -> GM -> L1` detour, reducing GM bandwidth pressure and also reducing the wait before PV starts.

- If AIC frequently waits for AIV, it usually means the Vector side has become the bottleneck: check Softmax, mask reads, RescaleO, frequent AIV scheduling caused by a small `qSeqBase`, and whether `KERNEL_TASK_NUM` is insufficient to cover the Vector latency first.
- If AIV frequently waits for AIC, it usually means Cube or data movement has become the bottleneck: check the QK/PV tile granularity, K/V layout contiguity, MTE2 transfers, and Paged Attention jump reads first.

Increasing `KERNEL_TASK_NUM` may improve pipeline coverage capability, but it also increases the `mm2AL1TensorList` occupancy in L1, the Softmax state occupancy in UB, synchronization event pressure, and the complexity of the final flush stage. Therefore, first confirm that the TileShape and multi-core partitioning are reasonable, and then consider adjusting the pipeline depth.

### Optimizing Multi-Core Load Balancing

In causal or sparse mask scenarios, different `qSeq` blocks can see different numbers of `kvSeq` blocks. If cores are simply split evenly by the number of qSeq rows, the effective computation amount of early and late q blocks may differ greatly, and the longest-tail core will determine the overall duration.

FAI performs greedy partitioning by the number of valid blocks in the tiling stage. The core logic is located in `examples/49_ascend950_flash_attention_infer/fai_tiling.h`:

| Function or field         | Role                                                                    |
| ------------------------- | ----------------------------------------------------------------------- |
| `GetCalcBlockNumsOneHead` | Counts the number of valid `qSeq x kvSeq` blocks after masking under a single head |
| `ComputeSplitNBSeq`       | Performs greedy partitioning along the Batch/Head/qSeq three axes        |
| `bnAxisStartIdx`          | Records the starting batch/head of each AI Core                          |
| `sparseStartIdx`          | Records the starting qSeq block of each AI Core                          |

The core partitioning strategy can be understood with the following abstract weights:

```text
core_weight[i] = sum(valid_kv_blocks for qSeq blocks assigned to core i)
target_weight = total_valid_blocks / blockDim
imbalance = max(core_weight) / average(core_weight)
```

If the number of valid blocks differs greatly across cores, check the mask mode, `actualSeqLengths`, `actualSeqLengthsKV`, and `blockDim` first. If the number of valid tasks itself is very small, using too many AI Cores only adds synchronization and scheduling overhead. In Paged Attention scenarios, note also that the cost of block table jump reads is not fully reflected in the number of valid blocks; even if the number of valid blocks is similar across cores, duration differences may still appear due to different physical KV cache layouts.

Greedy core partitioning also has limitations: when the weight of a single qSeq block is greater than the target core weight, it cannot further split that block; and when the batch is small, qSeq is very short, or there are very few valid blocks after mask clipping, it cannot fill all cores either. In such scenarios, consider reducing `blockDim`, adjusting `qSeqBase`, or redesigning a finer-grained partitioning strategy.

### Optimizing Input Layout, MTE2, and FixPipe

When performance is affected by data movement, first confirm that the business-side input physical layout is consistent with the kernel TLA layout. In the current sample, Q/O are contiguous along `[batch * qSeqlen, qHeads * embed]`, V is contiguous along `[batch * kvSeqlen, kvHeads * embed]`, and K uses ColumnMajor to adapt to the `Q * K^T` access.

MTE2-related issues usually come from:

- The upper-layer actual layout is inconsistent with the kernel TLA layout.
- K/V cross large strides on the QK or PV access path.
- In GQA scenarios, the `qHeads / kvHeads` grouping is not contiguous.
- The mask tensor is not stored contiguously as `[batch * qSeqlen, kvSeqlen]`.
- Under Paged Attention, the block table causes frequent KV cache jump reads.

If the business side cannot change the input physical layout, consider adding a dedicated `LayoutTag` or TileCopy adaptation. However, such modifications simultaneously affect the QK, PV, and mask access paths, and correctness and performance must be re-verified.

FixPipe-related issues usually appear on the O write-back path, especially when `embed` is small, there are many tail blocks, the O address is not aligned, or the workspace segments are not aligned. It is recommended to ensure that the start addresses of Q/K/V/O are aligned to 512 Bytes as much as possible, and that the contiguous-dimension strides meet the data movement alignment requirements as much as possible. If a GM workspace is later introduced to cache intermediate results, when allocating multiple workspace segments, use a method such as `RoundUp(offset, 512)` to ensure that the start address of each segment is aligned.

### Optimizing the Paged Attention Path

After `PAGED_CACHE_FLAG` is enabled, the kernel rounds `kvSeqlen` up by `blockSize` and accesses the KV cache indirectly through `blockTables`:

```cpp
kvSeqlen = RoundUp(kvSeqlen, blockSize);
```

A `blockSize` that is too small reduces tail-block waste, but increases the number of block table accesses and the probability of KV jump reads; a `blockSize` that is too large favors contiguous access within a page, but may increase invalid computation in the tail block. When optimizing Paged Attention, first ensure that the KV cache is contiguous within a page, that page start addresses are aligned, that the block table order matches the kernel access order, and that `actualSeqLengthsKV` participates accurately in core partitioning.

If high MTE2 usage or cross-core duration differences occur only in the Paged Attention scenario, check the physical layout of the block table and KV cache first, rather than directly shrinking the TileShape. TileShape can only change the granularity of a single computation; it cannot eliminate random jump reads caused by the page organization.

## Summary

Fused operator scenarios can refer to this case for optimization
