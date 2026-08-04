# Readme

基于catlass DSL框架，在Ascend950上实现的推理FlashAttention算子，对齐 C++ 参考实现 `examples/70_ascend950_flash_attention_chunk_prefill`

## 1. 代码组织

```text
├── flash_attention_infer/
│   ├── flash_attention_infer.py     # @tla.kernel + Host：构造输入、编译、调用kernel、精度校验
│   ├── fa_tiling.py                 # Tiling
│   └── README.md
```

`flash_attention_infer.py` 在同一文件内同时包含设备侧 `@tla.kernel` 函数与 host 侧运行/校验逻辑。编译期输入的 shape 参数集中定义在该文件顶部，kernel 与 host 同文件共享，改一处即两端同步生效。

## 2. 功能

### 2.1 算子功能

适配 Prefill 场景的 FlashAttention 算子，通过分块（Tiling）策略和 OnlineSoftmax 技术，避免物化完整的 $N \times N$ 注意力分数矩阵，将内存复杂度从 $O(N^2)$ 降至 $O(N)$，实现等价的attention计算。

计算公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

分块后单个 Q 基块 $Q_i$ 与 K 基块 $K_j$ 的注意力分数矩阵 $S_{ij}$（FP32）：

$$
S_{ij} = \text{scale} \cdot (Q_i K_j^T) \in \mathbb{R}^{B_r \times B_c}
$$

其中 $\text{scale} = 1/\sqrt{d_k}$（代码中 `QK_SCALE = 1.0 / (HEAD_DIM ** 0.5)`），$B_r$、$B_c$ 分别为 Q/KV 基块大小（`Q_BLOCK`、`KV_BLOCK`，Ascend950均适配为128）。

增量更新过程（Online Softmax）：

$$
m_{\text{new}} = \max(m,\ \text{rowmax}(S_{ij}))
$$

$$
P_{ij} = e^{S_{ij} - m_{\text{new}}}
$$

$$
O = O \cdot e^{m - m_{\text{new}}} + P_{ij} \cdot V_j
$$

$$
l = l \cdot e^{m - m_{\text{new}}} + \text{rowsum}(P_{ij})
$$

$$
m = m_{\text{new}}
$$

内层循环结束后归一化输出 $O_{\text{final}} = O / l$，结果转回 FP16 写回 GM。

### 2.2 已支持特性

| 特性 | 说明 |
| :--: | :-- |
| 完整 FA | QK + Online Softmax + PV + Rescale 四阶段全链路 |
| Prelaunch=2 流水 | Cube 超前 Vec 2 个基块，CV 跨迭代并行，避免PV计算因Vector侧softmax未完成而阻塞 |
| GQA | `KV_HEAD_NUM < HEAD_NUM`，`kv_head_idx = head_idx // GROUP_SIZE` |
| UBBank 冲突优化 | P 矩阵 UB 行 stride padding 1 个 32B block，避免因UBBank冲突带来的vec性能劣化 |
| Ascend950 新通路 | L0C→UB（S/OTmp）、UB→L1（P），中间矩阵不落 GM |
| Tiling 框架 | tilingdata / actual_seqlen 参数化，负载均衡（block=核数） |
| 多缓冲 | K/V double、P triple、S/OTmp double、L0C QK/PV 各自 double |

### 2.3 当前限制

- 形状为编译期常量驱动，运行期不支持动态 shape；改 shape 需改 `flash_attention_infer.py` 顶部常量并重新编译。
- `mask` 仅占位（全0），暂不支持0/1 mask。
- 暂不支持 PagedAttention（KV cache存放）。
- 序列长度需对齐到 `Q_BLOCK/KV_BLOCK`，尾部非对齐暂不支持。
- 仅支持 FP16 输入输出、FP32 中间计算、`HEAD_DIM=128`。

## 3. 接口

### 3.1 Kernel 接口

```python
@tla.kernel
def flash_attention_infer_kernel(
    mem_q: tla.Tensor,            # Q，GM，FP16，2D ND，[-1, HEAD_DIM]（BSND 展平），RowMajor
    mem_k: tla.Tensor,            # K，GM，FP16，2D ND，[-1, HEAD_DIM]（BSND 展平），ColumnMajor
    mem_v: tla.Tensor,            # V，GM，FP16，2D ND，[-1, HEAD_DIM]（BSND 展平），RowMajor
    mem_o: tla.Tensor,            # O，GM，FP16，2D ND，[-1, HEAD_DIM]（BSND 展平），输出，RowMajor
    mem_mask: tla.Tensor,         # mask，GM，INT8，2D ND [Q_SEQ, KV_SEQ]，暂不支持，只能为全 0
    tiling_data: tla.Tensor,      # tilingdata，Int32 1D（见 fa_tiling.pack_tiling_int）
    actual_q_seqlen: tla.Tensor,  # Q 前缀和序列，Int32，长度 batch+1（从0开始，actual_q_seqlen[0] = 0）
    actual_kv_seqlen: tla.Tensor, # KV 前缀和序列，Int32，长度 batch+1
)
```

> 注：Host 侧将 `[BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM]` 的 BSND 张量经 `reshape(-1, HEAD_DIM)` 展平为 2D 后再传入 kernel（见 `_reshape_qk_to_2d`）。K 以 ColumnMajor 传入以适配 $K^T$ 的矩阵乘布局。

### 3.2 编译期形状参数（`flash_attention_infer.py` 顶部）

| 参数 | 默认值 | 说明 |
| :-- | :-- | :-- |
| `HEAD_DIM` | 128 | 头维度 |
| `Q_BLOCK` | 128 | Q 基块（qBaseTile） |
| `KV_BLOCK` | 128 | KV 基块（kvBaseTile） |
| `Q_BLOCK_SUB` | 64 | UB 子块（Q_BLOCK // 2，配合 SPLIT_M 双 AIV及双缓冲） |
| `BATCH` | 2 | batch 数 |
| `HEAD_NUM` | 8 | Q 头数 |
| `KV_HEAD_NUM` | 2 | KV 头数（GQA） |
| `Q_SEQ` | 256 | Q 序列长度 |
| `KV_SEQ` | 256 | KV 序列长度 |
| `PRE_LAUNCH` | 2 | Cube 超前 Vec 的基块数 |
| `QK_SCALE` | $1/\sqrt{128}$ | 缩放系数 |

## 4. 如何执行

### 4.1 Host 参数

| 参数 | 类型 | 默认值 | 说明 |
| :-- | :-- | :-- | :-- |
| `--device` | int | `0` | NPU 设备 id |
| `--block` | int | `0` | 下发 block 数；`<=0` 表示使用设备全部 AICore（`tla.get_aicore_num`），实现负载均衡 |
| `--sentinel` | float | `-7.0` | O 的初始哨兵值，用于检测 kernel 是否真正写入 |
| `--cache-dir` | str | `./artifacts/runtime-cache` | 编译缓存与 `kernel.o` 输出目录 |
| `--force-recompile` | flag | — | 忽略已有缓存，强制重新编译 |
| `--no-cache` | flag | — | 禁用编译缓存复用 |

> 容差阈值为编译期常量，UNCHANGED_THRESHOLD判定 O 是否被写入、THRESHOLD判定精度是否通过。

### 4.2 执行命令

```bash
# 默认参数（全核、device 0）
python flash_attention_infer.py

# 指定 device 和 block 数
python flash_attention_infer.py --device 0 --block 28

# 强制重新编译
python flash_attention_infer.py --force-recompile
```

### 4.3 预期输出

```text
compile_ok=True host=torch_npu BATCH=2 Q_SEQ=256 KV_SEQ=256 HEAD_NUM=8 KV_HEAD_NUM=2 HEAD_DIM=128 ...
launch_ok=True
O unchanged (sentinel)? False
O changed count=... / ...
abs_err: max=0.00xxxx mean=0.0000xx
max_err_pos=[...] actual=... expected=... diff=0.00xxxx
passed? True
```

### 4.4 精度校验

Host 侧 (`flash_attention_infer.py`) 实现逐 Q 块 × KV 块的 Online Softmax 参考计算（FP32 累加），与 kernel 输出比对：

- `O unchanged (sentinel)?`：检测 O 是否被 kernel 写过（哨兵值未变即为异常）；
- `abs_err: max / mean`：最大/平均绝对误差及最大误差位置；
- `first_mismatch`：仅当最大绝对误差未通过精度阈值时打印前 5 个超阈值不匹配点（index/actual/expected）；
- `passed?`：通过判定。

## 5. 未来待实现的功能

以下功能已在 C++ 参考实现中支持，DSL 版本待开发。

### 5.1 seqlen 非对齐处理

当前要求 `Q_SEQ`、`KV_SEQ` 对齐到 `Q_BLOCK/KV_BLOCK`（128）。待支持尾部不满 128 行/列的处理：

- Q 尾部：`rowNum = q_seqlen - (q_block_count - 1) * Q_BLOCK`
- KV 尾部：`colNum = kv_seqlen - (kv_block_count - 1) * KV_BLOCK`
- 影响 L0 tile shape、SPLIT_M 拆分、UB store 的 mask 等多处，需结合运行期 seqlen 读取。

### 5.2 mask处理

当前 `mem_mask` 仅占位（全 0，GM→UB copy 但不参与计算），开发中。

### 5.3 PagedAttention / PagedCache

当前 K/V 在 GM 中连续存放（TND）。待支持 PagedAttention（KV cache 分页），K/V 的 shape 为 `PAGE_ND[num_blocks, kv_head_num, block_size, head_dim]` ，通过 `blockTable` 做逻辑到物理的地址映射：

```
blockTableIdx = kvSTileIdx * 128 / blockSize
blockOffset   = kvSTileIdx * 128 % blockSize
blockIdx      = blockTable[blockTableIdx]
物理地址       = blockIdx × blockSize + blockOffset
```

