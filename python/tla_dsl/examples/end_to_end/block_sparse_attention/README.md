# Block Sparse Attention 端到端示例

本目录下提供的样例演示 **CATLASS DSL** 下 Block Sparse Attention（BSA）的计算。

## 功能说明

Block Sparse Attention 算子实现带块级稀疏掩码（AnyMask）的注意力计算，计算公式为：

$$
\text{attentionOut} = \text{Softmax}(\text{scale} \times Q \times K_{\text{sparse}}^T) \times V_{\text{sparse}}
$$

BSA 通过将任意 mask 模式编码为块级位图、逐行边界、孔洞描述三组参数，在 kernel 运行时对每个 tile 做三分支判断（整块跳过 / 精细掩码 / 全计算），跳过全零 tile 的无效搬运和计算，从而在稀疏掩码场景下获得显著性能收益。

## 代码组织

本目录组织结构如下所示：

```plain
./block_sparse_attention
├── block_sparse_attention.py
├── bsa_tiling.py
├── mask_ref.py
└── README.md
```

各文件所承载功能概述如下：

| 文件 | 概述 |
|------|------|
| [**`block_sparse_attention.py`**](block_sparse_attention.py) | BSA kernel + host 运行/校验 + CLI，使用手动构造 `tla.Tensor` 创建设备张量。 |
| [**`bsa_tiling.py`**](bsa_tiling.py) | host 侧 Tiling 实现，产出 `FAInferTilingData` 并打包为 kernel 入参。 |
| [**`mask_ref.py`**](mask_ref.py) | 纯 CPU 的 mask 参考实现（maskgen 子集），支持 causal / doc_prefix / sliding_window / four_stage_forward 四种 mask 模式。 |

## 约束说明

- 支持的输入数据类型组合如下。

| 输入 dtype (Q/K/V) | 输出 dtype (O) | 累加类型 (S/OTMP/ACC) |
|---------|------------------|---------|
| f16 | f16 | f32 |
| bf16 | bf16 | f32 |

- 其他约束：

| 约束 | 说明 |
|---|---|
| `q_len <= kv_len`（每 batch） | 当前 kernel 不支持 q > kv 的语义。 |
| `num_heads % kv_heads == 0` | GQA 约束：Q 头数必须是 KV 头数的整数倍。 |

## 使用示例

要运行本路径下的样例，请参考[环境配置](../../docs/dev_guide/)完成部署。

### 命令行参数

```text
block_sparse_attention.py [-h] [--device DEVICE] [--qs QS] [--ks KS]
                          [--heads HEADS] [--kv-heads KV_HEADS]
                          [--head-dim HEAD_DIM]
                          [--dtype {fp16,bf16}]
                          [--pattern {causal,doc_prefix,sliding_window,four_stage_forward}]
                          [--format {BSND,TND}]
                          [--block-num BLOCK_NUM]
                          [--cache-dir CACHE_DIR]
                          [--force-recompile] [--no-cache]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--qs` / `--ks` | `128` / `128` | Q / KV 序列长度。 |
| `--heads` / `--kv-heads` | `1` / `1` | Q / KV 头数，需满足 `heads % kv-heads == 0`。 |
| `--head-dim` | `128` | head dim，当前固定为 128。 |
| `--dtype` | `"fp16"` | 输入数据类型，可选 `"fp16"` 或 `"bf16"`。 |
| `--pattern` | `"causal"` | mask 模式，可选 `"causal"` / `"doc_prefix"` / `"sliding_window"` / `"four_stage_forward"`。 |
| `--format` | `"BSND"` | Tensor 格式，可选 `"BSND"`（定长）或 `"TND"`（变长）。 |
| `--block-num` | `-1`（哨兵值，根据 NPU 设备采集满核值） | 所启用的 AI Core 核数。 |
| `--cache-dir` | `./artifacts/runtime-cache` | 编译缓存目录。 |
| `--force-recompile` | `False` | 强制重新编译。 |
| `--no-cache` | `False` | 禁用编译缓存。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 基础测试（默认 qs=128, ks=128, fp16, causal, BSND）
python examples/end_to_end/block_sparse_attention/block_sparse_attention.py

# 指定NPU ID、序列长度、数据类型、mask模式
python examples/end_to_end/block_sparse_attention/block_sparse_attention.py --device 1  --qs 256 --ks 512 --dtype bf16 --pattern doc_prefix
```

执行测试后，预期输出：
```plain
--- qs=<qs> ks=<ks> heads=<heads>/<kv_heads> d=<head_dim> dtype=<dtype> pattern=<pattern> fmt=<format> ---
[Step 1/4] mask_ref: 纯 CPU 生成 mask 六输出 ABI...
[Step 2/3] AnyMask 格式转换...
[Step 3/3] 准备 DSL kernel 输入并编译运行...
  编译 kernel...
kernel launch, start to run...
  kernel.o=<cache_dir>/<cache_key>/kernel.o
  kernel 执行完成 (<time>s)
[Step 4/5] 生成 dense mask 用于 golden...
[Step 5/5] 计算 golden reference 并比较...
  [分子] kernel vs 真值:  RMSE=<num_rmse>  MARE=<num_mare>  MERE=<num_mere>
  [分母] 标杆 vs 真值:    RMSE=<den_rmse>  MARE=<den_mare>  MERE=<den_mere>
  [比值] floor=<floor>  RMSE=<ratio_rmse>(<=1.2)  MARE=<ratio_mare>(<=2.0)  MERE=<ratio_mere>(<=1.2)

PASS  match_rate=<match_rate>  max_abs=<max_abs>  kernel=<time>s ratio_rmse=<...> ratio_mare=<...> ratio_mere=<...>
```

- 上述 `<qs>`、`<ks>` 等为占位符，具体依赖外部参数传入。
- `PASS` / `FAIL` 表明 NPU 计算结果与 golden 参考值精度校验是否通过。精度校验采用双标杆机制：以 `bsa_golden_attn` 为真值，`compute_golden_torch_bsnd` 为标杆，计算 kernel 与真值、标杆与真值的 RMSE / MARE / MERE 比值，判定标准为 `MARE比值 ≤ 2.0`、`MERE比值 ≤ 1.2`、`RMSE比值 ≤ 1.2`。
