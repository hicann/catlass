# 多核切 K Matmul（example 68 / 69）

本目录包含两个单文件示例（Kernel + Host），结构均与 [`basic_matmul.py`](../basic_mmad/basic_matmul.py) 对齐：

| 文件 | 算法 |
|------|------|
| `multi_core_splitk_matmul.py` | **ex68**：全部 M×N tile 做 K 维 split-K → workspace，AIV ReduceAdd → C |
| `tail_multi_core_splitk_matmul.py` | **ex69**：normal tile full-K 直写 C；tail tile（`mn_blocks % AIC != 0`）再 split-K + ReduceAdd |

CLI 与 basic 同形：`--dtype-a/b/c`、`--block-dim`（default `-1` → 平台 AICore 数）等。当前要求 **A/B/C dtype 同型**；`SWIZZLE_DIRECTION` 由 Host 按 `m>n`→Zn / 否则 Nz 注入。

## 运行

在 `python/tla_dsl` 目录下：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ex68
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --device 0 --m 256 --n 512 --k 1024

# ex69（诱导 tail：mn_blocks=32, block_dim=24 → normal=24, tail=8）
python examples/end_to_end/multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py \
  --device 0 --m 2048 --n 1024 --k 2048 --block-dim 24 --force-recompile
```

常用参数：`--device/--m/--n/--k/--layout-a/--layout-b/--dtype-a/--dtype-b/--dtype-c/--block-dim/--sentinel/--cache-dir/--force-recompile/--no-cache`。完整列表见各脚本 `--help`。

## ex69 尾轮调度摘要

- L1 `256×256×128`；L0 `256×256×32`；Zn/Nz swizzle 与 C++ `DynamicSplitkGemmIdentityBlockSwizzle` 一致。
- `compute_tail_scheduler`：`tail_block_num = mn_blocks % AIC`，`splitk_factor = min(AIC/tail, k_tiles)`。
- Workspace 按 AIC 行槽存放 tail 部分积；L0C / W / reduce UB 为 fp32。
