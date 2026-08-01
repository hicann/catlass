# 尾轮多核切 K Matmul（example 69 → catlass-dsl）

将 Catlass C++ 样例 `69_ascend950_tail_multi_core_splitk_matmul` 转为 TLA DSL：
**正常 MN 块**完整 K 直写 C；**尾轮 MN 块**切 K 写 per-AIC workspace，AIV 仅对尾块 ReduceAdd。

| 文件 | 作用 |
|------|------|
| `tail_multi_core_splitk_matmul.py` | Host：`--dump-tlair` / `--build-only` / `--run` |
| `tail_splitk_mmad_kernels.py` | 单 mixed kernel（cube + vector） |

对应 C++ 样例目录：`examples/69_ascend950_tail_multi_core_splitk_matmul`（仓库根下）。

## 与 example 68 的差异

| | 68 | 69（本目录） |
|--|----|--------------|
| 切 K | 全部 MN | 仅尾轮 MN |
| Workspace | `(factor·M, N)` | `aic · L1_M · L1_N`（≥10MB） |
| 正常块 | — | 写 GM C |
| Host | f16/bf16/f32 × 四种 A/B layout；任意正 shape | 同左（非对齐 MN/K 经 origin） |

`cross_flag` 使用上游 Ascend C-like API（`mode` 在 flag 上，pipe 在 set/wait 上），与 example 68 一致。

## 环境前提

CANN / conda / AscendNPU-IR / `build.sh` 等见仓库内 [`python/tla_dsl/README.md`](../../../README.md) **§2**。上板需 **PyTorch** 与 **torch_npu**。

## 运行

在 `python/tla_dsl` 目录下：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 默认（常无尾轮 / normal=0）
python examples/end_to_end/tail_multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py \
  --run --device 0 --m 256 --n 512 --k 1024 --block 24 --force-recompile

# 尾轮诱导（mn_blocks=32, C=24 → normal=24, tail=8）
python examples/end_to_end/tail_multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py \
  --run --device 0 --m 2048 --n 1024 --k 2048 --block 24 --force-recompile
```

`--device` 请换成实际空闲 NPU id。期望：打印 `ok=True`。更多参数见 `--help`。
