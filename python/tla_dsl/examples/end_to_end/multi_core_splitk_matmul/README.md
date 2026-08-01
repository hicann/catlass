# 多核切 K Matmul（example 68 → catlass-dsl）

将 Catlass C++ 样例 `68_ascend950_multi_core_splitk_matmul` 转为 TLA DSL：AIC 按 K 切片写 workspace 部分积，AIV ReduceAdd 得到最终 C。

| 文件 | 作用 |
|------|------|
| `multi_core_splitk_matmul.py` | Host：`--dump-tlair` / `--build-only` / `--run` |
| `splitk_mmad_kernels.py` | **单 mixed kernel**：`cube` + `vector` |

对应 C++ 样例目录：`examples/68_ascend950_multi_core_splitk_matmul`（仓库根下）。

## 同步（对齐 C++）

```text
AIC: work → cross_core_set_flag(aic_finish, FIX)           # mode 2
AIV: cross_core_wait_flag(aic_finish, MTE2)                # mode 2
     cross_core_set/wait_flag(aiv_ibarrier, MTE2)          # mode 0
     ReduceAdd（按行×VL，语义同 C++ flat M×N）
```

## 交付约定

| 项 | 支持 |
|----|------|
| dtype | **A/B/C 同型**：`f16` \| `bf16` \| `f32`（`--dtype`） |
| L0C / workspace | **恒 fp32** |
| A/B layout | **row / col**（各自独立） |
| C / W layout | **仅 RowMajor** |
| shape | **任意正整数 M/N/K**（尾块 in-kernel） |

- 默认 shape：`M=256, N=512, K=1024`
- L1：`256×256×128`；L0：`256×256×32`
- `splitkFactor`：与 C++ `GetSplitkFactor` 一致，由 `--block` 决定

## 环境前提

CANN / conda / AscendNPU-IR / `build.sh` 等见仓库内 [`python/tla_dsl/README.md`](../../../README.md) **§2**。上板需 **PyTorch** 与 **torch_npu**。

## 运行

在 `python/tla_dsl` 目录下：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 默认 f32 row/row
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --run --device 0 --m 256 --n 512 --k 1024 --block 24 --force-recompile

# ColumnMajor A/B
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --run --device 0 --layout-a col --layout-b col --force-recompile

# 同 dtype f16
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --run --device 0 --dtype f16 --force-recompile

# 不规则 shape
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --run --device 0 --m 333 --n 444 --k 555 --force-recompile
```

`--device` 请换成实际空闲 NPU id。期望：打印 `ok=True`。更多参数见 `--help`。
