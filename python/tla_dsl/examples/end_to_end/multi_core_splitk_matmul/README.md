# 多核切 K Matmul 端到端示例

本目录下提供的样例演示 **CATLASS DSL** 下多核切 K 矩阵乘的两种实现，对齐 C++ 参考实现 `examples/68_ascend950_multi_core_splitk_matmul` 和 `examples/69_ascend950_tail_multi_core_splitk_matmul` 。

## 功能说明

功能和基础矩阵乘一致，实现形如$(m, k)$和$(k, n)$的两矩阵的乘法运算，输出形如$(m, n)$，计算公式为：
$$
\begin{aligned}
C &= A \times B \\
C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
\end{aligned}
$$

Split-K 将 K 维度的计算切分到多个核上并行执行，各核的部分积累入 workspace，再由 AIV 做 ReduceAdd 归约得到最终结果 C。

## 代码组织

```plain
./multi_core_splitk_matmul
├── multi_core_splitk_matmul.py
├── tail_multi_core_splitk_matmul.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`multi_core_splitk_matmul.py`**](multi_core_splitk_matmul.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。全部 M×N tile 做 K 维 split-K 写入 workspace，AIV 做 ReduceAdd 写回 GM C。 |
| [**`tail_multi_core_splitk_matmul.py`**](tail_multi_core_splitk_matmul.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。normal tile full-K 直接写回 gmC；tail tile 再做 split-K + ReduceAdd。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
|---------|---------|------------------|
| f16 | f16 | f16 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |


## 使用示例

要运行本路径下的样例，请参考[环境配置](../../../docs/dev_guide/00_environment_setup.md)完成部署。

### 命令行参数

CLI 与 basic 同形，主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `512` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `1024` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `row` / `row` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f16"` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# multi_core_splitk
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --device 0 --m 256 --n 512 --k 1024

# tail_multi_core_splitk
python examples/end_to_end/multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py \
  --device 0 --m 2048 --n 1024 --k 2048
```
