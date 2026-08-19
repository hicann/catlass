# Batched Matmul 端到端示例

本目录下的样例演示 **CATLASS DSL** 下的批量矩阵乘，对齐 C++ 参考实现 `examples/67_ascend950_batched_matmul*`。

## 功能说明

对每个 batch 计算

$$
C_b = A_b @ B_b,\quad b = 0,\ldots,B-1
$$

各 batch 的 `(M, N, K)` 相同。



## 代码组织

```plain
./batched_matmul
├── batched_matmul.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`batched_matmul.py`**](batched_matmul.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。执行数据生成、compile/launch、以及golden比对。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

## 使用示例

要运行本路径下的样例，请参考[环境配置](../../../docs/zh/dev_guide/00_environment_setup.md)完成部署。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--batch` | `5` | batch 数 B。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `512` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `1024` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `f16` / `f16` / `f16` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |
| `--block` | `8` | 启用的核数。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd "${CATLASS_ROOT}/python/tla_dsl/examples/end_to_end/batched_matmul"

python batched_matmul.py --run --device 0 --batch 4 --m 256 --n 256 --k 256 --block 8

```

执行测试后，预期输出：

```text
compile_ok=True ...
launch_ok=True
C equals batched golden? True
first mismatch=None
```
