# Matmul EVG 端到端示例

本目录下提供的系列样例演示 **CATLASS DSL** 下 基于 Ascend950 的 GEMM + EVG（Epilogue Visitor Graph）尾处理，对齐 C++ 参考实现 `examples/64_ascend950_matmul_evg_*`。

## 功能说明

基础矩阵乘算子实现形如 `(m, k)` 和 `(k, n)` 两矩阵的乘法，输出形如 `(m, n)`：

$$
\begin{aligned}
D &= A \times B \oplus \text{Epilogue}
\end{aligned}
$$

各变体在 GEMM 之后接入不同的 EVG 尾处理算子（Add、Bias、LeakyRelu、Sigmoid、Silu、Tanh 等），由 AIV 融合完成。


## 代码组织

```plain
./basic_mmad_evg
├── matmul_add.py
├── matmul_add_ub.py
├── matmul_bias.py
├── matmul_leaky_relu.py
├── matmul_sigmoid.py
├── matmul_silu.py
├── matmul_tanh.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`matmul_add.py`**](matmul_add.py) | D = A×B + X；L0C→GM + AIV。 |
| [**`matmul_add_ub.py`**](matmul_add_ub.py) | D = A×B + X；L0C→UB + AIV。 |
| [**`matmul_bias.py`**](matmul_bias.py) | D = A×B + bias(1×N)；L0C→GM + RowBroadcast。 |
| [**`matmul_leaky_relu.py`**](matmul_leaky_relu.py) | D = LeakyRelu(A×B)，α=0.1；L0C→GM + AIV。 |
| [**`matmul_sigmoid.py`**](matmul_sigmoid.py) | D = Sigmoid(A×B)；L0C→GM + AIV。 |
| [**`matmul_silu.py`**](matmul_silu.py) | D = Silu(A×B)；L0C→GM + AIV。 |
| [**`matmul_tanh.py`**](matmul_tanh.py) | D = Tanh(A×B)；L0C→GM + AIV。 |

## 约束说明

| 算子 | `DTYPE_A` / `DTYPE_B` | `DTYPE_C` |
|------|-------------------|---------|
| add, bias, leaky_relu, sigmoid, silu | f16 / bf16 / f32 | f16 或 f32 |
| add_ub, tanh | f16 / bf16 / f32 | f32 |


## 使用示例

要运行本路径下的样例，请参考[环境配置](../../../docs/zh/dev_guide/00_environment_setup.md)完成部署。

### 命令行参数

主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `256` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `256` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `row` / `row` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `f32` / `f32` / `f32` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |


### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 其余变体替换文件名即可
python examples/end_to_end/basic_mmad_evg/matmul_add.py --device 0 \
  --m 256 --n 256 --k 256 \
  --layout-a row --layout-b row \
  --dtype-a f16 --dtype-b f16 --dtype-c f32

```
