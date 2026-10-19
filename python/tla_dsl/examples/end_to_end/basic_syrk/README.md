# BasicSyrk DSL Example Readme

## 功能说明

- 算子功能：计算输入矩阵与其转置的乘积，得到对称矩阵。
- 计算公式：

  $$
    \begin{aligned}
    Y &= X \times X^T \\
    Y_{i,j} &= \sum_k X_{i,k}X_{j,k}
    \end{aligned}
  $$

  其中 $X$ 是形如 `(m, k)` 的输入矩阵，$Y$ 是形如 `(m, m)` 的输出矩阵，满足 $Y=Y^T$。
- 支持产品型号：Ascend 950PR&950DT系列产品。

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `X` | Input | `(m, k)` | `float16/bfloat16` | 连续 NPU 输入张量，layout 仅支持 `RowMajor`
| `Y` | Output | `(m, m)` | `float16/bfloat16` | 对称矩阵乘结果，layout 为 `RowMajor`，数据类型与输入一致 |

Python 接口为 `syrk(x, block_num=-1)`，在输入所在设备的当前流上启动，函数内部不进行同步。
block_num 用于指定参与计算的 Cube 核数量，不设置时自动选择，设置为正整数时按指定数量执行。

## 使用范围说明

本样例按 `256×256` 输出块计算下三角，并通过 RowMajor/ColumnMajor 写回补全对称位置；L1/L0 的 K 分块分别为 128/64，使用双缓冲。默认采用 `GemmIdentityBlockSwizzle<3,1>` 调度，通过 CUBE/FIX 事件保护 L0C 双写，`unit_flag=0`。需要安装已包含 L0C→ColumnMajor GM 支持的 DSL wheel。

使用范围：

- 输入为连续二维 NPU 张量，`1 ≤ M ≤ 8192`、`1 ≤ K ≤ 65536`，且 `MK ≤ 8192² = 67108864`；M、K 可独立取值，支持尾块。越界检查在命令行分配测试张量前执行，Python 接口同样检查。
- 输入、输出使用 ND RowMajor 存储，计算使用 fp32 累加；`X^T` 由输入存储的 ColumnMajor 视图表示，无需独立转置输入。
- 默认启动核数取设备 Cube 核数与输出块网格大小中的较小值；显式指定正整数 `block_num` 时，取指定值与输出块网格大小中的较小值。
- 当前仅实现 `Y = X @ X.T`，不支持 alpha/beta、输入 C、fp32 输入或仅返回单侧三角结果。

SYRK 输出占 `2M²` 字节，M 翻倍时输出规模变为四倍，因此限制 `M ≤ 8192`；输入占 `2MK` 字节，另限制 `MK ≤ 8192²`。单份 fp16/bf16 输入、输出均不超过 128 MiB，为 10 GiB 主机内存预算下的参考计算及比较留出空间。例如允许 `(1024,65536)`，拒绝 `(8192,65536)`；该范围是交付测试约束，不是算子理论极限或进程内存的硬保证。

推荐范围：

- 在上述范围内，推荐 M 为 256 的整数倍、K 为 128 的整数倍，以匹配分块并避免尾块处理。
- 推荐采用默认自动核数；单核调试时可指定 `block_num=1`。小尺寸可能无法充分利用多核并行。

## 使用示例

### 命令行参数

```bash
python basic_syrk.py [m M] [k K] [dtype {f16,bf16}] [device DEVICE] [block-num BLOCK_NUM]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | `1024` | 输入行数及输出边长 |
| `k` | `1024` | 归约维度 |
| `dtype` | `bf16` | 输入和输出类型，可选 `f16`、`bf16` |
| `device` | `0` | 指定运行设备 ID |
| `block-num` | `-1` | 默认自动选择 Cube 核数，也可指定正整数 |

### 执行示例

1. 进入项目根目录，参考[快速上手](../../../docs/zh/quick_start.md)准备样例代码的运行环境。

2. 在项目根目录下执行算子样例程序。

   ```bash
   python python/tla_dsl/examples/end_to_end/basic_syrk/basic_syrk.py --m 1024 --k 1024 --dtype bf16 --device 0
   ```

   若提示 BC 缺失或版本不匹配，先执行 `python -m catlass.bc_compile`。

3. 执行结果如下，说明样例执行成功，精度和对称性检查通过：

   ```text
   passed=True symmetric=True
   ```
