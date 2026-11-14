# Ascend950BmmBroadcast Example Readme

## 功能说明

- 算子功能：完成支持 batch 广播、输入转置和可选 Bias 的批量矩阵乘。
- 计算公式：

  $$
    C_b = \operatorname{cast}_T\left(\operatorname{op}(A_{b_A}) \times \operatorname{op}(B_{b_B}) + \operatorname{Bias}_b\right)
  $$

  其中，A、B 的 batch 为 1 时共享输入，否则使用对应 batch 的输入；`op` 表示是否转置。
  Bias 可省略，或为共享的 `(n)`、逐 batch 的 `(batch,n)`，沿 M 维广播。累加类型为 FP32。

- 支持产品型号：Ascend 950PR&950DT系列产品。

## 样例参数说明

下表说明输入、输出及数据类型，shape 表示物理存储形状。可执行样例固定使用 FP16、未转置 A/B 和共享 Bias。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batchA,m,k)` 或 `(batchA,k,m)` | `float16`、`bfloat16` | 分别对应 `transA=false`、`transA=true`；物理存储连续，batchA 为 1 或 batch。 |
| `B` | Input | `(batchB,k,n)` 或 `(batchB,n,k)` | `float16`、`bfloat16` | 分别对应 `transB=false`、`transB=true`；物理存储连续，batchB 为 1 或 batch。 |
| `Bias` | Input | `(n)` 或 `(batch,n)` | `float16`、`bfloat16` | `bias_mode=0` 时不使用，1 时共享，2 时逐 batch；沿 M 维广播，dtype 与 A/B 一致。 |
| `C` | Output | `(batch,m,n)` | `float16`、`bfloat16` | 矩阵乘结果，连续 RowMajor，dtype 与 A/B 一致。 |

## 使用范围说明

### 参数约束关系

- 本节使用范围为 FP16、BF16。A/B/Bias 使用相同数据类型，位于同一设备且物理存储连续；输出类型与 A/B 一致，累加类型为 FP32。
- `batch`、`m`、`n`、`k` 为正整数。接口用 32 位无符号字段记录 batch 和总输出任务数，用 32 位有符号字段记录 M/N/K；后者的正值占 31 位。下表的维度字段边界由这些位数确定，各参数组合还须同时满足分块计数与存储容量要求。
- A/B 的逻辑形状分别为 `(batchA,m,k)` 和 `(batchB,k,n)`，K 维相等。`batchA`、`batchB` 各自为 `1` 或 `batch`：取 `1` 时对应输入在各输出 batch 间共享。转置只交换各输入的后两维，输出形状始终为 `(batch,m,n)`。
- `bias_mode=0` 时无 Bias；`1` 时 Bias 形状为 `(n)`，沿 batch 和 M 维广播；`2` 时 Bias 形状为 `(batch,n)`，仅沿 M 维广播。
- 输出按 `64*128` 的 M/N 分块调度。每个 batch 的任务数为 `ceil(m/64)*ceil(n/128)`，总任务数再乘 `batch`，且总数须能存入上述无符号任务字段；`ceil` 表示向上取整。例如 `m=4096,n=2048` 时，每个 batch 有 `(4096/64)*(2048/128)=64*16=1024` 个任务；`batch=3` 时共 `3072` 个任务。
- FP16/BF16 每元素占 `2 Byte`。A、B、C 的元素数分别为 `batchA*m*k`、`batchB*k*n`、`batch*m*n`；共享 Bias 为 `n`，逐 batch Bias 为 `batch*n`。分配容量须覆盖完整张量，输出不得与输入重叠。例如 `(batch,m,n,k)=(3,4096,2048,8)`、`batchA=3,batchB=1` 及共享 Bias 时，A 为 `3*4096*8*2 Byte=192 KiB`，B 为 `1*8*2048*2 Byte=32 KiB`，C 为 `3*4096*2048*2 Byte=48 MiB`，Bias 为 `2048*2 Byte=4 KiB`；其中 `1 KiB=1024 Byte`，`1 MiB=1024 KiB`。

### FP16 的 K 与数据幅值约束

FP16 最大有限值由其数值格式决定，为 `(2-1/1024)*32768=65504`。令 `a_max`、`b_max`、`bias_max` 分别为 A、B、Bias 元素的最大绝对值，无 Bias 时 `bias_max=0`。计入输入转换与 FP32 累加的舍入余量后，采用以下防溢出条件：

```text
u16 = 1/2048
u32 = 1/(4096*4096)
e = (k+1)*u32
output_bound = (k*a_max*b_max*(1+u16)**2 + bias_max*(1+u16))/(1-e)
e < 1
output_bound <= 65504
```

其中 `u16` 和 `u32` 分别是 FP16 的 11 位有效精度、FP32 的 24 位有效精度对应的单位舍入误差。给定输入幅值后，代入整数 K，取满足条件且未超过维度字段边界的最大值，即得到对应的 K 范围。

例如 A/B/Bias 元素均在 `[-5,5]` 时，`a_max=b_max=5`。无 Bias 时，K 为 `2617`、`2618` 的估计上界分别为 `65499.13`、`65524.16`，因此防溢出范围取到 `2617`；带 Bias 时取 `bias_max=5`，K 为 `2616`、`2617` 的估计上界分别为 `65479.10`、`65504.13`，范围取到 `2616`。数据幅值改变时，按同一公式重新计算 K 范围。

### 参数取值示例

下表分别列出维度字段边界、数据幅值条件和功能取值，示例使用 `batch=3,m=67,n=131,k=145`。维度字段边界需与前述分块、存储和防溢出条件共同满足。

| 参数 | 约束范围或允许取值 | 简例 |
| --- | --- | --- |
| `batch` | 最小 `1`；最大为 32 位无符号 batch 字段的最大值，同时受总输出任务数限制 | `batch=3`。 |
| `batchA` | `1` 或 `batch` | `batch=3` 时取 `1` 表示 A 广播，取 `3` 表示不广播。 |
| `batchB` | `1` 或 `batch` | `batch=3` 时可取 `1` 或 `3`。 |
| `m` | 最小 `1`；最大为 32 位有符号维度字段的最大正值 | `m=67`。 |
| `n` | 最小 `1`；最大为 32 位有符号维度字段的最大正值 | `n=131`。 |
| `k`，FP16，无 Bias | 最小 `1`；同时满足维度字段边界和上述防溢出条件 | A/B 元素值在 `[-5,5]` 时，防溢出范围为 `1~2617`。 |
| `k`，FP16，带 Bias | 最小 `1`；同时满足维度字段边界和上述防溢出条件 | A/B/Bias 元素值在 `[-5,5]` 时，防溢出范围为 `1~2616`。 |
| `k`，BF16 | 最小 `1`；最大为 32 位有符号维度字段的最大正值 | `k=145`。 |
| A 元素值 | 由所用数据类型确定；FP16 还须满足上述输出防溢出条件 | 以 `[-5,5]` 为例，可取 `-5`、`0`、`5`。 |
| B 元素值 | 由所用数据类型确定；FP16 还须满足上述输出防溢出条件 | 以 `[-5,5]` 为例，可取 `-5`、`0`、`5`。 |
| Bias 元素值 | 由所用数据类型确定；FP16 还须满足上述输出防溢出条件 | 以 `[-5,5]` 为例，可取 `-5`、`0`、`5`；无 Bias 时不传入。 |
| A/B/Bias 数据类型 | `float16`、`bfloat16`，三者一致 | A/B 为 `float16` 时，传入的 Bias 和输出 C 也为 `float16`。 |
| `transA` | `false`、`true` | A shape 分别为 `(batchA,67,145)`、`(batchA,145,67)`。 |
| `transB` | `false`、`true` | B shape 分别为 `(batchB,145,131)`、`(batchB,131,145)`。 |
| `bias_mode` | `0`、`1`、`2` | 分别对应无 Bias、Bias shape `(131)`、Bias shape `(3,131)`。 |

可执行样例固定配置为 `float16`、`transA=false`、`transB=false`、共享 Bias；命令行参数见下文。

## 使用示例

### 命令行参数

```bash
88_ascend950_bmm_broadcast [batch] [batchA] [batchB] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | 输出批次数 |
| `batchA` | 无 | A 的批次数，1 或 batch |
| `batchB` | 无 | B 的批次数，1 或 batch |
| `m` | 无 | 输出矩阵的 m 轴大小 |
| `n` | 无 | 输出矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 88_ascend950_bmm_broadcast -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./88_ascend950_bmm_broadcast 3 3 1 67 131 145 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
