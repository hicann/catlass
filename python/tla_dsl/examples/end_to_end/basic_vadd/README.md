# 基础 VADD 端到端示例

本目录下提供的系列样例演示 **TLA DSL** 下基础向量加法的计算，包括 GM→UB 拷贝、Vector 侧 SIMD 计算与写回 GM 的完整链路。

| 文件 | 概述 |
|------|------|
| [**`basic_vadd.py`**](basic_vadd.py) | 基础向量加法示例，含 4 个 kernel 变体：flag 同步（默认）、mutex（`--use-mutex`）、mutex_with（`--use-mutex-with`）和 atomic_add（`--use-atomic-add`）。 |


---

## 1. 基本介绍

向量加法算子实现两个一维向量的逐元素加法，计算公式为：
$$
\begin{aligned}
C &= A + B \\
C_i &= A_i + B_i
\end{aligned}
$$

### 1.1 问题规模与分块

- 向量长度 `N`：默认 400（ `VECTOR_ELE = 400` ），可通过 `--n` 指定。
- VL 分块：`VL_ELE` 按 dtype 自适应（`f32`=64、`f16`=128、`i8`=256、`i16`=128、`i32`=64）。
- 加和向量的排布格式均为 `tla.arch.RowMajor` ，即等价为 `(N, 1)` 的矩阵。

### 1.2 支持元素类型

- **`DTYPE_A` / `DTYPE_B`**：分别为加和向量的数据类型，要求二者类型相同，满足下述数据类型：

| dtype | VL_ELE | 浮点/整数 | 校验方式 |
|-------|--------|----------|---------|
| `f32` | 64 | 浮点 | 容差 |
| `f16` | 128 | 浮点 | 容差 |
| `i8` | 256 | 整数 | 精确匹配 |
| `i16` | 128 | 整数 | 精确匹配 |
| `i32` | 64 | 整数 | 精确匹配 |


### 1.3 执行流程

样例同样由 Host 侧数据准备、Kernel 编译与启动、Device 侧计算、Host 侧校验四部分组成，可参考 [执行流程](../basic_mmad/README.md#13-执行流程)一节，下面主要介绍 AIV 侧计算过程。


1. 空间预分配，预先读取向量长度 `n_ele`，在 UB 上分配 `ub_a`、`ub_b`、`ub_c` 三份空间，并绑定为 RowMajor 格式。
2. 数据搬运，启用 MTE2 将 GM 上的 A 和 B 搬入 UB。
3. 加法运算，AIV 核中按 VL 分块循环，每次加载一块 A 和 B 到寄存器，调用 `tla.add` 后将结果输出到目标 UB 位置。
4. 输出搬出，MTE3 将结果 从 UB 写回 GM C，最后流水线排空。

> 当选择 `--use-atomic-add` 时，执行 `basic_vadd_atomic_add`，加和操作在 GM 上完成（启用原子加）
> 当选择 `--use-mutex` 或 `--use-mutex-with` 时，执行 `basic_vadd_mutex` 或 `basic_vadd_mutex_with`， 关于 Mutex 原语的有关内容可参考 [Mutex 说明](../basic_mmad/README.md#32-basic_matmul_mutex) 一节。
---

## 2. 运行样例

要运行本路径下的样例，请参考[环境配置](../../../README.md#2.从零开始：环境与完整指令)一节的有关内容。

### 2.1 命令行参数

```text
basic_vadd.py [-h] [--device DEVICE] [--n N]
              [--dtype {f32,f16,i8,i16,i32}]
              [--block-num BLOCK_DIM]
              [--sentinel SENTINEL]
              [--use-mutex | --use-mutex-with | --use-atomic-add]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | TLA 和 PyTorch 使用的 NPU 设备号。 |
| `--n` | `400` | 向量长度，范围 `[1, VECTOR_ELE]`。 |
| `--block-num` | `-1`（哨兵：按算子类型取满核；本示例为纯 Vector，默认 `vector_core_num`） | 启动 block 数。非 `-1` 时用入参；`-1` 时纯 v 用 AIV 物理核数，cube/mix 用 AIC 物理核数。 |
| `--dtype` | `"f32"` | 数据类型，可选 `"f32"`、`"f16"`、`"i8"`、`"i16"`、`"i32"`。 |
| `--sentinel` | 按 dtype 自适应 | Kernel 启动前写入输出的哨兵值。 |
| `--use-mutex` | 关闭 | 切换到显式 Mutex lock/unlock 同步。 |
| `--use-mutex-with` | 关闭 | 切换到 `with tla.mutex_guard(...)` 同步。 |
| `--use-atomic-add` | 关闭 | 切换到原子加同步（先 copy A→C，再 atomic_add B→C）。 |

> `--use-mutex`、`--use-mutex-with`、`--use-atomic-add` 三者互斥。

### 2.3 调用示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 查看帮助
python examples/end_to_end/basic_vadd/basic_vadd.py --help

# 默认运行（flag 同步，f32，n=400）
python examples/end_to_end/basic_vadd/basic_vadd.py --device 0

# 指定长度和 dtype
python examples/end_to_end/basic_vadd/basic_vadd.py --device 0 --n 256 --dtype f16

# 执行带Mutex同步的样例
python examples/end_to_end/basic_vadd/basic_vadd.py --device 0 --use-mutex
python examples/end_to_end/basic_vadd/basic_vadd.py --device 0 --use-mutex-with

# 执行带原子加操作的样例
python examples/end_to_end/basic_vadd/basic_vadd.py --device 0 --use-atomic-add
```

执行上述测试后，预期输出：

```plain
--- dtype=f32 n=400 ---
passed=True cache_key=<CACHE_KEY>
kernel.o=/path/to/artifacts/runtime-cache/<CACHE_KEY>/kernel.o
```

其中 `passed` 结果为 `True` 或 `False`，表明 NPU 计算结果与 golden 参考值精度校验是否通过；`cache_key` 是编译缓存的哈希值，`kernel.o` 后续的路径即为编译后的二进制路径。
