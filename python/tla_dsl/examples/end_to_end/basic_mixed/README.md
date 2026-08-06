# 基础 Mixed（Cube + Vector）端到端示例

本目录展示 **TLA DSL** 下 CV 融合类算子的基础示例，使用了Ascend950代际新增通路，如 L0C -> UB 和 UB -> L1。

| 文件 | 作用 |
|------|------|
| **`basic_mixed.py`** | **基础 CV 融合示例**（set/wait 同步）：AIC 侧矩阵乘结果计算完成后通过 L0C→UB 通路向 UB 搬移，AIV 上执行加法计算。 |
| **`basic_mixed_fixpipe_nz2dn.py`** | **使用 L0C->UB 通路的基础矩阵乘示例**：AIC 侧矩阵乘计算结果完成后，经由 L0C->UB 通路，向 AIV0 定向搬运，到 GM 上输出为 ColumnMajor 排布 |
| **`basic_mixed_mutex.py`** | **CV 融合示例（Mutex 同步）**：功能与 `basic_mixed.py` 相同，使用 Mutex 原语替代 set/wait 实现流水同步。 |
| **`basic_mixed_ub2l1.py`** | **启用 UB->L1 通路的 CV 融合示例**：AIV 侧从 GM 加载矩阵 A 的 Tile块后再搬运到 L1A 上，不对数据排布格式做任何更改 |
| **`basic_mixed_store_zN.py`** | **启用 UB->L1 通路带排布优化的 CV 融合示例**：在基础的 CV 融合示例基础上，在AIV上额外进行将矩阵 A 转为 zN排布的操作，对于非对齐场景，补充padding数据。 |
| **`basic_mixed_store_zNUnAlign.py`** | **启用 UB->L1 通路带排布优化的 CV 融合示例（zNUnAlign）**：在上述样例优化基础上，针对非对齐场景，M 轴方向上不进行 fractal 分块，搬运时的 stride 按运行时行数做调整。 |

---

## 1. 基本介绍

本目录演示的CV融合算子实现基本的 `A @ B + C` 计算，计算公式如：
$$
\begin{aligned}
D &= A \times B + C \\
D_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j} + C_{i,j}
\end{aligned}
$$

执行流程与 `basic_mmad` 类似（详见[执行流程](../basic_mmad/README.md#13-执行流程)一节）区别在于 CV 融合除 AIC 上的矩阵计算外，还包含 AIV 的运算过程。在本目录归纳示例中，二者之间有两条数据通路：

 - `L0C` -> `UB` 数据通路：矩阵乘结果计算完成后，将其搬出到 UB 上以便后续 AIV 上开展加法运算，不再搬出到 GM上;
 - `UB` -> `L1` 数据通路：从GM 加载 A后搬运到 UB 上，随后通过该通路搬运到 L1 上，此外可以通过 AIV 将其转为 zN 或者 zNUnAlign 排布，有利于提高数据搬运带宽。

> [`basic_mixed_fixpipe_nz2dn`](basic_mixed_fixpipe_nz2dn.py) 实现基本矩阵乘功能。

## 1.1 问题规模与分块

- 逻辑 GEMM：`M × N × K` （默认为 `32`, `32`, `32`，其中 [`store_zN`](basic_mixed_store_zN.py) 和 [`store_zNUnAlign`](basic_mixed_store_zNUnAlign.py) 分别为 `64`, `64`, `128` 和 `60`, `64`, `128`）
- 左矩阵 A 和右矩阵 B 的排布均默认为 `tla.arch.RowMajor`，加和矩阵 C 的排布固定为 `tla.arch.RowMajor`。
- L1 分块：`l1_tm × l1_tn × l1_tk` （默认为 `64`, `64`, `128`，仅针对 [`store_zN`](basic_mixed_store_zN.py) 和 [`store_zNUnAlign`](basic_mixed_store_zNUnAlign.py)，其余直接全载）
- L0 分块：示例中同 L1 分块大小
- AIV 上的 Tile 分块：`tile_m x tile_n` （默认为 `16`, `32`）
- Vector tile：`VECTOR_TILE_M=16, VECTOR_TILE_N=32`，L0C -> UB 通路按 `SPLIT_M` 模式搬运。

## 1.2 AIV 执行流程

本 CV 融合示例中，AIV 承载除矩阵乘外的数据搬运与后处理计算。本目录各示例中 AIV 的任务包括三类：

**类型 1：后处理（Epilogue）**

AIV 对 Cube 侧已完成的矩阵乘结果进行逐元素加法，这一操作在本目录下的所有示例均有体现。

1. 两个 AIV 子核各取输出矩阵的上下半区（ `SPLIT_M` 模式，AIV0 取上半部分， AIV1取下半部分），等待 `cross_core_wait_flag(fix_done)` 确认 AIC 侧已将 `L0C` 结果搬入 `UB`。
2. 启动数据搬运， MTE2 将加数 C 从 GM 加载到 UB 。
3. 启动运算，逐行进行加法运算，结果写出到目的 UB 位置。
4. 数据搬出，MTE3 将UB 上存放的加和结果写回 GM。

**类型 2：数据中转搬运**

AIV 替代 MTE2 完成矩阵 A 的数据搬运，数据通路是由 GM 搬运至 UB， 再启动 UB 至 L1 的搬运，不改变数据排布，对应于示例 [`basic_mixed_ub2l1`](basic_mixed_ub2l1.py)。

1. 两个 Vector 子核各取矩阵 A 的上/下半区（ `SPLIT_M` 模式）。
2. 启动数据搬运，MTE2 将矩阵 A 数据从 GM 加载到 UB 上。
3. 输出搬出到L1A, AIV 侧等待 AIC `cross_core_wait_flag(ub2l1_ready)` 允许搬运，随后通过 MTE3 将 UB 直接 把搬运到 L1 上，做随路转换（L1 目的排布为zN）。
4. 同步收尾，搬运完成后，通过设置 `cross_core_set_flag(ub2l1_done)` 通知 AIC 侧可读取 L1 上的数据。

**类型 3：数据搬运 + 排布转换**

在数据搬运的基础上，在 AIV 上显式完成 `tla.arch.RowMajor` 到 `tla.arch.zN` （或 `tla.arch.zNUnAlign` ） 的转换再写入 L1，不同的点在于：

1. 在 UB 上创建 `tla.arch.zN`（或 `tla.arch.zNUnAlign` ）格式的 `tla.Tensor`，以便后续 stride 计算，二者区别在于：
   - `tla.arch.zN`：使用昇腾亲和的 zN 排布格式，M 轴向上对齐，后续搬运时 `stride` 为编译期常量（隐含 padding 动作）；

   - `tla.arch.zNUnAlign`：仍按照 zN 排布，但 M 轴不进行 fractal 操作，数据搬运时 `stride` 为运行时变量。

2. 启动数据重排，在 AIV 上逐行将加载的矩阵数据写入到目的位置，按 zN 排布格式，
`tla.vec.func(mode="simd")` 内逐行逐列（`vf_row_loops × vf_col_loops`）以 `BlockStore(block_stride=...)` 将 RowMajor chunk 写入 zN dest。
3. 启动数据搬运，MTE3 将 UB zN 写入 L1，再通知 AIC 侧可读取 L1 上的数据。

这一操作流程对应于示例 [`basic_mixed_store_zN`](basic_mixed_store_zN.py) 和 [`basic_mixed_store_zNUnAlign`](basic_mixed_store_zNUnAlign.py)。

## 2. 运行样例

要运行本路径下的样例，请参考[环境配置](../../../README.md#2.从零开始：环境与完整指令)一节的有关内容。

### 2.1 命令行参数

```text
basic_mixed.py [-h] [--device DEVICE] [--m M] [--n N] [--k K] 
               [--layout-a {row,col}] [--layout-b {row,col}] 
               [--block-dim BLOCK_DIM] 
               [--sentinel SENTINEL]
               [--cache-dir CACHE_DIR] 
               [--force-recompile] 
               [--no-cache]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | TLA 和 PyTorch 使用的 NPU 设备号。 |
| `--m` / `--n` / `--k` | `32`, `32`, `32` （基础版本[basic_mixed](basic_mixed.py)） | 自定义一组运行时矩阵尺寸。 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | A、B 的 GM 布局，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--block-dim` | `-1`（哨兵值，后续替换为`tla.get_aicore_num(device)`） | 表示使用 NPU 所启用的 AI Core 核数 |
| `--sentinel` | `-9.0` | Kernel 启动前写入 C 的哨兵值，用于暴露未写回的元素。 |
| `--cache-dir` | `artifacts/runtime-cache` | 编译缓存目录 |
| `--force-recompile` | `False` | 忽略已有缓存并强制重新编译。 |
| `--no-cache` | `False` | 禁用编译缓存。 |

### 2.2 调用示例

```bash
cd python/tla_dsl

# 默认运行
python examples/end_to_end/basic_mixed/basic_mixed.py

# 测试带Mutex同步
python examples/end_to_end/basic_mixed/basic_mixed_mutex.py

# 指定问题规模（由于 basic_mixed.py 实际全载，因此需保证问题尺寸符合硬件限制L1, L0A, L0B的大小）
python examples/end_to_end/basic_mixed/basic_mixed.py --m 32 --n 48 --k 64

###################
# 测试采用 UB -> L1 通路的样例
python examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py

###################
# 测试执行带排布转换的样例
python examples/end_to_end/basic_mixed/basic_mixed_store_zN.py
python examples/end_to_end/basic_mixed/basic_mixed_store_zNUnAlign.py

# 测试非对齐场景下的效果
python examples/end_to_end/basic_mixed/basic_mixed_store_zNUnAlign.py --m 63 --n 64 --k 128
```

成功运行输出如下。

```text
--- mnk=(32,32,32) ---
passed=True cache_key=<CACHE_KEY>
kernel.o=<CACHE_DIR>/<CACHE_KEY>/kernel.o
```

其中 `<CACHE_KEY>` 是编译缓存的哈希值。
- `passed=True` 表示输出精度校验。
- `kernel.o=...` 表示当前 Kernel 二进制路径。
- 如算子实现存在问题会打印相关报错信息。