# PlanarComplexMatmul

> **注意**：本样例位于 `experimental/` 目录下，如需编译运行，请先将样例目录拷贝至 `examples/` 下，并在 `examples/CMakeLists.txt` 中添加样例名称 `planar_complex_matmul`。

本文档用于说明 `experimental/matmul/planar_complex_matmul` 平面复数矩阵乘算子示例所依赖的 Catlass GEMM 模板库能力、外部接口、分层设计方案。

## 1. 功能说明

 - 算子功能：完成平面复数矩阵乘计算。复数矩阵以实部、虚部分离的 planar complex 形式输入，样例输出实部与虚部两路结果。
 - 计算公式：

$$
\begin{aligned}
    A &= A_{real} + iA_{imag} \\
    B &= B_{real} + iB_{imag} \\
    C &= A \times B \\
    C_{real} &= A_{real} \times B_{real} - A_{imag} \times B_{imag} \\
    C_{imag} &= A_{real} \times B_{imag} + A_{imag} \times B_{real}
    \end{aligned}
$$

  其中 `A_real`、`A_imag` 是形如 `(m, k)` 的左矩阵实部和虚部，`B_real`、`B_imag` 是形如 `(k, n)` 的右矩阵实部和虚部，`C_real`、`C_imag` 是形如 `(m, n)` 的输出矩阵实部和虚部。


## 2. 参数说明

以下是本样例可执行文件的运行参数：

| 参数名 | 描述 | 约束 |
| ----- | -------- | ------ |
| `m` | 复数矩阵乘中左矩阵 A 的行 | 正整数 |
| `n` | 复数矩阵乘中右矩阵 B 的列 | 正整数 |
| `k` | 复数矩阵乘中左矩阵 A 的列，也即右矩阵 B 的行 | 正整数 |
| `deviceId` | 使用的 NPU 卡 ID（默认 0） | 在设备 NPU 有效范围内 |
| `--datapath` | 输入数据与输出数据目录 | 可选；未指定时仅随机生成输入并统计 kernel 耗时 |

PlanarComplexMatmul 所涉及的关键模板参数如下：

| 模板参数 | 说明 | 当前样例取值 |
| ----- | -------- | -------------- |
| `ElementA` | 左矩阵实部/虚部的数据类型 | `half` |
| `ElementB` | 右矩阵实部/虚部的数据类型 | `half` |
| `ElementC` | 输出矩阵实部/虚部的数据类型 | `float` |
| `LayoutA` | 左矩阵排布方式 | `layout::RowMajor` |
| `LayoutB` | 右矩阵排布方式 | `layout::ColumnMajor` |
| `LayoutC` | 输出矩阵排布方式 | `layout::RowMajor` |
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |
| `L1TileShape` | L1 tile 形状 | `GemmShape<128, 256, 256>` |
| `L0TileShape` | L0 tile 形状 | `GemmShape<128, 256, 64>` |
| `DispatchPolicy` (Four-Pass) | 4-pass 路径的 MMAD 调度策略 | `Gemm::MmadPingpong<ArchTag, true>` |
| `DispatchPolicy` (Fused) | Fused 路径的 MMAD 调度策略 | `Gemm::MmadPlanarComplexFused<ArchTag, true>` |

## 3. 约束说明

 - 输入矩阵实部和虚部均为 fp16，输出实部和虚部均为 fp32。
 - `B_real`、`B_imag` 在设备侧按 `layout::ColumnMajor` 读取；使用 `gen_data_compare.py` 校验时脚本会将 NumPy 生成的 B 矩阵转置后写入输入文件。
 - 样例根据 shape 在 Host 侧选择执行路径：当 `k >= 6000` 且每个 AIC core 分到的 MN tile 数不少于 3 时选择 Four-Pass，否则选择 Fused。
 - 样例根据 `m` 与 `n` 的关系选择对 `A_imag` 或 `B_imag` 取负后写入 workspace，用于计算 `C_real` 中的负号项。

## 4. 具体设计方案

### 4.1 Host 层

#### 4.1.1 参数解析

`planar_complex_matmul` 命令执行参数：

```text
m, n, k, [device_id], [--datapath DATA_PATH]
```

Host 层除了常规的 GEMM shape，还需要根据m, n, k,动态选择路径：

1. 选择执行路径（Four-Pass 或 Fused）；
2. 选择对 `A_imag` 还是 `B_imag` 取负（`NEGATE_A`）；
3. 选择 block swizzle 方向（`m >= n` 时行优先扫描，`m < n` 时列优先扫描）。

#### 4.1.2 路径选择

Host 侧基于 cost-model 选择 kernel 变体：

```text
K >= 6000 AND per_core >= 3 tiles  -> Four-Pass
否则                                -> Fused
```

- `coreLoops = CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N)`
- `perCore = coreLoops / aicCoreNum`

Four-Pass 每 pass 只读写一路 C，适合 K 大、per-core tile 多的场景；Fused 单遍完成，适合 K 小或 per-core tile 少的场景。

#### 4.1.3 NEGATE_A 选择

`C_real = A_real * B_real - A_imag * B_imag` 中的负号项通过预先对 `A_imag` 或 `B_imag` 取负实现：

- `m < n`：对 `A_imag` 取负（`NEGATE_A=true`），workspace 尺寸为 `M*K*half`；
- `m >= n`：对 `B_imag` 取负（`NEGATE_A=false`），workspace 尺寸为 `K*N*half`。

选择较小的一侧可以减少 workspace 开销和 AIV 取负工作量。

#### 4.1.4 device memory 与 copy

设备内存分配：

```text
deviceAReal, deviceAImag          // 输入 A 实部/虚部
deviceBReal, deviceBImag          // 输入 B 实部/虚部
deviceCReal, deviceCImag          // 输出 C 实部/虚部
deviceWorkspace                   // AIV 取负后的 signed 工作区
```

Host 将 6 个输入指针 + 2 个输出指针 + workspace 传入 `DeviceGemm::Arguments`，由 kernel 层根据 `NEGATE_A` 决定 `ptrAImagSigned`/`ptrBImagSigned` 指向 workspace 还是原始指针。

### 4.2 Kernel 层

#### 4.2.1 统一 kernel 模板

```cpp
Gemm::Kernel::PlanarComplexGemm<
    USE_FOUR_PASS,
    NEGATE_A,
    BlockMmadFourPass,
    BlockMmadFused,
    BlockScheduler>
```

kernel 通过 `USE_FOUR_PASS` 编译期开关选择 block 类型，未选中的路径以 `void` 传入，不会被实例化。

#### 4.2.2 AIV 取负预处理

`NegateMatrixAiv` 是 kernel 内的 AIV 组件：

- 输入：原始 `A_imag`（或 `B_imag`）GM tensor；
- 输出：取负后的 signed GM workspace；
- 单缓冲：src/dst 各占 UB 的一半，compute（Muls）相对 GM 带宽可忽略，双缓冲无收益。

AIV 路径在 Mix kernel prologue 阶段执行取负，AIC 路径通过 `ptrAImagSigned`/`ptrBImagSigned` 消费结果。

#### 4.2.3 Four-Pass 编排（`USE_FOUR_PASS=true`）

4 次顺序 `BlockMmad` 调用，fixpipe atomic-add 把交叉项累加回 C：

```text
pass1: C_real  = A_real * B_real              (无 atomic)
pass2: C_real += signed_imag_cross_term    		(atomic add)
pass3: C_imag  = A_imag * B_real              (无 atomic)
pass4: C_imag += A_real * B_imag              (atomic add)
```

其中 pass 2 的 `signed_imag_cross_term` 为：

\- `NEGATE_A = true`：`A_imag_signed * B_imag`

\- `NEGATE_A = false`：`A_imag * B_imag_signed`

#### 4.2.4 Fused 编排（`USE_FOUR_PASS=false`）

单遍 K-loop，C_real 与 C_imag 分时复用同一块 L0C：

```text
Stage 1 (C_real): 2K 个子迭代
  even sub: A_real * B_real           -> l0C (initC)
  odd  sub: AImagSigned * BImagSigned -> l0C (accumulate)
  ... FixPipe l0C -> GM_C_real

Stage 2 (C_imag): 2K 个子迭代
  even sub: A_imag * B_real -> l0C (initC)
  odd  sub: A_real * B_imag -> l0C (accumulate)
  ... FixPipe l0C -> GM_C_imag
```

C_real 的 FixPipe 与 C_imag 首个子迭代的 MTE2（GM->L1）重叠，隐藏 fixpipe 延迟。

### 4.3 Block 层

#### 4.3.1 Four-Pass block

Four-Pass 复用通用 BlockMmadTla。kernel 层负责 4 次调用的编排和 atomic-add。

#### 4.3.2 Fused block

Fused 使用 `BlockMmadTla` 针对 `MmadPlanarComplexFused` policy 的偏特化（`block_mmad_planar_complex_fused_tla.hpp`）：

1. **4 路输入 tensor**：`A_real`、（`A_imag` or `A_imag_signed`)、`B_real`、(`B_imag` or `B_imag_signed`)。block 对 `NEGATE_A` 无感知，始终从 Signed 槽位读 C_real 交叉项，从原始槽位读 C_imag。
2. **L1 4 槽 K-pingpong**：`[A_K0 | A_K1 | B_K0 | B_K1]`，A/B 槽位通用，GM 来源按子迭代交替。
3. **L0A/L0B 双缓冲 pingpong**：重叠 L1->L0 搬运与 Cube MMAD。
4. **L0C 单缓冲**：C_real FixPipe 完成后 C_imag 才开始，分时复用。
5. **K-shuffle**：`ENABLE_SHUFFLE_K=true` 时按 `GetBlockIdx()` 偏移 K tile 顺序，分散 L2 访问热点。

### 4.4 DispatchPolicy 设计

#### 4.4.1 Fused: `MmadPlanarComplexFused`

模板参数：

| 参数 | 说明 | 当前样例取值 |
|---|---|---|
| `ArchTag` | 目标架构 | `Arch::AtlasA2` |
| `ENABLE_SHUFFLE_K` | 是否启用 K 维 shuffle | `true` |

## 5. 空间分配

### 5.1 Tile Shape 设计

```cpp
using L1TileShape = GemmShape<128, 256, 256>;  // M, N, K
using L0TileShape = GemmShape<128, 256, 64>;   // M, N, K
```

### 5.2 存储空间计算

数据类型：`ElementA = ElementB = half`（2 字节），`ElementAccumulator = float`（4 字节），pingpong `STAGES = 2`。硬件 buffer 容量：`L1 = 512 KB`、`L0A = 64 KB`、`L0B = 64 KB`、`L0C = 128 KB`。

**Fused 路径 `<128, 256, 256> / <128, 256, 64>`：**

| Buffer | 单级尺寸 | 计算 | 单级容量 | ×2 stages | 硬件上限 | 利用率 |
|---|---|---|---|---|---|---|
| L1A | 128 × 256 | × 2B | 64 KB | 128 KB | - | - |
| L1B | 256 × 256 | × 2B | 128 KB | 256 KB | - | - |
| **L1 合计** | - | - | 192 KB | **384 KB** | 512 KB | **75%** |
| L0A | 128 × 64 | × 2B | 16 KB | **32 KB** | 64 KB | **50%** |
| L0B | 64 × 256 | × 2B | 32 KB | **64 KB** | 64 KB | **100%** |
| L0C | 128 × 256 | × 4B | 128 KB | - (单缓冲) | 128 KB | **100%** |

L0C 单缓冲（C_real/C_imag 分时复用）打满 128 KB；L0B 双缓冲打满 64 KB；L1 合计 384/512 KB（75%）。L0A 利用率 50%，留出余量给对角 tile 等特殊处理。

**GM workspace（AIV 取负）：**

- `NEGATE_A=true`（`m < n`）：`M × K × 2B`（对 `A_imag` 取负）
- `NEGATE_A=false`（`m >= n`）：`K × N × 2B`（对 `B_imag` 取负）

Host 选择较小的一侧取负以减少 workspace 开销。

## 6. 代码组织

```
├── experimental
│   └── matmul
│       └── planar_complex_matmul
│           ├── CMakeLists.txt              # CMake 编译文件
│           ├── README.md                   # 本文档
│           ├── gen_data_compare.py         # NumPy golden 数据生成与精度比对脚本
│           └── planar_complex_matmul.cpp   # 样例主文件（Host 层入口）
└── include
    └── catlass
        ├── gemm
        │   ├── block
        │   │   └── block_mmad_planar_complex_fused_tla.hpp
        │   │       # BlockMmadTla 针对 MmadPlanarComplexFused 的偏特化
        │   │       # 实现 Fused 路径的双 stage K-loop 与 L0C 分时复用
        │   ├── device
        │   │   └── (复用主仓库 device_gemm.hpp)
        │   │       # Device 层薄封装：参数透传 + KernelAdapter 启动
        │   └── kernel
        │       └── planar_complex_gemm_tla.hpp
        │           # PlanarComplexGemm 统一 kernel（Four-Pass / Fused 编排）
        │           # 含 NegateMatrixAiv AIV 取负预处理组件
        └── dispatch_policy.hpp
            # MmadPlanarComplexFused policy（仅 ENABLE_SHUFFLE_K 参数）
```

## 7. 使用示例

1. 编译样例代码，并生成相应的算子可执行文件。

```
bash scripts/build.sh planar_complex_matmul
```

2. 切换到可执行文件的编译目录 `output/bin` 下，执行算子样例程序。该方式随机生成输入数据，只输出 kernel 调度路径与平均耗时，不做精度比对。

```
cd output/bin
./planar_complex_matmul 256 512 1024 0
```

• 256：矩阵 m 轴

• 512：矩阵 n 轴

• 1024：矩阵 k 轴

• 0：Device ID，可选，默认为 0

执行结果中包含如下信息，说明样例执行成功。

```
PlanarComplexGemm dispatch: M=256 N=512 K=1024 ... -> Fused ...
PlanarComplexGemm: M=256 N=512 K=1024 variant=Fused gemm=... ms (20 iters)
No --datapath provided, skipping validation.
```

3. 使用 `gen_data_compare.py` 生成输入数据、运行 NPU 可执行文件并与 NumPy golden 结果进行比对。脚本默认从仓库根目录自动定位 `output/bin/planar_complex_matmul`，默认在当前目录下生成 `data`、`golden` 目录并在结束后删除；如需指定保存路径可使用 `--save_path`，如需保留可指定 `--clean false`。

```
python examples/planar_complex_matmul/gen_data_compare.py 256 512 1024
```

执行结果如下，说明精度比对成功。

```
Data generated: M=256, N=512, K=1024, BLAS threads=8
------计算npu------
------ 计算相对误差 -----
Precision metric: ...
------ 开始比较 ------
比较结果：Compare success
```
